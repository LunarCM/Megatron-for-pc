# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# A minimal Megatron-Core GPT training loop that trains on the real
# SimpleStories dataset (simplestories-4k) instead of the mock data used in
# `run_simple_mcore_train_loop.py`.
#
# Usage (single GPU, recommended for the ~12 GB card the defaults target):
#   cd /home/lxk/PycharmProjects/My_megatron
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#     torchrun --nproc_per_node=1 examples/run_simplestories_train_loop.py
#
# A checkpoint + validation + samples run every SIMPLESTORIES_CKPT_INTERVAL
# steps (default 5000) into examples/ckpt/step_XXXXX, keeping only the 3 most
# recent. To evaluate any saved checkpoint later (e.g. after an interrupted
# run), use the standalone script:
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#     torchrun --nproc_per_node=1 examples/evaluate_checkpoint.py \
#       --checkpoint examples/ckpt/step_5000
#
# Environment variables (all optional):
#   SIMPLESTORIES_DATA_DIR : directory containing the .bin/.idx files
#                            (defaults to the repo-local simplestories-4k-megatron)
#   SIMPLESTORIES_TOKENIZER_DIR : directory containing vocab.json/merges.txt
#                            (defaults to <DATA_DIR>/tokenizer)
#   SIMPLESTORIES_SEQUENCE_LENGTH : model sequence length (defaults to 1024;
#                            attention memory is O(seq^2), so 2048 needs ~1 GiB
#                            more per attention step and OOMs on a 12 GB GPU)
#   SIMPLESTORIES_MICRO_BATCH_SIZE : micro batch size (defaults to 2;
#                            raise to 4 only if you have VRAM headroom)
#   SIMPLESTORIES_NUM_SAMPLES_TRAIN : number of training samples (defaults to 1000)
#   SIMPLESTORIES_NUM_EPOCHS : number of training epochs (defaults to 3)
#   SIMPLESTORIES_LOG_INTERVAL : print loss every N steps (defaults to 100)
#   SIMPLESTORIES_LEARNING_RATE : Adam learning rate (defaults to 1e-4;
#                            raise up to 3e-4 only with clipping+stable loss)
#   SIMPLESTORIES_GRAD_CLIP : global gradient L2-norm cap, 0 disables (defaults to 1.0)
#   SIMPLESTORIES_LR_WARMUP_STEPS : linear LR warmup steps, 0 disables (defaults to 2000)
#   SIMPLESTORIES_CKPT_DIR : directory for checkpoints (defaults to <script>/ckpt)
#   SIMPLESTORIES_LOG_DIR : directory for the timestamped local training log
#                            (defaults to <script>/logs, e.g. train_<ts>.log;
#                            every printed line is also written there)
#   SIMPLESTORIES_CKPT_INTERVAL : save a checkpoint + eval every N steps (defaults to 5000)
#   SIMPLESTORIES_CKPT_KEEP : keep only the N most recent step_* checkpoints,
#                            deleting older ones (defaults to 3)
#   SIMPLESTORIES_EVAL_NUM_BATCHES : number of validation batches per eval (defaults to 200)
#   SIMPLESTORIES_NUM_SAMPLES : generated samples per eval, 0 disables (defaults to 2)
#   SIMPLESTORIES_SAMPLE_PROMPT_LENGTH : prompt length in tokens (defaults to 32)
#   SIMPLESTORIES_SAMPLE_MAX_NEW_TOKENS : generated tokens per sample (defaults to 64)
#   SIMPLESTORIES_SAMPLE_TEMPERATURE : sampling temperature (defaults to 0.8)
#   SIMPLESTORIES_NUM_LAYERS : number of transformer layers (defaults to 14;
#                            raise to 16/20 to use more VRAM, watch nvidia-smi)
#   SIMPLESTORIES_HIDDEN_SIZE : hidden size (defaults to 1024)
#   SIMPLESTORIES_NUM_ATTENTION_HEADS : number of attention heads (defaults to 16)
#   SIMPLESTORIES_RECOMPUTE : activation recomputation: "full" (default) is the
#                            stable choice for ~12 GB GPUs; "selective" is faster
#                            but uses more VRAM; "none" uses the most VRAM
#
# Defaults are tuned for a ~12 GB GPU. Measured: seq 2048 OOMs (the attention
# step alone is ~1 GiB on top of ~9.5 GB steady state), so the default uses
# seq 1024 which cuts that transient to ~256 MiB. If you hit OOM, reduce (in
# order) MICRO_BATCH_SIZE, SEQUENCE_LENGTH, NUM_LAYERS, or HIDDEN_SIZE; to use
# more memory, raise NUM_LAYERS a step at a time while watching nvidia-smi.

import atexit
import gc
import math
import os
import shutil
import sys
from datetime import datetime

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from megatron.core import parallel_state
from megatron.core import dist_checkpointing
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.datasets.utils import compile_helpers
from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, GPTDataset
from megatron.core.distributed import DistributedDataParallel
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.tokenizers import MegatronTokenizer

# NOTE: attention memory grows as O(batch * seq_len^2), and stored activations
# grow with layers * batch * seq_len * hidden_size. Measured on the target GPU:
# seq 2048 / batch 4 OOMs (attention transient alone is ~1 GiB on top of
# ~9.5 GB steady state). seq 1024 is the stable default; scale up gradually.
_SEQUENCE_LENGTH: int = int(os.environ.get("SIMPLESTORIES_SEQUENCE_LENGTH", "1024"))
_MICRO_BATCH_SIZE: int = int(os.environ.get("SIMPLESTORIES_MICRO_BATCH_SIZE", "2"))
_NUM_SAMPLES_TRAIN: int = int(os.environ.get("SIMPLESTORIES_NUM_SAMPLES_TRAIN", "1000"))
_NUM_EPOCHS: int = int(os.environ.get("SIMPLESTORIES_NUM_EPOCHS", "3"))
_LOG_INTERVAL: int = int(os.environ.get("SIMPLESTORIES_LOG_INTERVAL", "100"))
# lr=1e-4 with clipping+warmup is the stable default; 3e-4 was observed to
# diverge at batch=2 around step ~5000 (loss rose from 3.9 to 4.9, grad_norm 20-100).
_LEARNING_RATE: float = float(os.environ.get("SIMPLESTORIES_LEARNING_RATE", "1e-4"))
# Gradient clipping: caps the global grad L2 norm each step. Without it, a
# few large-loss batches can push grads from ~1 to 20-100 and the run
# diverges (observed at lr=3e-4 / batch=2 around step ~5000).
_GRAD_CLIP: float = float(os.environ.get("SIMPLESTORIES_GRAD_CLIP", "1.0"))
# Linear LR warmup over the first N steps (0 disables).
_LR_WARMUP_STEPS: int = int(os.environ.get("SIMPLESTORIES_LR_WARMUP_STEPS", "2000"))
# Model size (tune via env vars; 14 layers + seq 1024 is the stable default
# for a ~12 GB GPU).
_NUM_LAYERS: int = int(os.environ.get("SIMPLESTORIES_NUM_LAYERS", "14"))
_HIDDEN_SIZE: int = int(os.environ.get("SIMPLESTORIES_HIDDEN_SIZE", "1024"))
_NUM_ATTENTION_HEADS: int = int(os.environ.get("SIMPLESTORIES_NUM_ATTENTION_HEADS", "16"))

# Activation recomputation: "full" recomputes every transformer layer during
# backward, which drastically cuts activation memory (stable default for a
# ~12 GB GPU). "selective" stores more activations (faster, but uses more
# VRAM); switch only if you have headroom and want speed.
_RECOMPUTE: str = os.environ.get("SIMPLESTORIES_RECOMPUTE", "full").lower()
if _RECOMPUTE not in ("full", "selective", "none"):
    _RECOMPUTE = "full"

# Cache directory for dataset indices (kept out of the data dir).
_SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR: str = os.environ.get(
    "SIMPLESTORIES_CACHE_DIR", os.path.join(_SCRIPT_DIR, ".cache", "dataset")
)

# Local log directory: every line printed during the run (loss, grad_norm,
# eval loss, samples, checkpoint messages) is also written to a
# timestamped file here via a stdout/stderr tee.
_LOG_DIR: str = os.environ.get(
    "SIMPLESTORIES_LOG_DIR", os.path.join(_SCRIPT_DIR, "logs")
)

# Checkpointing, validation and sample generation.
_CKPT_DIR: str = os.environ.get(
    "SIMPLESTORIES_CKPT_DIR", os.path.join(_SCRIPT_DIR, "ckpt")
)
_CKPT_INTERVAL: int = int(os.environ.get("SIMPLESTORIES_CKPT_INTERVAL", "5000"))
_CKPT_KEEP: int = int(os.environ.get("SIMPLESTORIES_CKPT_KEEP", "3"))
_EVAL_NUM_BATCHES: int = int(os.environ.get("SIMPLESTORIES_EVAL_NUM_BATCHES", "200"))
_NUM_SAMPLES: int = int(os.environ.get("SIMPLESTORIES_NUM_SAMPLES", "2"))
_SAMPLE_PROMPT_LENGTH: int = int(
    os.environ.get("SIMPLESTORIES_SAMPLE_PROMPT_LENGTH", "32")
)
_SAMPLE_MAX_NEW_TOKENS: int = int(
    os.environ.get("SIMPLESTORIES_SAMPLE_MAX_NEW_TOKENS", "64")
)
_SAMPLE_TEMPERATURE: float = float(
    os.environ.get("SIMPLESTORIES_SAMPLE_TEMPERATURE", "0.8")
)

# Paths to the SimpleStories (4k vocab) dataset and its BPE tokenizer.
_DEFAULT_DATA_DIR: str = "/home/lxk/PycharmProjects/My_megatron/simplestories-4k-megatron"
_DATA_DIR: str = os.environ.get("SIMPLESTORIES_DATA_DIR", _DEFAULT_DATA_DIR)
_TOKENIZER_DIR: str = os.environ.get(
    "SIMPLESTORIES_TOKENIZER_DIR", os.path.join(_DATA_DIR, "tokenizer")
)
_TRAIN_DATA_PREFIX: str = os.path.join(_DATA_DIR, "simplestories_train_text_document")
_TEST_DATA_PREFIX: str = os.path.join(_DATA_DIR, "simplestories_test_text_document")


def initialize_distributed(
    tensor_model_parallel_size: int = 1, pipeline_model_parallel_size: int = 1
) -> None:
    """
    Set up torch.distributed and Megatron-Core model parallel groups.

    Args:
        tensor_model_parallel_size (int): Number of GPUs to use for tensor model parallelism.
        pipeline_model_parallel_size (int): Number of GPUs to use for pipeline model parallelism.
    """
    parallel_state.destroy_model_parallel()

    # Torch setup for distributed training
    rank: int = int(os.environ["RANK"])
    world_size: int = int(os.environ["WORLD_SIZE"])
    local_rank: int = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size
    )

    # Megatron core distributed training initialization
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size, pipeline_model_parallel_size
    )


def get_tokenizer():
    """
    Load the SimpleStories 4k-vocab BPE tokenizer through HuggingFace.

    The tokenizer is a GPT2-style BPE tokenizer whose vocabulary lives in
    `<DATA_DIR>/tokenizer/` (standard HF names: vocab.json + merges.txt +
    tokenizer_config.json). The `<|endoftext|>` special token (id 4095) is
    used as the end-of-document token.
    """
    return MegatronTokenizer.from_pretrained(
        tokenizer_path=_TOKENIZER_DIR,
        metadata_path={"library": "huggingface"},
        use_fast=False,
    )


def model_provider(vocab_size: int) -> GPTModel:
    """
    Construct a GPT model for training on SimpleStories.

    The vocabulary size must match the SimpleStories tokenizer (4097), unlike
    the mock example which used a hard-coded vocab size of 100.

    Args:
        vocab_size (int): The tokenizer vocabulary size.

    Returns:
        GPTModel: A GPT model instance with _NUM_LAYERS layers.
    """
    transformer_config: TransformerConfig = TransformerConfig(
        num_layers=_NUM_LAYERS,
        hidden_size=_HIDDEN_SIZE,
        num_attention_heads=_NUM_ATTENTION_HEADS,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        # Recompute core attention during backward instead of storing all
        # activation tensors. This trades a little compute for a large reduction
        # in memory, letting a bigger model fit stably on the GPU.
        recompute_granularity=(None if _RECOMPUTE == "none" else _RECOMPUTE),
        recompute_method=("block" if _RECOMPUTE == "full" else None),
        recompute_num_layers=(1 if _RECOMPUTE == "full" else None),
    )

    gpt_model: GPTModel = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=vocab_size,
        max_sequence_length=_SEQUENCE_LENGTH,
    )

    return gpt_model


class _EpochCyclingIterator:
    """DataLoader wrapper that never exhausts: after one epoch it re-creates the
    DataLoader iterator (reshuffling the data) and starts the next epoch.

    This lets the training loop run multiple epochs over the same dataset, which
    mirrors how real Megatron training consumes data.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        self.dataloader = dataloader
        self.iterator: Iterator = iter(dataloader)
        self.epoch: int = 0

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self.iterator = iter(self.dataloader)
            return next(self.iterator)


def get_train_data_iterator(tokenizer) -> Tuple[_EpochCyclingIterator, int]:
    """
    Initialize and return an iterator over the training dataset for the GPT model.

    This function builds a real Megatron GPTDataset from the pre-tokenized
    SimpleStories `.bin` / `.idx` files (train and test splits), and returns a
    DataLoader iterator over the training split.

    Args:
        tokenizer: The tokenizer to use for the dataset (needed for the eod id).

    Returns:
        Tuple[_EpochCyclingIterator, int, DataLoader]: An iterator that yields
            training batches (cycling across epochs), the number of training
            samples, and a validation DataLoader built from the test split.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
    else:
        compile_helpers()

    config: GPTDatasetConfig = GPTDatasetConfig(
        random_seed=0,
        sequence_length=_SEQUENCE_LENGTH,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=tokenizer,
        path_to_cache=_CACHE_DIR,
        # Each split uses its own pre-tokenized data file:
        # [train_prefix, valid_prefix, test_prefix]
        blend_per_split=[
            ([_TRAIN_DATA_PREFIX], None),
            ([_TEST_DATA_PREFIX], None),
            None,
        ],
    )

    # sizes = [train, valid, test]; None means "one full epoch".
    datasets = BlendedMegatronDatasetBuilder(
        GPTDataset, [_NUM_SAMPLES_TRAIN, None, None], lambda: True, config
    ).build()

    train_dataset = datasets[0]
    train_dataloader: DataLoader = DataLoader(
        train_dataset, batch_size=_MICRO_BATCH_SIZE, shuffle=True
    )

    # Validation split uses the pre-tokenized test file (datasets[1]).
    eval_dataloader: DataLoader = DataLoader(
        datasets[1], batch_size=_MICRO_BATCH_SIZE, shuffle=False
    )

    return _EpochCyclingIterator(train_dataloader), len(train_dataset), eval_dataloader


def get_eval_dataloader(tokenizer) -> DataLoader:
    """
    Build a validation DataLoader from the test split only.

    Unlike get_train_data_iterator(), this does not build the (large) training
    split, so it is cheap enough to run standalone on any saved checkpoint.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
    else:
        compile_helpers()

    config: GPTDatasetConfig = GPTDatasetConfig(
        random_seed=0,
        sequence_length=_SEQUENCE_LENGTH,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=tokenizer,
        path_to_cache=_CACHE_DIR,
        # Only build the validation (test) split.
        blend_per_split=[None, ([_TEST_DATA_PREFIX], None), None],
    )
    datasets = BlendedMegatronDatasetBuilder(
        GPTDataset, [0, None, 0], lambda: True, config
    ).build()

    return DataLoader(datasets[1], batch_size=_MICRO_BATCH_SIZE, shuffle=False)


def forward_step_func(
    data_iterator: Iterator, model: torch.nn.Module
) -> Tuple[torch.Tensor, Callable[[torch.Tensor], Dict[str, torch.Tensor]]]:
    """
    Forward step for the Megatron pipeline schedule.

    Args:
        data_iterator (Iterator): Iterator over the training data.
        model (torch.nn.Module): The model.

    Returns:
        Tuple[torch.Tensor, Callable[[torch.Tensor], Dict[str, torch.Tensor]]]:
            The output tensor and a loss function.
    """

    def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
        """
        Compute the masked cross-entropy loss.

        Args:
            loss_mask (torch.Tensor): Mask to zero-out loss on non-target tokens.
            output_tensor (torch.Tensor): The model output.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: The loss and a dict of losses.
        """
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

        # If pipeline parallel, loss computation is done only in last stage.
        return loss, {"lm loss": loss}

    data: Dict[str, torch.Tensor] = next(data_iterator)
    tokens: torch.Tensor = data["tokens"].to(device)
    # Pass attention_mask=None: with AttnMaskType.causal the model generates the
    # proper bool causal mask internally. (The GPTDataset float tril mask would
    # both trip torch's bool-only masked_fill and, in the torch softmax fallback,
    # mask the wrong triangle.)
    attention_mask: Optional[torch.Tensor] = None
    position_ids: torch.Tensor = data["position_ids"].to(device)
    labels: torch.Tensor = data["labels"].to(device)
    loss_mask: torch.Tensor = data["loss_mask"].to(device)

    output_tensor: torch.Tensor = model(
        tokens, position_ids, attention_mask, labels=labels
    )

    return output_tensor, partial(loss_func, loss_mask)


def _extract_losses(losses_reduced) -> Dict[str, float]:
    """
    Convert the value returned by forward_backward_func() into a dict of floats.

    In current Megatron-Core, forward_backward_func() returns forward_data_store,
    a *list* with one entry per microbatch, each entry being the dict returned by
    the loss function (e.g. {"lm loss": tensor}). Older versions returned the dict
    directly. Average across microbatches and float() each entry so callers can
    always index with the metric name.
    """
    if isinstance(losses_reduced, dict):
        return {k: float(v.detach()) for k, v in losses_reduced.items()}
    out: Dict[str, float] = {}
    n: int = max(len(losses_reduced), 1)
    for entry in losses_reduced:
        for k, v in entry.items():
            out[k] = out.get(k, 0.0) + float(v.detach()) / n
    return out


@torch.no_grad()
def _gpu_mem_str() -> str:
    """Short current/peak GPU memory usage string for diagnostics."""
    if not torch.cuda.is_available():
        return "gpu_mem: n/a"
    cur: float = torch.cuda.memory_allocated() / 1024 ** 2
    reserved: float = torch.cuda.memory_reserved() / 1024 ** 2
    peak: float = torch.cuda.max_memory_allocated() / 1024 ** 2
    return (
        f"gpu_mem alloc={cur:.0f}MiB reserved={reserved:.0f}MiB "
        f"peak_alloc={peak:.0f}MiB"
    )


@torch.no_grad()
def evaluate_loss(
    gpt_model: torch.nn.Module,
    eval_dataloader: DataLoader,
    forward_backward_func: Callable[..., Dict[str, Any]],
    num_batches: Optional[int] = None,
) -> float:
    """
    Evaluate the average LM loss over a fixed number of validation batches.

    Uses forward_only=True so no gradients are computed, and eval mode disables
    dropout. num_batches overrides _EVAL_NUM_BATCHES when provided.
    """
    gpt_model.eval()
    eval_iterator: Iterator = iter(eval_dataloader)
    num_batches: int = min(
        num_batches if num_batches is not None else _EVAL_NUM_BATCHES,
        len(eval_dataloader),
    )
    losses: List[float] = []
    for _ in range(num_batches):
        losses_reduced: Dict[str, Any] = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=eval_iterator,
            model=gpt_model,
            num_microbatches=1,
            seq_length=_SEQUENCE_LENGTH,
            micro_batch_size=_MICRO_BATCH_SIZE,
            decoder_seq_length=_SEQUENCE_LENGTH,
            forward_only=True,
        )
        losses.append(_extract_losses(losses_reduced)["lm loss"])
    gpt_model.train()
    return sum(losses) / max(len(losses), 1)


def get_sample_prompts(
    eval_dataloader: DataLoader, num_prompts: int, prompt_length: int
) -> List[torch.Tensor]:
    """
    Pull a few short prompt sequences from the validation set for generation.
    """
    prompts: List[torch.Tensor] = []
    for batch in eval_dataloader:
        tokens: torch.Tensor = batch["tokens"]
        for i in range(tokens.shape[0]):
            prompts.append(tokens[i, :prompt_length])
            if len(prompts) >= num_prompts:
                return prompts
    return prompts


@torch.no_grad()
def generate_samples(
    gpt_model: torch.nn.Module,
    tokenizer,
    prompts: List[torch.Tensor],
    max_new_tokens: int,
    temperature: float,
) -> List[str]:
    """
    Autoregressively generate continuations for each prompt and return text.

    Sampling is done from the softmax over the last position with temperature.
    Note: no KV cache is used, so generation is O(seq^2) per step; fine for
    short samples.
    """
    raw_model: torch.nn.Module = (
        gpt_model.module if hasattr(gpt_model, "module") else gpt_model
    )
    was_training: bool = raw_model.training
    raw_model.eval()
    device: torch.device = next(raw_model.parameters()).device

    outputs: List[str] = []
    for prompt_ids in prompts:
        input_ids: torch.Tensor = prompt_ids.to(device).unsqueeze(0)  # [1, L]
        for _ in range(max_new_tokens):
            seq_len: int = input_ids.shape[1]
            position_ids: torch.Tensor = torch.arange(seq_len, device=device).unsqueeze(0)
            logits: torch.Tensor = raw_model(input_ids, position_ids, None)
            next_logits: torch.Tensor = logits[:, -1, :] / temperature
            probs: torch.Tensor = torch.softmax(next_logits, dim=-1)
            next_id: torch.Tensor = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        outputs.append(tokenizer.detokenize(input_ids[0].tolist()))
    raw_model.train(was_training)
    return outputs


def evaluate_and_generate(
    step_label: str,
    gpt_model: torch.nn.Module,
    eval_dataloader: DataLoader,
    tokenizer,
    forward_backward_func: Callable[..., Dict[str, Any]],
) -> float:
    """
    Evaluate validation loss and generate a few samples, printing both.
    """
    eval_loss: float = evaluate_loss(gpt_model, eval_dataloader, forward_backward_func)
    print(f"[eval] {step_label}: eval_loss={eval_loss:.4f}")

    if _NUM_SAMPLES > 0:
        prompts: List[torch.Tensor] = get_sample_prompts(
            eval_dataloader, _NUM_SAMPLES, _SAMPLE_PROMPT_LENGTH
        )
        if prompts:
            generated: List[str] = generate_samples(
                gpt_model,
                tokenizer,
                prompts,
                _SAMPLE_MAX_NEW_TOKENS,
                _SAMPLE_TEMPERATURE,
            )
            for j, text in enumerate(generated):
                print(f"[sample {j + 1} @ {step_label}] {text}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[mem] {_gpu_mem_str()}")
    return eval_loss


def save_distributed_checkpoint(
    checkpoint_path: str, gpt_model: torch.nn.Module
) -> None:
    """
    Save a distributed checkpoint of the GPT model using Megatron-Core utilities.

    Args:
        checkpoint_path (str): Directory path where the checkpoint will be saved.
        gpt_model (torch.nn.Module): The GPT model to checkpoint (may be wrapped with DDP).
    """
    # Access underlying model if wrapped with DDP
    model: torch.nn.Module = (
        gpt_model.module if hasattr(gpt_model, "module") else gpt_model
    )
    sharded_state_dict: Dict = model.sharded_state_dict(prefix="")
    dist_checkpointing.save(
        sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path
    )


def load_distributed_checkpoint(
    checkpoint_path: str, gpt_model: torch.nn.Module
) -> torch.nn.Module:
    """
    Load a distributed checkpoint into the GPT model using Megatron-Core utilities.

    Args:
        checkpoint_path (str): Directory path from which to load the checkpoint.
        gpt_model (torch.nn.Module): The GPT model to load the checkpoint into (may be wrapped with DDP).

    Returns:
        torch.nn.Module: The model with loaded checkpoint weights.
    """
    # Access underlying model if wrapped with DDP
    model: torch.nn.Module = (
        gpt_model.module if hasattr(gpt_model, "module") else gpt_model
    )
    sharded_state_dict: Dict = model.sharded_state_dict(prefix="")
    checkpoint: Dict = dist_checkpointing.load(
        sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path
    )
    model.load_state_dict(checkpoint)
    return gpt_model


class _Tee:
    """Duplicate writes to a stream (stdout/stderr) into an open log file.

    Every write is flushed immediately so the log stays up to date even if the
    process is killed or the GPU OOMs mid-run.
    """

    def __init__(self, stream, log_handle):
        self._stream = stream
        self._log_handle = log_handle

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def write(self, message):
        self._stream.write(message)
        # Skip the file side once it has been closed at interpreter shutdown
        # (late writes from e.g. sys.unraisablehook would otherwise raise).
        if not self._log_handle.closed:
            self._log_handle.write(message)
            self._log_handle.flush()

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
        if not self._log_handle.closed:
            try:
                self._log_handle.flush()
            except Exception:
                pass


def setup_logging() -> str:
    """
    Tee all console output (stdout + stderr) into a local log file.

    Returns the path of the log file that was created.
    """
    os.makedirs(_LOG_DIR, exist_ok=True)
    timestamp: str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path: str = os.path.join(_LOG_DIR, f"train_{timestamp}.log")
    log_handle = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, log_handle)
    sys.stderr = _Tee(sys.stderr, log_handle)
    atexit.register(log_handle.close)
    print(f"Training log saved to: {log_path}")
    return log_path


def _prune_checkpoints(ckpt_dir: str, keep: int = 3) -> None:
    """
    Keep only the `keep` most recent step_* checkpoints, deleting older ones.

    The "final" checkpoint is never pruned.
    """
    step_dirs = [
        d
        for d in Path(ckpt_dir).iterdir()
        if d.is_dir() and d.name.startswith("step_")
    ]
    step_dirs.sort(key=lambda d: int(d.name.split("_")[1]))
    for old_dir in step_dirs[:-keep] if keep > 0 else step_dirs:
        print(f"Removing old checkpoint: {old_dir}")
        shutil.rmtree(old_dir)


if __name__ == "__main__":
    initialize_distributed(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)

    # Record the whole run to a local log file (only rank 0 writes it).
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        setup_logging()

    tokenizer = get_tokenizer()
    gpt_model: GPTModel = model_provider(vocab_size=tokenizer.vocab_size)
    device: torch.device = torch.device("cuda")
    gpt_model.to(device)

    num_params: int = sum(p.numel() for p in gpt_model.parameters())
    print(
        f"Model: layers={_NUM_LAYERS} hidden={_HIDDEN_SIZE} heads={_NUM_ATTENTION_HEADS} "
        f"seq={_SEQUENCE_LENGTH} batch={_MICRO_BATCH_SIZE} recompute={_RECOMPUTE} "
        f"params={num_params / 1e6:.1f}M"
    )

    # Wrap model with DistributedDataParallel for proper gradient synchronization.
    # This provides the finish_grad_sync() method required by finalize_model_grads().
    config: TransformerConfig = gpt_model.config
    ddp_config: DistributedDataParallelConfig = DistributedDataParallelConfig(
        grad_reduce_in_fp32=False,
        overlap_grad_reduce=False,
        use_distributed_optimizer=False,
    )
    gpt_model = DistributedDataParallel(
        config=config,
        ddp_config=ddp_config,
        module=gpt_model,
    )

    optim: Adam = Adam(gpt_model.parameters(), lr=_LEARNING_RATE)

    train_iterator, num_train_samples, eval_dataloader = get_train_data_iterator(tokenizer)

    forward_backward_func: Callable[..., Dict[str, Any]] = get_forward_backward_func()

    # Train for multiple epochs: each epoch covers the whole training split once.
    steps_per_epoch: int = math.ceil(num_train_samples / _MICRO_BATCH_SIZE)
    total_iterations: int = _NUM_EPOCHS * steps_per_epoch
    print(
        f"Training for {_NUM_EPOCHS} epochs: {steps_per_epoch} steps/epoch, "
        f"{total_iterations} total steps ({num_train_samples} samples/epoch)"
    )

    # Baseline validation loss and samples before any training.
    evaluate_and_generate(
        step_label="step_0",
        gpt_model=gpt_model,
        eval_dataloader=eval_dataloader,
        tokenizer=tokenizer,
        forward_backward_func=forward_backward_func,
    )

    # Release cached blocks left by the baseline eval so training starts clean.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[mem] after baseline eval: {_gpu_mem_str()}")

    for iteration in range(total_iterations):
        epoch: int = iteration // steps_per_epoch + 1
        step_in_epoch: int = iteration % steps_per_epoch + 1

        optim.zero_grad()
        # Reset Megatron DDP's internal gradient buffers (param.main_grad).
        gpt_model.zero_grad_buffer()

        losses_reduced: Dict[str, Any] = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=train_iterator,
            model=gpt_model,
            num_microbatches=1,
            seq_length=_SEQUENCE_LENGTH,
            micro_batch_size=_MICRO_BATCH_SIZE,
            decoder_seq_length=_SEQUENCE_LENGTH,
            forward_only=False,
        )

        # Finalize model gradients: all-reduce across DP and TP groups.
        # This synchronizes gradients for non-tensor-parallel parameters (e.g., LayerNorm)
        # across tensor parallel ranks and all gradients across data parallel ranks.
        finalize_model_grads([gpt_model])

        # IMPORTANT: Megatron-Core DDP moves gradients out of param.grad into its
        # own param.main_grad buffer during backward and sets param.grad to None.
        # torch.optim.Adam only reads param.grad, so without this bridge the
        # optimizer never updates the model (loss stays at the random-init value).
        for param in gpt_model.parameters():
            main_grad = getattr(param, "main_grad", None)
            if main_grad is not None:
                param.grad = main_grad

        raw_grad_norm: float = -1.0
        if _GRAD_CLIP > 0:
            raw_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(gpt_model.parameters(), _GRAD_CLIP)
            )

        optim.step()

        # Linear LR warmup (helps stability in the first steps).
        if _LR_WARMUP_STEPS > 0 and iteration + 1 < _LR_WARMUP_STEPS:
            warm_lr: float = _LEARNING_RATE * (iteration + 1) / _LR_WARMUP_STEPS
        else:
            warm_lr = _LEARNING_RATE
        for param_group in optim.param_groups:
            param_group["lr"] = warm_lr

        if (iteration + 1) % _LOG_INTERVAL == 0 or (iteration + 1) == total_iterations:
            # Total L2 norm of all gradients (diagnostic: if it is ~0, gradients
            # are not flowing and the model cannot learn).
            grad_norm: float = 0.0
            for param in gpt_model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.float().norm().item() ** 2
            grad_norm = grad_norm ** 0.5
            # Show the RAW gradient norm as the primary number (post-clip is just
            # capped at _GRAD_CLIP and reads 1.0 whenever clipping engages).
            if _GRAD_CLIP > 0 and raw_grad_norm >= 0:
                grad_str: str = f"grad_norm={raw_grad_norm:.2f}"
                if raw_grad_norm > _GRAD_CLIP:
                    grad_str += f" [clipped to {_GRAD_CLIP:.1f}]"
            else:
                grad_str = f"grad_norm={grad_norm:.4f}"
            print(
                f"Epoch {epoch}/{_NUM_EPOCHS} step {step_in_epoch}/{steps_per_epoch} "
                f"(global step {iteration + 1}/{total_iterations}): "
                f"losses={_extract_losses(losses_reduced)} {grad_str} "
                f"lr={optim.param_groups[0]['lr']:.2e} {_gpu_mem_str()}"
            )

        # Periodic checkpoint + validation + samples.
        if (iteration + 1) % _CKPT_INTERVAL == 0 or (iteration + 1) == total_iterations:
            ckpt_step_dir: str = os.path.join(_CKPT_DIR, f"step_{iteration + 1}")
            Path(ckpt_step_dir).mkdir(parents=True, exist_ok=True)
            save_distributed_checkpoint(gpt_model=gpt_model, checkpoint_path=ckpt_step_dir)
            print(f"Checkpoint saved to {ckpt_step_dir}")
            _prune_checkpoints(_CKPT_DIR, keep=_CKPT_KEEP)
            evaluate_and_generate(
                step_label=f"step_{iteration + 1}",
                gpt_model=gpt_model,
                eval_dataloader=eval_dataloader,
                tokenizer=tokenizer,
                forward_backward_func=forward_backward_func,
            )

    # Final checkpoint
    final_ckpt_path: str = os.path.join(_CKPT_DIR, "final")
    Path(final_ckpt_path).mkdir(parents=True, exist_ok=True)
    save_distributed_checkpoint(gpt_model=gpt_model, checkpoint_path=final_ckpt_path)

    # Load back the final checkpoint to verify the save/load round-trip.
    gpt_model = load_distributed_checkpoint(
        gpt_model=gpt_model, checkpoint_path=final_ckpt_path
    )
    gpt_model.to(device)
    print("Successfully loaded the model")
