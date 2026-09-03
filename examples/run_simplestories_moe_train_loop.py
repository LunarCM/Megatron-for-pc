# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Minimal Megatron-Core GPT MoE training loop for SimpleStories.
# SIMPLESTORIES_NUM_EXPERTS=0 disables MoE and trains a plain dense GPT
# (same as run_simplestories_train_loop.py).
#
# Usage (single GPU, ~12 GB):
#   cd /home/lxk/PycharmProjects/My_megatron
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#   SIMPLESTORIES_NUM_EXPERTS=8 SIMPLESTORIES_MOE_FFN_HIDDEN_SIZE=512 \
#   SIMPLESTORIES_MOE_ROUTER_TOPK=2 SIMPLESTORIES_MOE_AUX_LOSS_COEFF=0.01 \
#     torchrun --nproc_per_node=1 examples/run_simplestories_moe_train_loop.py
#
# Defaults: 8 experts / top-2 / per-expert FFN 512, total params close to the
# 14L/1024H dense model. Checkpoints go to <script>/ckpt_moe/step_XXXXX,
# separate from the dense runs' <script>/ckpt.
#
# All settings are SIMPLESTORIES_* environment variables (defaults in parens):
#   DATA_DIR (repo-local simplestories-4k-megatron)  TOKENIZER_DIR (<DATA_DIR>/tokenizer)
#   SEQUENCE_LENGTH (1024)  MICRO_BATCH_SIZE (2)
#   NUM_SAMPLES_TRAIN (1000)  NUM_EPOCHS (3)  LOG_INTERVAL (100)
#   LEARNING_RATE (1e-4)  GRAD_CLIP (1.0, 0 disables)  LR_WARMUP_STEPS (2000, 0 disables)
#   CKPT_DIR (<script>/ckpt_moe)  LOG_DIR (<script>/logs)
#   CKPT_INTERVAL (5000)  CKPT_KEEP (3)  EVAL_NUM_BATCHES (200)
#   NUM_SAMPLES (2, 0 disables)  SAMPLE_PROMPT_LENGTH (32)
#   SAMPLE_MAX_NEW_TOKENS (64)  SAMPLE_TEMPERATURE (0.8)
#   NUM_LAYERS (14)  HIDDEN_SIZE (1024)  NUM_ATTENTION_HEADS (16)
#   RECOMPUTE (full | selective | none)
#   MoE: NUM_EXPERTS (8, 0 disables)  MOE_FFN_HIDDEN_SIZE (512)  MOE_ROUTER_TOPK (2)
#        MOE_ROUTER_LOAD_BALANCING_TYPE (aux_loss)  MOE_AUX_LOSS_COEFF (0.01, 0 disables)
#        MOE_TOKEN_DISPATCHER_TYPE (allgather)  MOE_GROUPED_GEMM (false)
#        MOE_LAYER_FREQ (1; 1 = every layer MoE, 2 = every other, ...)
# MoE mode runs bias-free linear layers (like official --disable-bias-linear).

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
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
)
from megatron.core.datasets.utils import compile_helpers
from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, GPTDataset
from megatron.core.distributed import DistributedDataParallel
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.tokenizers import MegatronTokenizer

# Seq 1024 fits ~12 GB VRAM; 2048 OOMs.
_SEQUENCE_LENGTH: int = int(os.environ.get("SIMPLESTORIES_SEQUENCE_LENGTH", "1024"))
_MICRO_BATCH_SIZE: int = int(os.environ.get("SIMPLESTORIES_MICRO_BATCH_SIZE", "2"))
_NUM_SAMPLES_TRAIN: int = int(os.environ.get("SIMPLESTORIES_NUM_SAMPLES_TRAIN", "1000"))
_NUM_EPOCHS: int = int(os.environ.get("SIMPLESTORIES_NUM_EPOCHS", "3"))
_LOG_INTERVAL: int = int(os.environ.get("SIMPLESTORIES_LOG_INTERVAL", "100"))
# Default LR; higher values diverge without clipping.
_LEARNING_RATE: float = float(os.environ.get("SIMPLESTORIES_LEARNING_RATE", "1e-4"))
# Global grad L2 clip; without it the run can diverge.
_GRAD_CLIP: float = float(os.environ.get("SIMPLESTORIES_GRAD_CLIP", "1.0"))
# Linear LR warmup over the first N steps (0 disables).
_LR_WARMUP_STEPS: int = int(os.environ.get("SIMPLESTORIES_LR_WARMUP_STEPS", "2000"))
# 14 layers fits ~12 GB VRAM.
_NUM_LAYERS: int = int(os.environ.get("SIMPLESTORIES_NUM_LAYERS", "14"))
_HIDDEN_SIZE: int = int(os.environ.get("SIMPLESTORIES_HIDDEN_SIZE", "1024"))
_NUM_ATTENTION_HEADS: int = int(os.environ.get("SIMPLESTORIES_NUM_ATTENTION_HEADS", "16"))

# Recompute: "full" saves VRAM, "selective" is faster, "none" uses most memory.
_RECOMPUTE: str = os.environ.get("SIMPLESTORIES_RECOMPUTE", "full").lower()
if _RECOMPUTE not in ("full", "selective", "none"):
    _RECOMPUTE = "full"

# --- MoE settings (inert when _NUM_EXPERTS == 0) ---
# 8 experts / top-2 / FFN 512 keeps total params near the dense 14L/1024H model.
_NUM_EXPERTS: int = int(os.environ.get("SIMPLESTORIES_NUM_EXPERTS", "8"))
_MOE_FFN_HIDDEN_SIZE: int = int(
    os.environ.get("SIMPLESTORIES_MOE_FFN_HIDDEN_SIZE", "512")
)
_MOE_ROUTER_TOPK: int = int(os.environ.get("SIMPLESTORIES_MOE_ROUTER_TOPK", "2"))
_MOE_ROUTER_LB: str = os.environ.get(
    "SIMPLESTORIES_MOE_ROUTER_LOAD_BALANCING_TYPE", "aux_loss"
).lower()
if _MOE_ROUTER_LB not in ("aux_loss", "seq_aux_loss"):
    _MOE_ROUTER_LB = "aux_loss"
_MOE_AUX_LOSS_COEFF: float = float(
    os.environ.get("SIMPLESTORIES_MOE_AUX_LOSS_COEFF", "0.01")
)
_MOE_TOKEN_DISPATCHER: str = os.environ.get(
    "SIMPLESTORIES_MOE_TOKEN_DISPATCHER_TYPE", "allgather"
).lower()
if _MOE_TOKEN_DISPATCHER not in ("allgather", "alltoall"):
    _MOE_TOKEN_DISPATCHER = "allgather"
_MOE_GROUPED_GEMM: bool = (
    os.environ.get("SIMPLESTORIES_MOE_GROUPED_GEMM", "false").lower()
    in ("1", "true", "yes")
)
# MoE layer spacing: 1 = every layer, 2 = every other, ...
_MOE_LAYER_FREQ: int = int(os.environ.get("SIMPLESTORIES_MOE_LAYER_FREQ", "1"))

# Dataset index cache (kept out of the data dir).
_SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR: str = os.environ.get(
    "SIMPLESTORIES_CACHE_DIR", os.path.join(_SCRIPT_DIR, ".cache", "dataset")
)

# Tee console output to a timestamped log file.
_LOG_DIR: str = os.environ.get(
    "SIMPLESTORIES_LOG_DIR", os.path.join(_SCRIPT_DIR, "logs")
)

# Checkpointing / eval / sampling settings.
# Separate dir so MoE checkpoints never overwrite the dense runs' ones.
_CKPT_DIR: str = os.environ.get(
    "SIMPLESTORIES_CKPT_DIR", os.path.join(_SCRIPT_DIR, "ckpt_moe")
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

# SimpleStories (4k vocab) dataset and tokenizer paths.
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
    """Initialize torch.distributed and Megatron-Core model parallel groups."""
    parallel_state.destroy_model_parallel()

    rank: int = int(os.environ["RANK"])
    world_size: int = int(os.environ["WORLD_SIZE"])
    local_rank: int = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size
    )

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size, pipeline_model_parallel_size
    )


def get_tokenizer():
    """Load the SimpleStories 4k-vocab BPE tokenizer from <DATA_DIR>/tokenizer."""
    return MegatronTokenizer.from_pretrained(
        tokenizer_path=_TOKENIZER_DIR,
        metadata_path={"library": "huggingface"},
        use_fast=False,
    )


def model_provider(vocab_size: int) -> GPTModel:
    """Build a dense or MoE GPT model (vocab size must match the tokenizer).

    _NUM_EXPERTS > 0 replaces each layer's MLP with a top-k MoE FFN.
    """
    transformer_config: TransformerConfig = TransformerConfig(
        num_layers=_NUM_LAYERS,
        hidden_size=_HIDDEN_SIZE,
        num_attention_heads=_NUM_ATTENTION_HEADS,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        # Recompute activations during backward to trade compute for GPU memory.
        recompute_granularity=(None if _RECOMPUTE == "none" else _RECOMPUTE),
        recompute_method=("block" if _RECOMPUTE == "full" else None),
        recompute_num_layers=(1 if _RECOMPUTE == "full" else None),
        # MoE expert path forbids bias (like official --disable-bias-linear).
        # Dense fallback keeps bias, matching the original dense script.
        add_bias_linear=(_NUM_EXPERTS <= 0),
        # --- Mixture-of-Experts settings (inert when _NUM_EXPERTS == 0) ---
        num_moe_experts=(None if _NUM_EXPERTS <= 0 else _NUM_EXPERTS),
        moe_ffn_hidden_size=(None if _NUM_EXPERTS <= 0 else _MOE_FFN_HIDDEN_SIZE),
        moe_router_topk=_MOE_ROUTER_TOPK,
        moe_router_load_balancing_type=(
            "aux_loss" if _NUM_EXPERTS <= 0 else _MOE_ROUTER_LB
        ),
        moe_aux_loss_coeff=(0.0 if _NUM_EXPERTS <= 0 else _MOE_AUX_LOSS_COEFF),
        moe_token_dispatcher_type=_MOE_TOKEN_DISPATCHER,
        moe_grouped_gemm=_MOE_GROUPED_GEMM,
        moe_layer_freq=(1 if _NUM_EXPERTS <= 0 else _MOE_LAYER_FREQ),
    )

    if _NUM_EXPERTS > 0:
        # Per-layer spec list; moe_layer_freq decides which layers are MoE.
        transformer_layer_spec = get_gpt_decoder_block_spec(
            config=transformer_config,
            use_transformer_engine=False,
            normalization=transformer_config.normalization,
            qk_l2_norm=transformer_config.qk_l2_norm,
        )
    else:
        # Dense fallback (same as run_simplestories_train_loop.py).
        transformer_layer_spec = get_gpt_layer_local_spec()

    gpt_model: GPTModel = GPTModel(
        config=transformer_config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=vocab_size,
        max_sequence_length=_SEQUENCE_LENGTH,
    )

    return gpt_model


class _EpochCyclingIterator:
    """DataLoader wrapper that cycles epochs, reshuffling the data each time."""

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
    """Build the training iterator, sample count, and eval DataLoader."""
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
        # Each split uses its own pre-tokenized file: [train, valid, test].
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

    # Validation split uses the test file (datasets[1]).
    eval_dataloader: DataLoader = DataLoader(
        datasets[1], batch_size=_MICRO_BATCH_SIZE, shuffle=False
    )

    return _EpochCyclingIterator(train_dataloader), len(train_dataset), eval_dataloader


def get_eval_dataloader(tokenizer) -> DataLoader:
    """Build a validation DataLoader from the test split (no training split)."""
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
    """Forward step for the Megatron pipeline schedule (output tensor + loss func)."""

    def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
        """Compute the masked cross-entropy loss."""
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

        # If pipeline parallel, loss computation is done only in last stage.
        return loss, {"lm loss": loss}

    data: Dict[str, torch.Tensor] = next(data_iterator)
    tokens: torch.Tensor = data["tokens"].to(device)
    # Model applies its own causal mask.
    attention_mask: Optional[torch.Tensor] = None
    position_ids: torch.Tensor = data["position_ids"].to(device)
    labels: torch.Tensor = data["labels"].to(device)
    loss_mask: torch.Tensor = data["loss_mask"].to(device)

    output_tensor: torch.Tensor = model(
        tokens, position_ids, attention_mask, labels=labels
    )

    return output_tensor, partial(loss_func, loss_mask)


def _extract_losses(losses_reduced) -> Dict[str, float]:
    """Flatten forward_backward_func()'s per-microbatch losses into a float dict."""
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
    """Average LM loss over validation batches (no gradients; eval mode)."""
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
    """Pull short prompt sequences from the validation set for generation."""
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
    """Generate continuations for each prompt (sampling, no KV cache)."""
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
    """Evaluate validation loss and print a few generated samples."""
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
    """Save a Megatron-Core distributed checkpoint (handles DDP-wrapped models)."""
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
    """Load a Megatron-Core distributed checkpoint into the model."""
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
    """Duplicate stream writes into a log file, flushing each write."""

    def __init__(self, stream, log_handle):
        self._stream = stream
        self._log_handle = log_handle

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def write(self, message):
        self._stream.write(message)
        # Skip the log once it has been closed at interpreter shutdown.
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
    """Tee console output into a timestamped log file; returns its path."""
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
    """Delete all but the `keep` most recent step_* checkpoints (never "final")."""
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

    # Log the run to a file (rank 0 only).
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        setup_logging()

    tokenizer = get_tokenizer()
    gpt_model: GPTModel = model_provider(vocab_size=tokenizer.vocab_size)
    device: torch.device = torch.device("cuda")
    gpt_model.to(device)

    num_params: int = sum(p.numel() for p in gpt_model.parameters())
    moe_desc: str = (
        f"MoE experts={_NUM_EXPERTS} topk={_MOE_ROUTER_TOPK} "
        f"expert_ffn={_MOE_FFN_HIDDEN_SIZE} layer_freq={_MOE_LAYER_FREQ} "
        f"lb={_MOE_ROUTER_LB} coeff={_MOE_AUX_LOSS_COEFF}"
        if _NUM_EXPERTS > 0
        else "MoE disabled (dense GPT)"
    )
    print(
        f"Model: layers={_NUM_LAYERS} hidden={_HIDDEN_SIZE} heads={_NUM_ATTENTION_HEADS} "
        f"seq={_SEQUENCE_LENGTH} batch={_MICRO_BATCH_SIZE} recompute={_RECOMPUTE} "
        f"params={num_params / 1e6:.1f}M | {moe_desc}"
    )

    # Wrap the model in Megatron DDP for gradient synchronization.
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

    # Total steps: one pass over the training split per epoch.
    steps_per_epoch: int = math.ceil(num_train_samples / _MICRO_BATCH_SIZE)
    total_iterations: int = _NUM_EPOCHS * steps_per_epoch
    print(
        f"Training for {_NUM_EPOCHS} epochs: {steps_per_epoch} steps/epoch, "
        f"{total_iterations} total steps ({num_train_samples} samples/epoch)"
    )

    # Baseline eval before any training.
    evaluate_and_generate(
        step_label="step_0",
        gpt_model=gpt_model,
        eval_dataloader=eval_dataloader,
        tokenizer=tokenizer,
        forward_backward_func=forward_backward_func,
    )

    # Free memory from the baseline eval.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[mem] after baseline eval: {_gpu_mem_str()}")

    for iteration in range(total_iterations):
        epoch: int = iteration // steps_per_epoch + 1
        step_in_epoch: int = iteration % steps_per_epoch + 1

        optim.zero_grad()
        # Reset DDP's internal gradient buffers.
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

        # All-reduce gradients across data/tensor parallel ranks.
        finalize_model_grads([gpt_model])

        # Megatron DDP keeps grads in main_grad; move them to param.grad for Adam.
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

        # Linear LR warmup.
        if _LR_WARMUP_STEPS > 0 and iteration + 1 < _LR_WARMUP_STEPS:
            warm_lr: float = _LEARNING_RATE * (iteration + 1) / _LR_WARMUP_STEPS
        else:
            warm_lr = _LEARNING_RATE
        for param_group in optim.param_groups:
            param_group["lr"] = warm_lr

        if (iteration + 1) % _LOG_INTERVAL == 0 or (iteration + 1) == total_iterations:
            # Grad norm diagnostic (0 => no gradients flowing).
            grad_norm: float = 0.0
            for param in gpt_model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.float().norm().item() ** 2
            grad_norm = grad_norm ** 0.5
            # Report the pre-clip norm.
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

        # Periodic checkpoint + eval + samples.
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

    # Verify the save/load round-trip.
    gpt_model = load_distributed_checkpoint(
        gpt_model=gpt_model, checkpoint_path=final_ckpt_path
    )
    gpt_model.to(device)
    print("Successfully loaded the model")
