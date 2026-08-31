# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Standalone validation script for run_simplestories_train_loop.py checkpoints.
# It loads a saved checkpoint, evaluates the validation (test split) loss, and
# generates a few samples, so you can check any mid-training weight even if the
# training run was interrupted.
#
# Usage (single GPU, after an interrupted run use any saved step_XXXXX):
#   cd /home/lxk/PycharmProjects/My_megatron
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#     torchrun --nproc_per_node=1 examples/evaluate_checkpoint.py \
#       --checkpoint examples/ckpt/step_5000
#
# Optional args:
#   --num-eval-batches N   number of validation batches (default 200)
#   --num-samples N        number of samples to generate, 0 disables (default 2)
#   --prompt-length N      prompt length in tokens (default 32)
#   --max-new-tokens N     generated tokens per sample (default 64)
#   --temperature T        sampling temperature (default 0.8)

import argparse
import os
import sys
import torch

# Make the sibling training module importable (this file lives in examples/).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_simplestories_train_loop as train_mod  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved SimpleStories checkpoint: validation loss + samples."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to a checkpoint directory (e.g. examples/ckpt/step_10000)"
    )
    parser.add_argument(
        "--num-eval-batches",
        type=int,
        default=train_mod._EVAL_NUM_BATCHES,
        help="Number of validation batches to evaluate (default: %(default)s)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=train_mod._NUM_SAMPLES,
        help="Number of samples to generate, 0 disables (default: %(default)s)",
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=train_mod._SAMPLE_PROMPT_LENGTH,
        help="Prompt length in tokens (default: %(default)s)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=train_mod._SAMPLE_MAX_NEW_TOKENS,
        help="Generated tokens per sample (default: %(default)s)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=train_mod._SAMPLE_TEMPERATURE,
        help="Sampling temperature (default: %(default)s)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint):
        sys.exit(f"Checkpoint directory not found: {args.checkpoint}")

    # Same distributed/seed setup as training.
    train_mod.initialize_distributed(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1
    )
    train_mod.model_parallel_cuda_manual_seed(123)

    tokenizer = train_mod.get_tokenizer()
    gpt_model = train_mod.model_provider(vocab_size=tokenizer.vocab_size)
    device: torch.device = torch.device("cuda")
    gpt_model.to(device)

    gpt_model = train_mod.load_distributed_checkpoint(
        checkpoint_path=args.checkpoint, gpt_model=gpt_model
    )
    gpt_model.to(device)
    print(f"Loaded checkpoint from {args.checkpoint}")

    eval_dataloader = train_mod.get_eval_dataloader(tokenizer)
    forward_backward_func = train_mod.get_forward_backward_func()

    # Validation loss.
    eval_loss = train_mod.evaluate_loss(
        gpt_model=gpt_model,
        eval_dataloader=eval_dataloader,
        forward_backward_func=forward_backward_func,
        num_batches=args.num_eval_batches,
    )
    print(f"[eval] {args.checkpoint}: eval_loss={eval_loss:.4f}")

    # Samples.
    if args.num_samples > 0:
        prompts = train_mod.get_sample_prompts(
            eval_dataloader, args.num_samples, args.prompt_length
        )
        generated = train_mod.generate_samples(
            gpt_model=gpt_model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        for j, text in enumerate(generated):
            print(f"[sample {j + 1}] {text}")


if __name__ == "__main__":
    main()
