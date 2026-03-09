"""
Evaluate the baseline (untrained) LLaDA-8B model to establish "before"
metrics for all exit layers across timestep buckets.

The baseline model has never been trained with early-exit loss, so the
shared ln_f + ff_out was only optimized for final-layer hidden states.
We expect:
  - Shallow exit layers produce poor/random logits
  - Deep exit layers (e.g. L24) may be partially useful
  - Final layer (L32) is the reference

This gives the "before" snapshot to compare against after LayerSkip training.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m skipllada.evaluate_baseline [--num_batches 50]
"""

import argparse
import logging
import os

import torch
from transformers import AutoTokenizer

from .model.layerskip_llada import LayerSkipLLaDA
from .metrics import (
    MetricsComputer,
    forward_process,
    plot_diagnostic_dashboard,
    plot_per_layer_loss_curves,
    print_summary_table,
    TIMESTEP_BUCKETS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate baseline LLaDA-8B at all exit layers")
    parser.add_argument("--model_name_or_path", type=str, default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--num_batches", type=int, default=50,
                        help="Number of evaluation batches per timestep bucket")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="./eval_results/baseline")
    parser.add_argument("--exit_layers", type=int, nargs="+", default=[4, 8, 16, 24, 32])
    parser.add_argument("--mask_id", type=int, default=126336)
    parser.add_argument("--t_samples_per_bucket", type=int, default=5,
                        help="Number of t values to sample within each bucket per batch")
    return parser.parse_args()


def generate_eval_data(tokenizer, num_samples: int, seq_length: int) -> torch.Tensor:
    """
    Generate evaluation data by packing tokenized text.
    Uses a fixed text to ensure reproducibility across runs.
    """
    eval_texts = [
        "The quick brown fox jumps over the lazy dog. ",
        "In a hole in the ground there lived a hobbit. ",
        "It was the best of times, it was the worst of times. ",
        "All happy families are alike; each unhappy family is unhappy in its own way. ",
        "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. ",
        "Call me Ishmael. Some years ago, never mind how long precisely, having little or no money in my purse. ",
        "The Transformer architecture has revolutionized natural language processing. ",
        "Masked diffusion language models generate text by iteratively denoising masked tokens. ",
    ]

    all_ids = []
    buffer = []
    text_idx = 0
    while len(all_ids) < num_samples:
        text = eval_texts[text_idx % len(eval_texts)]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        buffer.extend(tokens)
        text_idx += 1

        while len(buffer) >= seq_length and len(all_ids) < num_samples:
            all_ids.append(torch.tensor(buffer[:seq_length], dtype=torch.long))
            buffer = buffer[seq_length:]

    return torch.stack(all_ids)


@torch.no_grad()
def evaluate_at_timestep(
    model: LayerSkipLLaDA,
    data: torch.Tensor,
    t_value: float,
    batch_size: int,
    exit_layers: list,
    mask_id: int,
    metrics: MetricsComputer,
):
    """Run forward_with_exits at a specific t value and accumulate metrics."""
    device = model.device
    num_samples = data.shape[0]

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        input_ids = data[start:end].to(device)
        B = input_ids.shape[0]

        t = torch.full((B,), t_value, device=device)
        noisy_input, masked_indices, p_mask = forward_process(
            input_ids, t, mask_id=mask_id
        )

        outputs = model.forward_with_exits(
            input_ids=noisy_input,
            exit_layers=exit_layers,
            t=t_value,
        )

        metrics.update(
            exit_logits=outputs.exit_logits,
            input_ids=input_ids,
            masked_indices=masked_indices,
            p_mask=p_mask,
            t=t,
        )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    logger.info(f"Loading model from {args.model_name_or_path}")
    model = LayerSkipLLaDA.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
    )
    model.eval()
    model.layer_dropout_enabled = False

    if torch.cuda.is_available():
        model = model.cuda()

    total_samples = args.num_batches * args.batch_size
    logger.info(f"Generating {total_samples} evaluation sequences of length {args.seq_length}")
    eval_data = generate_eval_data(tokenizer, total_samples, args.seq_length)

    metrics = MetricsComputer(
        exit_layers=args.exit_layers,
        n_layers=model.n_layers,
    )

    t_values_per_bucket = []
    for lo, hi in TIMESTEP_BUCKETS:
        step = (hi - lo) / (args.t_samples_per_bucket + 1)
        t_values_per_bucket.append(
            [lo + step * (i + 1) for i in range(args.t_samples_per_bucket)]
        )

    total_evals = sum(len(tvs) for tvs in t_values_per_bucket)
    eval_idx = 0

    for b_idx, ((lo, hi), t_values) in enumerate(zip(TIMESTEP_BUCKETS, t_values_per_bucket)):
        logger.info(f"Evaluating timestep bucket [{lo}, {hi})")
        for t_val in t_values:
            eval_idx += 1
            logger.info(f"  [{eval_idx}/{total_evals}] t = {t_val:.3f}")
            evaluate_at_timestep(
                model=model,
                data=eval_data,
                t_value=t_val,
                batch_size=args.batch_size,
                exit_layers=args.exit_layers,
                mask_id=args.mask_id,
                metrics=metrics,
            )

    logger.info("Computing metrics...")
    results = metrics.compute()

    print_summary_table(results)

    dashboard_path = os.path.join(args.output_dir, "baseline_dashboard.png")
    logger.info(f"Generating diagnostic dashboard -> {dashboard_path}")
    plot_diagnostic_dashboard(
        results,
        title="Baseline LLaDA-8B — Before LayerSkip Training",
        save_path=dashboard_path,
    )

    loss_path = os.path.join(args.output_dir, "baseline_loss_breakdown.png")
    logger.info(f"Generating loss breakdown -> {loss_path}")
    plot_per_layer_loss_curves(
        results,
        title="Baseline LLaDA-8B — Per-Exit-Layer Loss",
        save_path=loss_path,
    )

    import json
    serializable = {}
    for key, val in results.items():
        if isinstance(val, dict):
            serializable[key] = {
                str(k): (
                    {str(kk): vv for kk, vv in v.items()} if isinstance(v, dict)
                    else v
                )
                for k, v in val.items()
            }
        else:
            serializable[key] = val

    json_path = os.path.join(args.output_dir, "baseline_metrics.json")
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    logger.info(f"Raw metrics saved to {json_path}")

    logger.info("Baseline evaluation complete.")


if __name__ == "__main__":
    main()
