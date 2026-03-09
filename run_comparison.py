"""
Run evaluation and speed benchmark on baseline and all checkpoints,
then produce a comparison report (accuracy + speedup).

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m skipllada.run_comparison --eval_batches 15
"""

import argparse
import json
import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EVAL_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eval_results")


def parse_args():
    p = argparse.ArgumentParser(description="Compare baseline vs checkpoints (accuracy + speedup)")
    p.add_argument("--baseline", type=str, default="GSAI-ML/LLaDA-8B-Base", help="Baseline model path")
    p.add_argument("--checkpoints", type=str, nargs="+", default=None,
                   help="Checkpoint dirs; default: auto-detect skipllada_100M checkpoint-500/1000")
    p.add_argument("--eval_batches", type=int, default=15, help="Num batches for diagnostic eval")
    p.add_argument("--eval_batch_size", type=int, default=2, help="Batch size for eval (reduce if OOM)")
    p.add_argument("--output_dir", type=str, default=EVAL_RESULTS_DIR)
    p.add_argument("--skip_eval", action="store_true", help="Skip re-running eval (use existing JSONs)")
    p.add_argument("--skip_bench", action="store_true", help="Skip speed benchmark")
    return p.parse_args()


def find_checkpoints():
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "skipllada_100M")
    if not os.path.isdir(base):
        return []
    out = []
    for name in ["checkpoint-500", "checkpoint-1000"]:
        path = os.path.join(base, name)
        if os.path.isdir(path) and (
            os.path.isfile(os.path.join(path, "config.json"))
            or os.path.isfile(os.path.join(path, "model.safetensors"))
        ):
            out.append(path)
    if os.path.isfile(os.path.join(base, "config.json")) or os.path.isfile(os.path.join(base, "model.safetensors")):
        out.append(base)
    return out


def run_eval(model_path: str, output_subdir: str, num_batches: int, batch_size: int = 2) -> bool:
    out_dir = os.path.join(EVAL_RESULTS_DIR, output_subdir)
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable, "-m", "skipllada.evaluate_baseline",
        "--model_name_or_path", model_path,
        "--output_dir", out_dir,
        "--num_batches", str(num_batches),
        "--batch_size", str(batch_size),
    ]
    logger.info("Running eval: %s -> %s", model_path, out_dir)
    r = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))
    return r.returncode == 0


def run_benchmark(model_path: str) -> dict:
    """Run in subprocess so GPU memory is freed between models."""
    logger.info("Running speed benchmark: %s", model_path)
    cmd = [
        sys.executable, "-m", "skipllada.benchmark_speed",
        "--model_name_or_path", model_path,
        "--steps", "16", "--runs", "5",
        "--json",
    ]
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        if out.returncode != 0:
            logger.warning("Benchmark failed: %s", out.stderr)
            return {"speedup": 0.0, "time_full_s": 0.0, "time_early_s": 0.0}
        # Last line is JSON when --json; log lines may precede it
        raw = out.stdout.strip()
        for line in raw.splitlines()[::-1]:
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        return {"speedup": 0.0, "time_full_s": 0.0, "time_early_s": 0.0}
    except Exception as e:
        logger.warning("Benchmark failed: %s", e)
        return {"speedup": 0.0, "time_full_s": 0.0, "time_early_s": 0.0}


def load_metrics(metrics_path: str) -> dict:
    if not os.path.isfile(metrics_path):
        return {}
    with open(metrics_path) as f:
        return json.load(f)


def build_report(rows: list, out_dir: str):
    report_path = os.path.join(out_dir, "comparison_report.md")
    metrics_path = os.path.join(out_dir, "comparison_metrics.json")

    with open(metrics_path, "w") as f:
        json.dump(rows, f, indent=2)

    lines = [
        "# LayerSkip-LLaDA: Baseline vs Checkpoints",
        "",
        "Comparison of **accuracy** (per-exit perplexity, agreement with final layer) and **speedup** (full-depth vs depth-scheduled decoding).",
        "",
        "## Summary",
        "",
        "| Model | L32 PPL (low t) | L16 PPL (low t) | L16 agreement | Speedup (early exit) |",
        "|-------|-----------------|-----------------|--------------|----------------------|",
    ]
    for r in rows:
        name = r.get("name", "?")
        ppl32 = r.get("perplexity_L32_low", "—")
        ppl16 = r.get("perplexity_L16_low", "—")
        agree = r.get("agreement_L16_low", "—")
        speedup = r.get("speedup", "—")
        if isinstance(ppl32, float):
            ppl32 = f"{ppl32:.2f}"
        if isinstance(ppl16, float):
            ppl16 = f"{ppl16:.2f}"
        if isinstance(agree, float):
            agree = f"{agree:.1%}"
        if isinstance(speedup, float):
            speedup = f"{speedup:.2f}x"
        lines.append(f"| {name} | {ppl32} | {ppl16} | {agree} | {speedup} |")

    lines.extend([
        "",
        "## Notes",
        "",
        "- **L32 PPL**: Final-layer perplexity (lower is better). Should stay ~same or improve after training.",
        "- **L16 PPL**: Early-exit layer 16 perplexity. Should decrease after LayerSkip training.",
        "- **L16 agreement**: Fraction of L16 predictions that match L32. Higher = safer early exit.",
        "- **Speedup**: Wall-clock speedup of depth-scheduled decoding (tau=0.5, exit=16) vs full 32 layers.",
        "",
    ])
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("Wrote %s and %s", report_path, metrics_path)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    models = [("baseline", args.baseline)]
    if args.checkpoints:
        for i, ckpt in enumerate(args.checkpoints):
            models.append((os.path.basename(ckpt.rstrip("/")), ckpt))
    else:
        for ckpt in find_checkpoints():
            models.append((os.path.basename(ckpt.rstrip("/")), ckpt))

    if not models:
        logger.warning("No checkpoints found; only baseline will be evaluated.")

    rows = []
    for name, path in models:
        row = {"name": name, "path": path}

        if not args.skip_eval:
            run_eval(path, name, args.eval_batches, args.eval_batch_size)
        metrics_path = os.path.join(args.output_dir, name, "baseline_metrics.json")
        metrics = load_metrics(metrics_path)
        if metrics:
            ppl = metrics.get("perplexity", {})
            agree = metrics.get("agreement", {})
            buckets = metrics.get("bucket_names", ["low (0–0.3)", "mid (0.3–0.7)", "high (0.7–1.0)"])
            low_b, mid_b, high_b = buckets[0], buckets[1], buckets[2]
            row["perplexity_L32_low"] = ppl.get("32", {}).get(low_b)
            row["perplexity_L32_mid"] = ppl.get("32", {}).get(mid_b)
            row["perplexity_L32_high"] = ppl.get("32", {}).get(high_b)
            row["perplexity_L16_low"] = ppl.get("16", {}).get(low_b)
            row["agreement_L16_low"] = agree.get("16", {}).get(low_b)
            row["agreement_L24_low"] = agree.get("24", {}).get(low_b)

        if not args.skip_bench:
            bench = run_benchmark(path)
            row["speedup"] = bench["speedup"]
            row["time_full_s"] = bench["time_full_s"]
            row["time_early_s"] = bench["time_early_s"]

        rows.append(row)

    build_report(rows, args.output_dir)
    logger.info("Done. See %s/comparison_report.md", args.output_dir)


if __name__ == "__main__":
    main()
