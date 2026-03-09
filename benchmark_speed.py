"""
Benchmark inference speed: full-depth vs early-exit (depth-scheduled) decoding.

Reports wall-clock time and speedup for a fixed number of diffusion steps,
so we can compare baseline vs trained checkpoints and confirm early exit
reduces compute without changing the decoding algorithm.
"""

import argparse
import json
import logging
import time

import torch
from transformers import AutoTokenizer

from .model.layerskip_llada import LayerSkipLLaDA
from .inference import depth_scheduled_generate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MASK_ID = 126336


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark full vs early-exit LLaDA decoding")
    p.add_argument("--model_name_or_path", type=str, default="GSAI-ML/LLaDA-8B-Base")
    p.add_argument("--steps", type=int, default=16, help="Diffusion steps per run")
    p.add_argument("--gen_length", type=int, default=64)
    p.add_argument("--block_length", type=int, default=64)
    p.add_argument("--warmup", type=int, default=2, help="Warmup runs before timing")
    p.add_argument("--runs", type=int, default=5, help="Timed runs for mean/std")
    p.add_argument("--tau", type=float, default=0.5, help="Depth-schedule threshold (early exit when t > tau)")
    p.add_argument("--shallow_exit", type=int, default=16, help="Exit layer for shallow steps")
    p.add_argument("--json", action="store_true", help="Print only JSON result to stdout")
    return p.parse_args()


@torch.no_grad()
def run_generation(model, prompt, full_depth: bool, steps: int, gen_length: int, block_length: int):
    """Run one generation. full_depth=True => tau=1.0 (always full); False => use tau/shallow_exit."""
    tau = 1.0 if full_depth else 0.5
    shallow = 32 if full_depth else 16
    return depth_scheduled_generate(
        model,
        prompt,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=0.0,
        remasking="low_confidence",
        mask_id=MASK_ID,
        tau=tau,
        shallow_exit=shallow,
    )


def _main_impl(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading tokenizer and model from %s", args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = LayerSkipLLaDA.from_pretrained(args.model_name_or_path, dtype=torch.bfloat16)
    model.eval()
    model.layer_dropout_enabled = False
    model = model.to(device)
    prompt_text = "The capital of France is"
    prompt = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
    if prompt.shape[1] > 32:
        prompt = prompt[:, :32]
    logger.info("Warmup (%d full + %d early-exit)...", args.warmup, args.warmup)
    for _ in range(args.warmup):
        run_generation(model, prompt, full_depth=True, steps=args.steps, gen_length=args.gen_length, block_length=args.block_length)
    torch.cuda.synchronize()
    for _ in range(args.warmup):
        run_generation(model, prompt, full_depth=False, steps=args.steps, gen_length=args.gen_length, block_length=args.block_length)
    torch.cuda.synchronize()
    times_full = []
    for _ in range(args.runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_generation(model, prompt, full_depth=True, steps=args.steps, gen_length=args.gen_length, block_length=args.block_length)
        torch.cuda.synchronize()
        times_full.append(time.perf_counter() - t0)
    mean_full = sum(times_full) / len(times_full)
    std_full = (sum((t - mean_full) ** 2 for t in times_full) / len(times_full)) ** 0.5
    times_early = []
    for _ in range(args.runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_generation(model, prompt, full_depth=False, steps=args.steps, gen_length=args.gen_length, block_length=args.block_length)
        torch.cuda.synchronize()
        times_early.append(time.perf_counter() - t0)
    mean_early = sum(times_early) / len(times_early)
    std_early = (sum((t - mean_early) ** 2 for t in times_early) / len(times_early)) ** 0.5
    speedup = mean_full / mean_early if mean_early > 0 else 0.0
    result = {
        "model": args.model_name_or_path,
        "steps": args.steps,
        "time_full_s": mean_full,
        "time_full_std": std_full,
        "time_early_s": mean_early,
        "time_early_std": std_early,
        "speedup": speedup,
    }
    if getattr(args, "json", False):
        print(json.dumps(result))
    else:
        logger.info("Full-depth:  %.3f ± %.3f s", mean_full, std_full)
        logger.info("Early-exit: %.3f ± %.3f s", mean_early, std_early)
        logger.info("Speedup:    %.2fx", speedup)
    return result


def run_benchmark(model_path: str, steps: int = 16, runs: int = 5, warmup: int = 2) -> dict:
    """Programmatic entry: run benchmark and return result dict."""
    class A:
        pass
    a = A()
    a.model_name_or_path = model_path
    a.steps = steps
    a.gen_length = 64
    a.block_length = 64
    a.warmup = warmup
    a.runs = runs
    a.tau = 0.5
    a.shallow_exit = 16
    a.json = False
    return _main_impl(a)


def main():
    args = parse_args()
    return _main_impl(args)


if __name__ == "__main__":
    args = parse_args()
    result = _main_impl(args)
    if args.json:
        print(json.dumps(result))
