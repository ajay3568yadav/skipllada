"""
Early-exit accuracy and speedup evaluation for LayerSkip-LLaDA.

Evaluates one or more models on an MMLU-style multiple-choice benchmark using:
  - full        : all 32 layers (standard LLaDA loglikelihood)
  - exit_L      : loglikelihood computed using only the first L layers
  - depth_sched : shallow exit (first L layers) for high-noise steps (t > tau),
                  full depth for low-noise steps (t <= tau)

For every mode, reports:
  - accuracy (%)
  - average wall-clock time per example (s)
  - speedup relative to the full-depth mode of the *same* model

Usage (single model, all modes):
    python -m skipllada.eval_early_exit \
        --models GSAI-ML/LLaDA-8B-Base \
        --model_names baseline \
        --exit_layers 8,16,24 \
        --subset abstract_algebra --limit 100

Usage (compare baseline vs. trained checkpoint):
    python -m skipllada.eval_early_exit \
        --models GSAI-ML/LLaDA-8B-Base  /path/to/checkpoint-500-hf \
        --model_names baseline trained_500 \
        --exit_layers 16 \
        --subset abstract_algebra --limit 200 \
        --output_json results.json

The script loads one model at a time and frees GPU memory between models so that
multiple 8-B models can be evaluated on a single GPU sequentially.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Allow running as `python skipllada/eval_early_exit.py` from the project root
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from skipllada.model.layerskip_llada import LayerSkipLLaDA

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MASK_ID = 126336
CHOICE_LETTERS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_mmlu_prompt(question: str, choices: List[str]) -> str:
    """Format an MMLU example as a completion-style prompt."""
    prompt = question.strip() + "\n"
    for letter, choice in zip(CHOICE_LETTERS, choices):
        prompt += f"{letter}. {choice.strip()}\n"
    prompt += "Answer:"
    return prompt


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

class EarlyExitEvaluator:
    """Evaluates a LayerSkipLLaDA model with different exit strategies."""

    def __init__(
        self,
        model_path: str,
        mc_num: int = 32,
        batch_size: int = 4,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.mc_num = mc_num
        self.batch_size = batch_size
        assert mc_num % batch_size == 0, "--mc_num must be divisible by --batch_size"

        logger.info("Loading model from %s ...", model_path)
        self.model = LayerSkipLLaDA.from_pretrained(model_path, dtype=torch.bfloat16)
        self.model.eval()
        self.model.layer_dropout_enabled = False  # always off at eval time
        self.model = self.model.to(self.device)
        self.n_layers = self.model.n_layers

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        logger.info("Model loaded (%d layers).", self.n_layers)

    def free(self):
        """Delete the model and release GPU memory (call between models)."""
        del self.model
        torch.cuda.empty_cache()
        logger.info("Model unloaded.")

    # ------------------------------------------------------------------
    # LLaDA-style masking forward process
    # ------------------------------------------------------------------

    def _forward_process(
        self,
        batch: torch.Tensor,
        prompt_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly mask `k` of the `target_len` target positions in each row of
        `batch` (LLaDA's training-distribution masking scheme).

        Returns:
            noisy_batch : same shape as batch, with masked positions set to MASK_ID
            p_mask      : proportion of target tokens masked (1/t reweight denominator)
        """
        b, l = batch.shape
        target_len = l - prompt_len

        k = torch.randint(1, target_len + 1, (), device=batch.device)
        # Stagger k across batch rows for diversity
        x = torch.round(
            torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)
        ).long()
        x = ((x - 1) % target_len) + 1

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)
        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len, device=batch.device)]

        is_mask = torch.cat(
            [torch.zeros(b, prompt_len, dtype=torch.bool, device=batch.device), is_mask], dim=1
        )
        noisy_batch = torch.where(is_mask, MASK_ID, batch)
        p_mask = (x.float() / target_len).unsqueeze(1).expand(b, l)
        return noisy_batch, p_mask

    # ------------------------------------------------------------------
    # Single-pair loglikelihood computation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _loglikelihood(
        self,
        prefix_ids: torch.Tensor,
        target_ids: torch.Tensor,
        exit_layer: Optional[int] = None,
        tau: Optional[float] = None,
        shallow_exit: Optional[int] = None,
    ) -> float:
        """
        Monte-Carlo estimate of log p(target | prefix) under LLaDA's masking
        distribution, optionally using an early-exit or depth-scheduled forward.

        Args:
            prefix_ids   : 1-D LongTensor, prefix tokens
            target_ids   : 1-D LongTensor, answer tokens to score
            exit_layer   : if set, run only the first `exit_layer` transformer blocks
            tau          : depth-scheduled threshold; if avg t > tau use shallow exit
            shallow_exit : number of layers for the shallow branch (used with tau)
        """
        seq = torch.cat([prefix_ids, target_ids]).unsqueeze(0)  # (1, L)
        seq = seq.repeat(self.batch_size, 1).to(self.device)
        prompt_len = prefix_ids.shape[0]

        loss_acc: List[float] = []
        for _ in range(self.mc_num // self.batch_size):
            noisy_batch, p_mask = self._forward_process(seq, prompt_len)
            mask_indices = noisy_batch == MASK_ID

            # --- choose forward mode ---
            if tau is not None and shallow_exit is not None:
                # Depth-scheduled: use avg fraction-masked as proxy for noise level t
                avg_t = p_mask[:, prompt_len:].mean().item()
                if avg_t > tau:
                    logits = self.model.forward_early_exit(noisy_batch, exit_layer=shallow_exit).logits
                else:
                    out = self.model.forward(noisy_batch)
                    logits = out.logits if hasattr(out, "logits") else out[0]
            elif exit_layer is not None:
                logits = self.model.forward_early_exit(noisy_batch, exit_layer=exit_layer).logits
            else:
                out = self.model.forward(noisy_batch)
                logits = out.logits if hasattr(out, "logits") else out[0]

            # Cross-entropy with 1/t re-weighting (LLaDA's ELBO objective)
            ce = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction="none")
            loss = (ce / p_mask[mask_indices]).sum() / self.batch_size
            loss_acc.append(loss.item())

        return -float(np.mean(loss_acc))   # higher = more likely

    # ------------------------------------------------------------------
    # Multiple-choice scoring
    # ------------------------------------------------------------------

    def _score_choices(
        self,
        prompt_ids: torch.Tensor,
        choice_ids: List[torch.Tensor],
        **fwd_kwargs,
    ) -> Tuple[int, float]:
        """
        Score each choice token sequence and return the predicted index and
        total wall-clock time.

        Returns:
            pred_idx : index of highest-likelihood choice (0-3)
            elapsed  : total wall-clock seconds for all 4 forward passes
        """
        lls: List[float] = []
        t0 = time.perf_counter()
        for cids in choice_ids:
            lls.append(self._loglikelihood(prompt_ids, cids, **fwd_kwargs))
        elapsed = time.perf_counter() - t0
        return int(np.argmax(lls)), elapsed

    # ------------------------------------------------------------------
    # Dataset evaluation loop
    # ------------------------------------------------------------------

    def evaluate(
        self,
        dataset_name: str,
        subset: str,
        split: str = "test",
        limit: Optional[int] = None,
        modes: Optional[List] = None,
    ) -> Dict:
        """
        Evaluate on a multiple-choice dataset.

        `modes` is a list of dicts, each with:
            name         : str label
            exit_layer   : int | None
            tau          : float | None
            shallow_exit : int | None

        Returns a summary dict keyed by mode name.
        """
        if modes is None:
            modes = [{"name": "full", "exit_layer": None, "tau": None, "shallow_exit": None}]

        logger.info("Loading %s / %s [%s] ...", dataset_name, subset, split)
        try:
            ds = load_dataset(dataset_name, subset, split=split, trust_remote_code=True)
        except Exception:
            # fallback: some datasets put subsets differently
            ds = load_dataset(dataset_name, split=split, trust_remote_code=True)

        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        logger.info("  %d examples to evaluate.", len(ds))

        results = {m["name"]: {"correct": 0, "total": 0, "times": []} for m in modes}

        for ex_idx, example in enumerate(ds):
            question = example["question"]
            choices  = example["choices"]
            answer   = int(example["answer"])

            prompt_text = format_mmlu_prompt(question, choices)
            prompt_ids = torch.tensor(
                self.tokenizer.encode(prompt_text), dtype=torch.long
            )
            choice_ids = [
                torch.tensor(self.tokenizer.encode(f" {letter}"), dtype=torch.long)
                for letter in CHOICE_LETTERS
            ]

            for mode in modes:
                pred, elapsed = self._score_choices(
                    prompt_ids,
                    choice_ids,
                    exit_layer=mode.get("exit_layer"),
                    tau=mode.get("tau"),
                    shallow_exit=mode.get("shallow_exit"),
                )
                results[mode["name"]]["correct"] += int(pred == answer)
                results[mode["name"]]["total"]   += 1
                results[mode["name"]]["times"].append(elapsed)

            if (ex_idx + 1) % 10 == 0 or ex_idx == 0:
                line_parts = []
                for m in modes:
                    r = results[m["name"]]
                    acc = r["correct"] / r["total"] * 100
                    avg_t = np.mean(r["times"])
                    line_parts.append(f"{m['name']}={acc:.0f}% ({avg_t:.1f}s)")
                logger.info("[%d/%d]  %s", ex_idx + 1, len(ds), "  |  ".join(line_parts))

        # Build summary with speedup relative to "full" mode
        full_time = np.mean(results.get("full", {}).get("times", [1.0])) or 1.0
        summary: Dict[str, Dict] = {}
        for m in modes:
            name = m["name"]
            r = results[name]
            avg_time = float(np.mean(r["times"]))
            summary[name] = {
                "accuracy_pct": round(r["correct"] / r["total"] * 100, 2),
                "correct": r["correct"],
                "total": r["total"],
                "avg_time_s": round(avg_time, 4),
                "speedup_vs_full": round(full_time / avg_time, 3) if avg_time > 0 else 1.0,
            }
        return summary


# ---------------------------------------------------------------------------
# Argument parsing and main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate LLaDA early-exit accuracy and speedup on MMLU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--models", nargs="+", required=True,
        help="One or more model paths (HuggingFace hub ID or local dir). "
             "Evaluated sequentially so multiple 8-B models fit on one GPU.",
    )
    p.add_argument(
        "--model_names", nargs="+", default=None,
        help="Display names for each model (defaults to the path).",
    )
    p.add_argument(
        "--exit_layers", type=str, default="8,16,24",
        help="Comma-separated list of exit layer numbers to evaluate.",
    )
    p.add_argument(
        "--tau", type=float, default=0.5,
        help="Depth-schedule threshold: use shallow exit when avg t > tau.",
    )
    p.add_argument(
        "--shallow_exit", type=int, default=16,
        help="Exit layer used for the shallow (high-noise) branch in depth-scheduled mode.",
    )
    p.add_argument(
        "--no_depth_scheduled", action="store_true",
        help="Skip the depth-scheduled mode.",
    )
    p.add_argument(
        "--dataset", type=str, default="cais/mmlu",
        help="HuggingFace dataset name.",
    )
    p.add_argument(
        "--subset", type=str, default="abstract_algebra",
        help="Dataset subset / config name (e.g., MMLU subject).",
    )
    p.add_argument(
        "--split", type=str, default="test",
        help="Dataset split to evaluate on.",
    )
    p.add_argument(
        "--limit", type=int, default=100,
        help="Limit number of examples (0 = all).",
    )
    p.add_argument(
        "--mc_num", type=int, default=32,
        help="Monte Carlo samples for loglikelihood estimation.",
    )
    p.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size per MC forward pass (must divide mc_num).",
    )
    p.add_argument(
        "--device", type=str, default="cuda",
    )
    p.add_argument(
        "--output_json", type=str, default=None,
        help="Path to save JSON results.",
    )
    return p.parse_args()


def build_modes(exit_layers_str: str, tau: float, shallow_exit: int, no_depth_sched: bool):
    """Build the list of evaluation mode configs."""
    modes = [{"name": "full", "exit_layer": None, "tau": None, "shallow_exit": None}]
    for l in exit_layers_str.split(","):
        l = l.strip()
        if l:
            modes.append({"name": f"exit_{l}", "exit_layer": int(l), "tau": None, "shallow_exit": None})
    if not no_depth_sched:
        modes.append({
            "name": f"depth_sched_tau{tau}_L{shallow_exit}",
            "exit_layer": None,
            "tau": tau,
            "shallow_exit": shallow_exit,
        })
    return modes


def save_plots(all_results: Dict[str, Dict], out_dir: Path):
    """
    Generate and save two plots:
      1. accuracy_vs_exit.png  – accuracy (%) per mode for each model
      2. speedup_vs_accuracy.png – scatter: x=speedup, y=accuracy, one point per (model, mode)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        logger.warning("matplotlib not installed; skipping plots. Run: pip install matplotlib")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    model_names = list(all_results.keys())

    # -----------------------------------------------------------------------
    # Collect ordered modes (exit_layer modes only + depth_sched + full)
    # -----------------------------------------------------------------------
    def _mode_sort_key(name: str):
        if name == "full":
            return (1, 999)
        if name.startswith("exit_"):
            return (0, int(name.split("_")[1]))
        return (2, 0)  # depth_sched last

    all_mode_names = []
    for res in all_results.values():
        for mn in res:
            if mn not in all_mode_names:
                all_mode_names.append(mn)
    all_mode_names.sort(key=_mode_sort_key)

    # Build x-axis labels: replace "exit_L" with "L layers", full=>"32 layers (full)"
    def _label(name: str):
        if name == "full":
            return "32\n(full)"
        if name.startswith("exit_"):
            return name.split("_")[1] + "\nlayers"
        return name.replace("depth_sched_", "depth\nsched\n")

    x_labels = [_label(mn) for mn in all_mode_names]
    x = np.arange(len(all_mode_names))

    colors = cm.tab10(np.linspace(0, 0.9, len(model_names)))

    # -----------------------------------------------------------------------
    # Plot 1: Accuracy vs exit layer
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(8, len(all_mode_names) * 1.4), 5))
    bar_w = 0.8 / max(len(model_names), 1)

    for i, (mname, color) in enumerate(zip(model_names, colors)):
        accs = [all_results[mname].get(mn, {}).get("accuracy_pct", float("nan"))
                for mn in all_mode_names]
        offsets = x + (i - len(model_names) / 2 + 0.5) * bar_w
        bars = ax.bar(offsets, accs, width=bar_w * 0.9, label=mname,
                      color=color, alpha=0.85, edgecolor="white")
        for bar, acc in zip(bars, accs):
            if not np.isnan(acc):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{acc:.0f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xlabel("Exit configuration", fontsize=11)
    ax.set_title("MMLU Accuracy — Baseline vs. Trained Models at Different Exit Depths", fontsize=12)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p1 = out_dir / "accuracy_vs_exit.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", p1)

    # -----------------------------------------------------------------------
    # Plot 2: Speedup vs accuracy scatter
    # -----------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    markers = ["o", "s", "^", "D", "v", "P", "*"]

    for i, (mname, color) in enumerate(zip(model_names, colors)):
        model_res = all_results[mname]
        for j, mn in enumerate(all_mode_names):
            entry = model_res.get(mn, {})
            acc = entry.get("accuracy_pct", float("nan"))
            spd = entry.get("speedup_vs_full", float("nan"))
            if np.isnan(acc) or np.isnan(spd):
                continue
            marker = markers[j % len(markers)]
            label = f"{mname} / {_label(mn).replace(chr(10), ' ')}"
            ax2.scatter(spd, acc, color=color, marker=marker, s=80,
                        label=label, zorder=3)
            ax2.annotate(
                _label(mn).replace("\n", " "),
                (spd, acc),
                textcoords="offset points", xytext=(4, 4),
                fontsize=7, color=color,
            )

    ax2.set_xlabel("Speedup vs. full-depth (same model)", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_title("Accuracy–Speedup Trade-off for Early Exit Configurations", fontsize=12)
    ax2.legend(fontsize=7, loc="lower left", ncol=2)
    ax2.grid(alpha=0.3)
    # Reference line: x=1 (no speedup)
    ax2.axvline(x=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    fig2.tight_layout()
    p2 = out_dir / "speedup_vs_accuracy.png"
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)
    logger.info("Saved %s", p2)

    # -----------------------------------------------------------------------
    # Plot 3: Per-model accuracy drop from full-depth (delta chart)
    # -----------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(max(8, len(all_mode_names) * 1.4), 5))
    non_full_modes = [mn for mn in all_mode_names if mn != "full"]

    for i, (mname, color) in enumerate(zip(model_names, colors)):
        full_acc = all_results[mname].get("full", {}).get("accuracy_pct", float("nan"))
        deltas = [
            all_results[mname].get(mn, {}).get("accuracy_pct", float("nan")) - full_acc
            for mn in non_full_modes
        ]
        offsets = np.arange(len(non_full_modes)) + (i - len(model_names) / 2 + 0.5) * bar_w
        bars3 = ax3.bar(offsets, deltas, width=bar_w * 0.9, label=mname,
                        color=color, alpha=0.85, edgecolor="white")
        for bar, d in zip(bars3, deltas):
            if not np.isnan(d):
                y_pos = bar.get_height() + (0.3 if d >= 0 else -1.2)
                ax3.text(bar.get_x() + bar.get_width() / 2, y_pos,
                         f"{d:+.0f}%", ha="center", va="bottom", fontsize=7)

    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_xticks(np.arange(len(non_full_modes)))
    ax3.set_xticklabels([_label(mn) for mn in non_full_modes], fontsize=9)
    ax3.set_ylabel("Δ Accuracy vs. full-depth (%)", fontsize=11)
    ax3.set_xlabel("Exit configuration", fontsize=11)
    ax3.set_title("Accuracy Drop from Full-Depth — How Much Does Early Exit Hurt?", fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(axis="y", alpha=0.3)
    fig3.tight_layout()
    p3 = out_dir / "accuracy_delta_vs_exit.png"
    fig3.savefig(p3, dpi=150)
    plt.close(fig3)
    logger.info("Saved %s", p3)

    print(f"\nPlots saved to: {out_dir}/")
    print(f"  {p1.name}  – accuracy per exit depth (bar chart)")
    print(f"  {p2.name}  – accuracy vs speedup scatter")
    print(f"  {p3.name}  – accuracy drop from full-depth per exit config")


def print_comparison_table(all_results: Dict[str, Dict]):
    """Print a formatted side-by-side comparison table."""
    # Collect all mode names (union across models)
    mode_names = []
    for model_res in all_results.values():
        for mn in model_res.keys():
            if mn not in mode_names:
                mode_names.append(mn)

    model_names = list(all_results.keys())

    # Header
    col_w = 20
    mode_w = 36
    header = f"{'Mode':<{mode_w}}" + "".join(f"{'Model: ' + n:^{col_w*2}}" for n in model_names)
    print()
    print("=" * (mode_w + col_w * 2 * len(model_names)))
    print(header)
    subheader = f"{'':>{mode_w}}" + "".join(
        f"{'Acc (%)':>{col_w}}{'Speedup':>{col_w}}" for _ in model_names
    )
    print(subheader)
    print("-" * (mode_w + col_w * 2 * len(model_names)))

    for mode in mode_names:
        row = f"{mode:<{mode_w}}"
        for mname in model_names:
            res = all_results[mname].get(mode, {})
            acc = res.get("accuracy_pct", float("nan"))
            spd = res.get("speedup_vs_full", float("nan"))
            acc_str = f"{acc:.1f}%" if not np.isnan(acc) else "N/A"
            spd_str = f"{spd:.2f}x" if not np.isnan(spd) else "N/A"
            row += f"{acc_str:>{col_w}}{spd_str:>{col_w}}"
        print(row)

    print("=" * (mode_w + col_w * 2 * len(model_names)))
    print()
    print("  Speedup = (time_full) / (time_this_mode)  for each model independently.")
    print("  Early-exit at L means only the first L transformer layers run.")
    print("  depth_sched uses shallow layers for high-noise steps, full depth for low-noise.")
    print()


def main():
    args = parse_args()

    model_names = args.model_names or args.models
    if len(model_names) != len(args.models):
        logger.warning("--model_names length mismatch with --models; using paths as names.")
        model_names = args.models

    modes = build_modes(args.exit_layers, args.tau, args.shallow_exit, args.no_depth_scheduled)
    logger.info("Evaluation modes: %s", [m["name"] for m in modes])

    limit = args.limit or None

    all_results: Dict[str, Dict] = {}

    for model_path, model_display_name in zip(args.models, model_names):
        logger.info("=" * 60)
        logger.info("Model: %s", model_display_name)
        logger.info("=" * 60)

        evaluator = EarlyExitEvaluator(
            model_path=model_path,
            mc_num=args.mc_num,
            batch_size=args.batch_size,
            device=args.device,
        )

        summary = evaluator.evaluate(
            dataset_name=args.dataset,
            subset=args.subset,
            split=args.split,
            limit=limit,
            modes=modes,
        )
        all_results[model_display_name] = summary

        evaluator.free()

        logger.info("Results for %s:", model_display_name)
        for mode_name, res in summary.items():
            logger.info(
                "  %-36s  acc=%.1f%%  speedup=%.2fx  avg_time=%.2fs",
                mode_name,
                res["accuracy_pct"],
                res["speedup_vs_full"],
                res["avg_time_s"],
            )

    print_comparison_table(all_results)

    final = {
        "config": {
            "models": args.models,
            "model_names": model_names,
            "modes": [m["name"] for m in modes],
            "dataset": args.dataset,
            "subset": args.subset,
            "split": args.split,
            "limit": limit,
            "mc_num": args.mc_num,
            "tau": args.tau,
            "shallow_exit": args.shallow_exit,
        },
        "results": all_results,
    }

    # Determine output directory for JSON + plots
    if args.output_json:
        out_path = Path(args.output_json)
    else:
        out_path = Path("results") / "early_exit_eval" / "results.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    logger.info("Results saved to %s", out_path)

    # Save plots next to the JSON
    save_plots(all_results, out_path.parent)

    return final


if __name__ == "__main__":
    main()
