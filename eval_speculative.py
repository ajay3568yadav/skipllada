"""
Self-Speculative Decoding evaluation for LayerSkip-LLaDA.

Evaluates accuracy and speedup on MMLU using three generation modes:

  full        -- all 32 layers every denoising step (baseline ceiling)
  depth_sched -- shallow layers for high-noise steps, full for low-noise (Algorithm 1)
  self_spec   -- draft with first draft_exit layers, then conditionally verify
                 with the remaining layers (Algorithm 2)

For every mode the script reports:
  - Accuracy (%)
  - Average wall-clock time per question (s)
  - Speedup vs. full-depth (same model)
  - (self_spec only) Draft skip rate: % of denoising steps where verify was skipped
  - (self_spec only) Average layers used per denoising step

How accuracy is measured
------------------------
For each MMLU question the model generates a short completion
  "<question text>\\nA. ...\\nAnswer: [gen_length masked tokens]"
then the first generated token is decoded and compared to the correct
choice letter (A/B/C/D).  A short gen_length (default 4) with
block_length=4 gives 1 block so all `steps` denoising steps operate
on the 4-token answer region.

Usage
-----
# Baseline only
python -m skipllada.eval_speculative \\
    --models GSAI-ML/LLaDA-8B-Base \\
    --model_names baseline \\
    --draft_exit 16 --subset abstract_algebra --limit 50

# Baseline vs. trained checkpoints
python -m skipllada.eval_speculative \\
    --models GSAI-ML/LLaDA-8B-Base \\
            /hdd1/ay8757/SkipDLM/checkpoints/skipllada_100M/checkpoint-1526-hf \\
    --model_names baseline ckpt-1526 \\
    --draft_exit 16 \\
    --subset abstract_algebra --limit 200 \\
    --output_json /hdd1/ay8757/SkipDLM/eval_results/speculative/results.json
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from skipllada.model.layerskip_llada import LayerSkipLLaDA
from skipllada.inference import (
    add_gumbel_noise,
    get_num_transfer_tokens,
    depth_scheduled_generate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MASK_ID = 126336
CHOICE_LETTERS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Self-speculative generate with per-step statistics tracking
# ---------------------------------------------------------------------------

@torch.no_grad()
def self_speculative_generate_tracked(
    model: LayerSkipLLaDA,
    prompt: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    steps: int = 16,
    gen_length: int = 4,
    block_length: int = 4,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = MASK_ID,
    draft_exit: int = 16,
    gamma_base: float = 0.85,
    gamma_low_t: float = 0.95,
    gamma_t_boundary: float = 0.3,
    delta_quantile: float = 0.1,
    periodic_verify_k: int = 8,
) -> Tuple[torch.Tensor, Dict]:
    """
    Self-Speculative Diffusion Decoding — same algorithm as
    inference.self_speculative_generate but also returns step-level stats.

    Returns:
        x     : (B, prompt_len + gen_length) generated token ids
        stats : dict with keys
                  total_steps     - total denoising steps across all blocks
                  draft_used      - steps where verify was skipped (draft accepted)
                  verify_used     - steps where verify ran
                  forced_verify   - subset of verify_used that were periodic forces
                  draft_skip_rate - draft_used / total_steps  (fraction)
                  avg_layers      - avg transformer layers evaluated per step
                  layers_per_step - list of layer counts, one per step
    """
    n_layers = model.n_layers
    B = prompt.shape[0]

    x = torch.full(
        (B, prompt.shape[1] + gen_length), mask_id,
        dtype=torch.long, device=prompt.device,
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((B, gen_length), dtype=attention_mask.dtype, device=prompt.device),
        ], dim=-1)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    stats = {
        "total_steps": 0,
        "draft_used": 0,
        "verify_used": 0,
        "forced_verify": 0,
        "layers_per_step": [],
    }

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end   = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            t_eff = 1.0 - (i / steps_per_block)
            stats["total_steps"] += 1

            gamma = gamma_base if t_eff > gamma_t_boundary else gamma_low_t
            force_verify = (i % periodic_verify_k == 0)

            # ---- Draft pass (first draft_exit layers) ----
            draft_out = model.forward_early_exit(x, exit_layer=draft_exit,
                                                  attention_mask=attention_mask)
            draft_logits  = draft_out.logits
            cached_hidden = draft_out.hidden_states

            draft_p    = F.softmax(draft_logits.float(), dim=-1)
            draft_x0   = torch.argmax(add_gumbel_noise(draft_logits, temperature), dim=-1)
            draft_conf = torch.gather(draft_p, -1, draft_x0.unsqueeze(-1)).squeeze(-1)

            # ---- Decide: skip verify? ----
            use_draft = True
            if force_verify:
                use_draft = False
            else:
                for j in range(B):
                    sample_mask = mask_index[j]
                    n_masked = sample_mask.sum().item()
                    if n_masked == 0:
                        continue
                    masked_conf = draft_conf[j, sample_mask]
                    q_idx = max(0, int(delta_quantile * n_masked) - 1)
                    q_val = torch.kthvalue(masked_conf, q_idx + 1).values.item()
                    if q_val < gamma:
                        use_draft = False
                        break

            if use_draft:
                # Accept draft — only draft_exit layers ran
                logits = draft_logits
                x0     = draft_x0
                x0_p   = draft_conf
                stats["draft_used"] += 1
                stats["layers_per_step"].append(draft_exit)
            else:
                # Verify — run remaining layers from cached h_E
                verify_logits = model.forward_remainder(
                    cached_hidden,
                    exit_layer=draft_exit,
                    attention_mask=attention_mask,
                    input_ids=x,
                )
                logits = verify_logits
                x0     = torch.argmax(add_gumbel_noise(logits, temperature), dim=-1)
                verify_p = F.softmax(logits.float(), dim=-1)
                x0_p   = torch.gather(verify_p, -1, x0.unsqueeze(-1)).squeeze(-1)
                stats["verify_used"] += 1
                if force_verify:
                    stats["forced_verify"] += 1
                # Draft already ran draft_exit layers; remainder ran n_layers - draft_exit
                stats["layers_per_step"].append(n_layers)

            # ---- Remask ----
            x0_p[:, block_end:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(
                mask_index, x0_p,
                torch.tensor(-np.inf, device=x.device, dtype=x0_p.dtype),
            )

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(B):
                k = int(num_transfer_tokens[j, i].item())
                if k > 0:
                    _, sel = torch.topk(confidence[j], k=k)
                    transfer_index[j, sel] = True
            x[transfer_index] = x0[transfer_index]

    # Finalize stats
    total = stats["total_steps"]
    stats["draft_skip_rate"] = stats["draft_used"] / total if total > 0 else 0.0
    stats["avg_layers"] = float(np.mean(stats["layers_per_step"])) if stats["layers_per_step"] else 0.0
    return x, stats


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def format_mmlu_prompt(question: str, choices: List[str]) -> str:
    prompt = question.strip() + "\n"
    for letter, choice in zip(CHOICE_LETTERS, choices):
        prompt += f"{letter}. {choice.strip()}\n"
    prompt += "Answer:"
    return prompt


def decode_answer(token_ids: torch.Tensor, tokenizer, prompt_len: int) -> Optional[str]:
    """
    Decode the first generated token after the prompt and return the choice
    letter (A/B/C/D) if it matches one, else None.
    """
    gen_part = token_ids[0, prompt_len:]
    for tok in gen_part:
        decoded = tokenizer.decode([tok.item()]).strip().upper()
        if decoded and decoded[0] in CHOICE_LETTERS:
            return decoded[0]
    return None


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class SpectulativeEvaluator:
    def __init__(
        self,
        model_path: str,
        draft_exit: int = 16,
        steps: int = 16,
        gen_length: int = 4,
        block_length: int = 4,
        tau: float = 0.5,
        gamma_base: float = 0.85,
        gamma_low_t: float = 0.95,
        gamma_t_boundary: float = 0.3,
        delta_quantile: float = 0.1,
        periodic_verify_k: int = 8,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.draft_exit = draft_exit
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.tau = tau
        self.gamma_base = gamma_base
        self.gamma_low_t = gamma_low_t
        self.gamma_t_boundary = gamma_t_boundary
        self.delta_quantile = delta_quantile
        self.periodic_verify_k = periodic_verify_k

        logger.info("Loading model from %s ...", model_path)
        self.model = LayerSkipLLaDA.from_pretrained(model_path, dtype=torch.bfloat16)
        self.model.eval()
        self.model.layer_dropout_enabled = False
        self.model = self.model.to(self.device)
        self.n_layers = self.model.n_layers
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        logger.info("Model loaded (%d layers, draft_exit=%d).", self.n_layers, draft_exit)

    def free(self):
        del self.model
        torch.cuda.empty_cache()
        logger.info("Model unloaded.")

    # ------------------------------------------------------------------
    # Single-example generation with timing
    # ------------------------------------------------------------------

    def _generate_full(self, prompt_ids: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Full-depth generation (tau=1 → always 32 layers)."""
        t0 = time.perf_counter()
        out = depth_scheduled_generate(
            self.model, prompt_ids,
            steps=self.steps, gen_length=self.gen_length,
            block_length=self.block_length,
            temperature=0.0, remasking="low_confidence",
            mask_id=MASK_ID,
            tau=1.0,            # t_eff is always <= 1 → never shallow
            shallow_exit=self.n_layers,
        )
        elapsed = time.perf_counter() - t0
        return out, elapsed

    def _generate_depth_sched(self, prompt_ids: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Depth-scheduled generation (Algorithm 1)."""
        t0 = time.perf_counter()
        out = depth_scheduled_generate(
            self.model, prompt_ids,
            steps=self.steps, gen_length=self.gen_length,
            block_length=self.block_length,
            temperature=0.0, remasking="low_confidence",
            mask_id=MASK_ID,
            tau=self.tau,
            shallow_exit=self.draft_exit,
        )
        elapsed = time.perf_counter() - t0
        return out, elapsed

    def _generate_self_spec(
        self, prompt_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, float, Dict]:
        """Self-speculative generation (Algorithm 2) with stats."""
        t0 = time.perf_counter()
        out, stats = self_speculative_generate_tracked(
            self.model, prompt_ids,
            steps=self.steps, gen_length=self.gen_length,
            block_length=self.block_length,
            temperature=0.0, remasking="low_confidence",
            mask_id=MASK_ID,
            draft_exit=self.draft_exit,
            gamma_base=self.gamma_base,
            gamma_low_t=self.gamma_low_t,
            gamma_t_boundary=self.gamma_t_boundary,
            delta_quantile=self.delta_quantile,
            periodic_verify_k=self.periodic_verify_k,
        )
        elapsed = time.perf_counter() - t0
        return out, elapsed, stats

    # ------------------------------------------------------------------
    # Dataset evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        dataset_name: str,
        subset: str,
        split: str = "test",
        limit: Optional[int] = None,
        run_full: bool = True,
        run_depth_sched: bool = True,
        run_self_spec: bool = True,
    ) -> Dict:
        logger.info("Loading %s / %s [%s] ...", dataset_name, subset, split)
        try:
            ds = load_dataset(dataset_name, subset, split=split, trust_remote_code=True)
        except Exception:
            ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        logger.info("  %d examples.", len(ds))

        # Accumulators
        modes_active = []
        if run_full:        modes_active.append("full")
        if run_depth_sched: modes_active.append("depth_sched")
        if run_self_spec:   modes_active.append("self_spec")

        acc    = {m: 0   for m in modes_active}
        total  = {m: 0   for m in modes_active}
        times  = {m: []  for m in modes_active}
        # self_spec extras
        skip_rates: List[float] = []
        avg_layers_list: List[float] = []
        all_step_layers: List[int] = []

        for ex_idx, example in enumerate(ds):
            question = example["question"]
            choices  = example["choices"]
            answer   = int(example["answer"])
            correct_letter = CHOICE_LETTERS[answer]

            prompt_text = format_mmlu_prompt(question, choices)
            prompt_ids  = self.tokenizer.encode(prompt_text)
            prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long,
                                         device=self.device).unsqueeze(0)
            prompt_len  = len(prompt_ids)

            # --- full ---
            if "full" in modes_active:
                out_f, t_f = self._generate_full(prompt_tensor)
                pred_f = decode_answer(out_f, self.tokenizer, prompt_len)
                acc["full"]   += int(pred_f == correct_letter)
                total["full"] += 1
                times["full"].append(t_f)

            # --- depth_sched ---
            if "depth_sched" in modes_active:
                out_d, t_d = self._generate_depth_sched(prompt_tensor)
                pred_d = decode_answer(out_d, self.tokenizer, prompt_len)
                acc["depth_sched"]   += int(pred_d == correct_letter)
                total["depth_sched"] += 1
                times["depth_sched"].append(t_d)

            # --- self_spec ---
            if "self_spec" in modes_active:
                out_s, t_s, stats = self._generate_self_spec(prompt_tensor)
                pred_s = decode_answer(out_s, self.tokenizer, prompt_len)
                acc["self_spec"]   += int(pred_s == correct_letter)
                total["self_spec"] += 1
                times["self_spec"].append(t_s)
                skip_rates.append(stats["draft_skip_rate"])
                avg_layers_list.append(stats["avg_layers"])
                all_step_layers.extend(stats["layers_per_step"])

            if (ex_idx + 1) % 10 == 0 or ex_idx == 0:
                parts = []
                for m in modes_active:
                    if total[m] == 0:
                        continue
                    a_pct = acc[m] / total[m] * 100
                    avg_t = np.mean(times[m])
                    parts.append(f"{m}={a_pct:.0f}% ({avg_t:.1f}s)")
                if skip_rates:
                    parts.append(f"skip_rate={np.mean(skip_rates)*100:.0f}%")
                logger.info("[%d/%d]  %s", ex_idx + 1, len(ds), "  |  ".join(parts))

        # Build summary
        full_mean_time = float(np.mean(times["full"])) if times.get("full") else 1.0
        summary: Dict[str, Dict] = {}

        for m in modes_active:
            if total[m] == 0:
                continue
            avg_time = float(np.mean(times[m]))
            summary[m] = {
                "accuracy_pct": round(acc[m] / total[m] * 100, 2),
                "correct": acc[m],
                "total": total[m],
                "avg_time_s": round(avg_time, 4),
                "speedup_vs_full": round(full_mean_time / avg_time, 3) if avg_time > 0 else 1.0,
            }

        if "self_spec" in summary and skip_rates:
            summary["self_spec"]["draft_skip_rate_pct"] = round(float(np.mean(skip_rates)) * 100, 2)
            summary["self_spec"]["avg_layers_per_step"] = round(float(np.mean(avg_layers_list)), 2)
            # Layer usage histogram: draft_exit vs n_layers
            total_steps = len(all_step_layers)
            shallow_steps = sum(1 for l in all_step_layers if l == self.draft_exit)
            summary["self_spec"]["shallow_step_pct"] = round(shallow_steps / total_steps * 100, 2)
            summary["self_spec"]["full_step_pct"]    = round((total_steps - shallow_steps) / total_steps * 100, 2)
            summary["self_spec"]["theoretical_max_speedup"] = round(
                self.n_layers / (
                    (shallow_steps * self.draft_exit + (total_steps - shallow_steps) * self.n_layers)
                    / total_steps
                ), 3
            ) if total_steps > 0 else 1.0

        return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plots(all_results: Dict[str, Dict], out_dir: Path, draft_exit: int, n_layers: int = 32):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        logger.warning("matplotlib not installed; skipping plots.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    model_names = list(all_results.keys())
    mode_order  = ["full", "depth_sched", "self_spec"]
    mode_labels = {
        "full":        f"Full (32L)",
        "depth_sched": f"Depth-Sched\n(shallow={draft_exit}L, tau=0.5)",
        "self_spec":   f"Self-Spec\n(draft={draft_exit}L)",
    }
    colors = cm.tab10(np.linspace(0, 0.9, len(model_names)))

    present_modes = [m for m in mode_order
                     if any(m in all_results[mn] for mn in model_names)]
    x = np.arange(len(present_modes))
    bar_w = 0.8 / max(len(model_names), 1)

    # --- Plot 1: Accuracy ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (mname, color) in enumerate(zip(model_names, colors)):
        accs = [all_results[mname].get(m, {}).get("accuracy_pct", float("nan"))
                for m in present_modes]
        offsets = x + (i - len(model_names) / 2 + 0.5) * bar_w
        bars = ax.bar(offsets, accs, width=bar_w * 0.9, label=mname,
                      color=color, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, accs):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                        f"{v:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([mode_labels[m] for m in present_modes], fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("MMLU Accuracy: Full vs. Depth-Scheduled vs. Self-Speculative", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p1 = out_dir / "accuracy_by_mode.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", p1)

    # --- Plot 2: Speedup ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for i, (mname, color) in enumerate(zip(model_names, colors)):
        spds = [all_results[mname].get(m, {}).get("speedup_vs_full", float("nan"))
                for m in present_modes]
        offsets = x + (i - len(model_names) / 2 + 0.5) * bar_w
        bars2 = ax2.bar(offsets, spds, width=bar_w * 0.9, label=mname,
                        color=color, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars2, spds):
            if not np.isnan(v):
                ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                         f"{v:.2f}x", ha="center", va="bottom", fontsize=8)

    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels([mode_labels[m] for m in present_modes], fontsize=9)
    ax2.set_ylabel("Speedup vs. Full-Depth", fontsize=11)
    ax2.set_title("Wall-Clock Speedup per Decoding Mode", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    p2 = out_dir / "speedup_by_mode.png"
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)
    logger.info("Saved %s", p2)

    # --- Plot 3: Self-spec draft skip rate per model ---
    skip_vals = [
        all_results[mn].get("self_spec", {}).get("draft_skip_rate_pct", float("nan"))
        for mn in model_names
    ]
    if any(not np.isnan(v) for v in skip_vals):
        fig3, ax3 = plt.subplots(figsize=(max(6, len(model_names) * 1.5), 5))
        xm = np.arange(len(model_names))
        bars3 = ax3.bar(xm, skip_vals, color=colors[:len(model_names)],
                        alpha=0.85, edgecolor="white", width=0.5)
        for bar, v in zip(bars3, skip_vals):
            if not np.isnan(v):
                ax3.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                         f"{v:.0f}%", ha="center", va="bottom", fontsize=10)
        ax3.set_xticks(xm)
        ax3.set_xticklabels(model_names, fontsize=9)
        ax3.set_ylabel("Draft Skip Rate (%)", fontsize=11)
        ax3.set_ylim(0, 100)
        ax3.set_title(
            f"Self-Speculative Draft Skip Rate\n"
            f"(% of steps where layer {draft_exit} draft was accepted, verify skipped)",
            fontsize=11,
        )
        ax3.grid(axis="y", alpha=0.3)

        # Annotate with avg layers
        for i, mn in enumerate(model_names):
            al = all_results[mn].get("self_spec", {}).get("avg_layers_per_step", None)
            if al is not None:
                ax3.text(i, 5, f"avg {al:.1f}L/step",
                         ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")

        fig3.tight_layout()
        p3 = out_dir / "speculative_skip_rate.png"
        fig3.savefig(p3, dpi=150)
        plt.close(fig3)
        logger.info("Saved %s", p3)

    # --- Plot 4: Accuracy vs Speedup scatter ---
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    markers = {"full": "o", "depth_sched": "s", "self_spec": "^"}
    for i, (mname, color) in enumerate(zip(model_names, colors)):
        for mode in present_modes:
            res = all_results[mname].get(mode, {})
            acc_v = res.get("accuracy_pct", float("nan"))
            spd_v = res.get("speedup_vs_full", float("nan"))
            if np.isnan(acc_v) or np.isnan(spd_v):
                continue
            ax4.scatter(spd_v, acc_v, color=color, marker=markers.get(mode, "o"),
                        s=100, zorder=3,
                        label=f"{mname}/{mode_labels[mode].split(chr(10))[0]}")
            ax4.annotate(
                f"{mname}\n{mode_labels[mode].split(chr(10))[0]}",
                (spd_v, acc_v),
                textcoords="offset points", xytext=(5, 3),
                fontsize=7, color=color,
            )

    ax4.axvline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax4.set_xlabel("Speedup vs. Full-Depth", fontsize=11)
    ax4.set_ylabel("Accuracy (%)", fontsize=11)
    ax4.set_title("Accuracy–Speedup Trade-off", fontsize=12)
    ax4.grid(alpha=0.3)
    fig4.tight_layout()
    p4 = out_dir / "accuracy_vs_speedup.png"
    fig4.savefig(p4, dpi=150)
    plt.close(fig4)
    logger.info("Saved %s", p4)

    print(f"\nPlots saved to: {out_dir}/")
    print(f"  {p1.name}  – accuracy per decoding mode")
    print(f"  {p2.name}  – wall-clock speedup per mode")
    print(f"  speculative_skip_rate.png  – % steps draft accepted (trained vs. baseline)")
    print(f"  {p4.name}  – accuracy vs speedup scatter")


# ---------------------------------------------------------------------------
# Argument parsing & main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate self-speculative decoding accuracy and speedup on MMLU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models", nargs="+", required=True,
                   help="Model paths (HF hub ID or local dir). Evaluated sequentially.")
    p.add_argument("--model_names", nargs="+", default=None,
                   help="Display names (defaults to paths).")
    p.add_argument("--draft_exit", type=int, default=16,
                   help="Number of layers for the draft (early-exit) pass.")
    p.add_argument("--tau", type=float, default=0.5,
                   help="Depth-schedule threshold for depth_sched mode.")
    p.add_argument("--gamma_base", type=float, default=0.85,
                   help="Confidence threshold for high-noise steps in self_spec.")
    p.add_argument("--gamma_low_t", type=float, default=0.95,
                   help="Confidence threshold for low-noise steps in self_spec.")
    p.add_argument("--gamma_t_boundary", type=float, default=0.3,
                   help="t below which gamma_low_t is used.")
    p.add_argument("--delta_quantile", type=float, default=0.1,
                   help="Fraction of masked-token confidences that must exceed gamma.")
    p.add_argument("--periodic_verify_k", type=int, default=8,
                   help="Force full-depth verify every k denoising steps.")
    p.add_argument("--steps", type=int, default=16,
                   help="Total denoising steps per block.")
    p.add_argument("--gen_length", type=int, default=4,
                   help="Number of answer tokens to generate (4 = one letter + padding).")
    p.add_argument("--block_length", type=int, default=4,
                   help="Block length (must divide gen_length).")
    p.add_argument("--dataset", type=str, default="cais/mmlu")
    p.add_argument("--subset", type=str, default="abstract_algebra",
                   help="MMLU subject (or 'all').")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--limit", type=int, default=100,
                   help="Max examples to evaluate (0 = all).")
    p.add_argument("--no_full",        action="store_true", help="Skip full-depth mode.")
    p.add_argument("--no_depth_sched", action="store_true", help="Skip depth-scheduled mode.")
    p.add_argument("--no_self_spec",   action="store_true", help="Skip self-speculative mode.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_json", type=str, default=None,
                   help="Output path for JSON results (plots saved in same directory).")
    return p.parse_args()


def print_table(all_results: Dict[str, Dict]):
    mode_order = ["full", "depth_sched", "self_spec"]
    model_names = list(all_results.keys())
    col = 18

    header_modes = [m for m in mode_order
                    if any(m in all_results[mn] for mn in model_names)]

    print()
    print("=" * (22 + col * 4 * len(model_names)))
    print(f"{'Mode':<22}" + "".join(f"{'Model: ' + n:^{col*4}}" for n in model_names))
    print(f"{'':>22}" + "".join(
        f"{'Acc':>{col}}{'Speedup':>{col}}{'SkipRate':>{col}}{'AvgLayers':>{col}}"
        for _ in model_names
    ))
    print("-" * (22 + col * 4 * len(model_names)))

    for mode in header_modes:
        row = f"{mode:<22}"
        for mn in model_names:
            res = all_results[mn].get(mode, {})
            acc  = res.get("accuracy_pct",      float("nan"))
            spd  = res.get("speedup_vs_full",   float("nan"))
            skip = res.get("draft_skip_rate_pct", float("nan"))
            alyr = res.get("avg_layers_per_step", float("nan"))
            row += f"{f'{acc:.1f}%':>{col}}{f'{spd:.2f}x':>{col}}"
            row += f"{f'{skip:.0f}%' if not np.isnan(skip) else 'N/A':>{col}}"
            row += f"{f'{alyr:.1f}L' if not np.isnan(alyr) else 'N/A':>{col}}"
        print(row)

    print("=" * (22 + col * 4 * len(model_names)))
    print()
    print("  SkipRate   = % of denoising steps where draft was accepted (verify skipped)")
    print("  AvgLayers  = mean transformer layers evaluated per denoising step")
    print("  Speedup    = time_full / time_this_mode (for the same model)")
    print()


def main():
    args = parse_args()
    model_names = args.model_names or args.models
    if len(model_names) != len(args.models):
        model_names = args.models

    limit = args.limit or None
    all_results: Dict[str, Dict] = {}
    draft_exit_used = args.draft_exit

    for model_path, display_name in zip(args.models, model_names):
        logger.info("=" * 60)
        logger.info("Model: %s", display_name)
        logger.info("=" * 60)

        evaluator = SpectulativeEvaluator(
            model_path=model_path,
            draft_exit=args.draft_exit,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            tau=args.tau,
            gamma_base=args.gamma_base,
            gamma_low_t=args.gamma_low_t,
            gamma_t_boundary=args.gamma_t_boundary,
            delta_quantile=args.delta_quantile,
            periodic_verify_k=args.periodic_verify_k,
            device=args.device,
        )
        draft_exit_used = args.draft_exit

        summary = evaluator.evaluate(
            dataset_name=args.dataset,
            subset=args.subset,
            split=args.split,
            limit=limit,
            run_full=not args.no_full,
            run_depth_sched=not args.no_depth_sched,
            run_self_spec=not args.no_self_spec,
        )
        all_results[display_name] = summary
        evaluator.free()

        logger.info("Results for %s:", display_name)
        for mode_name, res in summary.items():
            extra = ""
            if "draft_skip_rate_pct" in res:
                extra = (f"  skip={res['draft_skip_rate_pct']:.0f}%"
                         f"  avg_layers={res['avg_layers_per_step']:.1f}")
            logger.info("  %-14s  acc=%.1f%%  speedup=%.2fx%s",
                        mode_name, res["accuracy_pct"], res["speedup_vs_full"], extra)

    print_table(all_results)

    out_path = Path(args.output_json) if args.output_json \
               else Path("results") / "speculative_eval" / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    final = {
        "config": {
            "models": args.models,
            "model_names": model_names,
            "draft_exit": args.draft_exit,
            "tau": args.tau,
            "gamma_base": args.gamma_base,
            "gamma_low_t": args.gamma_low_t,
            "gamma_t_boundary": args.gamma_t_boundary,
            "delta_quantile": args.delta_quantile,
            "periodic_verify_k": args.periodic_verify_k,
            "steps": args.steps,
            "gen_length": args.gen_length,
            "dataset": args.dataset,
            "subset": args.subset,
            "limit": limit,
        },
        "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    logger.info("Results saved to %s", out_path)

    save_plots(all_results, out_path.parent, draft_exit=draft_exit_used)
    return final


if __name__ == "__main__":
    main()
