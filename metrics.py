"""
Per-exit-layer diagnostic metrics for LayerSkip-LLaDA.

Three diagnostic categories:
  1. Representation Quality — Is the depth hierarchy forming?
  2. Base Loss Protection — Is the final layer being degraded?
  3. Training Stability — Are gradients and losses behaving?

Plus inference-proxy metrics computed at phase boundaries.
"""

import os
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np


TIMESTEP_BUCKETS = [(0.0, 0.3), (0.3, 0.7), (0.7, 1.0)]
BUCKET_NAMES = ["low (0–0.3)", "mid (0.3–0.7)", "high (0.7–1.0)"]


@dataclass
class ExitLayerStats:
    """Accumulated statistics for a single exit layer within a single t bucket."""
    total_ce: float = 0.0
    total_reweighted_ce: float = 0.0
    total_masked_tokens: int = 0
    total_agreement_with_final: int = 0
    total_predictions: int = 0
    confidence_values: list = field(default_factory=list)
    cap_binding_count: int = 0
    cap_total_count: int = 0


class MetricsComputer:
    """
    Computes per-exit-layer diagnostics from forward_with_exits output.

    Usage:
        mc = MetricsComputer(exit_layers=[4, 8, 16, 24, 32])

        # For each evaluation batch:
        mc.update(exit_logits, input_ids, masked_indices, p_mask, t)

        # After all batches:
        results = mc.compute()
        mc.reset()
    """

    def __init__(
        self,
        exit_layers: List[int],
        n_layers: int = 32,
        c_cap: float = 20.0,
        eps_scale: float = 0.2,
        alpha: float = 0.1,
        max_confidence_samples: int = 50000,
    ):
        self.exit_layers = sorted(exit_layers)
        self.n_layers = n_layers
        self.c_cap = c_cap
        self.eps_scale = eps_scale
        self.alpha = alpha
        self.max_confidence_samples = max_confidence_samples

        self.stats: Dict[Tuple[int, int], ExitLayerStats] = {}
        self.reset()

    def reset(self):
        self.stats = {}
        for e in self.exit_layers:
            for b_idx in range(len(TIMESTEP_BUCKETS)):
                self.stats[(e, b_idx)] = ExitLayerStats()

    def _get_bucket_idx(self, t_val: float) -> int:
        for i, (lo, hi) in enumerate(TIMESTEP_BUCKETS):
            if lo <= t_val < hi:
                return i
        return len(TIMESTEP_BUCKETS) - 1

    @torch.no_grad()
    def update(
        self,
        exit_logits: Dict[int, torch.Tensor],
        input_ids: torch.Tensor,
        masked_indices: torch.Tensor,
        p_mask: torch.Tensor,
        t: torch.Tensor,
    ):
        """
        Accumulate metrics from one batch.

        Args:
            exit_logits: {layer_num: (B, n, V)} logits at each exit
            input_ids: (B, n) ground truth token ids
            masked_indices: (B, n) bool mask
            p_mask: (B, n) mask probabilities
            t: (B,) per-sample timesteps
        """
        B = t.shape[0]
        final_layer = max(self.exit_layers)
        final_logits = exit_logits[final_layer]
        final_preds = final_logits.argmax(dim=-1)

        for sample_idx in range(B):
            t_val = t[sample_idx].item()
            b_idx = self._get_bucket_idx(t_val)
            sample_mask = masked_indices[sample_idx]
            n_masked = sample_mask.sum().item()
            if n_masked == 0:
                continue

            sample_targets = input_ids[sample_idx]
            sample_p_mask = p_mask[sample_idx]
            sample_final_preds = final_preds[sample_idx]

            for e in self.exit_layers:
                if e not in exit_logits:
                    continue
                stats = self.stats[(e, b_idx)]

                logits_e = exit_logits[e][sample_idx]
                masked_logits = logits_e[sample_mask]
                masked_targets = sample_targets[sample_mask]

                ce = F.cross_entropy(masked_logits, masked_targets, reduction="none")
                stats.total_ce += ce.sum().item()

                reweight = 1.0 / sample_p_mask[sample_mask]
                cap_binding = (reweight > self.c_cap).sum().item()
                stats.cap_binding_count += int(cap_binding)
                stats.cap_total_count += int(n_masked)

                reweight_capped = torch.clamp(reweight, max=self.c_cap)
                stats.total_reweighted_ce += (ce * reweight_capped).sum().item()
                stats.total_masked_tokens += int(n_masked)

                preds_e = logits_e.argmax(dim=-1)
                agreement = (preds_e[sample_mask] == sample_final_preds[sample_mask]).sum().item()
                stats.total_agreement_with_final += int(agreement)
                stats.total_predictions += int(n_masked)

                probs_e = F.softmax(masked_logits.float(), dim=-1)
                confidence = probs_e.max(dim=-1).values
                n_to_keep = min(
                    int(n_masked),
                    max(0, self.max_confidence_samples - len(stats.confidence_values))
                )
                if n_to_keep > 0:
                    stats.confidence_values.extend(
                        confidence[:n_to_keep].cpu().tolist()
                    )

    def compute(self) -> dict:
        """
        Compute final metrics from accumulated statistics.

        Returns dict with structured results for logging and plotting.
        """
        results = {
            "exit_layers": self.exit_layers,
            "bucket_names": BUCKET_NAMES,
            "perplexity": {},
            "reweighted_loss": {},
            "agreement": {},
            "mean_confidence": {},
            "confidence_quantiles": {},
            "cap_utilization": {},
            "loss_ratio": {},
        }

        final_layer = max(self.exit_layers)
        base_losses = {}
        for b_idx in range(len(TIMESTEP_BUCKETS)):
            key = (final_layer, b_idx)
            s = self.stats[key]
            if s.total_masked_tokens > 0:
                base_losses[b_idx] = s.total_reweighted_ce / s.total_masked_tokens
            else:
                base_losses[b_idx] = None

        for e in self.exit_layers:
            results["perplexity"][e] = {}
            results["reweighted_loss"][e] = {}
            results["agreement"][e] = {}
            results["mean_confidence"][e] = {}
            results["confidence_quantiles"][e] = {}
            results["cap_utilization"][e] = {}
            results["loss_ratio"][e] = {}

            for b_idx, b_name in enumerate(BUCKET_NAMES):
                s = self.stats[(e, b_idx)]

                if s.total_masked_tokens > 0:
                    avg_ce = s.total_ce / s.total_masked_tokens
                    results["perplexity"][e][b_name] = math.exp(min(avg_ce, 20.0))
                    results["reweighted_loss"][e][b_name] = (
                        s.total_reweighted_ce / s.total_masked_tokens
                    )
                else:
                    results["perplexity"][e][b_name] = None
                    results["reweighted_loss"][e][b_name] = None

                if s.total_predictions > 0:
                    results["agreement"][e][b_name] = (
                        s.total_agreement_with_final / s.total_predictions
                    )
                else:
                    results["agreement"][e][b_name] = None

                if s.confidence_values:
                    conf = np.array(s.confidence_values)
                    results["mean_confidence"][e][b_name] = float(conf.mean())
                    results["confidence_quantiles"][e][b_name] = {
                        "p10": float(np.percentile(conf, 10)),
                        "p25": float(np.percentile(conf, 25)),
                        "p50": float(np.percentile(conf, 50)),
                        "p75": float(np.percentile(conf, 75)),
                        "p90": float(np.percentile(conf, 90)),
                    }
                else:
                    results["mean_confidence"][e][b_name] = None
                    results["confidence_quantiles"][e][b_name] = None

                if s.cap_total_count > 0:
                    results["cap_utilization"][e][b_name] = (
                        s.cap_binding_count / s.cap_total_count
                    )
                else:
                    results["cap_utilization"][e][b_name] = None

                exit_loss = results["reweighted_loss"][e].get(b_name)
                base_loss = base_losses.get(b_idx)
                if exit_loss is not None and base_loss is not None and base_loss > 0:
                    n_exits = len([x for x in self.exit_layers if x != final_layer])
                    w_e = (e / self.n_layers) ** 2 * (1.0 + self.alpha)
                    effective = (self.eps_scale / max(n_exits, 1)) * w_e * exit_loss
                    results["loss_ratio"][e][b_name] = effective / base_loss
                else:
                    results["loss_ratio"][e][b_name] = None

        return results


def forward_process(
    input_ids: torch.Tensor,
    t: torch.Tensor,
    mask_id: int = 126336,
    eps: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply forward masking at specified timestep values.
    Returns noisy_input, masked_indices, p_mask.
    """
    B, n = input_ids.shape
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].expand(B, n)
    masked_indices = torch.rand(B, n, device=input_ids.device) < p_mask
    noisy_input = torch.where(masked_indices, mask_id, input_ids)
    return noisy_input, masked_indices, p_mask


def plot_diagnostic_dashboard(
    results: dict,
    title: str = "LayerSkip-LLaDA Diagnostics",
    save_path: Optional[str] = None,
):
    """
    Generate the 3-panel diagnostic dashboard:
      Panel 1: Per-exit perplexity heatmap (depth hierarchy)
      Panel 2: Per-exit agreement heatmap (inference skip proxy)
      Panel 3: Per-exit effective loss ratio heatmap (base loss protection)

    Plus supplementary panels:
      Panel 4: Mean confidence heatmap
      Panel 5: Confidence CDF curves by exit layer (one subplot per t bucket)
      Panel 6: 1/t cap utilization heatmap
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    exit_layers = results["exit_layers"]
    bucket_names = results["bucket_names"]
    n_exits = len(exit_layers)
    n_buckets = len(bucket_names)

    def _build_matrix(metric_key):
        mat = np.full((n_buckets, n_exits), np.nan)
        for j, e in enumerate(exit_layers):
            for i, b in enumerate(bucket_names):
                val = results[metric_key].get(e, {}).get(b)
                if val is not None:
                    mat[i, j] = val
        return mat

    fig = plt.figure(figsize=(22, 28))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3, top=0.95, bottom=0.05)

    exit_labels = [f"L{e}" for e in exit_layers]

    # --- Panel 1: Perplexity heatmap ---
    ax1 = fig.add_subplot(gs[0, 0])
    ppl_mat = _build_matrix("perplexity")
    im1 = ax1.imshow(ppl_mat, aspect="auto", cmap="YlOrRd",
                      norm=LogNorm(vmin=max(1, np.nanmin(ppl_mat)),
                                   vmax=np.nanmax(ppl_mat)))
    ax1.set_xticks(range(n_exits))
    ax1.set_xticklabels(exit_labels)
    ax1.set_yticks(range(n_buckets))
    ax1.set_yticklabels(bucket_names)
    ax1.set_xlabel("Exit Layer")
    ax1.set_ylabel("Timestep Bucket")
    ax1.set_title("Panel 1: Masked Diffusion Perplexity\n(depth hierarchy)")
    for i in range(n_buckets):
        for j in range(n_exits):
            val = ppl_mat[i, j]
            if not np.isnan(val):
                ax1.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8,
                         color="white" if val > np.nanmedian(ppl_mat) else "black")
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # --- Panel 2: Agreement heatmap ---
    ax2 = fig.add_subplot(gs[0, 1])
    agree_mat = _build_matrix("agreement")
    im2 = ax2.imshow(agree_mat, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax2.set_xticks(range(n_exits))
    ax2.set_xticklabels(exit_labels)
    ax2.set_yticks(range(n_buckets))
    ax2.set_yticklabels(bucket_names)
    ax2.set_xlabel("Exit Layer")
    ax2.set_ylabel("Timestep Bucket")
    ax2.set_title("Panel 2: Prediction Agreement with Final Layer\n(inference skip proxy)")
    for i in range(n_buckets):
        for j in range(n_exits):
            val = agree_mat[i, j]
            if not np.isnan(val):
                ax2.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                         color="white" if val < 0.5 else "black")
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # --- Panel 3: Effective loss ratio heatmap ---
    ax3 = fig.add_subplot(gs[1, 0])
    ratio_mat = _build_matrix("loss_ratio")
    vmax_ratio = max(np.nanmax(ratio_mat), 1.0) if not np.all(np.isnan(ratio_mat)) else 1.0
    im3 = ax3.imshow(ratio_mat, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=vmax_ratio)
    ax3.set_xticks(range(n_exits))
    ax3.set_xticklabels(exit_labels)
    ax3.set_yticks(range(n_buckets))
    ax3.set_yticklabels(bucket_names)
    ax3.set_xlabel("Exit Layer")
    ax3.set_ylabel("Timestep Bucket")
    ax3.set_title("Panel 3: Effective Loss Ratio (exit/base)\n(base loss protection — should stay < 1)")
    for i in range(n_buckets):
        for j in range(n_exits):
            val = ratio_mat[i, j]
            if not np.isnan(val):
                ax3.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im3, ax=ax3, shrink=0.8)

    # --- Panel 4: Mean confidence heatmap ---
    ax4 = fig.add_subplot(gs[1, 1])
    conf_mat = _build_matrix("mean_confidence")
    im4 = ax4.imshow(conf_mat, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax4.set_xticks(range(n_exits))
    ax4.set_xticklabels(exit_labels)
    ax4.set_yticks(range(n_buckets))
    ax4.set_yticklabels(bucket_names)
    ax4.set_xlabel("Exit Layer")
    ax4.set_ylabel("Timestep Bucket")
    ax4.set_title("Panel 4: Mean Confidence (max softmax prob)\n(higher = more confident predictions)")
    for i in range(n_buckets):
        for j in range(n_exits):
            val = conf_mat[i, j]
            if not np.isnan(val):
                ax4.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8,
                         color="white" if val > 0.5 else "black")
    plt.colorbar(im4, ax=ax4, shrink=0.8)

    # --- Panel 5: Confidence CDF curves ---
    ax5 = fig.add_subplot(gs[2, 0])
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_exits))
    line_styles = ["-", "--", ":"]
    for b_idx, b_name in enumerate(bucket_names):
        for j, e in enumerate(exit_layers):
            quantiles = results["confidence_quantiles"].get(e, {}).get(b_name)
            if quantiles is None:
                continue
            x_vals = [0.10, 0.25, 0.50, 0.75, 0.90]
            y_vals = [quantiles["p10"], quantiles["p25"], quantiles["p50"],
                      quantiles["p75"], quantiles["p90"]]
            ax5.plot(y_vals, x_vals, color=colors[j],
                     linestyle=line_styles[b_idx % len(line_styles)],
                     label=f"L{e} {b_name}" if b_idx == 0 else None,
                     alpha=0.8, linewidth=1.5)
    ax5.set_xlabel("Confidence (max softmax prob)")
    ax5.set_ylabel("Cumulative Fraction")
    ax5.set_title("Panel 5: Confidence CDF by Exit Layer\n(solid=low-t, dashed=mid-t, dotted=high-t)")
    ax5.legend(fontsize=7, loc="upper left", ncol=2)
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.grid(True, alpha=0.3)

    # --- Panel 6: Cap utilization heatmap ---
    ax6 = fig.add_subplot(gs[2, 1])
    cap_mat = _build_matrix("cap_utilization")
    im6 = ax6.imshow(cap_mat, aspect="auto", cmap="Oranges", vmin=0, vmax=1)
    ax6.set_xticks(range(n_exits))
    ax6.set_xticklabels(exit_labels)
    ax6.set_yticks(range(n_buckets))
    ax6.set_yticklabels(bucket_names)
    ax6.set_xlabel("Exit Layer")
    ax6.set_ylabel("Timestep Bucket")
    ax6.set_title("Panel 6: 1/t Cap Utilization Rate\n(fraction where 1/t > C_cap)")
    for i in range(n_buckets):
        for j in range(n_exits):
            val = cap_mat[i, j]
            if not np.isnan(val):
                ax6.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im6, ax=ax6, shrink=0.8)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Dashboard saved to {save_path}")
    plt.close(fig)

    return fig


def plot_per_layer_loss_curves(
    results: dict,
    title: str = "Per-Exit-Layer Loss Breakdown",
    save_path: Optional[str] = None,
):
    """
    Bar chart showing reweighted loss per exit layer per timestep bucket.
    Gives a direct view of what each exit layer is predicting.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exit_layers = results["exit_layers"]
    bucket_names = results["bucket_names"]
    n_exits = len(exit_layers)
    n_buckets = len(bucket_names)

    fig, axes = plt.subplots(1, n_buckets, figsize=(6 * n_buckets, 5), sharey=True)
    if n_buckets == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=14, fontweight="bold")
    colors = plt.cm.tab10(np.linspace(0, 1, n_exits))

    for b_idx, (ax, b_name) in enumerate(zip(axes, bucket_names)):
        losses = []
        for e in exit_layers:
            val = results["reweighted_loss"].get(e, {}).get(b_name)
            losses.append(val if val is not None else 0)

        bars = ax.bar(range(n_exits), losses, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(n_exits))
        ax.set_xticklabels([f"L{e}" for e in exit_layers], rotation=45)
        ax.set_title(f"t ∈ {b_name}")
        ax.set_xlabel("Exit Layer")
        if b_idx == 0:
            ax.set_ylabel("Reweighted Masked Diffusion Loss")
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, losses):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Loss curves saved to {save_path}")
    plt.close(fig)

    return fig


def print_summary_table(results: dict):
    """Print a text summary of all metrics to stdout."""
    exit_layers = results["exit_layers"]
    bucket_names = results["bucket_names"]

    print("\n" + "=" * 90)
    print("LAYERSKIP-LLADA DIAGNOSTIC SUMMARY")
    print("=" * 90)

    headers = ["Exit"] + [f"{b}" for b in bucket_names]

    def _print_table(title, metric_key, fmt=".2f"):
        print(f"\n--- {title} ---")
        print(f"{'Layer':<8}", end="")
        for b in bucket_names:
            print(f"{b:>20}", end="")
        print()
        print("-" * (8 + 20 * len(bucket_names)))

        for e in exit_layers:
            print(f"L{e:<7}", end="")
            for b in bucket_names:
                val = results[metric_key].get(e, {}).get(b)
                if val is not None:
                    print(f"{val:>20{fmt}}", end="")
                else:
                    print(f"{'N/A':>20}", end="")
            print()

    _print_table("Masked Diffusion Perplexity", "perplexity", ".1f")
    _print_table("Prediction Agreement with Final Layer", "agreement", ".4f")
    _print_table("Mean Confidence (max softmax)", "mean_confidence", ".4f")
    _print_table("Effective Loss Ratio (exit/base)", "loss_ratio", ".4f")
    _print_table("1/t Cap Utilization Rate", "cap_utilization", ".4f")
    _print_table("Reweighted Loss", "reweighted_loss", ".4f")

    print("\n--- Confidence Quantiles (p10 / p50 / p90) ---")
    print(f"{'Layer':<8}", end="")
    for b in bucket_names:
        print(f"{b:>30}", end="")
    print()
    print("-" * (8 + 30 * len(bucket_names)))
    for e in exit_layers:
        print(f"L{e:<7}", end="")
        for b in bucket_names:
            q = results["confidence_quantiles"].get(e, {}).get(b)
            if q is not None:
                s = f"{q['p10']:.3f} / {q['p50']:.3f} / {q['p90']:.3f}"
                print(f"{s:>30}", end="")
            else:
                print(f"{'N/A':>30}", end="")
        print()

    print("\n" + "=" * 90)
