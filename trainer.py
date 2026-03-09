"""
Custom Trainer for LayerSkip-LLaDA.

Overrides compute_loss to implement:
  - Forward masking process (LLaDA diffusion)
  - Base masked-diffusion CE loss with 1/t reweighting (exact bound)
  - Capped early-exit losses at intermediate layers
  - Depth- and timestep-modulated weighting w(e, t)
  - Curriculum integration (rotational/gradual/timestep annealing)
  - Per-exit-layer diagnostic metrics (logged every N steps, plotted at phase boundaries)

Supports both pre-training and SFT modes.
"""

import logging
import math
import os
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from transformers import Trainer

from .config import TrainingConfig
from .curriculum import CurriculumScheduler
from .metrics import MetricsComputer, plot_diagnostic_dashboard, print_summary_table
from .model.layerskip_llada import LayerSkipLLaDA

logger = logging.getLogger(__name__)


class LayerSkipTrainer(Trainer):

    def __init__(
        self,
        ls_model: LayerSkipLLaDA,
        training_config: TrainingConfig,
        curriculum: CurriculumScheduler,
        metrics_every_n_steps: int = 50,
        dashboard_every_n_steps: int = 500,
        **kwargs,
    ):
        super().__init__(model=ls_model, **kwargs)

        self.ls_model = ls_model
        self.tc = training_config
        self.curriculum = curriculum
        self.ls_cfg = training_config.layerskip

        self.n_layers = ls_model.n_layers
        self.tokens_seen: float = 0.0

        self.metrics_every_n_steps = metrics_every_n_steps
        self.dashboard_every_n_steps = dashboard_every_n_steps
        self._metrics = MetricsComputer(
            exit_layers=self.ls_cfg.exit_layers,
            n_layers=self.n_layers,
            c_cap=self.ls_cfg.c_cap,
            eps_scale=self.ls_cfg.eps_scale,
            alpha=self.ls_cfg.alpha,
        )
        self._last_phase = -1

    # ------------------------------------------------------------------
    # Masking process (forward diffusion)
    # ------------------------------------------------------------------

    def forward_process(
        self,
        input_ids: torch.Tensor,
        prompt_lengths: Optional[torch.Tensor] = None,
    ):
        """
        Apply the LLaDA forward masking process.

        Returns:
            noisy_input: (B, n) with MASK tokens
            masked_indices: (B, n) bool — True where we masked
            p_mask: (B, n) per-position mask probabilities
            t: (B,) sampled timesteps
        """
        B, n = input_ids.shape
        MASK_ID = self.tc.mask_token_id
        eps = self.tc.mask_eps

        t = torch.rand(B, device=input_ids.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].expand(B, n)

        rand = torch.rand(B, n, device=input_ids.device)
        masked_indices = rand < p_mask

        if prompt_lengths is not None:
            positions = torch.arange(n, device=input_ids.device).unsqueeze(0).expand(B, n)
            prompt_mask = positions < prompt_lengths.unsqueeze(1)
            masked_indices = masked_indices & ~prompt_mask

        noisy_input = torch.where(masked_indices, MASK_ID, input_ids)
        return noisy_input, masked_indices, p_mask, t

    # ------------------------------------------------------------------
    # Masked diffusion cross-entropy
    # ------------------------------------------------------------------

    def masked_diffusion_ce(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        masked_indices: torch.Tensor,
        p_mask: torch.Tensor,
        cap: Optional[float] = None,
        answer_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute masked diffusion CE with 1/t reweighting.

        CE(logits[masked], targets[masked]) / min(1/p_mask, cap)
        normalized by B * n.

        For SFT mode, answer_lengths is provided and normalization
        divides each sample's tokens by its answer length.
        """
        if masked_indices.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        V = logits.size(-1)
        flat_logits = logits[masked_indices].view(-1, V)
        flat_targets = targets[masked_indices].view(-1)

        token_ce = F.cross_entropy(flat_logits, flat_targets, reduction="none")

        reweight = 1.0 / p_mask[masked_indices]
        if cap is not None:
            reweight = torch.clamp(reweight, max=cap)

        weighted_ce = token_ce * reweight

        if answer_lengths is not None:
            B = logits.size(0)
            flat_answer = answer_lengths[masked_indices].float()
            token_loss = weighted_ce / flat_answer
            loss = token_loss.sum() / B
        else:
            B, n = logits.shape[:2]
            loss = weighted_ce.sum() / (B * n)

        return loss

    # ------------------------------------------------------------------
    # Depth × timestep weighting w(e, t)
    # ------------------------------------------------------------------

    def compute_exit_weight(self, exit_layer: int, t_mean: float) -> float:
        """w(e, t) = (e/L)^2 * ((1-t) + alpha)"""
        depth_factor = (exit_layer / self.n_layers) ** 2
        time_factor = (1.0 - t_mean) + self.ls_cfg.alpha
        return depth_factor * time_factor

    # ------------------------------------------------------------------
    # Core loss computation
    # ------------------------------------------------------------------

    def _get_ls_model(self, model) -> "LayerSkipLLaDA":
        """Unwrap the model from any FSDP/DDP wrappers to get LayerSkipLLaDA."""
        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        return unwrapped

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        prompt_lengths = inputs.get("prompt_length", None)
        B, n = input_ids.shape

        self.tokens_seen += B * n

        ls_model = self._get_ls_model(model)

        noisy_input, masked_indices, p_mask, t = self.forward_process(
            input_ids, prompt_lengths
        )

        if self.tc.mode == "pretrain" and torch.rand(1).item() < self.tc.random_length_fraction:
            random_length = torch.randint(1, n + 1, (1,)).item()
            noisy_input = noisy_input[:, :random_length]
            input_ids = input_ids[:, :random_length]
            masked_indices = masked_indices[:, :random_length]
            p_mask = p_mask[:, :random_length]

        answer_lengths = None
        if prompt_lengths is not None:
            seq_len = input_ids.size(1)
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(B, seq_len)
            answer_mask = positions >= prompt_lengths.unsqueeze(1)
            al = answer_mask.sum(dim=-1, keepdim=True).expand(B, seq_len)
            answer_lengths = al

        t_mean = t.mean().item()

        if self.curriculum.config.mode == "rotational":
            rot_exit = self.curriculum.get_rotational_exit(
                self.state.global_step, self.tokens_seen
            )
            active_exits = [rot_exit]
        else:
            active_exits = self.curriculum.get_active_exit_layers(self.tokens_seen)

        all_exits = list(set(active_exits + [self.n_layers]))

        outputs = ls_model.forward_with_exits(
            input_ids=noisy_input,
            exit_layers=all_exits,
            t=t_mean,
        )

        # L_base: full 1/t, no cap
        l_base = self.masked_diffusion_ce(
            outputs.exit_logits[self.n_layers],
            input_ids,
            masked_indices,
            p_mask,
            cap=None,
            answer_lengths=answer_lengths,
        )

        # Early-exit losses
        exit_losses: Dict[int, torch.Tensor] = {}
        for e in active_exits:
            if e == self.n_layers:
                continue
            if e not in outputs.exit_logits:
                continue

            if not self.curriculum.should_apply_exit_loss(t_mean, self.tokens_seen):
                continue

            l_exit = self.masked_diffusion_ce(
                outputs.exit_logits[e],
                input_ids,
                masked_indices,
                p_mask,
                cap=self.ls_cfg.c_cap,
                answer_lengths=answer_lengths,
            )
            exit_losses[e] = l_exit

        # Combine: L_total = L_base + (eps_scale / |E_active|) * Σ w(e,t) * L_exit^e
        if exit_losses:
            weighted_sum = sum(
                self.compute_exit_weight(e, t_mean) * loss
                for e, loss in exit_losses.items()
            )
            n_active = len(exit_losses)
            loss = l_base + (self.ls_cfg.eps_scale / n_active) * weighted_sum
        else:
            loss = l_base

        # --- Per-exit scalar logging (cheap, every step) ---
        step = self.state.global_step if self.state else 0
        with torch.no_grad():
            self._step_logs = {"loss/base": l_base.item()}
            for e, le in exit_losses.items():
                w_e = self.compute_exit_weight(e, t_mean)
                self._step_logs[f"loss/exit_L{e}"] = le.item()
                eff = (self.ls_cfg.eps_scale / max(len(exit_losses), 1)) * w_e * le.item()
                self._step_logs[f"loss/effective_L{e}"] = eff

            # Feed the metrics accumulator for periodic diagnostics.
            # Run on all exit layers (not just active ones) when doing a
            # diagnostic step, to get a full picture.
            should_diagnose = (step > 0 and step % self.metrics_every_n_steps == 0)
            if should_diagnose:
                all_exit_layers = self.ls_cfg.exit_layers
                if set(all_exit_layers) != set(all_exits):
                    diag_out = ls_model.forward_with_exits(
                        input_ids=noisy_input,
                        exit_layers=all_exit_layers,
                        t=t_mean,
                    )
                    diag_logits = diag_out.exit_logits
                else:
                    diag_logits = outputs.exit_logits

                self._metrics.update(
                    exit_logits=diag_logits,
                    input_ids=input_ids,
                    masked_indices=masked_indices,
                    p_mask=p_mask,
                    t=t,
                )

        if return_outputs:
            return loss, {"final_logits": outputs.final_logits}
        return loss

    # ------------------------------------------------------------------
    # Logging & diagnostics
    # ------------------------------------------------------------------

    def log(self, logs: Dict, *args, **kwargs):
        logs["tokens_seen"] = self.tokens_seen
        curriculum_state = self.curriculum.get_state(
            self.state.global_step, self.tokens_seen
        )
        for k, v in curriculum_state.items():
            if isinstance(v, (int, float)):
                logs[k] = v

        if hasattr(self, "_step_logs"):
            logs.update(self._step_logs)

        step = self.state.global_step if self.state else 0

        if step > 0 and step % self.dashboard_every_n_steps == 0:
            self._emit_diagnostics(step)

        current_phase = self.curriculum.get_current_phase(self.tokens_seen)
        if current_phase != self._last_phase and self._last_phase >= 0:
            logger.info(f"Phase transition {self._last_phase} -> {current_phase}, generating dashboard")
            self._emit_diagnostics(step, phase_boundary=True)
        self._last_phase = current_phase

        super().log(logs, *args, **kwargs)

    def _emit_diagnostics(self, step: int, phase_boundary: bool = False):
        """Compute accumulated metrics, print summary, save dashboard, then reset."""
        results = self._metrics.compute()

        has_data = any(
            s.total_masked_tokens > 0
            for s in self._metrics.stats.values()
        )
        if not has_data:
            return

        print_summary_table(results)

        output_dir = self.args.output_dir
        diag_dir = os.path.join(output_dir, "diagnostics")
        suffix = f"_phase{self._last_phase}" if phase_boundary else ""
        save_path = os.path.join(diag_dir, f"dashboard_step{step}{suffix}.png")

        plot_diagnostic_dashboard(
            results,
            title=f"Step {step} — Phase {self._last_phase}",
            save_path=save_path,
        )

        self._metrics.reset()
