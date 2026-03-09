"""
Inference algorithms for LayerSkip-LLaDA.

Two decoding strategies from the training recipe:
  1. Depth-Scheduled Diffusion Decoding (Algorithm 1):
     Use a shallow exit for high-noise steps and full depth for low-noise steps.

  2. Self-Speculative Diffusion Decoding (Algorithm 2):
     Draft with early-exit layers, then conditionally verify with remaining
     layers using a percentile-based confidence criterion.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .model.layerskip_llada import LayerSkipLLaDA


# ---------------------------------------------------------------------------
# Utilities (borrowed from LLaDA's generate.py)
# ---------------------------------------------------------------------------

def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer[i, : remainder[i]] += 1
    return num_transfer


def low_confidence_remask(
    x0: torch.Tensor,
    confidence: torch.Tensor,
    mask_index: torch.Tensor,
    num_to_keep: torch.Tensor,
    mask_id: int = 126336,
) -> torch.Tensor:
    """
    Keep the top-num_to_keep most confident predictions, remask the rest.
    Works per sample in the batch.
    """
    result = x0.clone()
    for j in range(x0.size(0)):
        sample_mask = mask_index[j]
        if sample_mask.sum() == 0:
            continue
        sample_conf = confidence[j].clone()
        sample_conf[~sample_mask] = float("inf")

        k = int(num_to_keep[j].item())
        if k <= 0:
            result[j, sample_mask] = mask_id
            continue

        _, keep_idx = torch.topk(sample_conf, k=min(k, int(sample_mask.sum().item())))
        keep_mask = torch.zeros_like(sample_mask)
        keep_mask[keep_idx] = True

        remask_pos = sample_mask & ~keep_mask
        result[j, remask_pos] = mask_id

    return result


# ---------------------------------------------------------------------------
# Algorithm 1: Depth-Scheduled Diffusion Decoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def depth_scheduled_generate(
    model: LayerSkipLLaDA,
    prompt: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    tau: float = 0.5,
    shallow_exit: int = 16,
) -> torch.Tensor:
    """
    Depth-Scheduled Diffusion Decoding.

    For effective timestep t > tau, use `shallow_exit` layers (shallow/fast).
    For t <= tau, use full depth (accurate).

    Args:
        model: LayerSkipLLaDA model
        prompt: (1, L) prompt token ids
        steps: number of denoising steps
        gen_length: number of tokens to generate
        block_length: block length for semi-AR remasking
        temperature: Gumbel noise temperature
        remasking: 'low_confidence' or 'random'
        mask_id: mask token id
        tau: depth-schedule threshold
        shallow_exit: exit layer for shallow steps
    """
    B = prompt.shape[0]
    x = torch.full((B, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=prompt.device)
    x[:, : prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((B, gen_length), dtype=attention_mask.dtype, device=prompt.device),
        ], dim=-1)

    prompt_index = x != mask_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            t_eff = 1.0 - (i / steps_per_block)

            if t_eff > tau:
                out = model.forward_early_exit(x, exit_layer=shallow_exit, attention_mask=attention_mask)
                logits = out.logits
            else:
                out = model.forward(x, attention_mask=attention_mask)
                logits = out.logits if hasattr(out, "logits") else out[0]

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits.float(), dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                x0_p = torch.rand(x0.shape, device=x0.device)
            else:
                raise ValueError(f"Unknown remasking: {remasking}")

            x0_p[:, block_end:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x.device))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(B):
                _, select_idx = torch.topk(confidence[j], k=int(num_transfer_tokens[j, i].item()))
                transfer_index[j, select_idx] = True
            x[transfer_index] = x0[transfer_index]

    return x


# ---------------------------------------------------------------------------
# Algorithm 2: Self-Speculative Diffusion Decoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def self_speculative_generate(
    model: LayerSkipLLaDA,
    prompt: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    draft_exit: int = 8,
    gamma_base: float = 0.85,
    gamma_low_t: float = 0.95,
    gamma_t_boundary: float = 0.3,
    delta_quantile: float = 0.1,
    periodic_verify_k: int = 8,
) -> torch.Tensor:
    """
    Self-Speculative Diffusion Decoding with percentile-based verify-skip.

    At each step:
      1. Draft: run forward_early_exit(draft_exit) → get draft predictions + confidence
      2. Decide: check if δ-quantile of confidence > γ(t) → if yes, skip verify
      3. If not skipped: run forward_remainder to verify, use verify predictions
      4. Remask low-confidence tokens

    Periodic verification every k steps is always full-depth.

    Args:
        model: LayerSkipLLaDA model
        gamma_base: confidence threshold for high-t steps
        gamma_low_t: confidence threshold for low-t steps (more conservative)
        gamma_t_boundary: t below which gamma_low_t is used
        delta_quantile: fraction of masked tokens whose confidence must exceed gamma
        periodic_verify_k: force full-depth verify every k steps
    """
    B = prompt.shape[0]
    x = torch.full((B, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=prompt.device)
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

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            t_eff = 1.0 - (i / steps_per_block)

            gamma = gamma_base if t_eff > gamma_t_boundary else gamma_low_t
            force_verify = (i % periodic_verify_k == 0)

            # --- Draft ---
            draft_out = model.forward_early_exit(x, exit_layer=draft_exit, attention_mask=attention_mask)
            draft_logits = draft_out.logits
            cached_hidden = draft_out.hidden_states

            draft_p = F.softmax(draft_logits.float(), dim=-1)
            draft_x0 = torch.argmax(add_gumbel_noise(draft_logits, temperature=temperature), dim=-1)
            draft_conf = torch.gather(draft_p, dim=-1, index=draft_x0.unsqueeze(-1)).squeeze(-1)

            # --- Decide whether to verify ---
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
                logits = draft_logits
                x0 = draft_x0
                x0_p = draft_conf
            else:
                # --- Verify: run remaining layers ---
                verify_logits = model.forward_remainder(
                    cached_hidden,
                    exit_layer=draft_exit,
                    attention_mask=attention_mask,
                    input_ids=x,
                )
                logits = verify_logits
                x0 = torch.argmax(add_gumbel_noise(logits, temperature=temperature), dim=-1)
                verify_p = F.softmax(logits.float(), dim=-1)
                x0_p = torch.gather(verify_p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

            # --- Remask ---
            x0_p[:, block_end:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x.device))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(B):
                k = int(num_transfer_tokens[j, i].item())
                if k > 0:
                    _, select_idx = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_idx] = True
            x[transfer_index] = x0[transfer_index]

    return x
