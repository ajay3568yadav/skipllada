"""
LayerSkip wrapper for LLaDA: early-exit forward passes, activation caching,
and timestep-conditioned per-sample layer dropout.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .modeling_llada import LLaDAModel, LLaDAModelLM, LLaDAConfig, create_model_config_from_pretrained_config


@dataclass
class EarlyExitOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor


@dataclass
class MultiExitOutput:
    final_logits: torch.Tensor
    exit_logits: Dict[int, torch.Tensor]


class LayerSkipLLaDA(nn.Module):
    """
    Wraps a pre-trained LLaDAModelLM with LayerSkip capabilities:
      - forward_with_exits: full forward collecting logits at designated exit layers
      - forward_early_exit: partial forward up to exit_layer, returns logits + cached h_E
      - forward_remainder: resume from cached h_E through remaining layers
      - timestep-conditioned per-sample layer dropout during training
    """

    def __init__(
        self,
        base_model: LLaDAModelLM,
        exit_layers: List[int] = None,
        p_max: float = 0.1,
        beta: float = 0.3,
        layer_dropout_enabled: bool = True,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config

        self.n_layers = self.config.n_layers
        self.exit_layers = exit_layers or [4, 8, 16, 24, self.n_layers]
        self.p_max = p_max
        self.beta = beta
        self.layer_dropout_enabled = layer_dropout_enabled
        self._use_gradient_checkpointing = False

    @property
    def _llada(self) -> LLaDAModel:
        """Always access the inner LLaDAModel through base_model so FSDP
        parameter management is respected (no stale sharded-weight refs)."""
        return self.base_model.model

    @property
    def device(self) -> torch.device:
        return self._llada.device

    @property
    def dtype(self) -> torch.dtype:
        return self.base_model.dtype

    def _get_blocks(self) -> nn.ModuleList:
        return self._llada.transformer.blocks

    def _apply_norm(self, x: torch.Tensor) -> torch.Tensor:
        return self._llada.transformer.ln_f(x)

    def _apply_lm_head(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.weight_tying:
            return F.linear(x, self._llada.transformer.wte.weight, None)
        return self._llada.transformer.ff_out(x)

    def _get_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        normed = self._apply_norm(hidden)
        logits = self._apply_lm_head(normed)
        if self.config.scale_logits:
            logits = logits * (1.0 / math.sqrt(self.config.d_model))
        return logits

    def _embed(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Embed tokens and prepare attention bias (shared preamble)."""
        batch_size, seq_len = input_ids.shape

        x = self._llada.transformer.wte(input_ids)
        if self.config.input_emb_norm:
            x = x * (self.config.d_model ** 0.5)
        x = self._llada.transformer.emb_drop(x)

        if attention_mask is not None and 0.0 in attention_mask:
            attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min
        else:
            attention_mask = None

        attention_bias = self._llada.get_bidirectional_attention_bias(seq_len, x.device)
        mask_len = seq_len
        if attention_mask is not None:
            mask_len = attention_mask.shape[-1]
        attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)
        if attention_mask is not None:
            attention_bias = attention_bias + attention_mask
            from .modeling_llada import ensure_finite_
            ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)

        return x, attention_bias

    def _compute_layer_dropout_rate(self, layer_idx: int, t: float) -> float:
        """Exponential schedule modulated by diffusion timestep t."""
        if layer_idx == 0:
            return 0.0
        D_l = math.exp(layer_idx * math.log(2) / (self.n_layers - 1)) - 1
        p_l = self.p_max * D_l
        p_lt = p_l * (self.beta + (1.0 - self.beta) * t)
        return p_lt

    def _run_block(self, block: nn.Module, x: torch.Tensor, attention_bias: Optional[torch.Tensor]) -> torch.Tensor:
        """Run a single transformer block, optionally with gradient checkpointing."""
        if self._use_gradient_checkpointing and self.training:
            def _fwd(x, attn_bias):
                out, _ = block(x, attention_bias=attn_bias)
                return out
            return torch_checkpoint(_fwd, x, attention_bias, use_reentrant=False)
        out, _ = block(x, attention_bias=attention_bias)
        return out

    def _run_block_with_dropout(
        self,
        block: nn.Module,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor],
        layer_idx: int,
        t: float,
    ) -> torch.Tensor:
        """Run a single block with optional per-sample layer dropout."""
        if self.training and self.layer_dropout_enabled:
            p_lt = self._compute_layer_dropout_rate(layer_idx, t)
            if p_lt > 0.0:
                B = x.size(0)
                keep = torch.bernoulli(
                    torch.full((B, 1, 1), 1.0 - p_lt, device=x.device, dtype=x.dtype)
                )
                new_x = self._run_block(block, x, attention_bias)
                return keep * new_x + (1.0 - keep) * x

        return self._run_block(block, x, attention_bias)

    def forward_with_exits(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        exit_layers: Optional[List[int]] = None,
        t: float = 0.5,
    ) -> MultiExitOutput:
        """
        Full forward pass collecting logits at each designated exit layer.
        Used during training to compute L_base and all L_exit^e in one pass.

        Args:
            input_ids: (B, n) token ids with mask tokens
            attention_mask: (B, n) optional padding mask
            exit_layers: which layers to tap (defaults to self.exit_layers)
            t: diffusion timestep (scalar, used for layer dropout rate)

        Returns:
            MultiExitOutput with final_logits and exit_logits dict
        """
        if exit_layers is None:
            exit_layers = self.exit_layers

        x, attention_bias = self._embed(input_ids, attention_mask)
        blocks = self._get_blocks()

        exit_logits: Dict[int, torch.Tensor] = {}

        for layer_idx, block in enumerate(blocks):
            x = self._run_block_with_dropout(block, x, attention_bias, layer_idx, t)

            block_num = layer_idx + 1
            if block_num in exit_layers and block_num != self.n_layers:
                exit_logits[block_num] = self._get_logits(x)

        final_logits = self._get_logits(x)
        exit_logits[self.n_layers] = final_logits

        return MultiExitOutput(final_logits=final_logits, exit_logits=exit_logits)

    def forward_early_exit(
        self,
        input_ids: torch.LongTensor,
        exit_layer: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> EarlyExitOutput:
        """
        Run only the first `exit_layer` blocks, then apply shared norm + lm_head.
        Used at inference time for the draft stage.

        Returns:
            EarlyExitOutput with logits and cached hidden_states h_E
        """
        x, attention_bias = self._embed(input_ids, attention_mask)
        blocks = self._get_blocks()

        for layer_idx in range(exit_layer):
            x, _ = blocks[layer_idx](x, attention_bias=attention_bias)

        logits = self._get_logits(x)
        return EarlyExitOutput(logits=logits, hidden_states=x)

    def forward_remainder(
        self,
        cached_hidden: torch.Tensor,
        exit_layer: int,
        attention_bias: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Resume from cached hidden states at layer `exit_layer`, run remaining
        blocks through to L, apply norm + lm_head. Used at inference time for
        the verify stage after forward_early_exit.

        If attention_bias is None but input_ids is provided, it will be
        recomputed (needed for the first call).

        Returns:
            logits tensor (B, n, V)
        """
        if attention_bias is None and input_ids is not None:
            batch_size, seq_len = input_ids.shape
            if attention_mask is not None and 0.0 in attention_mask:
                attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
                attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min
            else:
                attention_mask = None
            attention_bias = self._llada.get_bidirectional_attention_bias(seq_len, cached_hidden.device)
            attention_bias = attention_bias[:, :, :seq_len, :seq_len].to(dtype=torch.float)
            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask
                from .modeling_llada import ensure_finite_
                ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)

        blocks = self._get_blocks()
        x = cached_hidden
        for layer_idx in range(exit_layer, self.n_layers):
            x, _ = blocks[layer_idx](x, attention_bias=attention_bias)

        return self._get_logits(x)

    # ------------------------------------------------------------------
    # Delegate HF Trainer-expected attributes to the base PreTrainedModel
    # ------------------------------------------------------------------

    def gradient_checkpointing_enable(self, **kwargs):
        self._use_gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self._use_gradient_checkpointing = False

    @property
    def gradient_checkpointing(self):
        return self._use_gradient_checkpointing

    def save_pretrained(self, *args, **kwargs):
        return self.base_model.save_pretrained(*args, **kwargs)

    @property
    def generation_config(self):
        return self.base_model.generation_config

    @generation_config.setter
    def generation_config(self, value):
        self.base_model.generation_config = value

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Standard full forward pass (no early exits). For compatibility."""
        return self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> "LayerSkipLLaDA":
        """Load a pre-trained LLaDA model and wrap it."""
        if "torch_dtype" in kwargs:
            kwargs["dtype"] = kwargs.pop("torch_dtype")
        base_model = LLaDAModelLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            **kwargs,
        )
        return cls(base_model=base_model)
