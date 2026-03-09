"""
Training curricula for LayerSkip-LLaDA.

Three strategies:
  1. Rotational: cycle through a single active exit layer every R steps
  2. Gradual: progressively enable shallower exits across training phases
  3. Timestep Annealing: restrict early-exit loss to high-t first, then expand

These can be combined (the Gradual curriculum gates which exit layers are
available; Rotational cycles among those; Timestep Annealing gates the loss
computation itself).
"""

from typing import List, Optional, Tuple

from .config import CurriculumConfig


class CurriculumScheduler:
    """
    Manages the training curriculum and answers two questions at each step:
      1. Which exit layers are active?
      2. For a given timestep t, should early-exit loss be applied?
    """

    def __init__(self, config: CurriculumConfig, n_layers: int = 32, rotation_period: int = 16):
        self.config = config
        self.n_layers = n_layers
        self.rotation_period = rotation_period

        self._build_phase_boundaries()

    def _build_phase_boundaries(self):
        """Compute token boundaries for each phase."""
        self.phase_boundaries: List[float] = []
        cumulative = 0.0
        for frac in self.config.phase_fractions:
            cumulative += frac * self.config.total_tokens
            self.phase_boundaries.append(cumulative)

    def get_current_phase(self, tokens_seen: float) -> int:
        """Return the 0-indexed phase for the given token count."""
        for i, boundary in enumerate(self.phase_boundaries):
            if tokens_seen < boundary:
                return i
        return len(self.phase_boundaries) - 1

    def get_active_exit_layers(self, tokens_seen: float) -> List[int]:
        """
        Return the list of exit layers active in the current phase.
        Phase 0 = deepest only, progressively adding shallower exits.
        Note: n_layers (=32) is always active via L_base, not via this list.
        """
        phase = self.get_current_phase(tokens_seen)
        exits = self.config.phase_exit_layers[phase]
        return [e for e in exits if e != self.n_layers]

    def get_rotational_exit(self, step: int, tokens_seen: float) -> int:
        """
        For the rotational curriculum, return a single active exit layer
        for this step (cycling every R steps among available exits).
        """
        available = self.get_active_exit_layers(tokens_seen)
        if not available:
            return self.n_layers
        idx = (step // self.rotation_period) % len(available)
        return available[idx]

    def get_timestep_threshold(self, tokens_seen: float) -> Optional[float]:
        """
        Return the minimum timestep t for which early-exit losses are active.
        None means no restriction (all t values produce early-exit loss).
        """
        phase = self.get_current_phase(tokens_seen)
        return self.config.timestep_thresholds[phase]

    def should_apply_exit_loss(self, t_val: float, tokens_seen: float) -> bool:
        """
        Given a single timestep value and token count, decide whether to
        apply early-exit loss. Respects timestep annealing.
        """
        threshold = self.get_timestep_threshold(tokens_seen)
        if threshold is None:
            return True
        return t_val >= threshold

    def get_state(self, step: int, tokens_seen: float) -> dict:
        """Return a summary dict for logging."""
        phase = self.get_current_phase(tokens_seen)
        return {
            "curriculum/phase": phase,
            "curriculum/active_exits": self.get_active_exit_layers(tokens_seen),
            "curriculum/timestep_threshold": self.get_timestep_threshold(tokens_seen),
            "curriculum/rotational_exit": self.get_rotational_exit(step, tokens_seen),
        }
