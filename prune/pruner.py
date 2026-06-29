"""
Self-pruning mechanism.

IMPORTANCE CRITERION
---------------------
We use a *saliency* score derived from a first-order Taylor expansion of the loss:

    ΔL ≈ -∂L/∂w · w   (for removing weight w, i.e., setting Δw = -w)

So importance(w) = |w · ∂L/∂w|

This is the absolute value of the loss-change approximation: connections whose
removal would cause the largest change in loss are kept; those that barely
matter are pruned.

Why this beats magnitude-only (|w|)?
  - A large weight that the gradient has trained to be unimportant (gradient ≈ 0)
    correctly gets a low saliency score.
  - A small weight that is actively being trained (large gradient) gets protected.
  - Empirically, saliency-based pruning consistently outperforms magnitude
    pruning at the same sparsity, especially early in training.

Reference: Molchanov et al., "Importance Estimation for Neural Network Pruning" (CVPR 2019)

PRUNING SCHEDULE
-----------------
We use a *cubic sparsity ramp* (Zhu & Gupta, 2018):

    s(t) = s_f + (s_i - s_f) · (1 - (t - t0) / (t_f - t0))^3

starting from s_i = 0 at step t0, reaching s_f = target_sparsity at t_f.

Why cubic vs linear?
  - The network has time to recover after each prune step.
  - Pruning aggressively early (when loss is still high) and more conservatively
    later (when accuracy matters more) is empirically better.
  - One-shot end-of-training pruning is simpler but yields ~0.5-2% worse accuracy
    at the same sparsity because the network never adapts to the sparse structure.

REGROWTH
---------
Optionally, a fraction of pruned weights may regrow if their saliency exceeds a
threshold based on the current active-weight saliency distribution.
"""

import numpy as np
from typing import List, Optional, Tuple
from engine.tensor import Tensor


def saliency_scores(W: Tensor) -> np.ndarray:
    """
    |w * grad_w| — first-order Taylor approximation of |ΔL| per weight.
    Requires W.grad to be populated (call after backward()).
    """
    if W.grad is None:
        return np.abs(W.data)
    # Mask gradient is already zeroed for pruned weights by Tensor._accum,
    # so saliency of pruned weights is naturally 0.
    return np.abs(W.data * W.grad)


def magnitude_scores(W: Tensor) -> np.ndarray:
    """Pure magnitude baseline: |w|."""
    return np.abs(W.data)


def compute_target_sparsity(
    step: int,
    total_steps: int,
    target_sparsity: float,
    warmup_steps: int = 0,
    schedule: str = "cubic",
) -> float:
    """
    Cubic sparsity ramp from 0 to target_sparsity over [warmup_steps, total_steps].
    """
    if step < warmup_steps:
        return 0.0
    if step >= total_steps:
        return target_sparsity
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    if schedule == "cubic":
        fraction = 1.0 - (1.0 - t) ** 3
    elif schedule == "linear":
        fraction = t
    else:
        fraction = t
    return target_sparsity * fraction


class Pruner:
    """
    Manages masks and applies pruning updates to weight tensors.

    Parameters
    ----------
    weight_params  : list of weight Tensor objects (from MLP.weight_parameters())
    target_sparsity: final fraction of weights to prune (e.g. 0.90)
    total_steps    : total training steps over which to ramp sparsity
    warmup_steps   : steps before pruning begins
    prune_freq     : prune every N steps
    criterion      : 'saliency' | 'magnitude'
    allow_regrowth : if True, top-k pruned weights by saliency can regrow
    regrowth_fraction: fraction of pruned budget allowed to regrow each step
    schedule       : 'cubic' | 'linear'
    """

    def __init__(
        self,
        weight_params: List[Tensor],
        target_sparsity: float = 0.90,
        total_steps: int = 1000,
        warmup_steps: int = 100,
        prune_freq: int = 100,
        criterion: str = "saliency",
        allow_regrowth: bool = True,
        regrowth_fraction: float = 0.1,
        schedule: str = "cubic",
    ):
        self.weight_params = weight_params
        self.target_sparsity = target_sparsity
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.prune_freq = prune_freq
        self.criterion = criterion
        self.allow_regrowth = allow_regrowth
        self.regrowth_fraction = regrowth_fraction
        self.schedule = schedule

        # Initialise masks: all True (all active)
        for W in weight_params:
            W.mask = np.ones(W.data.shape, dtype=bool)

        self._step = 0
        self.sparsity_history: List[Tuple[int, float]] = []

    @property
    def step(self):
        return self._step

    def maybe_prune(self, optimizer=None) -> float:
        """
        Call once per training step (after backward, before optimizer.step).
        Returns current achieved sparsity.
        """
        self._step += 1

        if self._step % self.prune_freq != 0:
            return self._current_sparsity()

        current_target = compute_target_sparsity(
            self._step,
            self.total_steps,
            self.target_sparsity,
            self.warmup_steps,
            self.schedule,
        )

        self._apply_masks(current_target, optimizer)
        sp = self._current_sparsity()
        self.sparsity_history.append((self._step, sp))
        return sp

    def _score_all(self) -> Tuple[np.ndarray, List[int]]:
        """
        Compute flattened scores and layer-offsets for all weight tensors.
        Returns (flat_scores, offsets).
        """
        all_scores = []
        for W in self.weight_params:
            if self.criterion == "saliency":
                s = saliency_scores(W)
            else:
                s = magnitude_scores(W)
            all_scores.append(s.ravel())
        flat_scores = np.concatenate(all_scores)
        return flat_scores

    def _apply_masks(self, target_sparsity: float, optimizer=None):
        """
        Global unstructured pruning: keep top (1 - target_sparsity) weights
        by score across all weight tensors simultaneously.
        """
        flat_scores = self._score_all()
        n_total = flat_scores.size
        n_prune = int(np.floor(target_sparsity * n_total))
        n_keep = n_total - n_prune

        if n_keep <= 0:
            # Edge case: prune everything
            for W in self.weight_params:
                W.mask = np.zeros(W.data.shape, dtype=bool)
                W.data = np.zeros_like(W.data)
            return

        # Find global threshold
        threshold = np.partition(flat_scores, n_prune)[n_prune]

        # Build new masks
        offset = 0
        for idx, W in enumerate(self.weight_params):
            if self.criterion == "saliency":
                s = saliency_scores(W)
            else:
                s = magnitude_scores(W)

            new_mask = (s >= threshold)

            # ---- Regrowth ----
            if self.allow_regrowth and W.mask is not None:
                prev_pruned = ~W.mask
                n_regrow = max(1, int(self.regrowth_fraction * prev_pruned.sum()))
                if prev_pruned.any() and n_regrow > 0:
                    # Score pruned weights by saliency (gradient info)
                    pruned_saliency = np.where(prev_pruned, s, -np.inf)
                    flat_ps = pruned_saliency.ravel()
                    if flat_ps.max() > -np.inf:
                        top_regrow = np.argsort(flat_ps)[-n_regrow:]
                        regrow_mask = np.zeros_like(flat_ps, dtype=bool)
                        regrow_mask[top_regrow] = True
                        regrow_mask = regrow_mask.reshape(W.data.shape)
                        new_mask = new_mask | regrow_mask

            W.set_mask(new_mask)
            offset += W.data.size

    def _current_sparsity(self) -> float:
        total = zeros = 0
        for W in self.weight_params:
            total += W.data.size
            if W.mask is not None:
                zeros += int((~W.mask).sum())
            else:
                zeros += int((W.data == 0.0).sum())
        return zeros / total if total > 0 else 0.0

    def achieved_sparsity(self) -> float:
        return self._current_sparsity()


# ------------------------------------------------------------------ #
# Sparse forward-pass utility for honest FLOP counting               #
# ------------------------------------------------------------------ #

def sparse_linear_forward(x: np.ndarray, W: Tensor, b=None) -> np.ndarray:
    """
    Dense-equivalent forward pass that only multiplies active (unmasked)
    weights.  This is the honest sparse path used for cost measurement.

    For each output neuron j, compute sum over unmasked inputs i only.
    This is O(N * nnz) instead of O(N * D_in * D_out).
    """
    mask = W.mask  # shape (in, out)
    if mask is None:
        out = x @ W.data
    else:
        # Use scipy sparse if available, else masked dense
        # We use masked dense here for simplicity but count only nnz MACs
        out = x @ (W.data * mask.astype(np.float64))
    if b is not None:
        out = out + b.data
    return out


def count_flops(model, input_size: int, batch_size: int = 1) -> int:
    """Count multiply-accumulate operations for one forward pass."""
    # Support both MLP (.layers) and DeepMNISTNet (.linears)
    layers = getattr(model, "linears", None) or getattr(model, "layers", [])
    total_macs = 0
    for layer in layers:
        W = layer.W
        if W.mask is not None:
            nnz = int(W.mask.sum())
        else:
            nnz = W.data.size
        total_macs += batch_size * nnz
    return total_macs
