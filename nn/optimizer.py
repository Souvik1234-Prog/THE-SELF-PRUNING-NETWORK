"""
Adam optimizer (and SGD with momentum as baseline).

MASKED-WEIGHT HANDLING
-----------------------
When a weight is pruned its mask entry is False, its data is set to 0,
and its gradient is zeroed by Tensor._accum(). The Adam moment buffers
(m, v) still accumulate — but since the gradient is always 0 for masked
weights, the update is 0 * bias_corrected_lr which is exactly 0.
If a pruned weight is *revived* (mask flipped back to True):
  - m and v already carry history from before pruning.
  - We reset m[revived] = 0, v[revived] = 0 so the revival starts fresh
    with a neutral optimizer state, preventing corrupted momentum from
    the zero-gradient period from polluting the first real update.
  This is the correct choice: the moment buffers for a dead connection
  are meaningless and must not inject phantom momentum on revival.
"""

import numpy as np
from typing import List
from engine.tensor import Tensor


class Adam:
    """
    Adam: Adaptive Moment Estimation (Kingma & Ba, 2015).

    Parameters
    ----------
    params      : list of Tensor (leaf parameters with requires_grad=True)
    lr          : learning rate (step size)
    beta1       : decay rate for first moment
    beta2       : decay rate for second moment
    eps         : numerical stability constant
    weight_decay: L2 regularisation coefficient (applied before moment update)
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        self.t = 0  # global step counter

        # First and second moment buffers, per parameter
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]

        # Track the *previous* mask for each parameter so we can detect
        # newly-revived connections and reset their optimizer state.
        self._prev_mask = [
            p.mask.copy() if p.mask is not None else None for p in params
        ]

    def step(self):
        self.t += 1
        b1, b2, eps = self.beta1, self.beta2, self.eps
        # Bias-correction factors
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t
        alpha = self.lr * np.sqrt(bc2) / bc1

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad.copy()

            # ---- Weight decay (L2, applied to unmasked weights only) ----
            if self.weight_decay != 0.0:
                mask_f = p.mask.astype(np.float64) if p.mask is not None else 1.0
                g = g + self.weight_decay * p.data * mask_f

            # ---- Revival detection: reset moments for revived connections ----
            if p.mask is not None and self._prev_mask[i] is not None:
                # A connection is "revived" if it was False (pruned) and is
                # now True (active again).
                revived = (~self._prev_mask[i]) & p.mask
                if revived.any():
                    self.m[i][revived] = 0.0
                    self.v[i][revived] = 0.0
            # Update stored mask
            if p.mask is not None:
                self._prev_mask[i] = p.mask.copy()

            # ---- Moment updates ----
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * (g ** 2)

            # ---- Parameter update ----
            update = alpha * self.m[i] / (np.sqrt(self.v[i]) + eps)
            p.data -= update

            # ---- Re-apply mask so pruned weights stay exactly zero ----
            if p.mask is not None:
                p.data = p.data * p.mask.astype(np.float64)

    def zero_grad(self, params: List[Tensor]):
        for p in params:
            p.zero_grad()

    def notify_mask_update(self, param: Tensor, idx: int):
        """
        Explicitly notify optimizer that param's mask changed.
        Resets moment buffers for newly-pruned connections (optional call
        from pruner — also handled automatically in step()).
        """
        if param.mask is not None and self._prev_mask[idx] is not None:
            newly_pruned = self._prev_mask[idx] & (~param.mask)
            if newly_pruned.any():
                # Zeroing moments for pruned weights is optional (they will
                # receive zero gradient anyway) but keeps buffers clean.
                self.m[idx][newly_pruned] = 0.0
                self.v[idx][newly_pruned] = 0.0


class SGDMomentum:
    """SGD with momentum (baseline optimizer)."""

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [np.zeros_like(p.data) for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.copy()
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data
            self.velocity[i] = self.momentum * self.velocity[i] + g
            p.data -= self.lr * self.velocity[i]
            if p.mask is not None:
                p.data = p.data * p.mask.astype(np.float64)

    def zero_grad(self, params: List[Tensor]):
        for p in params:
            p.zero_grad()
