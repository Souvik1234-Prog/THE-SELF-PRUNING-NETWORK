"""
Reverse-mode automatic differentiation engine.
Pure Python + NumPy, no external autodiff libraries.
"""

import numpy as np
from typing import Optional, Tuple, List, Callable, Set


class Tensor:
    """
    A multidimensional array with reverse-mode autodiff support.

    Each Tensor tracks:
      - data  : the numpy array of values
      - grad  : accumulated gradient (same shape as data)
      - _parents : the (Tensor, ...) inputs that produced this tensor
      - _backward : closure that pushes gradients to _parents
      - mask  : optional boolean mask; masked (False) positions are treated
                as structural zeros in both forward and backward passes.
    """

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        _parents: Tuple["Tensor", ...] = (),
        _op: str = "",
        mask: Optional[np.ndarray] = None,
    ):
        if isinstance(data, np.ndarray):
            self.data = data.astype(np.float64)
        else:
            self.data = np.array(data, dtype=np.float64)

        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        self._parents = _parents
        self._backward: Callable[[], None] = lambda: None
        self._op = _op

        # Mask: True = active connection, False = pruned
        # Stored on leaf tensors (parameters). None means "no mask".
        self.mask: Optional[np.ndarray] = mask

        # Apply mask to data immediately so forward pass is honest
        if self.mask is not None:
            self.data = self.data * self.mask.astype(np.float64)

    # ------------------------------------------------------------------ #
    # Gradient initialisation / accumulation helpers                       #
    # ------------------------------------------------------------------ #

    def _init_grad(self):
        if self.grad is None:
            self.grad = np.zeros_like(self.data)

    def _accum(self, delta: np.ndarray):
        """Accumulate gradient, respecting mask on leaf parameters."""
        self._init_grad()
        # Unbroadcast delta to match self.data shape
        delta = _unbroadcast(delta, self.data.shape)
        # Masked weights receive ZERO gradient — they are structurally absent.
        # This is the correct choice: a removed connection has no effect on
        # the loss, so d_loss/d_w = 0 for masked weights.
        if self.mask is not None:
            delta = delta * self.mask.astype(np.float64)
        self.grad += delta

    # ------------------------------------------------------------------ #
    # Topological sort & backward                                          #
    # ------------------------------------------------------------------ #

    def backward(self, grad: Optional[np.ndarray] = None):
        """Run reverse-mode AD from this tensor."""
        if grad is None:
            assert self.data.shape == (), "grad required for non-scalar tensors"
            grad = np.ones_like(self.data)

        # Build topological order
        topo: List["Tensor"] = []
        visited: Set[int] = set()

        def build(t: "Tensor"):
            if id(t) not in visited:
                visited.add(id(t))
                for p in t._parents:
                    build(p)
                topo.append(t)

        build(self)

        # Seed gradient
        self._init_grad()
        self.grad += grad

        # Reverse topological order
        for t in reversed(topo):
            t._backward()

    # ------------------------------------------------------------------ #
    # Differentiable operations                                            #
    # ------------------------------------------------------------------ #

    def __add__(self, other: "Tensor") -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(self.data + other.data, _parents=(self, other), _op="add")

        def _backward():
            if self.requires_grad:
                self._accum(out.grad)
            if other.requires_grad:
                other._accum(out.grad)

        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out

    def __radd__(self, other):
        return self.__add__(_ensure_tensor(other))

    def __sub__(self, other: "Tensor") -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(self.data - other.data, _parents=(self, other), _op="sub")

        def _backward():
            if self.requires_grad:
                self._accum(out.grad)
            if other.requires_grad:
                other._accum(-out.grad)

        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out

    def __rsub__(self, other):
        return _ensure_tensor(other).__sub__(self)

    def __mul__(self, other: "Tensor") -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(self.data * other.data, _parents=(self, other), _op="mul")

        def _backward():
            if self.requires_grad:
                self._accum(out.grad * other.data)
            if other.requires_grad:
                other._accum(out.grad * self.data)

        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out

    def __rmul__(self, other):
        return self.__mul__(_ensure_tensor(other))

    def __truediv__(self, other: "Tensor") -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(self.data / other.data, _parents=(self, other), _op="div")

        def _backward():
            if self.requires_grad:
                self._accum(out.grad / other.data)
            if other.requires_grad:
                other._accum(-out.grad * self.data / (other.data ** 2))

        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out

    def __rtruediv__(self, other):
        return _ensure_tensor(other).__truediv__(self)

    def __neg__(self):
        return self.__mul__(_ensure_tensor(-1.0))

    def matmul(self, other: "Tensor") -> "Tensor":
        """Matrix multiplication: self @ other."""
        other = _ensure_tensor(other)
        out = Tensor(self.data @ other.data, _parents=(self, other), _op="matmul")

        def _backward():
            if self.requires_grad:
                # d_self = out.grad @ other.data.T
                self._accum(out.grad @ other.data.T)
            if other.requires_grad:
                # d_other = self.data.T @ out.grad
                other._accum(self.data.T @ out.grad)

        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    # ------------------------------------------------------------------ #
    # Reductions                                                           #
    # ------------------------------------------------------------------ #

    def sum(self, axis=None, keepdims=False) -> "Tensor":
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            _parents=(self,),
            _op="sum",
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                self._accum(np.broadcast_to(grad, self.data.shape).copy())

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def mean(self, axis=None, keepdims=False) -> "Tensor":
        n = self.data.size if axis is None else self.data.shape[axis]
        out = Tensor(
            self.data.mean(axis=axis, keepdims=keepdims),
            _parents=(self,),
            _op="mean",
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad / n
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                self._accum(np.broadcast_to(grad, self.data.shape).copy())

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    # ------------------------------------------------------------------ #
    # Non-linearities                                                      #
    # ------------------------------------------------------------------ #

    def relu(self) -> "Tensor":
        mask = (self.data > 0).astype(np.float64)
        out = Tensor(self.data * mask, _parents=(self,), _op="relu")

        def _backward():
            if self.requires_grad:
                self._accum(out.grad * mask)

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = Tensor(t, _parents=(self,), _op="tanh")

        def _backward():
            if self.requires_grad:
                self._accum(out.grad * (1.0 - t ** 2))

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def sigmoid(self) -> "Tensor":
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -500, 500)))
        out = Tensor(s, _parents=(self,), _op="sigmoid")

        def _backward():
            if self.requires_grad:
                self._accum(out.grad * s * (1.0 - s))

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def gelu(self) -> "Tensor":
        """GELU approximation: 0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))."""
        c = np.sqrt(2.0 / np.pi)
        x = self.data
        inner = c * (x + 0.044715 * x ** 3)
        t = np.tanh(inner)
        g = 0.5 * x * (1.0 + t)
        out = Tensor(g, _parents=(self,), _op="gelu")

        def _backward():
            if self.requires_grad:
                dtanh = 1.0 - t ** 2
                dinner = c * (1.0 + 3 * 0.044715 * x ** 2)
                dg = 0.5 * (1.0 + t) + 0.5 * x * dtanh * dinner
                self._accum(out.grad * dg)

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    # ------------------------------------------------------------------ #
    # Loss                                                                 #
    # ------------------------------------------------------------------ #

    def softmax_cross_entropy(self, targets: np.ndarray) -> "Tensor":
        """
        Numerically stable softmax + cross-entropy loss.
        targets: integer class indices of shape (N,).
        Returns scalar mean loss.
        """
        # Stable softmax
        logits = self.data
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)

        N = logits.shape[0]
        log_probs = np.log(probs[np.arange(N), targets] + 1e-12)
        loss_val = -log_probs.mean()

        out = Tensor(loss_val, _parents=(self,), _op="softmax_ce")

        def _backward():
            if self.requires_grad:
                d = probs.copy()
                d[np.arange(N), targets] -= 1.0
                d /= N
                self._accum(out.grad * d)

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    # ------------------------------------------------------------------ #
    # Utilities                                                            #
    # ------------------------------------------------------------------ #

    def apply_mask(self):
        """Re-apply mask to data (call after loading weights from disk)."""
        if self.mask is not None:
            self.data = self.data * self.mask.astype(np.float64)

    def set_mask(self, mask: np.ndarray):
        """Update mask and immediately zero out pruned positions."""
        self.mask = mask.astype(bool)
        self.data = self.data * self.mask.astype(np.float64)

    def zero_grad(self):
        self.grad = None

    def __repr__(self):
        return (
            f"Tensor(shape={self.data.shape}, op={self._op!r}, "
            f"requires_grad={self.requires_grad})"
        )

    # Allow indexing for convenience
    def __getitem__(self, idx):
        return Tensor(self.data[idx])


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _ensure_tensor(x) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return Tensor(np.array(x, dtype=np.float64))


def _unbroadcast(grad: np.ndarray, shape: Tuple) -> np.ndarray:
    """
    Sum grad over axes that were broadcast to produce shape `shape`.
    Handles the common case (N, D) + (D,) -> grad for (D,) bias.
    """
    if grad.shape == shape:
        return grad
    # Sum over leading dimensions that were added
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # Sum over dimensions that were broadcast (size 1 in target)
    for i, (gs, ts) in enumerate(zip(grad.shape, shape)):
        if ts == 1 and gs != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad
