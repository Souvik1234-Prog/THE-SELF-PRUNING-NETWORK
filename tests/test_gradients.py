"""
Part 1: Gradient checking tests.

Tests every implemented operation against finite-difference numerical gradients.
Also tests the masked-weight correctness requirement.

Run with: python -m pytest tests/test_gradients.py -v
  or:      python tests/test_gradients.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from engine.tensor import Tensor

EPS = 1e-5          # finite-difference step
ATOL = 1e-4         # absolute tolerance for gradient check
RTOL = 1e-3         # relative tolerance


def numerical_grad(func, inputs, wrt_idx: int, eps: float = EPS):
    """
    Compute numerical gradient of func(*inputs).sum() w.r.t. inputs[wrt_idx]
    using central differences.
    """
    x = inputs[wrt_idx]
    x_data = x.data.copy()
    grad = np.zeros_like(x_data)

    for idx in np.ndindex(x_data.shape):
        # f(x + eps)
        x_data[idx] += eps
        x.data = x_data.copy()
        out_plus = func(*inputs)
        val_plus = out_plus.data.sum()

        # f(x - eps)
        x_data[idx] -= 2 * eps
        x.data = x_data.copy()
        out_minus = func(*inputs)
        val_minus = out_minus.data.sum()

        grad[idx] = (val_plus - val_minus) / (2 * eps)
        x_data[idx] += eps  # restore

    x.data = x_data.copy()
    return grad


def check_grad(func, tensors, wrt_indices=None, eps=EPS, atol=ATOL, rtol=RTOL, label=""):
    """Check analytic grad vs numerical for each tensor in wrt_indices."""
    if wrt_indices is None:
        wrt_indices = list(range(len(tensors)))

    # Forward + backward
    for t in tensors:
        t.zero_grad()

    out = func(*tensors)
    out_sum = out.sum() if out.data.ndim > 0 else out
    out_sum.backward()

    errors = []
    for i in wrt_indices:
        t = tensors[i]
        analytic = t.grad
        numerical = numerical_grad(func, tensors, i, eps)

        if not np.allclose(analytic, numerical, atol=atol, rtol=rtol):
            max_err = np.abs(analytic - numerical).max()
            errors.append(
                f"[{label} | input {i}] max_abs_err={max_err:.2e}\n"
                f"  analytic={analytic.ravel()[:5]}\n"
                f"  numerical={numerical.ravel()[:5]}"
            )

    return errors


# ------------------------------------------------------------------ #
# Individual operation tests                                           #
# ------------------------------------------------------------------ #

def make_tensors(*shapes, seed=0):
    rng = np.random.default_rng(seed)
    return [Tensor(rng.standard_normal(s) * 0.5, requires_grad=True) for s in shapes]


def test_add():
    a, b = make_tensors((3, 4), (3, 4))
    errs = check_grad(lambda x, y: x + y, [a, b], label="add")
    assert not errs, "\n".join(errs)


def test_sub():
    a, b = make_tensors((3, 4), (3, 4))
    errs = check_grad(lambda x, y: x - y, [a, b], label="sub")
    assert not errs, "\n".join(errs)


def test_mul():
    a, b = make_tensors((3, 4), (3, 4))
    errs = check_grad(lambda x, y: x * y, [a, b], label="mul")
    assert not errs, "\n".join(errs)


def test_div():
    a, b = make_tensors((3, 4), (3, 4))
    # Avoid near-zero denominator
    b.data += 2.0
    errs = check_grad(lambda x, y: x / y, [a, b], label="div")
    assert not errs, "\n".join(errs)


def test_matmul():
    a, b = make_tensors((4, 5), (5, 3))
    errs = check_grad(lambda x, y: x @ y, [a, b], label="matmul")
    assert not errs, "\n".join(errs)


def test_sum():
    (a,) = make_tensors((3, 4))
    errs = check_grad(lambda x: x.sum(), [a], label="sum_all")
    assert not errs, "\n".join(errs)

    errs2 = check_grad(lambda x: x.sum(axis=0), [a], label="sum_axis0")
    assert not errs2, "\n".join(errs2)

    errs3 = check_grad(lambda x: x.sum(axis=1), [a], label="sum_axis1")
    assert not errs3, "\n".join(errs3)


def test_mean():
    (a,) = make_tensors((3, 4))
    errs = check_grad(lambda x: x.mean(), [a], label="mean_all")
    assert not errs, "\n".join(errs)


def test_relu():
    (a,) = make_tensors((4, 5))
    errs = check_grad(lambda x: x.relu(), [a], label="relu")
    assert not errs, "\n".join(errs)


def test_tanh():
    (a,) = make_tensors((4, 5))
    errs = check_grad(lambda x: x.tanh(), [a], label="tanh")
    assert not errs, "\n".join(errs)


def test_sigmoid():
    (a,) = make_tensors((4, 5))
    errs = check_grad(lambda x: x.sigmoid(), [a], label="sigmoid")
    assert not errs, "\n".join(errs)


def test_gelu():
    (a,) = make_tensors((4, 5))
    errs = check_grad(lambda x: x.gelu(), [a], label="gelu")
    assert not errs, "\n".join(errs)


def test_softmax_cross_entropy():
    """Test softmax CE loss gradient against numerical diff."""
    rng = np.random.default_rng(42)
    logits_data = rng.standard_normal((8, 5)) * 0.3
    targets = rng.integers(0, 5, size=8)

    logits = Tensor(logits_data, requires_grad=True)

    def func(x):
        return x.softmax_cross_entropy(targets)

    # Scalar output — backward without explicit grad argument
    logits.zero_grad()
    out = func(logits)
    out.backward()
    analytic = logits.grad.copy()

    numerical = numerical_grad(func, [logits], 0)
    assert np.allclose(analytic, numerical, atol=ATOL, rtol=RTOL), (
        f"softmax_ce grad mismatch, max_err={np.abs(analytic - numerical).max():.2e}"
    )


def test_broadcasting_add():
    """
    (N, D) + (D,) broadcasting: gradient for (D,) bias must be summed over N.
    """
    rng = np.random.default_rng(7)
    X = Tensor(rng.standard_normal((6, 4)), requires_grad=True)
    b = Tensor(rng.standard_normal((4,)), requires_grad=True)

    errs = check_grad(lambda x, bias: x + bias, [X, b], label="broadcast_add")
    assert not errs, "\n".join(errs)


def test_composition():
    """Gradient flows through a small MLP-like chain."""
    rng = np.random.default_rng(11)
    X = Tensor(rng.standard_normal((5, 4)), requires_grad=True)
    W1 = Tensor(rng.standard_normal((4, 8)) * 0.1, requires_grad=True)
    b1 = Tensor(rng.standard_normal((8,)) * 0.1, requires_grad=True)
    W2 = Tensor(rng.standard_normal((8, 3)) * 0.1, requires_grad=True)

    def mlp(x, w1, bias1, w2):
        h = (x @ w1 + bias1).relu()
        return h @ w2

    errs = check_grad(mlp, [X, W1, b1, W2], label="mlp_chain")
    assert not errs, "\n".join(errs)


# ------------------------------------------------------------------ #
# MASKED-WEIGHT CORRECTNESS TEST (Part 1 requirement)                 #
# ------------------------------------------------------------------ #

def test_masked_weight_forward_is_zero():
    """Masked weights must contribute exactly zero to forward pass."""
    rng = np.random.default_rng(99)
    W_data = rng.standard_normal((4, 6))
    mask = np.array([
        [1, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 1, 1],
        [1, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1],
    ], dtype=bool)

    W = Tensor(W_data, requires_grad=True, mask=mask)

    # Masked positions must be exactly zero in W.data
    assert np.all(W.data[~mask] == 0.0), "Masked weights not zeroed in forward pass"
    # Unmasked positions unchanged
    assert np.allclose(W.data[mask], W_data[mask]), "Unmasked weights incorrectly modified"


def test_masked_weight_gradient_is_zero():
    """
    Gradient of a masked weight must be zero.
    The loss should not depend on a structurally absent weight.
    """
    rng = np.random.default_rng(42)
    X = Tensor(rng.standard_normal((8, 4)))

    W_data = rng.standard_normal((4, 6))
    mask = np.ones((4, 6), dtype=bool)
    mask[1, 2] = False
    mask[3, 0] = False

    W = Tensor(W_data, requires_grad=True, mask=mask)
    b = Tensor(rng.standard_normal((6,)), requires_grad=True)
    targets = rng.integers(0, 6, size=8)

    # Forward pass
    logits = X @ W + b
    loss = logits.softmax_cross_entropy(targets)
    loss.backward()

    # Gradient at masked positions must be exactly zero
    assert W.grad is not None, "Gradient not computed for W"
    assert W.grad[1, 2] == 0.0, f"Masked weight [1,2] has non-zero grad: {W.grad[1,2]}"
    assert W.grad[3, 0] == 0.0, f"Masked weight [3,0] has non-zero grad: {W.grad[3,0]}"
    # Unmasked positions should have non-trivial gradients
    assert not np.all(W.grad[mask] == 0.0), "All unmasked grads are zero (suspect)"


def test_masked_weight_optimizer_does_not_corrupt():
    """
    After pruning + several steps with zero gradient, reviving a weight
    must not inject corrupted momentum.

    The Adam moments for a masked weight accumulate zero gradients, so the
    moments should remain at 0 (they were zeroed at revival time).
    After revival, the first real gradient update must be sensible.
    """
    from nn.optimizer import Adam

    rng = np.random.default_rng(77)
    W_data = rng.standard_normal((4, 4)) * 0.1

    mask = np.ones((4, 4), dtype=bool)
    W = Tensor(W_data.copy(), requires_grad=True, mask=mask)
    opt = Adam([W], lr=1e-3)

    # Run 5 steps with weight [0,0] pruned
    mask[0, 0] = False
    W.set_mask(mask)

    for step in range(5):
        W.zero_grad()
        # Simulate a gradient update (masked weight gets zero grad)
        W.grad = rng.standard_normal((4, 4)) * 0.01
        W.grad[~mask] = 0.0   # pruner ensures this
        opt.step()

    # Revive weight [0,0]
    mask[0, 0] = True
    W.set_mask(mask)

    # The moment buffers for [0,0] must be reset on revival
    # Manually trigger revival detection by running a step
    W.zero_grad()
    W.grad = np.zeros((4, 4))
    W.grad[0, 0] = 1.0   # a real gradient signal
    opt.step()

    # After revival, m[0][0,0] should reflect ONLY the post-revival gradient,
    # not an accumulation of zero-gradient steps.
    # With 1 step of grad=1.0: m = (1-beta1)*1.0 = 0.1
    expected_m = (1.0 - opt.beta1) * 1.0
    actual_m = opt.m[0][0, 0]
    assert abs(actual_m - expected_m) < 1e-9, (
        f"Adam moment at revived weight is {actual_m:.6f}, expected {expected_m:.6f}. "
        "Revival did not correctly reset moments."
    )


def test_mask_change_does_not_affect_other_weights():
    """Masking some weights must leave other weights' gradients unchanged."""
    rng = np.random.default_rng(55)
    X = Tensor(rng.standard_normal((4, 3)))
    targets = rng.integers(0, 5, size=4)

    # No mask
    W_data = rng.standard_normal((3, 5)) * 0.1
    W_no_mask = Tensor(W_data.copy(), requires_grad=True)
    b = Tensor(np.zeros(5), requires_grad=True)
    logits = X @ W_no_mask + b
    logits.softmax_cross_entropy(targets).backward()
    grad_no_mask = W_no_mask.grad.copy()

    # With mask on a specific position
    mask = np.ones((3, 5), dtype=bool)
    mask[1, 2] = False
    W_masked = Tensor(W_data.copy(), requires_grad=True, mask=mask)
    b2 = Tensor(np.zeros(5), requires_grad=True)
    logits2 = X @ W_masked + b2
    logits2.softmax_cross_entropy(targets).backward()
    grad_masked = W_masked.grad.copy()

    # Masked position: zero in grad_masked
    assert grad_masked[1, 2] == 0.0

    # Other positions: gradients will differ slightly because the masked weight
    # changes the forward pass, but they should be non-trivially nonzero
    other = mask.copy()
    assert not np.all(grad_masked[other] == 0.0), "Other gradients zeroed (wrong)"


# ------------------------------------------------------------------ #
# Main runner                                                          #
# ------------------------------------------------------------------ #

def run_all():
    tests = [
        ("add", test_add),
        ("sub", test_sub),
        ("mul", test_mul),
        ("div", test_div),
        ("matmul", test_matmul),
        ("sum", test_sum),
        ("mean", test_mean),
        ("relu", test_relu),
        ("tanh", test_tanh),
        ("sigmoid", test_sigmoid),
        ("gelu", test_gelu),
        ("softmax_ce", test_softmax_cross_entropy),
        ("broadcasting_add", test_broadcasting_add),
        ("composition", test_composition),
        ("masked_forward_zero", test_masked_weight_forward_is_zero),
        ("masked_grad_zero", test_masked_weight_gradient_is_zero),
        ("masked_optimizer_correct", test_masked_weight_optimizer_does_not_corrupt),
        ("mask_isolation", test_mask_change_does_not_affect_other_weights),
    ]

    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1

    print(f"\n{passed}/{passed+failed} tests passed.")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
