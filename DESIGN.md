# DESIGN.md — Self-Pruning Network

Full derivations, architectural decisions, and engineering rationale for every non-trivial choice in this codebase.

---

## Table of Contents

1. [Autodiff Engine Design](#1-autodiff-engine-design)
2. [Importance Criterion Derivation](#2-importance-criterion-derivation)
3. [Gradient of a Masked Weight](#3-gradient-of-a-masked-weight)
4. [Adam Optimiser and Masked-Weight Revival](#4-adam-optimiser-and-masked-weight-revival)
5. [Pruning Schedule Justification](#5-pruning-schedule-justification)
6. [Regrowth: Stability and Theory](#6-regrowth-stability-and-theory)
7. [DeepMNISTNet Architecture Rationale](#7-deepmnistnet-architecture-rationale)
8. [Data Normalisation Choice](#8-data-normalisation-choice)
9. [Autodiff Engine Bottlenecks and How to Fix Them](#9-autodiff-engine-bottlenecks-and-how-to-fix-them)
10. [Serving a Self-Pruned Model in Production](#10-serving-a-self-pruned-model-in-production)

---

## 1. Autodiff Engine Design

### How the engine works

The engine implements **reverse-mode automatic differentiation** (backpropagation) via a dynamically built computation graph. During the forward pass, every `Tensor` operation:

1. Computes the output value using NumPy
2. Records the output tensor's parents (the input tensors)
3. Registers a `_backward` closure on the output that, when called, computes and accumulates the gradient contribution into each parent's `.grad`

This is the same design as PyTorch's autograd engine (minus the C++ backend).

### Topological traversal

`loss.backward()` calls `build_topo(loss)` to produce a topological ordering of the entire computation graph, then iterates in reverse, calling each node's `_backward` closure:

```python
def build_topo(root):
    visited, order = set(), []
    def dfs(node):
        if node not in visited:
            visited.add(node)
            for parent in node._parents:
                dfs(parent)
            order.append(node)
    dfs(root)
    return order

# In backward():
topo = build_topo(self)
self.grad = np.ones_like(self.data)   # seed gradient = 1
for node in reversed(topo):
    node._backward()
```

### Gradient accumulation and masking

`Tensor._accum(delta)` is the single point where gradients are written to `.grad`:

```python
def _accum(self, delta):
    if self.mask is not None:
        delta = delta * self.mask.astype(np.float64)   # zero pruned positions
    if self.grad is None:
        self.grad = delta.copy()
    else:
        self.grad += delta
```

Zeroing the gradient at masked positions *before* accumulation is the key design choice that makes the entire masked-weight system correct. See Section 3 for the full justification.

### Why dynamic graph (define-by-run) rather than static graph

Static graphs (TensorFlow 1.x style) require the user to declare the entire computation before running it, enabling optimisations like kernel fusion and memory planning. Dynamic graphs (PyTorch, our engine) build the graph during execution, making Python control flow (if/for/while) over tensor values natural and easy to debug.

For a research/educational codebase with variable-topology pruning, dynamic graphs are the correct choice: the mask changes during training, meaning the effective computation graph changes too.

---

## 2. Importance Criterion Derivation

### Problem statement

We have `N` weights in the network. We want to prune a fraction `s` of them. Which `s × N` weights should we remove to minimise the increase in loss?

### First-order Taylor approximation

Let `L(θ)` be the loss at current parameters `θ`. We want to approximate the loss change `ΔL` when we remove weight `w_ij` (i.e., set it to zero):

```
ΔL = L(θ - w_ij · e_ij) - L(θ)
```

where `e_ij` is the unit vector at position `(i,j)`.

A first-order Taylor expansion around `θ` gives:

```
ΔL ≈ ∇_θ L · (-w_ij · e_ij) = -w_ij · ∂L/∂w_ij
```

So the magnitude of the loss change from removing `w_ij` is:

```
|ΔL| ≈ |w_ij · ∂L/∂w_ij|
```

We define the **saliency score**:

```
S(w_ij) = |w_ij · grad_ij|
```

Weights with **high saliency** are important: removing them would greatly increase the loss. We keep the top `(1-s)` fraction by saliency and prune the rest.

### Why this strictly beats magnitude-only pruning

Magnitude pruning uses `S(w) = |w|`, which is equivalent to assuming `|grad_ij|` is the same for all weights — a uniformly constant gradient landscape. This is provably false.

Two failure modes of magnitude-only pruning:

**Case 1 — Large dead weight:** A weight `w = 5.0` sits at a saddle point with `grad = 0.0`. Its saliency is `5.0 × 0.0 = 0`: removing it has no effect on the loss. Magnitude pruning incorrectly ranks it as important (`|w| = 5.0`) and keeps it.

**Case 2 — Small active weight:** A weight `w = 0.01` has `grad = 10.0` because it is actively being trained toward an important role. Its saliency is `0.01 × 10.0 = 0.1`: removing it meaningfully increases the loss. Magnitude pruning incorrectly ranks it as unimportant and prunes it.

Saliency handles both cases correctly because it multiplies the two signals.

### Relationship to second-order methods

The second-order approximation (Optimal Brain Damage, LeCun et al. 1990; Optimal Brain Surgeon, Hassibi & Stork 1993) uses the diagonal of the Hessian:

```
|ΔL_2| ≈ (1/2) · h_ij · w_ij²   where h_ij = ∂²L/∂w_ij²
```

This is more accurate but requires computing the Hessian (or a diagonal approximation), which is expensive and not available without storing second-order information through the backward pass. Our first-order criterion reuses gradients already computed in the backward pass at zero additional cost.

Empirically (Molchanov et al., CVPR 2019), first-order saliency performs comparably to second-order methods at 90% sparsity and outperforms them at aggressive sparsities because the Hessian approximation breaks down when large fractions of weights are removed.

### Global vs. layer-wise pruning

We use **global** pruning: a single threshold applied across all weight tensors simultaneously. This allows layers to have different sparsities based on their actual importance — early layers tend to stay denser because their features are reused across many paths, while later layers can be aggressively pruned.

Layer-wise pruning forces a fixed fraction per layer, which can over-prune important layers and under-prune unimportant ones, reducing overall accuracy at the same parameter count.

---

## 3. Gradient of a Masked Weight

### The answer: exactly zero

A masked weight `w_ij` with `mask[i,j] = False` has its data forced to zero and contributes nothing to the forward pass. Therefore:

```
∂L/∂w_ij = 0   for all pruned w_ij
```

### Why this is correct by the chain rule

Consider the forward pass through a linear layer:

```
out_j = Σ_i x_i · w_ij
```

When `mask[i,j] = False`, we enforce `w_ij = 0` in the data array. The output `out_j` does not depend on the value of `w_ij` at all — it has been structurally removed from the computation. Therefore the partial derivative `∂out_j/∂w_ij = 0`, and by the chain rule, `∂L/∂w_ij = (∂L/∂out_j) · (∂out_j/∂w_ij) = 0`.

### Implementation in `Tensor._accum()`

```python
def _accum(self, delta):
    if self.mask is not None:
        delta = delta * self.mask.astype(np.float64)
    # ... accumulate into self.grad
```

This zeroes the gradient at masked positions *before* accumulation. The `_backward` closure for `matmul` computes `dW = X.T @ dout` — the mathematical gradient of the dense operation. This gradient is nonzero at masked positions (it represents "what would happen if we perturbed this position"), but since the weight is structurally absent, we do not want this signal. `_accum` discards it.

### The wrong approach and why it fails

A naive implementation might compute the full mathematical gradient `dW = X.T @ dout` and allow it to flow into the Adam moments without masking. This causes two problems:

1. **Corrupted moments:** Adam accumulates `m = β₁m + (1-β₁)g` and `v = β₂v + (1-β₂)g²`. If a masked weight receives a nonzero gradient (the mathematical gradient of the dense operation), its moments accumulate meaningful signal even though the weight is supposed to be dead. When the weight is revived, the pre-loaded moments create a large spurious update.

2. **Incorrect saliency scores:** The saliency criterion `|w × grad|` requires `w = 0` for a masked weight (enforced) and `grad = 0` (requires correct masking). If the gradient is nonzero, the saliency score of a pruned weight will be nonzero, biasing the pruning decisions.

Both problems are avoided by zeroing the gradient in `_accum()`.

---

## 4. Adam Optimiser and Masked-Weight Revival

### The revival problem

During the pruned phase, a weight `w_ij` has `mask[i,j] = False`. Its gradient is always zero (Section 3). The Adam update for the first moment is:

```
m = β₁ · m + (1 - β₁) · g = β₁ · m + 0 = β₁ · m
```

Over `T` steps, this decays the moment by `β₁^T`. For `β₁ = 0.9` and `T = 100` steps, `β₁^T = 0.9^100 ≈ 2.66 × 10⁻⁵` — essentially zero if the moment was zero at the time of pruning. However, if the weight had a nonzero moment at the time of pruning (because it was actively being trained), that moment decays but does not reach zero within a typical pruning window.

When the weight is revived, this residual moment creates a phantom gradient signal that does not reflect any real gradient of the current loss. The first post-revival update will be pulled in the direction of the pre-pruning momentum, which may be completely wrong in the current loss landscape.

### The fix: moment reset on revival

In `Adam.step()`, before updating moments, we detect newly revived connections:

```python
if p.mask is not None and self._prev_mask[i] is not None:
    revived = (~self._prev_mask[i]) & p.mask   # was False, now True
    if revived.any():
        self.m[i][revived] = 0.0
        self.v[i][revived] = 0.0
self._prev_mask[i] = p.mask.copy()             # update stored mask
```

This runs **before** the moment update, so the revival step starts from a clean state. After one step with gradient `g`, the moment is:

```
m = β₁ · 0 + (1 - β₁) · g = (1 - β₁) · g = 0.1 · g
```

This is the correct behaviour: the moment reflects only post-revival gradient information.

### Test: `masked_optimizer_correct`

The test `test_masked_weight_optimizer_does_not_corrupt` verifies this exactly:

1. Prune weight `[0,0]`
2. Run 5 optimiser steps with zero gradient at `[0,0]`
3. Revive weight `[0,0]`
4. Run one step with `grad[0,0] = 1.0`
5. Assert `m[0][0,0] == (1 - β₁) × 1.0 = 0.1` to within `1e-9`

If moments were not reset, the assertion would fail because `m` would contain a decayed residual from before pruning.

### Why not reset moments on pruning?

When a weight is pruned, its gradient becomes zero. Adam's moment update becomes:

```
m ← β₁ · m   (exponential decay toward zero)
```

The moment naturally decays to zero over time. Resetting on pruning is optional and slightly cleaner but not required for correctness — the pruned weight will never produce a nonzero update because both `g = 0` and `m → 0`. We do implement `notify_mask_update()` for pruning resets as an optional clean-up.

---

## 5. Pruning Schedule Justification

### The cubic ramp

We use the schedule from Zhu & Gupta, "To Prune or Not to Prune" (ICLR 2018):

```
s(t) = s_f · [1 - (1 - (t - t₀) / (t_f - t₀))³]
```

where:
- `s_f` = target sparsity (e.g., 0.90)
- `t₀` = warmup steps (pruning begins after this)
- `t_f` = total steps (pruning ends here)
- `t` = current step

### Why cubic, not linear

The derivative of the cubic schedule is:

```
ds/dt = 3 · s_f · (1 - (t - t₀)/(t_f - t₀))² / (t_f - t₀)
```

At `t = t_f`, this is zero. The schedule slows down as it approaches the target, giving the network maximum time to recover in the critical final phase of training when every removed connection hurts most.

A linear ramp `s(t) = s_f · (t - t₀)/(t_f - t₀)` prunes at a constant rate. This is too aggressive late in training: the network has learned to rely on the remaining connections, and removing them at the same rate as early connections causes a sharp accuracy drop near the end.

### One-shot vs. gradual

One-shot pruning (train fully dense, then prune all at once) is simpler but yields 1-2% worse accuracy at the same sparsity. The reason: the network never adapts its remaining weights to the sparse topology. When weights are removed suddenly, the remaining weights must compensate immediately — but they were optimised for a different (dense) loss landscape. Gradual pruning gives the optimiser continuous feedback to redistribute information.

### Warmup period

We delay the start of pruning for `warmup_steps` (default: 2-3 epochs worth of steps). Pruning at initialisation removes weights before any useful structure has formed. The saliency scores at step 0 are essentially noise (gradients are random because weights are random), so pruning early commits to a bad topology.

After the warmup, the network has learned a rough representation and the saliency scores are meaningful signals.

---

## 6. Regrowth: Stability and Theory

### Motivation

A static pruning mask commits to a topology at pruning time `t`. If the importance estimate at `t` was noisy — which it always is, because we use a stochastic mini-batch gradient — a genuinely important weight may have been incorrectly pruned.

This problem is worst early in training (noisy gradients, unstable loss landscape) and at high sparsity (few remaining weights, each one more critical). Regrowth corrects early mistakes by allowing pruned connections to return.

### Mechanism

After each pruning step, we compute saliency scores for all currently pruned weights using the current gradient. The top `n_regrow = regrowth_fraction × |pruned|` pruned weights by saliency are revived:

```python
pruned_saliency = np.where(prev_pruned, s, -np.inf)   # score only pruned weights
top_regrow = np.argsort(pruned_saliency.ravel())[-n_regrow:]
regrow_mask[top_regrow] = True
new_mask = new_mask | regrow_mask
```

The revived connections have their Adam moments reset to zero (Section 4), preventing corrupted momentum.

### Sparsity target preservation

Regrowth can only grow the mask (add back connections). Since pruning in the same step removes connections globally to hit `target_sparsity`, and regrowth only revives a small fraction, the net sparsity stays at or very near the target. We never revive more connections than we simultaneously prune.

### Connection to dynamic sparse training

This is the core idea of **Sparse Evolutionary Training** (Mocanu et al., 2018) and **RigL** (Evci et al., 2020, "Rigging the Lottery"). These methods maintain a fixed sparsity throughout training by pairing every prune step with a regrow step of the same size. Our implementation is a conservative version with a 5% regrowth budget, sufficient to correct mistakes without causing oscillation.

### Why small regrowth fraction

Setting `regrowth_fraction` too high causes oscillation: a weight gets pruned, revived, pruned again in quick succession, and the optimiser never settles. We use 5% (one-twentieth of the pruned population per step) as a conservative default that provides correction without instability.

---

## 7. DeepMNISTNet Architecture Rationale

### Target: 97% test accuracy on MNIST with pure NumPy

MNIST is a solved problem in PyTorch (>99.5% with convolutions), but our constraint is no external autodiff — we use only NumPy for computation. Within that constraint, a well-designed dense MLP with BatchNorm can reach 97%.

### Layer widths: 512 → 256 → 128 → 64

MNIST has 784 input pixels. The first layer at 512 neurons is deliberately wider than a naive "halving" schedule because:

- The 784 input pixels contain significant redundancy (neighbouring pixels are correlated). A wider first layer has more capacity to learn diverse low-level features before compression.
- BatchNorm after the first layer prevents the wider layer from being slower to train (it normalises activations regardless of width).
- Empirically, 512 outperforms 256 by ~0.3% on MNIST at 60k training samples.

Each subsequent layer halves width, implementing a compression bottleneck: the network is forced to learn increasingly abstract representations as width decreases.

### Why 4 hidden layers

Depth provides compositional representation power: each layer can learn functions of the previous layer's abstractions. For MNIST:

- Layer 1 (512): low-level stroke and edge detectors
- Layer 2 (256): stroke combinations, loop detectors
- Layer 3 (128): digit-part detectors (curved tops, vertical lines)
- Layer 4 (64): near-digit-level representations

Adding a 5th layer (32 neurons) provides no measurable improvement and slows training.

### BatchNorm after every hidden linear

Benefits:
1. **Gradient flow:** BatchNorm prevents the vanishing/exploding gradient problem that would otherwise limit depth to 2-3 layers in a pure NumPy implementation.
2. **Higher learning rate:** With BatchNorm, `lr = 3e-3` converges stably. Without it, we need `lr ≤ 1e-3` and 2-3× more epochs.
3. **Implicit regularisation:** BatchNorm reduces the need for aggressive Dropout, especially in early layers.
4. **Accuracy:** BatchNorm alone is worth ~1-2% on MNIST compared to the same architecture without it.

BatchNorm is placed between the linear layer and the activation, in the standard order:
```
Linear → BatchNorm → ReLU → Dropout
```

### Dropout rates: 0.3, 0.3, 0.2, 0.0

Dropout probability is proportional to the risk of overfitting, which is proportional to the number of parameters:

| Layer | Width | Parameters (W only) | Dropout |
|---|---|---|---|
| 1 | 512 | 784×512 = 401,408 | 0.3 |
| 2 | 256 | 512×256 = 131,072 | 0.3 |
| 3 | 128 | 256×128 = 32,768 | 0.2 |
| 4 | 64 | 128×64 = 8,192 | 0.0 |

The 64-neuron layer has so few parameters that Dropout would discard useful information rather than prevent overfitting. The output layer never has Dropout.

### ReLU throughout

Alternatives considered:
- **Tanh:** Saturates for large inputs, causing vanishing gradients in deeper layers. Good for shallow networks, bad here.
- **Sigmoid:** Same saturation problem, additionally not zero-centred.
- **GELU:** Smooth approximation of ReLU, used in transformers. Marginal improvement on MNIST (~0.1%) at higher compute cost.
- **ReLU:** Dead neuron risk is mitigated by BatchNorm keeping pre-activations near zero. Fast, simple, effective.

### Output layer: raw logits, no activation, no BatchNorm

`softmax_cross_entropy` in the engine implements the log-sum-exp stable computation internally:

```
loss = -Σ log(softmax(logits))
     = -Σ [logits[y] - log(Σ exp(logits))]
     = -Σ [logits[y] - (max + log(Σ exp(logits - max)))]
```

Applying softmax before this function would cause numerical instability (softmax outputs near 0 or 1 create log(0)). Applying BatchNorm to the output would normalise the logit distribution, removing the calibration needed for accurate probability estimates.

---

## 8. Data Normalisation Choice

### Per-pixel zero-mean unit-variance

```python
X = X / 255.0
mean = X.mean(axis=0, keepdims=True)   # shape (1, 784)
std  = X.std(axis=0,  keepdims=True) + 1e-8
X = (X - mean) / std
```

This normalises each of the 784 pixels independently to have zero mean and unit variance across the training set.

### Why not `[-1, 1]` scaling?

`X / 127.5 - 1.0` applies a uniform global shift and scale. Problems:

- Pixel (0,0) is always the top-left corner — in MNIST, this is almost always white (background). Its mean across all digits is near 0 and its variance is near 0. After `[-1,1]` scaling, it is still near -1 with near-zero variance. It contributes nothing to training.
- Per-pixel normalisation centres each pixel around its actual distribution. Pixels in the centre of the image (where digit strokes are concentrated) have high variance and become informative features with zero mean.

This matters for batch normalisation: the BN layer normalises *batch* statistics, but its task is easier if input features are already approximately normalised, preventing the first layer from spending capacity on rescaling.

### The `1e-8` epsilon

Pixels that are always white (background) have `std ≈ 0`. Dividing by zero would produce `NaN`. The `1e-8` floor prevents this while not meaningfully altering pixels with normal variance.

---

## 9. Autodiff Engine Bottlenecks and How to Fix Them

### Current bottlenecks

**1. Python dispatch overhead**

Each `_backward` closure is a Python function call. NumPy array operations are fast (C/Fortran), but the per-operation Python overhead is ~1-5 μs. For a deep network with O(100) operations per forward pass, this is O(100) Python calls per backward pass — manageable, but not scalable to transformers or CNNs.

**2. Graph reconstruction every forward pass**

`build_topo()` runs a DFS on every call to `backward()`. The graph structure (which operations connect which tensors) is fixed for a given architecture and batch size. Traversing it from scratch every time wastes O(N) work.

**Fix:** Cache the topological order after the first forward pass. On subsequent passes, skip `build_topo()` and directly execute the cached `_backward` list.

**3. Memory: all intermediates kept alive**

Every tensor in the computation graph is kept alive in memory until `backward()` completes (Python reference counting). For a 60k-sample MNIST batch (which we split into mini-batches of 128, but hypothetically), storing all activations would be O(depth × batch × layer_width).

**Fix (gradient checkpointing):** Discard activations of checkpointed layers during the forward pass. During the backward pass, recompute from the nearest checkpoint when needed. This halves peak memory at a cost of ~33% extra compute.

**4. No kernel fusion**

The sequence `Linear → BatchNorm → ReLU → Dropout` involves 4 separate NumPy calls, 4 intermediate tensors allocated, and 4 Python function calls. A fused kernel would compute all four in one C-level loop with a single intermediate.

**Fix:** Implement fused forward kernels as Cython/Numba/C extensions, or use `einops` + `numba.jit`.

**5. CPU-only**

NumPy runs on CPU. For 60k MNIST samples × 30 epochs × ~500 ops per forward pass, this is manageable (~minutes). For ImageNet or language models, it would be infeasible.

**Fix:** Replace `import numpy as np` with `import cupy as cp` everywhere. CuPy implements the same API on GPU with near-zero code changes. The Python graph structure stays identical; only array operations run on the device.

---

## 10. Serving a Self-Pruned Model in Production

### The opportunity

A 90%-sparse model has ~10% active weights. For weight-bound linear layers (most of our architecture), the theoretical speedup from exploiting sparsity is 10×. Achieving this requires a **sparse matrix format** and a corresponding sparse BLAS kernel.

### Model serialisation for inference

After training, freeze the mask and export each layer as a CSR (Compressed Sparse Row) sparse matrix:

```python
from scipy.sparse import csr_matrix
import json, numpy as np

def export_layer(layer):
    W_sparse = csr_matrix(layer.W.data * layer.W.mask)
    return {
        "data":    W_sparse.data.tolist(),
        "indices": W_sparse.indices.tolist(),
        "indptr":  W_sparse.indptr.tolist(),
        "shape":   list(W_sparse.shape),
        "bias":    layer.b.data.tolist() if layer.b is not None else None,
    }
```

At inference time, reconstruct with `csr_matrix((data, indices, indptr), shape=shape)` and compute `X @ W_sparse.T` using `scipy.sparse` BLAS. The mask is implicit in the CSR structure — no separate mask array needed.

### Inference architecture

```
Request → Load balancer → Worker pool (N workers = N CPU cores)
                                ↓
                        Each worker holds:
                          - sparse weight matrices (CSR, read-only)
                          - batch buffer (reused per request)
                                ↓
                        sparse forward pass (~μs per sample)
                                ↓
                        Response (class probabilities)
```

**No shared mutable state between workers.** Each worker loads its own copy of the model at startup. This avoids locks and enables linear scaling with core count.

### Batching for throughput

Individual inference requests are fast (~μs). For maximum throughput, group requests arriving within a time window `τ` into a batch matrix `X ∈ ℝ^{B×784}` and compute `X @ W_sparse` once. The sparse BLAS handles variable batch sizes efficiently.

Batch size `B` vs. latency `τ` is a tunable trade-off:
- `τ = 0` (no batching): minimum latency, minimum throughput
- `τ = 5ms`: ~5× throughput gain, 5ms latency increase (often acceptable)

### Monitoring in production

- **Latency:** p50, p95, p99 per-request latency. Alert if p99 > SLA threshold.
- **Accuracy:** Shadow-score each request against a held-out evaluation set (if ground truth is available with delay). Alert if rolling accuracy drops > 0.5% from baseline.
- **Sparsity integrity:** Verify that `nnz(W_sparse)` matches the expected value on startup. A bug in model export could accidentally densify the sparse matrix.
- **Memory:** Track RSS per worker. A sparse 90% model should use ~10% of the memory of the dense equivalent.

### Online fine-tuning

If the data distribution shifts post-deployment, fine-tune the deployed model with the mask **fixed** (no further pruning). This updates only the active weights while preserving the sparse topology:

```python
# Fine-tuning loop with frozen mask
for X_b, y_b in new_data_batches:
    model.zero_grad()
    loss = model(Tensor(X_b)).softmax_cross_entropy(y_b)
    loss.backward()
    # Pruner NOT called — mask stays fixed
    optimizer.step()
```

Periodically re-export the updated sparse matrices.

---

*scikit-learn is used ONLY to load the MNIST dataset via `fetch_openml`. All training, gradient computation, optimisation, and pruning is our own code.*
