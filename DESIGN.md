# DESIGN.md — Self-Pruning Network

## 1. Importance Criterion Derivation

### Why saliency beats magnitude

We want to approximate **ΔL**, the change in loss when we remove weight `w_ij`.
Removing `w_ij` means applying the perturbation `Δw_ij = -w_ij`.

A first-order Taylor expansion of `L` around the current parameters gives:

```
ΔL ≈ ∂L/∂w_ij · Δw_ij = ∂L/∂w_ij · (-w_ij) = -w_ij · ∂L/∂w_ij
```

So the magnitude of the loss change from removing weight `w_ij` is approximately:

```
|ΔL| ≈ |w_ij · ∂L/∂w_ij|
```

We define the **saliency score**:

```
S(w_ij) = |w_ij · grad_ij|
```

and keep the weights with the **largest** saliency scores (the ones that, if removed, would increase the loss the most). Weights with low saliency — either small magnitude *or* small gradient — are pruned first.

### Why this is strictly better than magnitude-only

Magnitude pruning uses `S(w) = |w|`, which assumes the gradient is uniform across all weights. This is false:
- A **large** weight sitting at a local minimum in the loss landscape (gradient ≈ 0) may safely be removed.
- A **small** weight that is actively changing (large gradient) is being trained toward importance and should be protected.

The saliency criterion captures both signals. Empirically (Molchanov et al., CVPR 2019), saliency-based pruning retains ~1-3% more accuracy at 90% sparsity compared to magnitude pruning.

### Relationship to higher-order methods

The second-order approximation (Optimal Brain Damage / OBD, LeCun 1990) uses the Hessian diagonal:

```
|ΔL_2| ≈ (1/2) · h_ij · w_ij²
```

This is more accurate but requires computing the Hessian (expensive). The first-order saliency we use is a computationally cheap approximation that reuses the already-computed gradients from the backward pass, making it essentially free.

---

## 2. What the Engine Computes as "the Gradient of a Masked Weight"

### The answer: zero

A masked weight is structurally absent from the computation. Its gradient is **zero**, enforced in `Tensor._accum()`:

```python
if self.mask is not None:
    delta = delta * self.mask.astype(np.float64)
```

This zeroes out the gradient at masked positions *before* accumulating.

### Why this is correct

The gradient `∂L/∂w_ij` measures how much the loss changes as `w_ij` varies. But a masked weight has no effect on the forward pass — it's been zeroed out. Therefore:

```
∂L/∂w_ij = 0   for all pruned w_ij
```

This is precisely what our implementation computes. There is no other defensible answer: a weight that does not participate in computation cannot influence the loss, and so its gradient is zero by the chain rule.

### The alternative (wrong) approach

A naive implementation might let the gradient flow through `w_ij` by computing the mathematical gradient of the dense `X @ W` operation — even for positions that happen to be zero. This gives the *mathematical* gradient for "what would happen if we revived and then perturbed `w_ij`", which is not what we want. It silently couples pruned weights to the optimiser, causing corrupted momentum.

---

## 3. Autodiff Engine Bottleneck and Optimisation

### Current bottleneck

The engine builds a Python object graph (linked `Tensor` nodes with closures) and traverses it in Python during both `build_topo()` and the backward pass. For large graphs:

1. **Python overhead**: each `_backward` closure is a Python function call. NumPy operations themselves are fast (C), but the dispatch overhead per operation is ~1-5μs, which dominates for small tensors.
2. **Memory**: every intermediate tensor is kept alive (for gradient computation) until `backward()` completes. This is `O(depth × batch × layer_size)`.
3. **Topological sort**: re-run on every `backward()` call; the graph is fixed per forward pass so this is O(N) but wasted work.

### How to optimise

1. **Cache the topological order** (the graph doesn't change between batches). Record it once after the first forward pass and reuse.
2. **Fuse operations**: replace the separate `matmul → add_bias → relu` chain with a single fused kernel that computes all three in one C-level call. This cuts Python overhead by 3×.
3. **Recompute-vs-store tradeoff (gradient checkpointing)**: discard activations of large layers during forward and recompute them during backward. Halves peak memory at ~33% compute overhead.
4. **Vectorise the backward loop**: compile the backward pass closures into a static schedule (e.g., a list of `(op, parent_indices, kwargs)` tuples) and dispatch via a lookup table, removing Python function-call overhead.
5. **NumPy→CuPy**: swap `np` for `cupy` and the same Python graph runs on GPU with near-zero code changes.

---

## 4. Serving a Self-Pruned Model in a Multi-Tenant Inference Service

### The opportunity

A 90%-sparse model has ~10% of active weights. The honest speedup comes from using a **sparse matrix format** (CSR/CSC) and a sparse BLAS kernel (`scipy.sparse.csr_matrix @ dense_vector`), which skips the zero-multiplication work entirely, giving theoretical 10× speedup on weight-bound layers.

### Deployment plan

**Model serialisation**
- Export each layer as `(values, row_indices, col_indices)` in CSR format.
- Store alongside the bias and a metadata record of input/output dimensions and sparsity.
- At load time, reconstruct the sparse matrix. No mask needed at inference: the structural zeros are implicit in the sparse format.

**Serving architecture**
- Deploy behind a model server (e.g., FastAPI + Uvicorn, or Triton Inference Server).
- Use a worker pool sized to CPU core count; each request is a synchronous sparse forward pass (~μs for small models).
- For multi-tenant isolation: one model instance per worker thread; no shared mutable state.

**Batching**
- Group requests arriving within a time window into a batch (`X` matrix) and compute `X @ W_sparse` once. Sparse × dense BLAS handles variable-width batch efficiently.
- The theoretical throughput scales as `O(batch × nnz)` vs `O(batch × D_in × D_out)` for the dense path.

**Monitoring**
- Track per-request latency, active-connection count, and accuracy on a held-out evaluation split (shadow scoring).
- Alert if sparsity diverges from the trained level (e.g., due to a model hot-swap bug).

**Regrowth at serving time**
- A self-pruning model trained with regrowth has a fixed mask post-training. The mask is frozen at export.
- If online fine-tuning is required, run one training step per request batch with the pruner disabled (mask fixed), then periodically re-export.

---

## 5. Pruning Schedule Justification

We use the **cubic sparsity ramp** (Zhu & Gupta, "To Prune or Not to Prune", ICLR 2018):

```
s(t) = s_f · [1 - (1 - (t - t₀)/(t_f - t₀))³]
```

**Why cubic, not linear?**
- The derivative is zero at `t = t_f`, meaning the final sparsity increase is gradual, giving the network maximal time to recover accuracy near the target.
- Linear ramps prune at a constant rate, which is too aggressive late in training when every removed connection hurts more.
- One-shot pruning (prune all at epoch `t_f`) is 1-2% worse at the same sparsity because the network never adapts to the sparse topology during training.

**Warmup period**
We delay pruning for 2-3 epochs so the network first learns a reasonable representation. Pruning at initialisation would remove weights before any useful structure forms.

---

## 6. Regrowth: Stability and Theory

### Why allow regrowth?

Static pruning commits to a topology at step `t₀`. If the initial importance estimate was noisy (which it is, because we use a single mini-batch gradient), a truly important weight may have been incorrectly pruned.

Regrowth — allowing pruned connections to return if their saliency later exceeds a threshold — corrects early mistakes. This is the key idea in **dynamic sparse training** (Mocanu et al. 2018, Evci et al. 2020 "Rigged Lottery").

### Stability implications

- Reviving a connection with corrupted Adam moments (nonzero m, v accumulated from zero gradients) injects phantom momentum that can destabilise training. Our implementation **resets m and v to zero** on revival.
- The regrowth fraction is kept small (5% of pruned budget per step) to prevent oscillation.
- We never revive more connections than we prune in the same step, maintaining the sparsity target monotonically.

---

*scikit-learn is used ONLY to load the MNIST dataset via `fetch_openml`. All training, gradient computation, optimisation, and pruning is our own code.*
