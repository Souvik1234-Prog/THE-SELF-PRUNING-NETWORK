# Self-Pruning Network

A neural network that **prunes itself during training** — built entirely from scratch in pure Python and NumPy, with a custom reverse-mode automatic differentiation engine.

> **No PyTorch. No TensorFlow. No JAX. No Keras. No autodiff or pruning libraries.**
> scikit-learn is used *only* to load the MNIST dataset via `fetch_openml`. Every gradient, optimiser update, mask operation, and pruning decision is hand-written NumPy code.

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Running the Tests](#running-the-tests)
- [Reproducing All Results](#reproducing-all-results)
- [Architecture Overview](#architecture-overview)
- [Key Design Decisions](#key-design-decisions)
- [Results Summary](#results-summary)
- [Requirements](#requirements)

---

## What This Project Does

This project implements a complete deep learning pipeline — from raw tensor operations and automatic differentiation, through neural network layers with batch normalisation and dropout, to a self-pruning training loop that gradually removes unimportant weights during training.

The core idea: instead of training a dense network and pruning afterwards, the network **prunes itself during training** using a saliency criterion derived from a first-order Taylor expansion of the loss. Pruned weights can also **regrow** if they later become important, correcting noisy early pruning decisions.

The pipeline is validated on MNIST handwritten digit classification, achieving:
- **~97% test accuracy** with a dense `DeepMNISTNet`
- **~95.5% test accuracy** at 50% sparsity (negligible loss)
- **~94% test accuracy** at 90% sparsity (10× fewer active weights)
- Saliency pruning **consistently outperforms** magnitude pruning at every sparsity level

---

## Repository Structure

```
self_pruning_net/
│
├── engine/                        # Reverse-mode autodiff engine
│   ├── tensor.py                  # Tensor class: data, grad, mask, backward graph
│   └── __init__.py
│
├── nn/                            # Neural network layers and optimisers
│   ├── layers.py                  # Linear, BatchNorm1d, Dropout, MLP, DeepMNISTNet
│   ├── optimizer.py               # Adam (with masked-weight revival), SGDMomentum
│   └── __init__.py
│
├── prune/                         # Pruning engine
│   ├── pruner.py                  # Pruner, saliency_scores, magnitude_scores,
│   │                              #   compute_target_sparsity, sparse_linear_forward,
│   │                              #   count_flops
│   └── __init__.py
│
├── train/                         # Training utilities
│   ├── trainer.py                 # train(), evaluate(), load_mnist(), make_spirals(),
│   │                              #   batch_iter(), cosine_lr()
│   └── __init__.py
│
├── tests/                         # Test suite
│   ├── test_gradients.py          # Numerical gradient checks for all ops
│   └── test_masked_weights.py     # Masked-weight correctness + optimizer revival test
│
├── results/                       # Committed output numbers (JSON)
│   ├── dense_history.json         # Part 2: loss/accuracy curves
│   ├── pruned_sp90_saliency.json  # Part 3: saliency run at 90% sparsity
│   ├── pruned_sp90_magnitude.json # Part 3: magnitude baseline at 90% sparsity
│   ├── pareto_raw.json            # Part 4: all sweep runs
│   └── pareto_summary.json        # Part 4: mean ± std per (sparsity, criterion)
│
├── plots/                         # Committed figures (PNG)
│   ├── part2_learning_curves.png  # Dense training loss + accuracy
│   ├── part3_sp90_saliency.png    # Pruned training curves (saliency)
│   ├── part3_sp90_magnitude.png   # Pruned training curves (magnitude)
│   └── part4_pareto.png           # Accuracy vs FLOPs Pareto curve
│
├── train_dense.py                 # Part 2: dense training entry point
├── train_pruned.py                # Part 3: self-pruning entry point
├── pareto_sweep.py                # Part 4: full Pareto sweep
├── DESIGN.md                      # Full derivations and architectural decisions
├── README.md                      # This file
└── requirements.txt
```

---

## Installation

### Using pip (standard)

```bash
python -m venv auto_diff_v2
# Windows:
auto_diff_v2\Scripts\activate
# macOS/Linux:
source auto_diff_v2/bin/activate

pip install numpy matplotlib scikit-learn pandas pytest
```

> **pandas is required** for `sklearn`'s `fetch_openml` to parse MNIST. Without it, training falls back to a synthetic spiral dataset.

### Using uv (faster)

```bash
uv venv && source .venv/bin/activate
uv pip install numpy matplotlib scikit-learn pandas pytest
```

### Windows PowerShell note

```powershell
# Activate venv on Windows:
.\auto_diff_v2\Scripts\Activate.ps1
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install numpy matplotlib scikit-learn pandas pytest

# 2. Run gradient checks (verifies the autodiff engine)
pytest tests/ -v

# 3. Train a dense model (~97% accuracy on MNIST)
python train_dense.py

# 4. Train with self-pruning at 90% sparsity
python train_pruned.py --sparsity 0.90 --criterion saliency

# 5. Run the full Pareto sweep
python pareto_sweep.py
```

---

## Running the Tests

```bash
pytest tests/ -v
```

Expected output:

```
PASS  composition
PASS  masked_forward_zero
PASS  masked_grad_zero
PASS  masked_optimizer_correct
PASS  mask_isolation
```

### What each test checks

| Test | What it verifies |
|---|---|
| `composition` | Reverse-mode gradients match numerical finite-difference estimates for composed ops (matmul, relu, softmax, etc.) |
| `masked_forward_zero` | A masked weight produces exactly zero in the forward pass |
| `masked_grad_zero` | The gradient of a masked weight is exactly zero (chain rule: dead weight has no effect on loss) |
| `masked_optimizer_correct` | Adam moment buffers are reset to zero when a pruned weight is revived, preventing corrupted momentum from zero-gradient accumulation |
| `mask_isolation` | Masking one weight does not affect the gradients of neighbouring weights |

---

## Reproducing All Results

### Part 2 — Dense training baseline

```bash
python train_dense.py
```

What it does:
- Loads full MNIST (60,000 train / 10,000 test)
- Trains `DeepMNISTNet` (784→512→256→128→64→10) with BatchNorm, Dropout, and cosine LR schedule
- 30 epochs, batch size 128, lr 3e-3 → 1e-5

Outputs:
- `results/dense_history.json` — per-epoch loss, train accuracy, test accuracy
- `plots/part2_learning_curves.png` — training curves

Expected: **~97% test accuracy**

---

### Part 3 — Self-pruning run

```bash
# Saliency criterion (recommended)
python train_pruned.py --sparsity 0.90 --criterion saliency

# Magnitude baseline
python train_pruned.py --sparsity 0.90 --criterion magnitude
```

What it does:
- Same model and data as Part 2
- Adds a `Pruner` that gradually increases sparsity from 0% to 90% via a cubic ramp
- Prunes every 100 steps; allows 5% regrowth of pruned connections per step
- Adam moments are reset for revived connections

Outputs:
- `results/pruned_sp90_saliency.json`
- `results/pruned_sp90_magnitude.json`
- `plots/part3_sp90_saliency.png`
- `plots/part3_sp90_magnitude.png`

Expected: **saliency ~94%, magnitude ~92.5%** at 90% sparsity

---

### Part 4 — Full Pareto sweep

```bash
python pareto_sweep.py
```

What it does:
- Sweeps sparsities: [0%, 50%, 75%, 90%, 95%]
- Criteria: [saliency, magnitude]
- 3 random seeds per combination (15 runs total)
- Reports mean ± std accuracy and FLOPs at each point

Outputs:
- `results/pareto_raw.json` — all individual run results
- `results/pareto_summary.json` — aggregated mean ± std
- `plots/part4_pareto.png` — accuracy vs FLOPs Pareto curve

---

## Architecture Overview

### Autodiff Engine (`engine/tensor.py`)

The `Tensor` class wraps a NumPy array and builds a dynamic computation graph during the forward pass. Each operation registers a `_backward` closure on its output tensor. Calling `.backward()` on a scalar loss traverses the graph in reverse topological order, accumulating gradients via the chain rule.

Supported operations and their backward passes:

| Operation | Forward | Backward |
|---|---|---|
| `__matmul__` | `X @ W` | `dX = dout @ W.T`, `dW = X.T @ dout` |
| `__add__` | `a + b` | `da = dout`, `db = dout.sum(axis=0)` (broadcast) |
| `__mul__` | `a * b` | `da = dout * b`, `db = dout * a` |
| `relu` | `max(0, x)` | `dout * (x > 0)` |
| `tanh` | `tanh(x)` | `dout * (1 - tanh²(x))` |
| `sigmoid` | `1/(1+e^-x)` | `dout * s * (1 - s)` |
| `gelu` | `x * Φ(x)` | chain rule through CDF approximation |
| `softmax_cross_entropy` | log-sum-exp stable | `(softmax - one_hot) / N` |

**Masked weight handling:** `Tensor._accum()` zeroes gradients at masked positions before accumulation, ensuring pruned weights receive zero gradient and their Adam moments stay at zero.

---

### Layers (`nn/layers.py`)

#### `Linear`
Fully-connected layer `out = X @ W + b`. Weight initialisation:
- Kaiming He uniform for ReLU/GELU: `bound = sqrt(6 / fan_in)`
- Xavier/Glorot uniform for tanh/sigmoid: `bound = sqrt(6 / (fan_in + fan_out))`

#### `BatchNorm1d`
Batch normalisation for 2D inputs `(N, D)`. During training, normalises over the batch dimension and updates running statistics via exponential moving average. During inference, uses frozen running statistics. Learnable `gamma` (scale) and `beta` (shift) parameters allow the network to undo normalisation if optimal.

**Why BatchNorm helps:**
- Keeps activations in the linear region of ReLU, preventing dead neurons
- Acts as a regulariser
- Enables higher learning rates (3e-3 vs 1e-3), reducing epochs needed

#### `Dropout`
Inverted dropout: zero activations with probability `p` during training, scale survivors by `1/(1-p)` so expected value is unchanged at inference. Never applied to the output layer.

#### `DeepMNISTNet`
The primary model for MNIST, targeting 97%+ accuracy:

```
Input (784)
  → Linear(784, 512) → BatchNorm1d(512) → ReLU → Dropout(0.3)
  → Linear(512, 256) → BatchNorm1d(256) → ReLU → Dropout(0.3)
  → Linear(256, 128) → BatchNorm1d(128) → ReLU → Dropout(0.2)
  → Linear(128, 64)  → BatchNorm1d(64)  → ReLU
  → Linear(64, 10)   [raw logits]
```

**Design rationale:**
- 512 first layer: wide enough to capture pixel co-occurrence patterns
- 4 hidden layers: each learns progressively more abstract features
- BatchNorm after every linear: stabilises training at lr=3e-3
- Dropout decreasing with depth: strongest regularisation where overfitting risk is highest
- No dropout on 64-neuron layer: too small to benefit
- Output layer: raw logits fed directly into `softmax_cross_entropy`

#### `MLP`
Simpler multi-layer perceptron kept for backward compatibility. No BatchNorm or Dropout. Use `DeepMNISTNet` for serious accuracy targets.

---

### Optimiser (`nn/optimizer.py`)

#### `Adam`
Standard Adam (Kingma & Ba, 2015) with two extensions:

**Weight decay:** L2 regularisation applied to unmasked weights only, before the moment update:
```
g = grad + weight_decay * w * mask
```

**Masked-weight revival detection:** On every `step()`, the optimiser compares the current mask to the previous mask. For any weight that was `False` (pruned) and is now `True` (revived), it resets:
```python
self.m[i][revived] = 0.0
self.v[i][revived] = 0.0
```
This prevents phantom momentum — accumulated zero-gradient steps would otherwise inject a large spurious update on the first post-revival gradient.

---

### Pruner (`prune/pruner.py`)

#### Importance criterion

**Saliency** (default): `S(w) = |w × grad_w|`

Derived from the first-order Taylor expansion of the loss when removing weight `w`:
```
|ΔL| ≈ |w · ∂L/∂w|
```

Weights with low saliency — either small magnitude *or* small gradient — are pruned first.

**Magnitude** (baseline): `S(w) = |w|`

Assumes uniform gradient across all weights. Strictly inferior to saliency.

#### Pruning schedule

Cubic sparsity ramp (Zhu & Gupta, ICLR 2018):
```
s(t) = s_f · [1 - (1 - (t - t₀)/(t_f - t₀))³]
```

The cubic schedule has zero derivative at `t = t_f`, meaning pruning slows near the target sparsity, giving the network maximum time to recover accuracy in the critical final phase.

#### Regrowth

After each pruning step, up to `regrowth_fraction` (default 5%) of currently pruned connections can revive if their saliency — computed using the current gradient — exceeds the saliency of currently active connections. This corrects noisy early pruning decisions without violating the sparsity target.

#### `count_flops`

Counts multiply-accumulate operations (MACs) for one forward pass, counting only non-zero (unmasked) weights:
```
MACs = sum over layers of (batch_size × nnz)
```
Works with both `MLP` (`.layers`) and `DeepMNISTNet` (`.linears`) via `getattr` fallback.

---

### Training Loop (`train/trainer.py`)

#### `load_mnist`
Loads full MNIST (60,000/10,000) via `sklearn.fetch_openml`. Applies per-pixel zero-mean unit-variance normalisation (better than `[-1,1]` scaling for deep networks). Falls back to synthetic spiral data if MNIST is unavailable.

#### `cosine_lr`
Cosine annealing schedule:
```
lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))
```

#### `train`
Mini-batch training loop supporting:
- Cosine LR schedule (enabled by default)
- `train_mode()` / `eval_mode()` switching for BatchNorm and Dropout
- Optional `Pruner` integration (called after `backward()`, before `optimizer.step()`)
- Per-epoch logging of loss, train accuracy, test accuracy, sparsity, FLOPs

---

## Key Design Decisions

| Decision | Choice | Justification |
|---|---|---|
| Importance criterion | `\|w × grad\|` saliency | First-order Taylor approx of `\|ΔL\|`; captures both magnitude and gradient signal |
| Baseline criterion | `\|w\|` magnitude | Assumes uniform gradient; strictly weaker than saliency |
| Pruning schedule | Cubic sparsity ramp | Zero-derivative at target; network adapts gracefully near final sparsity |
| Gradient of masked weight | Exactly zero | Dead weight has no effect on loss; chain rule gives zero |
| Optimiser on revival | Reset m, v to zero | Prevents phantom momentum from zero-gradient accumulation period |
| Regrowth | Enabled, 5% budget | Corrects noisy early pruning; maintains sparsity target |
| Normalisation | Per-pixel zero-mean unit-variance | Outperforms `[-1,1]` for deep nets; keeps activations well-conditioned |
| LR schedule | Cosine annealing | Smooth decay; zero-derivative at end prevents overshooting |
| BatchNorm | After every hidden linear | Worth ~1-2% accuracy; enables lr=3e-3 |
| Dropout | 0.3/0.3/0.2/0.0 per layer | Stronger regularisation where parameter count is highest |

---

## Results Summary

All numbers are committed to `results/pareto_summary.json` and reproducible by running `pareto_sweep.py`.

| Sparsity | Criterion | Test Accuracy | FLOPs vs Dense |
|---|---|---|---|
| 0% | — | ~97.0% | 1.00× |
| 50% | saliency | ~95.5% | 0.50× |
| 75% | saliency | ~95.0% | 0.25× |
| 90% | saliency | ~94.0% | 0.10× |
| 90% | magnitude | ~92.5% | 0.10× |
| 95% | saliency | ~93.0% | 0.05× |

**Falsifiable claim:** At 90% target sparsity, saliency pruning retains higher accuracy than magnitude pruning, averaged over 3 random seeds. Exact numbers are in `results/pareto_summary.json`.

---

## Requirements

- Python ≥ 3.10
- `numpy` — all tensor operations and autodiff
- `matplotlib` — learning curve and Pareto plots
- `scikit-learn` — MNIST data loading only (`fetch_openml`)
- `pandas` — required by sklearn's OpenML parser
- `pytest` — test runner

```txt
# requirements.txt
numpy
matplotlib
scikit-learn
pandas
pytest
```

---

*Built from scratch. No autodiff libraries. No ML frameworks. Just NumPy.*
