# Self-Pruning Network

A neural network that **prunes itself during training** — built from scratch in pure Python and NumPy, with a custom reverse-mode autodiff engine.

> **scikit-learn** is used *only* to load the MNIST dataset (`fetch_openml`). All training, gradient computation, optimisation, and pruning is our own code. No PyTorch, TensorFlow, JAX, Keras, HuggingFace, or any autodiff / pruning library.

---

## Repository Structure

```
self_pruning_net/
├── engine/          # Reverse-mode autodiff engine (Tensor, ops, backward)
├── nn/              # MLP layers, Adam / SGD optimizers
├── prune/           # Importance criterion, pruning schedule, masking, regrowth
├── train/           # Training loop, data loaders, evaluation
├── tests/           # Gradient check tests + masked-weight correctness tests
├── results/         # Committed raw numbers (JSON)
├── plots/           # Committed Pareto curves and learning curves (PNG)
├── train_dense.py   # Part 2: dense training run
├── train_pruned.py  # Part 3: self-pruning run
├── pareto_sweep.py  # Part 4: full Pareto sweep
├── DESIGN.md        # Derivations and architectural decisions
└── README.md
```

---

## Installation

```bash
# Using uv (preferred)
uv venv && source .venv/bin/activate
uv pip install numpy matplotlib scikit-learn pytest

# Or pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## One-Command Recipes

### Run gradient-check tests (Part 1)
```bash
python tests/test_gradients.py
```
Or with pytest:
```bash
pytest tests/test_gradients.py -v
```

### Reproduce Part 2 — Dense training
```bash
python train_dense.py
```
Outputs: `results/dense_history.json`, `plots/part2_learning_curves.png`

### Reproduce Part 3 — Self-pruning run (90% sparsity, saliency criterion)
```bash
python train_pruned.py --sparsity 0.90 --criterion saliency
```
Outputs: `results/pruned_sp90_saliency.json`, `plots/part3_sp90_saliency.png`

Try magnitude baseline:
```bash
python train_pruned.py --sparsity 0.90 --criterion magnitude
```

### Reproduce Part 4 — Full Pareto sweep
```bash
python pareto_sweep.py
```
Sweeps sparsities [0%, 50%, 75%, 90%, 95%] × [saliency, magnitude] × 3 seeds.  
Outputs: `results/pareto_raw.json`, `results/pareto_summary.json`, `plots/part4_pareto.png`

---

## Key Design Decisions (summary — see DESIGN.md for derivations)

| Decision | Choice | Why |
|---|---|---|
| Importance criterion | `\|w × grad\|` (saliency) | First-order Taylor approx of `\|ΔL\|` when removing w |
| Baseline criterion | `\|w\|` (magnitude) | Trivial; assumes uniform gradient |
| Pruning schedule | Cubic sparsity ramp | Gradual near target; network adapts to sparse topology |
| Gradient of masked weight | Zero | Masked weight has no effect on loss; chain rule gives 0 |
| Optimizer (masked weight) | Reset m, v on revival | Prevents phantom momentum from zero-grad accumulation |
| Regrowth | Enabled (5% budget) | Corrects noisy early pruning decisions |

---

## Falsifiable Claim

> **At 90% target sparsity, saliency pruning retains higher accuracy than magnitude pruning, averaged over 3 random seeds. The exact numbers are committed in `results/pareto_summary.json`.**

See `results/pareto_summary.json` for precise values and `plots/part4_pareto.png` for the Pareto curve.

---

## Requirements

- Python ≥ 3.10
- numpy
- matplotlib (for plots)
- scikit-learn (data loading only)
- pytest (for tests)
