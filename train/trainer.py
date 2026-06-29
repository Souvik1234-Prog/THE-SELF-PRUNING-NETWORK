"""
Training loop, data loading, and evaluation utilities.
scikit-learn is used ONLY to load datasets (not for training, models, or gradients).
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from engine.tensor import Tensor
from nn.optimizer import Adam
from prune.pruner import Pruner, count_flops


# ------------------------------------------------------------------ #
# Data utilities                                                       #
# ------------------------------------------------------------------ #

def load_mnist(n_train: int = 60000, n_test: int = 10000, seed: int = 42):
    """
    Load full MNIST via sklearn (data loading only).
    Falls back to spirals if unavailable.
    """
    try:
        from sklearn.datasets import fetch_openml
        print("Loading MNIST via sklearn (data loading only)...")
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X, y = mnist.data.astype(np.float64), mnist.target.astype(int)
        # Normalise: zero mean, unit variance per pixel (better than [-1,1] for deep nets)
        X = X / 255.0
        mean = X.mean(axis=0, keepdims=True)
        std  = X.std(axis=0,  keepdims=True) + 1e-8
        X = (X - mean) / std

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))
        X, y = X[idx], y[idx]
        X_train, y_train = X[:n_train], y[:n_train]
        X_test,  y_test  = X[n_train: n_train + n_test], y[n_train: n_train + n_test]
        print(f"MNIST loaded: train={len(X_train)}, test={len(X_test)}, features={X_train.shape[1]}")
        return X_train, y_train, X_test, y_test, 10
    except Exception as e:
        print(f"MNIST unavailable ({e}), falling back to spiral dataset.")
        return make_spirals(n_train, n_test, seed=seed)


def make_spirals(n_train=2000, n_test=500, n_classes=4, seed=42):
    rng = np.random.default_rng(seed)
    def _spiral(n, cls, total):
        offset = 2 * np.pi * cls / total
        r = np.linspace(0.1, 1.0, n)
        theta = np.linspace(0, 4 * np.pi, n) + offset
        noise = rng.normal(0, 0.05, (n, 2))
        return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1) + noise

    n_per = (n_train + n_test) // n_classes
    X = np.vstack([_spiral(n_per, c, n_classes) for c in range(n_classes)]).astype(np.float64)
    y = np.concatenate([np.full(n_per, c) for c in range(n_classes)]).astype(int)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    return X[:n_train], y[:n_train], X[n_train:n_train+n_test], y[n_train:n_train+n_test], n_classes


def batch_iter(X, y, batch_size, rng=None):
    n = len(X)
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.permutation(n)
    for start in range(0, n, batch_size):
        b = idx[start: start + batch_size]
        yield X[b], y[b]


# ------------------------------------------------------------------ #
# LR scheduler                                                         #
# ------------------------------------------------------------------ #

def cosine_lr(epoch: int, total_epochs: int, lr_max: float, lr_min: float = 1e-5) -> float:
    """Cosine annealing: smoothly reduces LR from lr_max to lr_min."""
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / total_epochs))


# ------------------------------------------------------------------ #
# Training loop                                                        #
# ------------------------------------------------------------------ #

def train(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    epochs:      int   = 30,
    batch_size:  int   = 128,
    lr:          float = 3e-3,
    lr_min:      float = 1e-5,
    weight_decay: float = 1e-4,
    pruner:    Optional[Pruner] = None,
    optimizer = None,
    use_lr_schedule: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    rng = np.random.default_rng(seed)
    params = model.parameters()

    if optimizer is None:
        from nn.optimizer import Adam
        optimizer = Adam(params, lr=lr, weight_decay=weight_decay)

    history: Dict[str, List] = {
        "train_loss": [], "train_acc": [],
        "test_acc":   [], "sparsity":  [], "flops": [],
    }

    for epoch in range(epochs):
        # ---- LR schedule ----
        if use_lr_schedule:
            current_lr = cosine_lr(epoch, epochs, lr, lr_min)
            optimizer.lr = current_lr

        # ---- Training mode ----
        if hasattr(model, "train_mode"):
            model.train_mode()

        epoch_loss = epoch_correct = epoch_samples = 0

        for X_b, y_b in batch_iter(X_train, y_train, batch_size, rng):
            optimizer.zero_grad(params)

            logits = model(Tensor(X_b))
            loss   = logits.softmax_cross_entropy(y_b)
            loss.backward()

            if pruner is not None:
                pruner.maybe_prune(optimizer)

            optimizer.step()

            preds = logits.data.argmax(axis=1)
            epoch_correct += (preds == y_b).sum()
            epoch_samples += len(y_b)
            epoch_loss    += float(loss.data) * len(y_b)

        # ---- Eval mode for test accuracy ----
        if hasattr(model, "eval_mode"):
            model.eval_mode()

        train_loss = epoch_loss / epoch_samples
        train_acc  = epoch_correct / epoch_samples
        test_acc   = evaluate(model, X_test, y_test)
        sparsity   = model.sparsity() if hasattr(model, "sparsity") else 0.0
        flops      = count_flops(model, X_train.shape[1]) if hasattr(model, "linears") or hasattr(model, "layers") else 0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["sparsity"].append(sparsity)
        history["flops"].append(flops)

        if verbose:
            lr_str = f"lr={optimizer.lr:.5f} | " if use_lr_schedule else ""
            print(
                f"Epoch {epoch+1:3d}/{epochs} | {lr_str}"
                f"loss={train_loss:.4f} | "
                f"train={train_acc:.4f} | "
                f"test={test_acc:.4f} | "
                f"sparsity={sparsity:.3f}"
            )

    return history


def evaluate(model, X: np.ndarray, y: np.ndarray, batch_size: int = 512) -> float:
    correct = total = 0
    for start in range(0, len(X), batch_size):
        Xb, yb = X[start:start+batch_size], y[start:start+batch_size]
        logits = model(Tensor(Xb)).data
        correct += (logits.argmax(axis=1) == yb).sum()
        total   += len(yb)
    return correct / total if total > 0 else 0.0
