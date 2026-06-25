"""
Training loop, data loading, and evaluation utilities.
scikit-learn is used ONLY to load datasets (not for training, models, or gradients).
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from engine.tensor import Tensor
from nn.layers import MLP
from nn.optimizer import Adam
from prune.pruner import Pruner, count_flops


# ------------------------------------------------------------------ #
# Data utilities                                                       #
# ------------------------------------------------------------------ #

def load_mnist(n_train: int = 10000, n_test: int = 2000, seed: int = 42):
    """
    Load MNIST via sklearn's fetch_openml (allowed: sklearn only for data).
    Falls back to a synthetic 2-spiral dataset if MNIST is unavailable.
    """
    try:
        from sklearn.datasets import fetch_openml
        print("Loading MNIST via sklearn (data loading only)...")
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X, y = mnist.data.astype(np.float32), mnist.target.astype(int)
        # Normalise to [-1, 1]
        X = X / 127.5 - 1.0
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))
        X, y = X[idx], y[idx]
        X_train, y_train = X[:n_train], y[:n_train]
        X_test, y_test = X[n_train: n_train + n_test], y[n_train: n_train + n_test]
        print(f"MNIST loaded: train={len(X_train)}, test={len(X_test)}, features={X_train.shape[1]}")
        return X_train, y_train, X_test, y_test, 10
    except Exception as e:
        print(f"MNIST unavailable ({e}), falling back to synthetic spiral dataset.")
        return make_spirals(n_train, n_test, seed=seed)


def make_spirals(n_train: int = 2000, n_test: int = 500, n_classes: int = 4, seed: int = 42):
    """Multi-class spiral dataset."""
    rng = np.random.default_rng(seed)

    def _spiral_class(n, cls, total_classes):
        angle_offset = 2 * np.pi * cls / total_classes
        r = np.linspace(0.1, 1.0, n)
        theta = np.linspace(0, 4 * np.pi, n) + angle_offset
        noise = rng.normal(0, 0.05, (n, 2))
        x = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1) + noise
        return x

    n_per = (n_train + n_test) // n_classes
    Xs, ys = [], []
    for c in range(n_classes):
        Xs.append(_spiral_class(n_per, c, n_classes))
        ys.append(np.full(n_per, c, dtype=int))

    X = np.vstack(Xs).astype(np.float32)
    y = np.concatenate(ys)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    return X[:n_train], y[:n_train], X[n_train: n_train + n_test], y[n_train: n_train + n_test], n_classes


def batch_iter(X: np.ndarray, y: np.ndarray, batch_size: int, rng=None):
    """Yield mini-batches of (X_batch, y_batch)."""
    n = len(X)
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.permutation(n)
    for start in range(0, n, batch_size):
        b = idx[start: start + batch_size]
        yield X[b], y[b]


# ------------------------------------------------------------------ #
# Training loop                                                        #
# ------------------------------------------------------------------ #

def train(
    model: MLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    pruner: Optional[Pruner] = None,
    optimizer: Optional[Adam] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Mini-batched training loop with optional pruning.

    Returns a history dict with loss/accuracy curves and sparsity.
    """
    rng = np.random.default_rng(seed)
    params = model.parameters()

    if optimizer is None:
        optimizer = Adam(params, lr=lr, weight_decay=weight_decay)

    history: Dict[str, List] = {
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
        "sparsity": [],
        "flops": [],
    }

    global_step = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        for X_b, y_b in batch_iter(X_train, y_train, batch_size, rng):
            optimizer.zero_grad(params)

            x_t = Tensor(X_b)
            logits = model(x_t)
            loss = logits.softmax_cross_entropy(y_b)
            loss.backward()

            # Pruning step (uses saliency = |w * grad|, computed right after backward)
            if pruner is not None:
                pruner.maybe_prune(optimizer)

            optimizer.step()

            # Metrics
            preds = logits.data.argmax(axis=1)
            epoch_correct += (preds == y_b).sum()
            epoch_samples += len(y_b)
            epoch_loss += float(loss.data) * len(y_b)
            global_step += 1

        train_loss = epoch_loss / epoch_samples
        train_acc = epoch_correct / epoch_samples
        test_acc = evaluate(model, X_test, y_test)
        sparsity = model.sparsity()
        flops = count_flops(model, X_train.shape[1])

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["sparsity"].append(sparsity)
        history["flops"].append(flops)

        if verbose:
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"loss={train_loss:.4f} | "
                f"train_acc={train_acc:.3f} | "
                f"test_acc={test_acc:.3f} | "
                f"sparsity={sparsity:.3f}"
            )

    return history


def evaluate(model: MLP, X: np.ndarray, y: np.ndarray, batch_size: int = 512) -> float:
    """Compute accuracy on dataset in batches."""
    correct = 0
    total = 0
    for start in range(0, len(X), batch_size):
        Xb = X[start: start + batch_size]
        yb = y[start: start + batch_size]
        logits = model(Tensor(Xb)).data
        preds = logits.argmax(axis=1)
        correct += (preds == yb).sum()
        total += len(yb)
    return correct / total if total > 0 else 0.0
