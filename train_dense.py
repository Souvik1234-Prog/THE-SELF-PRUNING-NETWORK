"""
Part 2: Train a dense MLP and report learning curves.
Usage: python train_dense.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import json

from nn.layers import MLP
from nn.optimizer import Adam
from train.trainer import train, load_mnist, make_spirals

SEED = 42
np.random.seed(SEED)

def main():
    print("=" * 60)
    print("PART 2 — Dense MLP Training")
    print("=" * 60)

    # Try MNIST first; fall back to spirals
    X_train, y_train, X_test, y_test, n_classes = load_mnist(
        n_train=8000, n_test=2000, seed=SEED
    )

    input_size = X_train.shape[1]
    print(f"\nDataset: input_size={input_size}, classes={n_classes}")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Build MLP
    np.random.seed(SEED)
    model = MLP(
        input_size=input_size,
        hidden_sizes=[256, 128],
        output_size=n_classes,
        activation="relu",
    )
    print(f"\nModel: {model.num_parameters()} parameters")
    print("Architecture: Linear(in, 256) -> ReLU -> Linear(256, 128) -> ReLU -> Linear(128, out)")
    print("Init: Kaiming He uniform (appropriate for ReLU networks)")

    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    history = train(
        model,
        X_train, y_train,
        X_test, y_test,
<<<<<<< HEAD
        epochs=100,
=======
        epochs=20,
>>>>>>> 2c0b5a126d2c4f2d8f4b18795143acd310e37ad5
        batch_size=128,
        optimizer=optimizer,
        seed=SEED,
        verbose=True,
    )

    print(f"\nFinal test accuracy: {history['test_acc'][-1]:.4f}")
    print(f"Final sparsity (should be ~0): {history['sparsity'][-1]:.4f}")

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/dense_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("\nResults saved to results/dense_history.json")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        epochs_range = range(1, len(history["train_loss"]) + 1)

        axes[0].plot(epochs_range, history["train_loss"], label="Train Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs_range, history["train_acc"], label="Train Acc")
        axes[1].plot(epochs_range, history["test_acc"], label="Test Acc")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/part2_learning_curves.png", dpi=150, bbox_inches="tight")
        print("Plot saved to plots/part2_learning_curves.png")
        plt.close()
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
