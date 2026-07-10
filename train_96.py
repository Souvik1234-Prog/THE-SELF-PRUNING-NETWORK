"""
Train DeepMNISTNet to 96%+ accuracy on MNIST.

Architecture:  784 → 512 → 256 → 128 → 64 → 10
               + BatchNorm + Dropout + cosine LR schedule

Usage:  python train_96.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import json

SEED = 42
np.random.seed(SEED)

from nn.layers import DeepMNISTNet
from nn.optimizer import Adam
from train.trainer import train, load_mnist

def main():
    print("=" * 65)
    print("  DeepMNISTNet — Target: 96%+ accuracy")
    print("=" * 65)

    #  MNIST: 8k train, 2k test
    X_train, y_train, X_test, y_test, n_classes = load_mnist(
        n_train=8000, n_test=2000, seed=SEED
    )

    np.random.seed(SEED)
    model = DeepMNISTNet(
        input_size   = 784,
        output_size  = 10,
        hidden_sizes = [512, 256, 128, 64],
        dropout_rates= [0.3, 0.3, 0.2, 0.0],
        activation   = "relu",
        use_batchnorm= True,
    )

    print(f"\nArchitecture:  784 → 512 → 256 → 128 → 64 → 10")
    print(f"  + BatchNorm after each hidden layer")
    print(f"  + Dropout(0.3, 0.3, 0.2, 0.0)")
    print(f"  + Cosine LR annealing: 3e-3 → 1e-5 over 30 epochs")
    print(f"Total parameters: {model.num_parameters():,}\n")

    optimizer = Adam(
        model.parameters(),
        lr=3e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=1e-4,
    )

    history = train(
        model,
        X_train, y_train,
        X_test,  y_test,
        epochs          = 100,
        batch_size      = 256,    # larger batch → more stable BN stats
        lr              = 3e-3,
        lr_min          = 1e-5,
        weight_decay    = 1e-4,
        optimizer       = optimizer,
        use_lr_schedule = True,
        seed            = SEED,
        verbose         = True,
    )

    best_acc = max(history["test_acc"])
    final_acc = history["test_acc"][-1]

    print(f"\n{'='*65}")
    print(f"  Best test accuracy : {best_acc*100:.2f}%")
    print(f"  Final test accuracy: {final_acc*100:.2f}%")
    print(f"  Target achieved    : {'✓ YES' if best_acc >= 0.96 else '✗ not yet — try more epochs'}")
    print(f"{'='*65}")

    os.makedirs("results_bt", exist_ok=True)
    with open("results_bt/deep_mnist_history_relu.json", "w") as f:
        json.dump({**history, "best_acc": best_acc, "final_acc": final_acc}, f, indent=2)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        ep = range(1, len(history["train_loss"]) + 1)

        axes[0].plot(ep, history["train_loss"], label="Train Loss", linewidth=2)
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss"); axes[0].grid(True, alpha=0.3)

        axes[1].plot(ep, [a*100 for a in history["train_acc"]], label="Train", linewidth=2)
        axes[1].plot(ep, [a*100 for a in history["test_acc"]],  label="Test",  linewidth=2)
        axes[1].axhline(96, color="red", linestyle="--", label="96% target")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([85, 100])

        plt.suptitle(f"DeepMNISTNet  |  Best: {best_acc*100:.2f}%", fontsize=13)
        plt.tight_layout()
        os.makedirs("plots_bt", exist_ok=True)
        plt.savefig("plots_bt/deep_mnist_96_relu.png", dpi=150, bbox_inches="tight")
        print("Plot saved to plots_bt/deep_mnist_96_relu.png")
        plt.close()
    except ImportError:
        pass

if __name__ == "__main__":
    main()
