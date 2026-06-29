"""
Part 3: Train a self-pruning MLP.
Usage: python train_pruned.py [--sparsity 0.9] [--criterion saliency]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import json
import argparse

from nn.layers import MLP
from nn.optimizer import Adam
from train.trainer import train, load_mnist, make_spirals
from prune.pruner import Pruner, count_flops

SEED = 42
np.random.seed(SEED)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparsity", type=float, default=0.90)
    parser.add_argument("--criterion", type=str, default="saliency", choices=["saliency", "magnitude"])
    parser.add_argument("--regrowth", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    print("=" * 60)
    print(f"PART 3 — Self-Pruning MLP")
    print(f"  target_sparsity = {args.sparsity}")
    print(f"  criterion       = {args.criterion}")
    print(f"  regrowth        = {args.regrowth}")
    print("=" * 60)

    X_train, y_train, X_test, y_test, n_classes = load_mnist(
        n_train=8000, n_test=2000, seed=SEED
    )
    input_size = X_train.shape[1]

    np.random.seed(SEED)
    model = MLP(
        input_size=input_size,
        hidden_sizes=[256, 128],
        output_size=n_classes,
        activation="relu",
    )

    # Steps per epoch * total epochs
    steps_per_epoch = len(X_train) // 128 + 1
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * 3   # 3 epoch warmup before pruning starts

    pruner = Pruner(
        weight_params=model.weight_parameters(),
        target_sparsity=args.sparsity,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        prune_freq=steps_per_epoch,      # prune once per epoch
        criterion=args.criterion,
        allow_regrowth=args.regrowth,
        regrowth_fraction=0.05,
        schedule="cubic",
    )

    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    history = train(
        model,
        X_train, y_train,
        X_test, y_test,
        epochs=args.epochs,
        batch_size=128,
        optimizer=optimizer,
        pruner=pruner,
        seed=SEED,
        verbose=True,
    )

    final_sparsity = model.sparsity()
    final_acc = history["test_acc"][-1]
    dense_flops = count_flops(model, input_size)

    print(f"\n{'='*60}")
    print(f"Final test accuracy : {final_acc:.4f}")
    print(f"Achieved sparsity   : {final_sparsity:.4f}")
    print(f"Active params (weights): {sum(int(l.W.mask.sum()) for l in model.layers if l.W.mask is not None)}")
    print(f"FLOPs (sparse path) : {dense_flops:,}")
    print(f"{'='*60}")

    # Save
    os.makedirs("results", exist_ok=True)
    tag = f"sp{int(args.sparsity*100)}_{args.criterion}"
    with open(f"results/pruned_{tag}.json", "w") as f:
        json.dump({
            "history": history,
            "final_sparsity": final_sparsity,
            "final_acc": final_acc,
            "sparsity_history": pruner.sparsity_history,
            "criterion": args.criterion,
            "target_sparsity": args.sparsity,
        }, f, indent=2)
    print(f"Results saved to results/pruned_{tag}.json")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        ep = range(1, len(history["train_loss"]) + 1)

        axes[0].plot(ep, history["train_loss"])
        axes[0].set_title("Train Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(ep, history["train_acc"], label="Train")
        axes[1].plot(ep, history["test_acc"], label="Test")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(ep, history["sparsity"], color="orange")
        axes[2].axhline(args.sparsity, linestyle="--", color="red", label=f"Target {args.sparsity}")
        axes[2].set_title("Sparsity During Training")
        axes[2].set_xlabel("Epoch")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.suptitle(f"Self-Pruning ({args.criterion}, target={args.sparsity})")
        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/part3_{tag}.png", dpi=150, bbox_inches="tight")
        print(f"Plot saved to plots/part3_{tag}.png")
        plt.close()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
