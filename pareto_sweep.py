"""
Part 4: Sparsity–accuracy Pareto sweep.
Sweeps multiple sparsity levels and both criteria across multiple seeds.
Usage: python pareto_sweep.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import json
import time

from nn.layers import MLP
from nn.optimizer import Adam
from train.trainer import train, load_mnist, make_spirals, evaluate
from prune.pruner import Pruner, count_flops
from engine.tensor import Tensor

SPARSITIES = [0.0, 0.50, 0.75, 0.90, 0.95]
SEEDS = [42, 123, 7]
CRITERIA = ["saliency", "magnitude"]
EPOCHS = 25


def run_one(X_train, y_train, X_test, y_test, n_classes,
            target_sparsity, criterion, seed, epochs=EPOCHS):
    np.random.seed(seed)
    input_size = X_train.shape[1]

    model = MLP(
        input_size=input_size,
        hidden_sizes=[256, 128],
        output_size=n_classes,
        activation="relu",
    )

    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    pruner = None
    if target_sparsity > 0:
        steps_per_epoch = len(X_train) // 128 + 1
        total_steps = steps_per_epoch * epochs
        pruner = Pruner(
            weight_params=model.weight_parameters(),
            target_sparsity=target_sparsity,
            total_steps=total_steps,
            warmup_steps=steps_per_epoch * 2,
            prune_freq=steps_per_epoch,
            criterion=criterion,
            allow_regrowth=True,
            regrowth_fraction=0.05,
            schedule="cubic",
        )

    history = train(
        model,
        X_train, y_train,
        X_test, y_test,
        epochs=epochs,
        batch_size=128,
        optimizer=optimizer,
        pruner=pruner,
        seed=seed,
        verbose=False,
    )

    final_acc = history["test_acc"][-1]
    achieved_sparsity = model.sparsity()
    flops = count_flops(model, input_size)

    # Measure sparse inference time
    t0 = time.perf_counter()
    N_BENCH = 100
    for _ in range(N_BENCH):
        _ = evaluate(model, X_test[:64], y_test[:64])
    sparse_time = (time.perf_counter() - t0) / N_BENCH * 1000  # ms

    return {
        "target_sparsity": target_sparsity,
        "achieved_sparsity": achieved_sparsity,
        "test_acc": final_acc,
        "flops": flops,
        "inference_ms": sparse_time,
        "criterion": criterion,
        "seed": seed,
    }


def main():
    print("=" * 70)
    print("PART 4 — Pareto Sweep: Sparsity vs Accuracy")
    print(f"Sparsities: {SPARSITIES}")
    print(f"Criteria:   {CRITERIA}")
    print(f"Seeds:      {SEEDS}")
    print("=" * 70)

    X_train, y_train, X_test, y_test, n_classes = load_mnist(
        n_train=8000, n_test=2000, seed=42
    )

    all_results = []

    for criterion in CRITERIA:
        for sparsity in SPARSITIES:
            for seed in SEEDS:
                print(f"\n  criterion={criterion}, sparsity={sparsity}, seed={seed}")
                result = run_one(
                    X_train, y_train, X_test, y_test, n_classes,
                    sparsity, criterion, seed,
                )
                all_results.append(result)
                print(f"    -> acc={result['test_acc']:.4f}, "
                      f"sparsity={result['achieved_sparsity']:.4f}, "
                      f"flops={result['flops']:,}")

    # Summarise by criterion x sparsity
    print("\n" + "=" * 70)
    print("SUMMARY (mean ± std over seeds)")
    print(f"{'Criterion':<12} {'Target':>8} {'Acc Mean':>10} {'Acc Std':>8} {'Sparsity':>10} {'FLOPs':>12}")
    print("-" * 70)

    summary = {}
    for criterion in CRITERIA:
        summary[criterion] = {}
        for sparsity in SPARSITIES:
            rows = [r for r in all_results
                    if r["criterion"] == criterion and r["target_sparsity"] == sparsity]
            accs = [r["test_acc"] for r in rows]
            sps = [r["achieved_sparsity"] for r in rows]
            flops = [r["flops"] for r in rows]
            summary[criterion][sparsity] = {
                "acc_mean": float(np.mean(accs)),
                "acc_std": float(np.std(accs)),
                "sparsity_mean": float(np.mean(sps)),
                "flops_mean": float(np.mean(flops)),
            }
            print(f"{criterion:<12} {sparsity:>8.2f} {np.mean(accs):>10.4f} "
                  f"{np.std(accs):>8.4f} {np.mean(sps):>10.4f} {int(np.mean(flops)):>12,}")

    # Falsifiable claim
    print("\n" + "=" * 70)
    print("FALSIFIABLE CLAIM")
    sp90_sal = summary["saliency"].get(0.90, {})
    sp90_mag = summary["magnitude"].get(0.90, {})
    if sp90_sal and sp90_mag:
        print(
            f"At 90% sparsity target, saliency pruning achieves "
            f"{sp90_sal['acc_mean']*100:.2f}% ± {sp90_sal['acc_std']*100:.2f}% accuracy "
            f"vs {sp90_mag['acc_mean']*100:.2f}% ± {sp90_mag['acc_std']*100:.2f}% for "
            f"magnitude pruning, averaged over {len(SEEDS)} seeds."
        )
        delta = sp90_sal["acc_mean"] - sp90_mag["acc_mean"]
        print(f"Advantage of saliency: {delta*100:+.2f} percentage points.")

    # Save all results
    os.makedirs("results", exist_ok=True)
    with open("results/pareto_raw.json", "w") as f:
        json.dump(all_results, f, indent=2)
    with open("results/pareto_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nRaw results: results/pareto_raw.json")
    print("Summary:     results/pareto_summary.json")

    # Plot Pareto curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = {"saliency": "#1f77b4", "magnitude": "#ff7f0e"}
        markers = {"saliency": "o", "magnitude": "s"}

        for ax_idx, criterion in enumerate(CRITERIA):
            sp_vals = sorted(SPARSITIES)
            means = [summary[criterion][sp]["acc_mean"] * 100 for sp in sp_vals]
            stds = [summary[criterion][sp]["acc_std"] * 100 for sp in sp_vals]
            sp_achieved = [summary[criterion][sp]["sparsity_mean"] * 100 for sp in sp_vals]

            axes[0].errorbar(
                sp_achieved, means, yerr=stds,
                marker=markers[criterion],
                color=colors[criterion],
                label=criterion,
                capsize=4,
                linewidth=2,
            )

        axes[0].set_xlabel("Achieved Sparsity (%)")
        axes[0].set_ylabel("Test Accuracy (%)")
        axes[0].set_title("Sparsity–Accuracy Pareto Curve")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].invert_xaxis()

        # FLOPs vs sparsity
        for criterion in CRITERIA:
            sp_vals = sorted(SPARSITIES)
            flops_vals = [summary[criterion][sp]["flops_mean"] for sp in sp_vals]
            sp_achieved = [summary[criterion][sp]["sparsity_mean"] * 100 for sp in sp_vals]
            axes[1].plot(
                sp_achieved, flops_vals,
                marker=markers[criterion],
                color=colors[criterion],
                label=criterion,
                linewidth=2,
            )

        axes[1].set_xlabel("Achieved Sparsity (%)")
        axes[1].set_ylabel("FLOPs (MACs)")
        axes[1].set_title("FLOPs vs Sparsity")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].invert_xaxis()

        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/part4_pareto.png", dpi=150, bbox_inches="tight")
        print("Pareto plot saved to plots/part4_pareto.png")
        plt.close()
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
