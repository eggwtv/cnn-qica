"""
workflow_plots.py
=========================================================
Shared plotting utilities for W1-W7.

Creates

    <prefix>_convergence.png
    <prefix>_boxplot.png
    <prefix>_histogram.png
    <prefix>_sensitivity.png   (if sensitivity supplied)
"""

import numpy as np
import matplotlib.pyplot as plt

def save_workflow_plots(prefix, results, train_floor, raw, sens=None):

    # =====================================================
    # 1. Convergence
    # =====================================================

    plt.figure(figsize=(8,5))

    for r in results:

        hist = r.get("history", None)

        if hist is None:
            continue

        if isinstance(hist, dict):
            series = hist.get('best_raw_ppf') or hist.get('best_fitness')
            if series is None:
                continue
        else:
            series = hist

        plt.plot(
            series,
            linewidth=2,
            alpha=0.8,
            label=f"Seed {r['seed']}"
        )

    plt.xlabel("Generation")
    plt.ylabel("Best PPF")
    plt.title(f"{prefix.upper()} Convergence")

    if len(results) <= 10:
        plt.legend()

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{prefix}_convergence.png", dpi=300)
    plt.close()

    # (rest of the function unchanged — boxplot / histogram / sensitivity)


    # =====================================================
    # 2. Boxplot
    # =====================================================

    fits = np.array([r["best_fitness"] for r in results])

    plt.figure(figsize=(4,6))

    plt.boxplot(
        fits,
        labels=["Best PPF"]
    )

    plt.ylabel("PPF")

    plt.title(f"{prefix.upper()} Final Results")

    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{prefix}_boxplot.png", dpi=300)
    plt.close()


    # =====================================================
    # 3. Histogram
    # =====================================================

    plt.figure(figsize=(7,5))

    plt.hist(
        raw,
        bins=max(5, min(len(raw), 10)),
        edgecolor="black",
    )

    plt.axvline(
        train_floor,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Training floor"
    )

    plt.xlabel("Predicted PPF")
    plt.ylabel("Count")
    plt.title(f"{prefix.upper()} Best PPF Distribution")

    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{prefix}_histogram.png", dpi=300)
    plt.close()


    # =====================================================
    # 4. Sensitivity (optional)
    # =====================================================

    if sens is not None:

        plt.figure(figsize=(10,4))

        x = np.arange(1, len(sens)+1)

        plt.bar(x, sens)

        plt.xlabel("Fuel Position")
        plt.ylabel("Sensitivity")

        plt.title(f"{prefix.upper()} Position Sensitivity")

        plt.grid(axis="y")

        plt.tight_layout()
        plt.savefig(f"{prefix}_sensitivity.png", dpi=300)
        plt.close()


    print(f"[PLOTS] Saved plots for {prefix}")