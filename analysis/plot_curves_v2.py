"""
Plot eval-return curves for the 4 variants on a given game.
Prefers reeval.csv (low-variance, 20+ episodes) if present, else falls back to metrics.csv.
Adds shaded std bands when reeval.csv is available.

Usage:
    python plot_curves_v2.py --runs-dir runs --game pong --out pong_curves.png
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


VARIANT_ORDER = ["dqn", "double", "dueling", "double_duel"]
VARIANT_LABELS = {
    "dqn": "DQN (baseline)",
    "double": "Double DQN",
    "dueling": "Dueling DQN",
    "double_duel": "Double + Dueling DQN",
}
VARIANT_COLORS = {
    "dqn": "#888888",
    "double": "#d62728",
    "dueling": "#2ca02c",
    "double_duel": "#1f77b4",
}


def smooth(series, window=3):
    return series.rolling(window=window, min_periods=1).mean()


def load_curve(run_dir: Path):
    """Returns (steps, mean_return, std_return_or_None, source_label)."""
    reeval = run_dir / "reeval.csv"
    if reeval.exists():
        df = pd.read_csv(reeval)
        return df["step"].values, df["return_mean"].values, df["return_std"].values, "reeval"
    metrics = run_dir / "metrics.csv"
    if metrics.exists():
        df = pd.read_csv(metrics)
        evdf = df.dropna(subset=["eval_return"])
        return evdf["step"].values, evdf["eval_return"].values, None, "metrics"
    return None, None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--game", required=True)
    ap.add_argument("--out", default="curves.png")
    ap.add_argument("--smooth", type=int, default=3)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for v in VARIANT_ORDER:
        exp = f"{args.game}_{v}"
        run_dir = runs_dir / exp
        if not run_dir.exists():
            print(f"Skipping missing dir: {run_dir}")
            continue
        steps, mean, std, src = load_curve(run_dir)
        if steps is None or len(steps) == 0:
            print(f"  No data for {exp}")
            continue
        steps_m = steps / 1e6
        mean_smooth = smooth(pd.Series(mean), args.smooth).values
        ax.plot(steps_m, mean_smooth, label=VARIANT_LABELS[v],
                color=VARIANT_COLORS[v], linewidth=2.2)
        if std is not None:
            std_smooth = smooth(pd.Series(std), args.smooth).values
            ax.fill_between(steps_m,
                            mean_smooth - std_smooth,
                            mean_smooth + std_smooth,
                            color=VARIANT_COLORS[v], alpha=0.15)
        print(f"  {exp}: source={src}, n_points={len(steps)}, "
              f"final_mean={mean[-1]:.2f}")

    ax.set_xlabel("Environment frames (millions)", fontsize=12)
    ax.set_ylabel("Evaluation return", fontsize=12)
    ax.set_title(f"{args.game.capitalize()} — Double/Dueling ablation",
                 fontsize=13)
    ax.legend(loc="best", frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", lw=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()