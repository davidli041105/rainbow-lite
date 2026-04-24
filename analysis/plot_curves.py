"""
Plot eval-return curves for the 4 variants on a given game.
Reads metrics.csv files from runs/ directory.

Usage:
    python plot_curves.py --runs-dir runs --game pong --out pong_curves.png
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


VARIANT_ORDER = ["dqn", "double", "dueling", "double_duel"]
VARIANT_LABELS = {
    "dqn": "DQN",
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


def smooth(series, window=5):
    return series.rolling(window=window, min_periods=1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--game", required=True, help="e.g. pong, breakout")
    ap.add_argument("--out", default="curves.png")
    ap.add_argument("--smooth", type=int, default=3)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    fig, ax = plt.subplots(figsize=(8, 5))

    for v in VARIANT_ORDER:
        exp = f"{args.game}_{v}"
        csv = runs_dir / exp / "metrics.csv"
        if not csv.exists():
            print(f"Skipping missing: {csv}")
            continue
        df = pd.read_csv(csv)
        evdf = df.dropna(subset=["eval_return"])
        if len(evdf) == 0:
            print(f"No eval data in {csv}")
            continue
        y = smooth(evdf["eval_return"], args.smooth)
        ax.plot(evdf["step"] / 1e6, y, label=VARIANT_LABELS[v],
                color=VARIANT_COLORS[v], linewidth=2)

    ax.set_xlabel("Environment frames (millions)")
    ax.set_ylabel("Evaluation return (mean over 5 eps)")
    ax.set_title(f"{args.game.capitalize()} — ablation over Double & Dueling")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
