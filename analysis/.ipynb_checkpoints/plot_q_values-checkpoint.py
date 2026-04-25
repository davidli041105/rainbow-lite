"""
Plot training Q-mean over time to visualize the overestimation
behavior of vanilla DQN vs. Double DQN.

The hypothesis from van Hasselt et al. 2016: vanilla DQN's max-operator
in target computation systematically inflates Q estimates, while Double DQN
reduces this bias.

Usage:
    python plot_q_values.py --runs-dir runs --game pong --out pong_qvalues.png
    python plot_q_values.py --runs-dir runs --game breakout --out breakout_qvalues.png
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

VARIANT_LABELS = {
    "dqn": "DQN (no Double)",
    "double": "Double DQN",
    "dueling": "Dueling DQN (no Double)",
    "double_duel": "Double + Dueling",
}
VARIANT_COLORS = {
    "dqn": "#888888",
    "double": "#d62728",
    "dueling": "#2ca02c",
    "double_duel": "#1f77b4",
}
VARIANT_ORDER = ["dqn", "double", "dueling", "double_duel"]


def load_q_means(run_dir: Path):
    """Read TensorBoard event files for q_mean, fall back to metrics.csv."""
    metrics_csv = run_dir / "metrics.csv"
    if not metrics_csv.exists():
        return None, None

    # We logged training Q-mean in q_mean column when grad_steps fire (every train_freq).
    # The CSV format from train.py only has q_mean implicitly, so let's try TensorBoard:
    try:
        from tensorboard.backend.event_processing import event_accumulator
        events_files = sorted(run_dir.glob("events.out.tfevents.*"))
        if not events_files:
            return None, None
        ea = event_accumulator.EventAccumulator(str(events_files[-1]),
                                                size_guidance={"scalars": 0})
        ea.Reload()
        if "train/q_mean" not in ea.Tags()["scalars"]:
            return None, None
        events = ea.Scalars("train/q_mean")
        steps = [e.step for e in events]
        values = [e.value for e in events]
        return steps, values
    except ImportError:
        print("tensorboard not installed; cannot read event files")
        return None, None


def smooth(values, window=20):
    return pd.Series(values).rolling(window=window, min_periods=1).mean().values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--game", required=True)
    ap.add_argument("--out", default="qvalues.png")
    ap.add_argument("--smooth", type=int, default=30)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for v in VARIANT_ORDER:
        run_dir = runs_dir / f"{args.game}_{v}"
        if not run_dir.exists():
            continue
        steps, values = load_q_means(run_dir)
        if steps is None or len(steps) == 0:
            print(f"  No q_mean data for {run_dir.name}")
            continue
        # ls = "--" if v in ("dqn", "dueling") else "-"  # dashed for non-Double
        ls = "-"
        sm = smooth(values, args.smooth)
        ax.plot([s / 1e6 for s in steps], sm,
                label=VARIANT_LABELS[v],
                color=VARIANT_COLORS[v],
                linestyle=ls, linewidth=2)
        print(f"  {run_dir.name}: max q_mean={max(values):.3f}, "
              f"final q_mean={values[-1]:.3f}")

    ax.set_xlabel("Environment frames (millions)", fontsize=12)
    ax.set_ylabel(r"Mean predicted Q-value (training batch)", fontsize=12)
    ax.set_title(f"{args.game.capitalize()} — Q-value estimates: "
                 "DQN vs Double DQN", fontsize=13)
    ax.legend(loc="best", frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", lw=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()