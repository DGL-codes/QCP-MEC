from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from paper4_sim.reporting import summarize_payment_discipline


RESULT_DIR = Path("results")
FIG_DIR = RESULT_DIR / "figures"
PROPOSED = "QCP"

PAPER_FIGURES = {
    "default_advantage_heatmap.png",
    "default_advantage_radar.png",
    "ablation_metric_bars.png",
    "arrival_combined.png",
    "reputation_combined.png",
    "payment_discipline_trace.png",
    "deposit_combined.png",
    "reservation_lyapunov_combined.png",
}

POLICY_ORDER = [
    "QCP",
    "DP-MEC",
    "Stackelberg-EPRA",
    "TARFO",
    "DoubleAuction",
    "Delay-Energy",
    "NoReputation",
]

COLORS = {
    "QCP": "#D34A3A",
    "DP-MEC": "#4C78A8",
    "Stackelberg-EPRA": "#72B7B2",
    "TARFO": "#59A14F",
    "DoubleAuction": "#F28E2B",
    "Delay-Energy": "#B07AA1",
    "NoReputation": "#7F7F7F",
    "Ablation-NoReservation": "#4C78A8",
    "Ablation-NoDeposit": "#59A14F",
    "Ablation-NoPostSettlement": "#F28E2B",
    "Ablation-NoVirtualQueues": "#B07AA1",
}

MARKERS = {
    "QCP": "o",
    "DP-MEC": "s",
    "Stackelberg-EPRA": "^",
    "TARFO": "D",
    "DoubleAuction": "P",
    "Delay-Energy": "X",
    "NoReputation": "v",
}

LINESTYLES = {
    "QCP": "-",
    "DP-MEC": "--",
    "Stackelberg-EPRA": "-.",
    "TARFO": ":",
    "DoubleAuction": (0, (4, 1.5, 1, 1.5)),
    "Delay-Energy": (0, (5, 2)),
    "NoReputation": (0, (2, 1.4)),
}

METRIC_LABELS = {
    "average_delay": "Average delay (s)",
    "average_energy": "Average energy (J)",
    "sla_satisfaction_ratio": "SLA satisfaction ratio",
    "average_user_payment": "Average user payment",
    "average_user_utility": "Average user utility",
    "average_social_welfare": "Average social welfare",
    "broker_utility": "Broker utility",
    "bad_debt_loss": "Bad-debt loss",
    "settlement_deviation": "Settlement deviation",
    "task_acceptance_ratio": "Task acceptance ratio",
    "individual_rational_ratio": "Nonnegative realized\nutility ratio",
    "budget_balance_ratio": "Nonnegative broker utility ratio",
}

PANEL_TITLES = {
    "average_delay": "Delay",
    "average_energy": "Energy",
    "sla_satisfaction_ratio": "SLA satisfaction",
    "average_user_payment": "User payment",
    "average_user_utility": "User utility",
    "average_social_welfare": "Social welfare",
    "task_acceptance_ratio": "Task acceptance",
    "individual_rational_ratio": "Nonnegative realized\nutility ratio",
    "budget_balance_ratio": "Nonnegative broker\nutility ratio",
}

LOWER_IS_BETTER = {
    "average_delay",
    "average_energy",
    "average_user_payment",
    "bad_debt_loss",
    "settlement_deviation",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 7.5,
            "figure.dpi": 170,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the publication figures used by the QCP paper.")
    parser.add_argument("--results", default="results", help="Directory containing generated CSV files.")
    parser.add_argument("--clean-stale", action="store_true",
                        help="Explicitly remove non-paper PNG files from the selected figures directory.")
    return parser.parse_args()


def set_result_dir(path: str) -> None:
    global RESULT_DIR, FIG_DIR
    RESULT_DIR = Path(path)
    FIG_DIR = RESULT_DIR / "figures"


def cleanup_unused_figures() -> None:
    if not FIG_DIR.exists():
        return
    for path in FIG_DIR.glob("*.png"):
        if path.name not in PAPER_FIGURES:
            path.unlink()


def ordered_policies(df: pd.DataFrame) -> List[str]:
    return [p for p in POLICY_ORDER if p in set(df["policy"])]


def add_grid(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D8DEE9", linewidth=0.65, alpha=0.78)
    ax.grid(axis="x", color="#EDF0F5", linewidth=0.45, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def refine_line_axis(ax: plt.Axes, values: Iterable[float], metric: str) -> None:
    series = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if series.empty:
        return
    lo = float(series.min())
    hi = float(series.max())
    span = hi - lo
    ratio_metric = metric.endswith("_ratio")

    if ratio_metric:
        if span < 0.08 or lo > 0.85:
            margin = max(span * 0.28, 0.004)
            lower = max(0.0, lo - margin)
            upper = min(1.02, hi + margin)
            if upper - lower < 0.015:
                center = 0.5 * (lo + hi)
                lower = max(0.0, center - 0.008)
                upper = min(1.02, center + 0.008)
            ax.set_ylim(lower, upper)
            ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3f"))
        else:
            ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
    elif span > 0:
        center = 0.5 * (lo + hi)
        relative_span = span / max(abs(center), 1e-9)
        if relative_span < 0.18:
            margin = max(span * 0.25, 0.002)
            lower = max(0.0, lo - margin)
            upper = hi + margin
            ax.set_ylim(lower, upper)
            ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3f"))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))


def put_legend_below(ax: plt.Axes, ncol: int = 2, y: float = -0.24, fontsize: float = 6.9) -> None:
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=True,
        fancybox=False,
        framealpha=0.94,
        edgecolor="#D0D5DD",
        fontsize=fontsize,
    )


def annotate_edge(ax: plt.Axes, x: float, y: float, text: str, color: str = "#D34A3A") -> None:
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(8, 9),
        textcoords="offset points",
        color=color,
        fontsize=8.5,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": color, "lw": 0.9},
    )


def plot_spotlight_lines(
    df: pd.DataFrame,
    x_col: str,
    metrics: Iterable[str],
    prefix: str,
    x_label: str,
) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    policies = ordered_policies(df)
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(6.55, 4.35))
        for policy in policies:
            group = df[df["policy"] == policy].sort_values(x_col)
            if group.empty:
                continue
            is_prop = policy == PROPOSED
            ax.plot(
                group[x_col],
                group[metric],
                marker=MARKERS[policy],
                markersize=6.5 if is_prop else 4.7,
                linewidth=2.85 if is_prop else 1.15,
                linestyle=LINESTYLES.get(policy, "-"),
                color=COLORS[policy],
                markerfacecolor="white",
                markeredgecolor=COLORS[policy],
                markeredgewidth=1.15 if is_prop else 0.8,
                alpha=1.0 if is_prop else 0.46,
                label=policy,
                zorder=5 if is_prop else 2,
            )
            if is_prop:
                ax.scatter(
                    group[x_col],
                    group[metric],
                    s=62,
                    facecolor="white",
                    edgecolor=COLORS[policy],
                    linewidth=1.5,
                    zorder=6,
                )

        prop = df[df["policy"] == PROPOSED].sort_values(x_col)
        if len(prop):
            ax.fill_between(
                prop[x_col].to_numpy(),
                prop[metric].to_numpy(),
                alpha=0.085,
                color=COLORS[PROPOSED],
                zorder=1,
            )
            mid = prop.iloc[len(prop) // 2]
            compare = df[(df[x_col] == mid[x_col]) & (df["policy"] != PROPOSED)]
            if not compare.empty:
                if metric in LOWER_IS_BETTER:
                    ref = compare[metric].min()
                    gain = (ref - mid[metric]) / max(abs(ref), 1e-9) * 100.0
                    if gain > 1:
                        annotate_edge(ax, mid[x_col], mid[metric], f"{gain:.1f}% lower")
                else:
                    ref = compare[metric].max()
                    gain = (mid[metric] - ref) / max(abs(ref), 1e-9) * 100.0
                    if gain > 1:
                        annotate_edge(ax, mid[x_col], mid[metric], f"{gain:.1f}% higher")

        ax.set_xlabel(x_label)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=12, fontweight="bold", pad=8)
        refine_line_axis(ax, df[metric], metric)
        add_grid(ax)
        put_legend_below(ax, ncol=2, y=-0.24)
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        fig.savefig(FIG_DIR / f"{prefix}_{metric}.png")
        plt.close(fig)


def add_shared_legend(
    fig: plt.Figure,
    axes: np.ndarray,
    ncol: int = 4,
    y: float = 0.010,
    fontsize: float = 7.2,
    handlelength: float = 2.2,
    columnspacing: float = 1.1,
) -> None:
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        edgecolor="#D0D5DD",
        fontsize=fontsize,
        handlelength=handlelength,
        columnspacing=columnspacing,
    )


def plot_metric_lines_on_axis(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    metric: str,
    policies: List[str],
    title_fontsize: float = 10.4,
) -> None:
    for policy in policies:
        group = df[df["policy"] == policy].sort_values(x_col)
        if group.empty:
            continue
        is_prop = policy == PROPOSED
        ax.plot(
            group[x_col],
            group[metric],
            marker=MARKERS[policy],
            markersize=5.3 if is_prop else 3.9,
            linewidth=2.25 if is_prop else 1.05,
            linestyle=LINESTYLES.get(policy, "-"),
            color=COLORS[policy],
            markerfacecolor="white",
            markeredgecolor=COLORS[policy],
            markeredgewidth=1.05 if is_prop else 0.72,
            alpha=1.0 if is_prop else 0.48,
            label=policy,
            zorder=5 if is_prop else 2,
        )
    prop = df[df["policy"] == PROPOSED].sort_values(x_col)
    if len(prop):
        ax.fill_between(
            prop[x_col].to_numpy(),
            prop[metric].to_numpy(),
            alpha=0.075,
            color=COLORS[PROPOSED],
            zorder=1,
        )
    ax.set_title(
        PANEL_TITLES.get(metric, METRIC_LABELS.get(metric, metric)),
        fontsize=title_fontsize,
        fontweight="bold",
        pad=5 if title_fontsize < 9 else 6,
    )
    if title_fontsize < 9:
        ax.tick_params(axis="both", labelsize=6.3)
    refine_line_axis(ax, df[metric], metric)
    add_grid(ax)


def plot_spotlight_grid(
    df: pd.DataFrame,
    x_col: str,
    metrics: List[str],
    output_name: str,
    x_label: str,
    ncols: int,
    figsize: tuple[float, float],
    legend_ncol: int = 4,
    legend_y: float = 0.012,
    legend_fontsize: float = 7.0,
    bottom: float = 0.115,
    title_fontsize: float = 10.4,
    h_pad: float = 1.0,
    w_pad: float = 0.85,
) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    policies = ordered_policies(df)
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    axes = np.asarray(axes).ravel()

    for idx, (ax, metric) in enumerate(zip(axes, metrics)):
        plot_metric_lines_on_axis(ax, df, x_col, metric, policies, title_fontsize=title_fontsize)
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel(x_label)
            if title_fontsize < 9:
                ax.xaxis.label.set_size(7.0)
        else:
            ax.set_xlabel("")
    for ax in axes[len(metrics) :]:
        ax.set_visible(False)

    fig.tight_layout(rect=(0.0, bottom, 1.0, 1.0), h_pad=h_pad, w_pad=w_pad)
    add_shared_legend(
        fig,
        axes,
        ncol=legend_ncol,
        y=legend_y,
        fontsize=legend_fontsize,
        handlelength=1.55 if legend_ncol <= 2 else 2.2,
        columnspacing=0.55 if legend_ncol <= 2 else 1.1,
    )
    fig.savefig(FIG_DIR / output_name)
    plt.close(fig)


def plot_reputation_bars(df: pd.DataFrame, metrics: Iterable[str]) -> None:
    policies = ordered_policies(df)
    xs = sorted(df["low_reputation_ratio"].unique())
    width = 0.105
    offsets = np.linspace(-width * (len(policies) - 1) / 2, width * (len(policies) - 1) / 2, len(policies))

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(6.7, 4.25))
        x = np.arange(len(xs))
        for offset, policy in zip(offsets, policies):
            values = [
                float(df[(df["policy"] == policy) & (df["low_reputation_ratio"] == ratio)][metric].iloc[0])
                for ratio in xs
            ]
            is_prop = policy == PROPOSED
            ax.bar(
                x + offset,
                values,
                width=width,
                label=policy,
                color=COLORS[policy],
                alpha=0.96 if is_prop else 0.52,
                edgecolor="#FFFFFF" if is_prop else "#E6E8EF",
                linewidth=1.0 if is_prop else 0.45,
                zorder=4 if is_prop else 2,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:.2f}" for v in xs])
        ax.set_xlabel("Low-reputation user ratio")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=12, fontweight="bold", pad=8)
        add_grid(ax)
        put_legend_below(ax, ncol=2, y=-0.23)
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        fig.savefig(FIG_DIR / f"reputation_{metric}.png")
        plt.close(fig)


def plot_reputation_grid(df: pd.DataFrame, metrics: List[str], output_name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    policies = ordered_policies(df)
    xs = sorted(df["low_reputation_ratio"].unique())
    width = 0.105
    offsets = np.linspace(-width * (len(policies) - 1) / 2, width * (len(policies) - 1) / 2, len(policies))
    x = np.arange(len(xs))

    fig, axes = plt.subplots(2, 3, figsize=(10.35, 5.45), sharex=True)
    axes = axes.ravel()
    for idx, (ax, metric) in enumerate(zip(axes, metrics)):
        for offset, policy in zip(offsets, policies):
            values = [
                float(df[(df["policy"] == policy) & (df["low_reputation_ratio"] == ratio)][metric].iloc[0])
                for ratio in xs
            ]
            is_prop = policy == PROPOSED
            ax.bar(
                x + offset,
                values,
                width=width,
                label=policy,
                color=COLORS[policy],
                alpha=0.96 if is_prop else 0.52,
                edgecolor="#FFFFFF" if is_prop else "#E6E8EF",
                linewidth=0.9 if is_prop else 0.4,
                zorder=4 if is_prop else 2,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:.2f}" for v in xs])
        if idx >= 3:
            ax.set_xlabel("Low-reputation user ratio")
        ax.set_title(PANEL_TITLES.get(metric, METRIC_LABELS.get(metric, metric)), fontsize=10.4, fontweight="bold", pad=6)
        refine_line_axis(ax, df[metric], metric)
        add_grid(ax)

    fig.tight_layout(rect=(0.0, 0.115, 1.0, 1.0), h_pad=1.0, w_pad=0.85)
    add_shared_legend(fig, axes, ncol=4, y=0.012, fontsize=7.0)
    fig.savefig(FIG_DIR / output_name)
    plt.close(fig)


def plot_payment_discipline(transactions: pd.DataFrame) -> None:
    plot_payment_discipline_summary(summarize_payment_discipline(transactions))


def plot_payment_discipline_summary(trace: pd.DataFrame) -> None:
    required = {"policy", "slot_bin", "nonpayment", "user_reputation", "deposit_quote_ratio"}
    if not required.issubset(trace.columns):
        raise ValueError(f"Payment-discipline summary is missing columns: {sorted(required - set(trace.columns))}")
    if trace.empty:
        return
    policies = [PROPOSED, "NoReputation"]

    fig, axes = plt.subplots(1, 3, figsize=(9.25, 3.35))
    panels = [
        ("nonpayment", "Non-payment ratio (%)", lambda values: 100.0 * values),
        ("user_reputation", "Average user reputation", lambda values: values),
        ("deposit_quote_ratio", "Deposit / quote ratio", lambda values: values),
    ]
    for ax, (metric, ylabel, transform) in zip(axes, panels):
        for policy in policies:
            group = trace[trace["policy"] == policy].sort_values("slot_bin")
            if group.empty:
                continue
            is_prop = policy == PROPOSED
            x = group["slot_bin"].to_numpy()
            y = transform(group[metric].to_numpy())
            color = COLORS[policy]
            ax.plot(
                x,
                y,
                marker=MARKERS.get(policy, "o"),
                markersize=6.0 if is_prop else 4.8,
                linewidth=2.35 if is_prop else 1.45,
                linestyle=LINESTYLES.get(policy, "-"),
                color=color,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.1,
                alpha=1.0 if is_prop else 0.72,
                label=policy,
                zorder=4 if is_prop else 3,
            )
            if is_prop:
                ax.scatter(x, y, s=52, facecolor="white", edgecolor=color, linewidth=1.15, zorder=5)
        ax.set_xlabel("Time slot")
        ax.set_ylabel(ylabel)
        add_grid(ax)
    axes[0].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
    axes[1].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3f"))
    axes[2].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
    axes[0].set_title("Payment default", fontsize=11.2, fontweight="bold", pad=7)
    axes[1].set_title("Reputation recovery", fontsize=11.2, fontweight="bold", pad=7)
    axes[2].set_title("Future cost pressure", fontsize=11.2, fontweight="bold", pad=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        frameon=True,
        fancybox=False,
        framealpha=0.94,
        edgecolor="#D0D5DD",
        fontsize=7.1,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(FIG_DIR / "payment_discipline_trace.png")
    plt.close(fig)


def normalized_scores(single: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    score = single[["policy"]].copy()
    for metric in metrics:
        values = single[metric].astype(float)
        lo, hi = values.min(), values.max()
        denom = max(hi - lo, 1e-12)
        if metric in LOWER_IS_BETTER:
            score[metric] = (hi - values) / denom
        else:
            score[metric] = (values - lo) / denom
    return score


def plot_advantage_heatmap(single: pd.DataFrame) -> None:
    metrics = [
        "average_delay",
        "average_energy",
        "sla_satisfaction_ratio",
        "average_user_payment",
        "average_user_utility",
        "task_acceptance_ratio",
    ]
    scores = normalized_scores(single, metrics)
    scores = scores.set_index("policy").loc[ordered_policies(single)]
    fig, ax = plt.subplots(figsize=(7.2, 3.55))
    im = ax.imshow(scores[metrics].to_numpy(), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(
        ["Delay", "Energy", "SLA", "Payment", "User\nutility", "Accept."],
        fontsize=9,
    )
    ax.set_yticks(np.arange(len(scores)))
    ax.set_yticklabels(
        scores.index.tolist(),
        fontsize=8.5,
    )
    for i, policy in enumerate(scores.index):
        for j, metric in enumerate(metrics):
            val = scores.loc[policy, metric]
            color = "white" if val < 0.28 or val > 0.78 else "#111827"
            weight = "bold" if policy == PROPOSED else "normal"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.8, color=color, fontweight=weight)
    ax.set_title("Core advantage map (higher is better)", fontsize=12, fontweight="bold", pad=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "default_advantage_heatmap.png")
    plt.close(fig)


def plot_radar(single: pd.DataFrame) -> None:
    metrics = [
        "average_delay",
        "average_energy",
        "sla_satisfaction_ratio",
        "average_user_payment",
        "average_user_utility",
        "task_acceptance_ratio",
    ]
    labels = ["Low delay", "Low energy", "High SLA", "Low payment", "User\nutility", "Acceptance"]
    scores = normalized_scores(single, metrics).set_index("policy")
    policies = [PROPOSED, "DP-MEC", "Stackelberg-EPRA", "TARFO", "DoubleAuction", "NoReputation"]
    policies = [p for p in policies if p in scores.index]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(5.25, 4.35))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 4.0)
    for policy in policies:
        vals = scores.loc[policy, metrics].tolist()
        vals += vals[:1]
        is_prop = policy == PROPOSED
        ax.plot(
            angles,
            vals,
            color=COLORS[policy],
            linewidth=2.8 if is_prop else 1.05,
            linestyle=LINESTYLES.get(policy, "-"),
            alpha=1.0 if is_prop else 0.42,
            label=policy,
        )
        ax.fill(angles, vals, color=COLORS[policy], alpha=0.16 if is_prop else 0.035)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8.7)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7.2, color="#596273")
    ax.grid(color="#D8DEE9", linewidth=0.75)
    ax.spines["polar"].set_color("#AAB2C0")
    ax.set_title("Overall advantage profile", fontsize=12, fontweight="bold", pad=16)
    ax.legend(loc="center left", bbox_to_anchor=(1.10, 0.5), ncol=1, frameon=False, fontsize=7.1)
    fig.tight_layout(rect=(0, 0, 0.80, 1))
    fig.savefig(FIG_DIR / "default_advantage_radar.png")
    plt.close(fig)


def plot_ablation_bars(ablation: pd.DataFrame) -> None:
    metrics = [
        ("average_delay", "Delay", True),
        ("sla_satisfaction_ratio", "SLA", False),
        ("average_user_payment", "Payment", True),
        ("average_user_utility", "User utility", False),
        ("average_social_welfare", "Welfare", False),
        ("task_acceptance_ratio", "Acceptance", False),
    ]
    policies = [
        "QCP",
        "Ablation-NoReservation",
        "NoReputation",
        "Ablation-NoDeposit",
        "Ablation-NoPostSettlement",
        "Ablation-NoVirtualQueues",
    ]
    policies = [p for p in policies if p in set(ablation["policy"])]
    ordered = ablation.set_index("policy").loc[policies].reset_index()
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.2))
    axes = axes.ravel()
    for ax, (metric, title, lower) in zip(axes, metrics):
        values = ordered[metric].to_numpy()
        y = np.arange(len(ordered))
        colors = [COLORS.get(p, "#9AA4B2") for p in ordered["policy"]]
        ax.barh(y, values, color=colors, alpha=0.88, edgecolor="#FFFFFF", linewidth=0.7)
        best = values.min() if lower else values.max()
        ax.axvline(best, color="#111827", linewidth=0.8, linestyle=":", alpha=0.72)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_yticks(y)
        if ax in axes[::3]:
            ax.set_yticklabels(ordered["policy"], fontsize=7.0)
        else:
            ax.set_yticklabels([])
        ax.invert_yaxis()
        ax.grid(axis="x", color="#D8DEE9", linewidth=0.55, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Ablation study", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ablation_metric_bars.png")
    plt.close(fig)


def plot_reservation_lyapunov_combined(reservation: pd.DataFrame, lyapunov: pd.DataFrame) -> None:
    reservation = reservation[reservation["policy"] == PROPOSED].sort_values(
        "proposed_reservation_capacity_fraction"
    )
    lyapunov = lyapunov[lyapunov["policy"] == PROPOSED].sort_values("lyapunov_v")

    def plot_dual_axis(
        ax: plt.Axes,
        x: pd.Series,
        y_left: pd.Series,
        y_right: pd.Series,
        xlabel: str,
        left_label: str,
        right_label: str,
        left_title: str,
        right_title: str,
        left_color: str,
        right_color: str,
        left_marker: str,
        right_marker: str,
    ) -> None:
        ax_r = ax.twinx()
        line_left = ax.plot(
            x,
            y_left,
            marker=left_marker,
            markersize=4.8,
            linewidth=2.0,
            color=left_color,
            markerfacecolor="white",
            markeredgecolor=left_color,
            markeredgewidth=1.0,
            label=left_title,
            zorder=3,
        )[0]
        line_right = ax_r.plot(
            x,
            y_right,
            marker=right_marker,
            markersize=4.8,
            linewidth=2.0,
            linestyle="--",
            color=right_color,
            markerfacecolor="white",
            markeredgecolor=right_color,
            markeredgewidth=1.0,
            label=right_title,
            zorder=3,
        )[0]
        ax.set_xlabel(xlabel, fontsize=7.4)
        ax.set_ylabel(left_label, fontsize=7.2, color=left_color)
        ax_r.set_ylabel(right_label, fontsize=7.2, color=right_color)
        ax.tick_params(axis="both", labelsize=6.8)
        ax.tick_params(axis="y", colors=left_color)
        ax_r.tick_params(axis="y", labelsize=6.8, colors=right_color)
        add_grid(ax)
        ax_r.grid(False)
        ax.legend(
            [line_left, line_right],
            [left_title, right_title],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.28),
            ncol=2,
            fontsize=6.0,
            frameon=True,
            fancybox=False,
            framealpha=0.88,
            borderaxespad=0.0,
        )

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.75))
    plot_dual_axis(
        axes[0],
        reservation["proposed_reservation_capacity_fraction"],
        reservation["average_delay"],
        reservation["sla_satisfaction_ratio"],
        r"Reserved-service budget fraction ($\varphi_{\rm res}$)",
        "Average delay (s)",
        "SLA satisfaction",
        "Delay",
        "SLA",
        COLORS[PROPOSED],
        "#4C78A8",
        "o",
        "s",
    )
    plot_dual_axis(
        axes[1],
        lyapunov["lyapunov_v"],
        lyapunov["average_delay"],
        lyapunov["average_social_welfare"],
        r"Lyapunov parameter ($V$)",
        "Average delay (s)",
        "Social welfare",
        "Delay",
        "Welfare",
        "#59A14F",
        "#F28E2B",
        "^",
        "D",
    )
    axes[1].set_xscale("log", base=2)
    v_ticks = lyapunov["lyapunov_v"].astype(float).to_numpy()
    axes[1].set_xticks(v_ticks)
    axes[1].set_xticklabels([f"{int(v)}" for v in v_ticks])

    fig.tight_layout(pad=0.35, w_pad=2.2)
    fig.subplots_adjust(bottom=0.30)
    fig.savefig(FIG_DIR / "reservation_lyapunov_combined.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_result_dir(args.results)
    configure_style()
    single = pd.read_csv(RESULT_DIR / "single_summary.csv")
    arrival = pd.read_csv(RESULT_DIR / "arrival_sweep_summary.csv")
    reputation = pd.read_csv(RESULT_DIR / "reputation_sweep_summary.csv")
    deposit = pd.read_csv(RESULT_DIR / "deposit_sweep_summary.csv")
    ablation_path = RESULT_DIR / "ablation_summary.csv"
    reservation_path = RESULT_DIR / "reservation_sweep_summary.csv"
    lyapunov_path = RESULT_DIR / "lyapunov_sweep_summary.csv"

    plot_spotlight_grid(
        arrival,
        "arrival_rate",
        [
            "average_delay",
            "average_energy",
            "sla_satisfaction_ratio",
            "average_user_payment",
            "average_user_utility",
            "task_acceptance_ratio",
        ],
        "arrival_combined.png",
        "Task arrival rate",
        ncols=3,
        figsize=(10.35, 5.45),
    )
    plot_reputation_grid(
        reputation,
        [
            "sla_satisfaction_ratio",
            "task_acceptance_ratio",
            "average_user_utility",
            "average_user_payment",
            "individual_rational_ratio",
            "budget_balance_ratio",
        ],
        "reputation_combined.png",
    )
    plot_spotlight_grid(
        deposit,
        "deposit_price_coeff",
        [
            "average_user_payment",
            "task_acceptance_ratio",
            "average_user_utility",
            "individual_rational_ratio",
        ],
        "deposit_combined.png",
        "Deposit coefficient",
        ncols=2,
        figsize=(3.45, 4.3),
        legend_ncol=2,
        legend_y=0.035,
        legend_fontsize=5.2,
        bottom=0.165,
        title_fontsize=7.4,
        h_pad=1.0,
        w_pad=0.35,
    )
    plot_advantage_heatmap(single)
    plot_radar(single)
    if ablation_path.exists():
        plot_ablation_bars(pd.read_csv(ablation_path))
    if reservation_path.exists() and lyapunov_path.exists():
        plot_reservation_lyapunov_combined(
            pd.read_csv(reservation_path),
            pd.read_csv(lyapunov_path),
        )
    transactions_path = RESULT_DIR / "single_transactions.csv"
    if transactions_path.exists():
        plot_payment_discipline(pd.read_csv(transactions_path))
    else:
        summary_path = RESULT_DIR / "payment_discipline_summary.csv"
        if summary_path.exists():
            plot_payment_discipline_summary(pd.read_csv(summary_path))
        else:
            print("Skipped payment-discipline figure: neither trace nor compact summary is available.")
    if args.clean_stale:
        cleanup_unused_figures()
    print(f"Figures written to: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
