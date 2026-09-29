from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from paper4_sim.config import SimConfig
from paper4_sim.policies import ProposedLyapunovPolicy, make_ablation_policies, make_default_policies
from paper4_sim.simulator import MecSimulator
from paper4_sim.reporting import summarize_payment_discipline


METRIC_LABELS = {
    "average_delay": "Average delay (s)",
    "average_energy": "Average user energy (J)",
    "sla_satisfaction_ratio": "SLA satisfaction ratio",
    "average_user_payment": "Average user payment",
    "average_user_utility": "Average user utility",
    "average_social_welfare": "Average social welfare",
    "broker_utility": "Broker utility per accepted task",
    "bad_debt_loss": "Bad debt loss per accepted task",
    "settlement_deviation": "Average settlement deviation",
    "task_acceptance_ratio": "Task acceptance ratio",
    "individual_rational_ratio": "Nonnegative realized-utility ratio",
    "budget_balance_ratio": "Nonnegative broker utility ratio",
}

PRIMARY_METRICS = [
    "average_delay",
    "average_energy",
    "sla_satisfaction_ratio",
    "average_user_payment",
    "average_user_utility",
    "average_social_welfare",
    "task_acceptance_ratio",
]

LOWER_IS_BETTER = {
    "average_delay",
    "average_energy",
    "average_user_payment",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Paper 4 MEC pricing/offloading simulations.")
    parser.add_argument("--out", default="results", help="Output directory for generated CSV files.")
    parser.add_argument("--quick", action="store_true", help="Run a small smoke-test experiment.")
    parser.add_argument("--time-slots", type=int, default=None, help="Override number of time slots.")
    parser.add_argument("--repetitions", type=int, default=None, help="Number of Monte Carlo repetitions.")
    parser.add_argument(
        "--experiment",
        choices=["all", "arrival", "reputation", "deposit", "single", "ablation", "reservation", "lyapunov"],
        default="all",
        help="Experiment family to run.",
    )
    args = parser.parse_args()
    if args.time_slots is not None and args.time_slots <= 10:
        parser.error("--time-slots must be greater than 10 to retain samples after warm-up")
    if args.repetitions is not None and args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    return args


class ProgressBar:
    def __init__(self, total: int, enabled: bool = True) -> None:
        self.total = max(int(total), 1)
        self.enabled = enabled
        self.count = 0
        self.start_time = time.time()

    def update(self, label: str) -> None:
        self.count += 1
        if not self.enabled:
            return
        width = 32
        ratio = min(self.count / self.total, 1.0)
        filled = int(round(width * ratio))
        bar = "#" * filled + "-" * (width - filled)
        elapsed = time.time() - self.start_time
        message = (
            f"\r[{bar}] {self.count:>4}/{self.total:<4} "
            f"{100.0 * ratio:6.2f}% | elapsed {elapsed:6.1f}s | {label[:76]}"
        )
        sys.stdout.write(message)
        sys.stdout.flush()
        if self.count >= self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()


def selected_experiments(name: str) -> List[str]:
    order = ["single", "arrival", "reputation", "deposit", "ablation", "reservation", "lyapunov"]
    if name == "all":
        return order
    return [name]


def planned_runs(experiment: str, reps: int) -> int:
    default_policy_count = len(list(make_default_policies()))
    if experiment == "single":
        return reps * default_policy_count
    if experiment == "arrival":
        return 5 * reps * default_policy_count
    if experiment == "reputation":
        return 4 * reps * default_policy_count
    if experiment == "deposit":
        return 5 * reps * default_policy_count
    if experiment == "ablation":
        return reps * len(list(make_ablation_policies()))
    if experiment == "reservation":
        return 5 * reps
    if experiment == "lyapunov":
        return 5 * reps
    raise ValueError(f"Unknown experiment: {experiment}")


def run_many_with_progress(
    cfg: SimConfig,
    policies: Iterable,
    repetitions: int,
    progress: ProgressBar,
    context: str,
) -> pd.DataFrame:
    rows = []
    policy_list = list(policies)
    for rep in range(repetitions):
        for policy in policy_list:
            result = MecSimulator(cfg).run(policy, seed_offset=1000 * rep + 17)
            row = dict(result.summary)
            row["rep"] = rep
            rows.append(row)
            progress.update(f"{context} | rep {rep + 1}/{repetitions} | {policy.name}")
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    metric_cols = [col for col in METRIC_LABELS if col in df.columns]
    return df.groupby(by, as_index=False)[metric_cols].mean()


def aggregate_std(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    metric_cols = [col for col in METRIC_LABELS if col in df.columns]
    return df.groupby(by, as_index=False)[metric_cols].std(ddof=1).fillna(0.0)


def compute_overall_scores(df: pd.DataFrame) -> pd.DataFrame:
    score = df[["policy"]].copy()
    for metric in PRIMARY_METRICS:
        values = df[metric].astype(float)
        lo = values.min()
        hi = values.max()
        span = max(hi - lo, 1e-12)
        if metric in LOWER_IS_BETTER:
            score[f"{metric}_score"] = (hi - values) / span
        else:
            score[f"{metric}_score"] = (values - lo) / span
    score_cols = [f"{metric}_score" for metric in PRIMARY_METRICS]
    score["overall_score"] = score[score_cols].mean(axis=1)
    score["overall_rank"] = score["overall_score"].rank(method="min", ascending=False).astype(int)
    return score.sort_values(["overall_rank", "policy"]).reset_index(drop=True)
def run_arrival_sweep(base: SimConfig, reps: int, out_dir: Path, progress: ProgressBar) -> pd.DataFrame:
    rates = [0.2, 0.35, 0.5, 0.65, 0.8]
    rows = []
    for rate in rates:
        cfg = base.with_updates(arrival_rate=rate)
        rows.append(run_many_with_progress(cfg, make_default_policies(), reps, progress, f"arrival={rate}"))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out_dir / "arrival_sweep_raw.csv", index=False)
    agg = aggregate(df, ["policy", "arrival_rate"])
    agg.to_csv(out_dir / "arrival_sweep_summary.csv", index=False)
    return agg


def run_reputation_sweep(base: SimConfig, reps: int, out_dir: Path, progress: ProgressBar) -> pd.DataFrame:
    ratios = [0.0, 0.25, 0.5, 0.75]
    rows = []
    for ratio in ratios:
        cfg = base.with_updates(low_reputation_ratio=ratio)
        rows.append(run_many_with_progress(cfg, make_default_policies(), reps, progress, f"low_rep_ratio={ratio}"))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out_dir / "reputation_sweep_raw.csv", index=False)
    agg = aggregate(df, ["policy", "low_reputation_ratio"])
    agg.to_csv(out_dir / "reputation_sweep_summary.csv", index=False)
    return agg


def run_deposit_sweep(base: SimConfig, reps: int, out_dir: Path, progress: ProgressBar) -> pd.DataFrame:
    coeffs = [0.0, 0.2, 0.35, 0.5, 0.7]
    rows = []
    for coeff in coeffs:
        cfg = base.with_updates(deposit_price_coeff=coeff)
        rows.append(run_many_with_progress(cfg, make_default_policies(), reps, progress, f"deposit_coeff={coeff}"))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out_dir / "deposit_sweep_raw.csv", index=False)
    agg = aggregate(df, ["policy", "deposit_price_coeff"])
    agg.to_csv(out_dir / "deposit_sweep_summary.csv", index=False)
    return agg


def run_ablation(base: SimConfig, reps: int, out_dir: Path, progress: ProgressBar) -> pd.DataFrame:
    df = run_many_with_progress(base, make_ablation_policies(), reps, progress, "ablation")
    df.to_csv(out_dir / "ablation_raw.csv", index=False)
    agg = aggregate(df, ["policy"])
    agg.to_csv(out_dir / "ablation_summary.csv", index=False)
    aggregate_std(df, ["policy"]).to_csv(out_dir / "ablation_std.csv", index=False)
    compute_overall_scores(agg).to_csv(out_dir / "ablation_overall_scores.csv", index=False)
    return agg


def run_reservation_sweep(base: SimConfig, reps: int, out_dir: Path, progress: ProgressBar) -> pd.DataFrame:
    fractions = [0.0, 0.15, 0.25, 0.35, 0.5]
    rows = []
    for fraction in fractions:
        cfg = base.with_updates(proposed_reservation_capacity_fraction=fraction)
        rows.append(run_many_with_progress(cfg, [ProposedLyapunovPolicy()], reps, progress, f"reservation_fraction={fraction}"))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out_dir / "reservation_sweep_raw.csv", index=False)
    agg = aggregate(df, ["policy", "proposed_reservation_capacity_fraction"])
    agg.to_csv(out_dir / "reservation_sweep_summary.csv", index=False)
    aggregate_std(df, ["policy", "proposed_reservation_capacity_fraction"]).to_csv(
        out_dir / "reservation_sweep_std.csv", index=False
    )
    return agg


def run_lyapunov_sweep(base: SimConfig, reps: int, out_dir: Path, progress: ProgressBar) -> pd.DataFrame:
    values = [2.0, 4.0, 8.0, 16.0, 32.0]
    rows = []
    for value in values:
        cfg = base.with_updates(lyapunov_v=value)
        rows.append(run_many_with_progress(cfg, [ProposedLyapunovPolicy()], reps, progress, f"lyapunov_v={value}"))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out_dir / "lyapunov_sweep_raw.csv", index=False)
    agg = aggregate(df, ["policy", "lyapunov_v"])
    agg.to_csv(out_dir / "lyapunov_sweep_summary.csv", index=False)
    aggregate_std(df, ["policy", "lyapunov_v"]).to_csv(out_dir / "lyapunov_sweep_std.csv", index=False)
    return agg


def run_single(base: SimConfig, out_dir: Path, reps: int, progress: ProgressBar) -> None:
    rows = []
    traces = []
    for rep in range(reps):
        for policy in make_default_policies():
            result = MecSimulator(base).run(policy, seed_offset=1000 * rep + 17)
            row = dict(result.summary)
            row["rep"] = rep
            rows.append(row)
            tx = result.transactions.copy()
            tx["policy"] = policy.name
            tx["rep"] = rep
            traces.append(tx)
            progress.update(f"single | rep {rep + 1}/{reps} | {policy.name}")
    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "single_summary_raw.csv", index=False)
    summary = aggregate(raw, ["policy"])
    summary.to_csv(out_dir / "single_summary.csv", index=False)
    aggregate_std(raw, ["policy"]).to_csv(out_dir / "single_summary_std.csv", index=False)
    compute_overall_scores(summary).to_csv(out_dir / "default_overall_scores.csv", index=False)
    transactions = pd.concat(traces, ignore_index=True)
    transactions.to_csv(out_dir / "single_transactions.csv", index=False)
    summarize_payment_discipline(transactions).to_csv(
        out_dir / "payment_discipline_summary.csv", index=False
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    time_slots = args.time_slots if args.time_slots is not None else (180 if args.quick else 600)
    reps = args.repetitions if args.repetitions is not None else (1 if args.quick else 3)
    base = SimConfig(time_slots=time_slots, warmup_slots=max(10, time_slots // 12))
    experiments = selected_experiments(args.experiment)
    progress = ProgressBar(sum(planned_runs(exp, reps) for exp in experiments))

    if args.experiment in ("all", "single"):
        run_single(base, out_dir, reps, progress)
    if args.experiment in ("all", "arrival"):
        run_arrival_sweep(base, reps, out_dir, progress)
    if args.experiment in ("all", "reputation"):
        run_reputation_sweep(base, reps, out_dir, progress)
    if args.experiment in ("all", "deposit"):
        run_deposit_sweep(base, reps, out_dir, progress)
    if args.experiment in ("all", "ablation"):
        run_ablation(base, reps, out_dir, progress)
    if args.experiment in ("all", "reservation"):
        run_reservation_sweep(base, reps, out_dir, progress)
    if args.experiment in ("all", "lyapunov"):
        run_lyapunov_sweep(base, reps, out_dir, progress)

    print(f"Finished. Results written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
