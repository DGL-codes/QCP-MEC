# QCP: QoS-Contingent Pricing for Mobile Edge Computing

This repository contains the simulation code, publication-figure scripts, and
reference outputs for **Reputation-Aware Post-Service Pricing for Mobile Edge
Computing: A QoS-Contingent Pricing Framework**.

QCP is a broker-assisted mobile edge computing mechanism. The broker first
announces a quotation and a deposit or credit requirement. The user then
decides whether to participate. After task execution, the final payment is
settled according to realized service quality. If the user defaults after
service, user reputation decreases and future quotation, deposit, and risk
pressure increase.

## What Is Included

- `paper4_sim/`: simulator, system model, policies, and configuration.
- `run_experiments.py`: runs the default, sensitivity, ablation, reservation,
  and Lyapunov experiments, writes CSV files, and shows a text progress bar.
- `plot_publication_figures.py`: regenerates the publication-style figures used
  in the paper. Existing unrelated PNG files are retained unless `--clean-stale`
  is explicitly supplied.
- `verify_outputs.py`: compares generated CSV files with references.
- `audit_consistency.py`: checks policy flags and selected configuration values;
  this is not a proof of theoretical or numerical correctness.
- `reference_results/`: reference CSV summaries and figures generated from the
  current code.
- `docs/IMPLEMENTATION_AUDIT.md`: mapping between paper mechanisms and code.
- `docs/REPRODUCIBILITY.md`: exact metric definitions, parameter mappings,
  figure inputs, and interpretation of the reputation ablation.

## Fairness Notes

- `QCP` denotes the proposed method with reputation-aware quotation, deposit
  reservation, post-service settlement, adaptive lightweight reserved-service
  acceleration, and Lyapunov virtual queues.
- The comparison includes four literature-inspired proxy baselines:
  `DP-MEC`, `Stackelberg-EPRA`, `TARFO`, and `DoubleAuction`. Their pricing
  and selection rules are adapted to the common MEC simulator; they are not
  reproductions of the cited algorithms.
- `Delay-Energy` is a deterministic heuristic without policy training. It
  selects the local or edge option minimizing estimated
  `delay / deadline + 0.25 * energy` and uses fixed pre-service pricing.
- All five baselines use pre-service settlement and therefore do not incur
  post-service payment default loss.
- Common random numbers are used for arrivals, task attributes, and channels so
  policies face the same environmental sample paths in each repetition.

## Naming Compatibility

The paper and figure labels use `Delay-Energy`, short for the Delay-Energy
heuristic. The legacy identifiers `DRL-Offloading` and `drl_offloading` are
retained for code and result-file compatibility. They do not denote a trained
DRL policy.

The `DISPLAY_NAMES` mapping in `plot_publication_figures.py` converts the legacy
CSV policy label to `Delay-Energy` when drawing legends and heatmap labels.
Keep the existing CSV identifiers and numerical values when replotting.

## Setup

Use **Python 3.9–3.12** with the pinned dependencies in `requirements.txt`.
Python **3.12** is recommended for the reviewed release. NumPy 1.26.4 supports
Python 3.9–3.12, so the pinned environment is not intended for Python 3.13+.
The simulations run on a CPU; no GPU, external dataset, or online service is
required. Run all commands from the directory containing this README.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Quick Check

```bash
python audit_consistency.py
python run_experiments.py --quick --experiment all --out results_quick
python plot_publication_figures.py --results results_quick
```

On Windows, the examples below can use `.\.venv\Scripts\python.exe` in place
of `python`; activating PowerShell scripts is not required.

The quick check uses 180 slots, 15 warm-up slots, and one Monte Carlo repetition. It verifies
that the simulator and plotting pipeline run correctly, but its numerical
values are not expected to match the paper tables.

## Full Reproduction

```bash
python audit_consistency.py
python run_experiments.py --experiment all --out results
python plot_publication_figures.py --results results
python verify_outputs.py --results results --reference reference_results
```

The full run performs 363 policy simulations across seven experiment families.
Repetition seeds are 24, 1024, and 2024: the base seed is 7 and the runner adds
`1000 * rep + 17`. Each seed is split into initialization, exogenous-input,
and event generators. Only exogenous arrivals, task attributes, channels, and
initial resources are aligned across policies; policy-dependent event draws
need not be identical.

The full run creates `results/single_transactions.csv`, a large per-task trace
(about 82.5 MiB in the review environment). It is excluded from Git and is not
required by `verify_outputs.py`. A 30-row `payment_discipline_summary.csv`
contains the exact aggregated series used by Figure 6.

`verify_outputs.py` requires every reference CSV to have a matching generated
CSV and uses `atol=rtol=1e-6`. The reviewed bundle has 21 reference CSVs: the
20 original numerical files and the added compact Figure 6 data. Missing or
mismatched files cause a nonzero exit code. The quick-run outputs should not
be compared with the full-run reference results.

To redraw figures from the bundled CSV files without rerunning simulations:

```bash
python plot_publication_figures.py --results reference_results
```

The output directory is `reference_results/figures/`. All eight result PNGs
can be regenerated without the full trace. If a trace is present it is used;
otherwise Figure 6 uses the compact summary. Font availability and Matplotlib
versions can change PNG appearance without changing the plotted numbers.
The conceptual system illustration (paper Figure 1) is not a simulation plot
and is not generated here.

The default, arrival, reputation, and deposit summary files must be present
when calling the publication plotting script. Use `--experiment all`, or use
the complete `reference_results/` directory.

## Main Outputs

- `single_summary.csv`: default performance table.
- `default_overall_scores.csv`: normalized seven-metric overall score.
- `arrival_sweep_summary.csv`: sensitivity to task arrival rate.
- `reputation_sweep_summary.csv`: sensitivity to low-reputation user ratio.
- `deposit_sweep_summary.csv`: sensitivity to deposit coefficient.
- `ablation_summary.csv`: module-level ablation.
- `reservation_sweep_summary.csv`: sensitivity to the reserved-service budget
  fraction.
- `lyapunov_sweep_summary.csv`: sensitivity to the Lyapunov parameter.
- `payment_discipline_summary.csv`: the pooled 40-slot Figure 6 time series.
- `figures/*.png`: publication-style figures used by the paper.

The plotting script intentionally does not generate intermediate diagnostic
figures such as separate reservation or Lyapunov sweeps. Those quantities are
still available in the CSV files, while the paper uses the combined
`reservation_lyapunov_combined.png` figure.

Expected paper figures:

- `default_advantage_heatmap.png`
- `default_advantage_radar.png`
- `ablation_metric_bars.png`
- `arrival_combined.png`
- `reputation_combined.png`
- `payment_discipline_trace.png`
- `deposit_combined.png`
- `reservation_lyapunov_combined.png`

## Default Table

The default setting uses arrival rate `0.4`, `600` slots, `50` warm-up slots,
and `3` Monte Carlo repetitions.

| Scheme | Delay | Energy | SLA | Pay. | User util. | Welfare | Accept. | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| QCP | 0.692 | 0.307 | 0.675 | 0.064 | 0.713 | 0.728 | 0.995 | 0.971 |
| DP-MEC | 0.764 | 0.383 | 0.629 | 0.110 | 0.647 | 0.716 | 0.990 | 0.506 |
| Stackelberg-EPRA | 0.771 | 0.384 | 0.624 | 0.094 | 0.660 | 0.714 | 0.994 | 0.499 |
| TARFO | 0.764 | 0.414 | 0.629 | 0.048 | 0.706 | 0.714 | 0.997 | 0.545 |
| DoubleAuction | 0.758 | 0.406 | 0.634 | 0.336 | 0.423 | 0.719 | 0.971 | 0.189 |
| Delay-Energy | 0.798 | 0.320 | 0.612 | 0.097 | 0.654 | 0.709 | 0.993 | 0.473 |
| NoReputation | 0.701 | 0.297 | 0.665 | 0.068 | 0.707 | 0.724 | 0.994 | 0.904 |

## Interpretation of the Results

The main tables exclude the first 50 slots. Figure 6 includes the initial
transient and pools accepted edge transactions from all repetitions in
40-slot bins. Its historical label "Non-payment ratio" means the fraction
with **positive residual bad debt** (`bad_debt_loss > 1e-12`), not all sampled
default events. Fully deposit-covered defaults are not counted in this series.

`NoReputation` substitutes user reputation 0.75 and server reputation 0.80
where the policy calls `_effective_reps`, including the post-service default
probability in `settle`. The physical server-failure model continues to use
the simulator's evolving server reputation. Thus the payment-discipline plot
compares the specified stochastic mechanisms; it does not independently
establish strategic behavior change under an identical default process.

The default SLA satisfaction is about 0.675, so the observed violation rate
is about 0.325, above the configured budget of 0.08. These finite-horizon runs
demonstrate performance tradeoffs, not empirical satisfaction of that
asymptotic constraint or virtual-queue mean-rate stability. The manuscript's
theoretical conclusions remain conditional on its assumptions.

## Verification Record

The September 29, 2026 review reran all seven full experiment families and
matched all 20 original reference CSVs. The review used Python 3.12.14,
NumPy 2.3.5, pandas 2.2.3, and Matplotlib 3.10.8. This is an additional
verified environment, distinct from the original pinned dependency set.
Installation of that original pinned set was not verified in the restricted
review package source. The simulation configuration, policies, and simulator
are unchanged in the reviewed release; the additions concern reporting,
documentation, argument validation, and comparison ordering.

See `docs/REPRODUCIBILITY.md` for the detailed interpretation and reproduction
mapping. Publication metadata and a reuse license can be added by the authors
when the repository is released; no venue, DOI, or software license is inferred
by this bundle.
