# Reproducibility and implementation details

This document describes the code's implemented semantics. Equation and figure
numbers refer to the 17-page revised manuscript supplied for the September 29,
2026 review. The original configuration, policy, model, and simulator files
are unchanged in the reviewed release.

## Mapping to the manuscript

| Manuscript component | Implementation |
|---|---|
| Local and edge delay/energy, Eqs. (8)–(18), (67) | `BrokerPolicy._build_candidates`, `MecSimulator._execute_task`; full successful-task CPU demand is added to backlog. |
| Bilateral reputation, Eqs. (19)–(21) | `MecSimulator._update_reputations`; local accepted tasks update user reputation, rejected tasks update neither side. |
| QCP quotation, Eq. (22) | `BrokerPolicy._estimate_price` and `ProposedLyapunovPolicy._estimate_price`; scaling factor `proposed_price_multiplier=0.40`. |
| Deposit, Eqs. (24)–(25) | `BrokerPolicy._deposit`; clipped to the user's credit limit before acceptance. |
| QoS settlement and collection, Eqs. (26)–(30) | `BrokerPolicy.settle`; final payment is clipped to 0.75–1.25 times quotation. |
| Server reward and accounting, Eqs. (31)–(34) | `_server_payment`, `settle`; collected payment already reflects bad debt, which is not deducted a second time. |
| Sequential virtual queues, Eqs. (47)–(50) | `ProposedLyapunovPolicy.observe_transaction`; each accepted local or edge task consumes its relevant per-task budget. |
| Practical selection, Eqs. (55)–(58) | `_score_candidate`, `decide`; estimates conditional on acceptance, plus a normalized deadline-excess term. |
| Reserved service, Eqs. (59)–(60) | `_adaptive_reservation_factor`, `decide`; per-server capacity ledger resets every slot. |
| Seven-metric score, Eq. (68) | `run_experiments.compute_overall_scores`; within-table min–max normalization. |

The reserved-service factor changes the modeled queue/execution delay; it does
not reduce the CPU cycles entered in the physical backlog. This is an abstract
reserved-service model, not an implementation of a hardware scheduler.

## Numerical settings omitted from the compact paper table

The full configuration is in `paper4_sim/config.py`. Some useful mappings are:

| Symbol or role | Configuration | Default |
|---|---|---:|
| Quotation scale, nu | `proposed_price_multiplier` | 0.40 |
| wD in Eq. (56) | `objective_delay_weight` | 1.20 |
| wE | `proposed_energy_weight_multiplier * objective_energy_weight` | 5.00 * 0.25 = 1.25 |
| wT | `proposed_sla_weight_multiplier * objective_sla_weight` | 1.15 * 2.00 = 2.30 |
| wP | `objective_settlement_weight` | 0.60 |
| wR | `objective_risk_weight` | 1.35 |
| wB | `proposed_utility_weight_multiplier * objective_utility_weight` | 0.25 * 0.40 = 0.10 |
| Nominal remuneration factor | `proposed_reservation_delay_factor` | 0.58 |
| Reservation coefficient | `proposed_reservation_cost_per_gcycle` | 0.010 |
| Reservation budget fraction | `proposed_reservation_capacity_fraction` | 0.35 |
| DPP control V | `lyapunov_v` | 8.0 |
| Per-accepted-task SLA/risk/deviation budgets | `sla_violation_budget`, `risk_budget_per_task`, `settlement_budget_per_task` | 0.08 / 0.015 / 0.020 |

The target reservation factor uses pressure weights 0.35 (SLA backlog), 0.25
(normalized **deadline slack**, despite the variable name `urgency`), 0.20
(queue load), and 0.20 (bilateral reliability). Its target range is 0.50–0.92;
the remaining-budget adjustment can raise it to 1.00. The fixed nominal factor
0.58 is used only for server remuneration. `NoReservation` retains that
remuneration, as the manuscript's Table IV note specifies.

## Reference experiments and seeds

The full command uses 600 slots, 50 warm-up slots, and three repetitions.
The quick command uses 180 slots, 15 warm-up slots, and one repetition.
The seed in repetition r (zero-based) is `7 + 1000*r + 17`.

| Family | Values | Policy simulations in full run |
|---|---|---:|
| Default | arrival=0.4; seven policies | 21 |
| Arrival | 0.2, 0.35, 0.5, 0.65, 0.8 | 105 |
| Initial low-reputation fraction | 0, 0.25, 0.5, 0.75 | 84 |
| Deposit price coefficient | 0, 0.2, 0.35, 0.5, 0.7 | 105 |
| Ablation | six variants | 18 |
| Reserved capacity | 0, 0.15, 0.25, 0.35, 0.5 | 15 |
| Lyapunov V | 2, 4, 8, 16, 32 | 15 |
| Total | | 363 |

Policy-dependent event generators are not synchronized after every branch.
The common-random-number claim covers initial resources and exogenous task
arrivals, task attributes, and channel samples. Changing initial population
parameters can also change later initialization draws; cross-parameter sweeps
should not be interpreted as controlled single-user causal experiments.

## Metric denominators

| Metric | Denominator / interpretation |
|---|---|
| Delay, energy, payment, utility, SLA satisfaction | Accepted local and edge tasks after warm-up. |
| Acceptance | All arrivals after warm-up. |
| Broker utility and bad debt | Sum divided by the number of accepted local and edge tasks. |
| Reported settlement deviation | Accepted edge tasks only. The virtual-queue budget still applies to all accepted tasks. |
| Nonnegative utility ratios | Accepted tasks; local tasks have zero broker utility. These are realized ratios, not proofs of conditional expected guarantees. |
| Social welfare | Average user utility + average broker utility, as explicitly defined in the manuscript. |
| `completed` in trace CSV | A settled execution attempt, including an edge execution failure. It is not an execution-success flag; SLA also checks success. |

The CSV identifier `DRL-Offloading` denotes the deterministic Delay-Energy
heuristic. It is retained for historical result compatibility. No neural model
is trained. The four pricing/auction proxies do not reproduce the original
cited algorithms or inherit their equilibrium/truthfulness properties.

## Figure 6: payment-discipline data

The plot selects accepted edge transactions (`est_price > 1e-12`) from QCP
and NoReputation, pools the repetitions, and bins time in 40-slot intervals.
Unlike the summary tables it includes warm-up, because it displays recovery
from the initial state. Reputation is measured after the transaction update.

The plotted `nonpayment` statistic is `mean(bad_debt_loss > 1e-12)`.
It measures **positive residual bad-debt incidence**. A sampled default event
can leave zero bad debt if the deposit covers the final payment. Such an event
still lowers the user's reputation but is absent from this plotted statistic.
The inherited panel title "Payment default" must be read with this definition.
The compact `payment_discipline_summary.csv` preserves the original plotted
series; it does not add a different default metric or modify the figure data.

In NoReputation, `_effective_reps` returns user reputation 0.75 and server
reputation 0.80. These constants affect policy estimates, quotation, deposit,
reservation, and also the realized default probability in `BrokerPolicy.settle`.
The physical server-success/delay model still uses evolving server reputation.
The ablation therefore is not a comparison of two selectors under an identical
payment-default process. Figure 6 illustrates the specified stochastic
feedback models; it is not independent evidence that users strategically
change their payment behavior. No behavioral learning or strategic payment
optimization is simulated.

## Figure and table inputs

| Manuscript output | Source files |
|---|---|
| Table III | `single_summary.csv`, `default_overall_scores.csv` |
| Table IV | `ablation_summary.csv`, `ablation_overall_scores.csv` |
| Fig. 2 | `single_summary.csv`; heatmap and radar PNGs |
| Fig. 3 | `ablation_summary.csv` |
| Fig. 4 | `arrival_sweep_summary.csv` |
| Fig. 5 | `reputation_sweep_summary.csv` |
| Fig. 6 | `single_transactions.csv`, or the compact `payment_discipline_summary.csv` |
| Fig. 7 | `deposit_sweep_summary.csv` |
| Fig. 8 | `reservation_sweep_summary.csv`, `lyapunov_sweep_summary.csv` |

The default score uses seven metrics; the heatmap/radar present six dimensions
and omit welfare. The inherited radar lists six policies and omits Delay-Energy;
the heatmap and default table include all seven. Table-specific scores should not be compared across the
default and ablation tables. Error bars are not claimed by this plotting code;
raw repetition-level results and the supplied standard-deviation files permit
further statistical analysis.

## Limits of the verification

All 20 original CSVs matched a fresh run at `atol=rtol=1e-6` in the reviewed
environment; the largest numeric absolute difference was approximately
4.55e-13. Functional checks covered common exogenous samples, reservation
limits, deposits, settlement intervals, accounting identities, prepaid losses,
finite outputs, reputation bounds, and virtual-queue recurrences across the
11 distinct default/ablation policies in a separate high-load run.

The default QCP SLA satisfaction is 0.675125, hence the realized violation
rate is 0.324875 rather than at most 0.08. As stated in the manuscript, the
finite-horizon experiments do not establish mean-rate stability or empirical
satisfaction of the asymptotic SLA budget. Verification of files and accounting
does not prove the analytical approximation-gap assumption or optimality.

The original pinned dependency installation could not be exercised using the
review environment's available package source. All reported executions used
Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, and Matplotlib 3.10.8. They required
no external data or network requests. The existing pins remain unchanged.
