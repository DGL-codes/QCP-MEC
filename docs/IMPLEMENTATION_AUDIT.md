# Implementation Audit

This note maps the paper design to the reproducibility code. It is intended to
make the GitHub release transparent and reproducible.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact metric denominators,
the Figure 6 bad-debt definition, NoReputation's default-probability treatment,
and the independent execution record. `audit_consistency.py` only checks flags
and selected configuration values; it is not a mathematical proof.

## QCP Mechanism

| Paper component | Code location | Implementation status |
|---|---|---|
| Task arrivals and MEC system state | `paper4_sim/simulator.py` | Bernoulli arrivals, sampled task size, CPU cycles, delay tolerance, channel rates, server queues, and reputations. |
| Pre-service quotation | `BrokerPolicy._estimate_price` and `ProposedLyapunovPolicy._estimate_price` | The broker computes an estimated quote before acceptance. QCP applies a proposed price multiplier after the reputation-aware base quote. |
| User participation decision | `BrokerPolicy._accepts` | The user accepts probabilistically according to estimated utility and deposit feasibility. |
| Reputation-aware deposit or credit reservation | `BrokerPolicy._deposit` | Low-reputation users face higher deposit pressure. High-reputation users receive lower deposit requirements. |
| Adaptive reserved-service acceleration | `ProposedLyapunovPolicy._adaptive_reservation_factor` and `MecSimulator._execute_task` | The reserved-service factor reduces only the queueing/execution part of edge delay, is budget constrained, and adds server compensation. |
| Post-service settlement | `BrokerPolicy.settle` | Final payment is adjusted after actual delay, energy, and SLA outcome are observed, then clipped to the announced interval. |
| Payment default and bad-debt loss | `BrokerPolicy.settle` | Default probability depends on base risk, user reputation, and uncovered final payment. Bad debt is measured after deposit recovery. |
| User and server reputation update | `MecSimulator._update_reputations` | User reputation decreases after default and recovers after successful payment. Server reputation follows SLA success. |
| Lyapunov virtual queues | `ProposedLyapunovPolicy._score_candidate` and `observe_transaction` | SLA, risk, and settlement-deviation virtual queues enter the per-task score and are updated after each accepted transaction. |
| Metrics and warm-up removal | `MecSimulator._summarize` | Metrics are aggregated after the warm-up period. |

## Baseline Mapping

The comparison follows the paper's common MEC simulation setting and includes
four literature-inspired pricing and selection schemes, the deterministic
Delay-Energy heuristic, and QCP ablation variants. The table lists the
implemented rules using the manuscript's method names.

| Baseline | High-level idea | Code flags and behavior |
|---|---|---|
| `DP-MEC` | Dynamic pricing reacts to congestion and urgency. | `dynamic_pricing=True`, pre-service settlement, no reputation or deposit. |
| `Stackelberg-EPRA` | Edge server pricing uses cost, load, and user urgency markup. | `stackelberg_pricing=True`, pre-service settlement, no reputation or deposit. |
| `TARFO` | Auction-inspired bid and surplus scoring. | `truthful_auction=True`, load-adjusted server-cost bid plus broker commission, pre-service settlement. |
| `DoubleAuction` | Bid--ask midpoint pricing when bid covers ask, otherwise bid pricing. | `double_auction=True`, negative surplus penalized, pre-service settlement. |
| `Delay-Energy` | Select the local or edge option minimizing estimated `delay / deadline + 0.25 * energy`. | `delay_energy_heuristic=True`, `delay_energy_only=True`, `post_service=False`, `prepaid=True`; fixed pricing, no neural network or policy training. |
| `NoReputation` | QCP using fixed reputation references in the policy. | `ProposedLyapunovPolicy(use_reputation=False)` substitutes user 0.75 and server 0.80, including user reputation in the post-service default probability; the QCP settlement/deposit structure remains. |

The pricing and auction proxies use prescribed quotation and scoring rules.
They do not solve a Stackelberg equilibrium or perform global auction clearing,
and the auction proxies provide no truthfulness guarantee.

`make_default_policies()` uses `score_label="Delay-Energy"` and the
`delay_energy_heuristic` flag. The audit script, CSV policy column, figure
ordering, and legends use the same method name directly.

## Reproducibility Safeguards

- All compared policies use the same exogenous sample paths per repetition
  through common random numbers.
- Pre-service baselines are not assigned post-service bad-debt loss because
  they do not use deferred settlement.
- QCP pays an additional reservation cost for reserved-service acceleration, so
  the delay advantage is not free.
- Reference results are regenerated from the current code and can be checked by
  `python verify_outputs.py --results results --reference reference_results`.
