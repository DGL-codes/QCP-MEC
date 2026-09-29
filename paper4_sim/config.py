from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple


@dataclass(frozen=True)
class SimConfig:
    """Default parameters use seconds, bits, cycles, Joules, and abstract money units."""

    seed: int = 7
    num_users: int = 50
    num_servers: int = 5
    time_slots: int = 600
    warmup_slots: int = 50
    arrival_rate: float = 0.4
    slot_duration: float = 1.0

    data_size_mb_range: Tuple[float, float] = (0.5, 10.0)
    cpu_cycles_range: Tuple[float, float] = (0.2e9, 2.0e9)
    tau_ms_range: Tuple[float, float] = (150.0, 1800.0)
    priority_levels: int = 3

    local_cpu_hz_range: Tuple[float, float] = (0.5e9, 1.5e9)
    server_cpu_hz_range: Tuple[float, float] = (5.0e9, 20.0e9)
    user_tx_power_w_range: Tuple[float, float] = (0.2, 1.0)
    kappa_range: Tuple[float, float] = (0.8e-27, 2.5e-27)

    bandwidth_hz: float = 10.0e6
    noise_power_w: float = 1.0e-10
    distance_m_range: Tuple[float, float] = (40.0, 350.0)
    rate_error_sigma: float = 0.10
    actual_delay_jitter_sigma: float = 0.08

    initial_user_rep_range: Tuple[float, float] = (0.5, 0.9)
    initial_server_rep_range: Tuple[float, float] = (0.6, 0.95)
    low_reputation_ratio: float = 0.20
    low_reputation_range: Tuple[float, float] = (0.1, 0.45)
    user_rep_eta: float = 0.08
    server_rep_eta: float = 0.06

    price_delay: float = 0.15
    price_energy: float = 0.08
    price_resource: float = 0.055
    price_risk: float = 0.080
    price_server_discount: float = 0.025
    fixed_price_per_gcycle: float = 0.070
    fixed_price_per_mb: float = 0.006
    broker_commission_rate: float = 0.18
    server_cost_per_gcycle: float = 0.040
    server_load_cost: float = 0.015

    settlement_alpha_delay: float = 0.18
    settlement_alpha_energy: float = 0.10
    settlement_sla_penalty_rate: float = 0.22
    settlement_lower_mult: float = 0.75
    settlement_upper_mult: float = 1.25

    min_deposit: float = 0.015
    deposit_price_coeff: float = 0.45
    deposit_risk_coeff: float = 0.24
    high_rep_threshold: float = 0.72
    credit_limit_range: Tuple[float, float] = (0.25, 1.2)

    default_base_prob: float = 0.020
    default_rep_sensitivity: float = 0.28
    default_uncovered_sensitivity: float = 0.45
    server_failure_base_prob: float = 0.010
    server_failure_rep_sensitivity: float = 0.12

    value_base: float = 0.55
    value_priority_step: float = 0.20
    value_noise_sigma: float = 0.08
    user_delay_cost: float = 0.24
    user_energy_cost: float = 0.05
    acceptance_temperature: float = 0.10

    objective_delay_weight: float = 1.20
    objective_energy_weight: float = 0.25
    objective_settlement_weight: float = 0.60
    objective_risk_weight: float = 1.35
    objective_utility_weight: float = 0.40
    objective_sla_weight: float = 2.00
    objective_reputation_weight: float = 0.35

    proposed_price_multiplier: float = 0.40
    proposed_energy_weight_multiplier: float = 5.00
    proposed_utility_weight_multiplier: float = 0.25
    proposed_sla_weight_multiplier: float = 1.15
    proposed_reservation_delay_factor: float = 0.58
    proposed_reservation_min_factor: float = 0.50
    proposed_reservation_max_factor: float = 0.92
    proposed_reservation_capacity_fraction: float = 0.35
    proposed_reservation_cost_per_gcycle: float = 0.010

    lyapunov_v: float = 8.0
    sla_violation_budget: float = 0.08
    risk_budget_per_task: float = 0.015
    settlement_budget_per_task: float = 0.020

    max_server_queue_s: float = 3.0

    def with_updates(self, **kwargs: object) -> "SimConfig":
        return replace(self, **kwargs)
