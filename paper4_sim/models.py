from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


LOCAL_SERVER = -1


@dataclass
class Task:
    slot: int
    user: int
    data_mb: float
    data_bits: float
    cycles: float
    tau_s: float
    priority: int
    value: float


@dataclass
class Candidate:
    server: int
    est_delay: float
    est_energy: float
    est_price: float
    deposit: float
    expected_risk: float
    expected_settlement_dev: float
    expected_broker_utility: float
    lower_price: float
    upper_price: float
    score: float
    reservation_delay_factor: float = 1.0


@dataclass
class Decision:
    policy_name: str
    candidate: Candidate
    accepted: bool = False
    acceptance_prob: float = 0.0


@dataclass
class Transaction:
    policy: str
    slot: int
    user: int
    server: int
    arrived: int
    accepted: int
    completed: int
    delay: float
    energy: float
    sla_met: int
    est_price: float
    final_price: float
    deposit: float
    collected_payment: float
    server_payment: float
    broker_utility: float
    bad_debt_loss: float
    settlement_deviation: float
    user_reputation: float
    server_reputation: float
    user_utility: float
    individual_rational: int
    settlement_bounded: int
    risk_bounded: int
    budget_balanced: int
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class UserState:
    local_cpu_hz: np.ndarray
    tx_power_w: np.ndarray
    kappa: np.ndarray
    credit_limit: np.ndarray
    reputation: np.ndarray


@dataclass
class ServerState:
    cpu_hz: np.ndarray
    reputation: np.ndarray
    backlog_cycles: np.ndarray
    cost_per_gcycle: np.ndarray


@dataclass
class SlotContext:
    slot: int
    user_state: UserState
    server_state: ServerState
    est_rates_bps: np.ndarray
    actual_rates_bps: np.ndarray
    rng: np.random.Generator
