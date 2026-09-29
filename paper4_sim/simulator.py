from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .config import SimConfig
from .models import LOCAL_SERVER, ServerState, SlotContext, Task, Transaction, UserState
from .policies import Policy


@dataclass
class SimulationResult:
    policy_name: str
    config: SimConfig
    transactions: pd.DataFrame
    summary: Dict[str, float]


class MecSimulator:
    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg

    def run(self, policy: Policy, seed_offset: int = 0) -> SimulationResult:
        cfg = self.cfg
        seed_seq = np.random.SeedSequence(cfg.seed + seed_offset)
        init_rng, exogenous_rng, event_rng = [
            np.random.default_rng(child) for child in seed_seq.spawn(3)
        ]
        user_state = self._init_users(init_rng)
        server_state = self._init_servers(init_rng)
        distances = init_rng.uniform(
            cfg.distance_m_range[0], cfg.distance_m_range[1], size=(cfg.num_users, cfg.num_servers)
        )
        policy.reset(cfg, event_rng)
        records: List[Transaction] = []

        for slot in range(cfg.time_slots):
            self._drain_server_backlogs(server_state)
            actual_rates = self._sample_rates(exogenous_rng, user_state, distances)
            est_rates = actual_rates * exogenous_rng.lognormal(
                mean=0.0, sigma=cfg.rate_error_sigma, size=actual_rates.shape
            )
            ctx = SlotContext(
                slot=slot,
                user_state=user_state,
                server_state=server_state,
                est_rates_bps=est_rates,
                actual_rates_bps=actual_rates,
                rng=event_rng,
            )
            arrivals = exogenous_rng.random(cfg.num_users) < cfg.arrival_rate
            for user in np.flatnonzero(arrivals):
                task = self._sample_task(exogenous_rng, slot, int(user))
                decision = policy.decide(task, ctx)
                if not decision.accepted:
                    tx = self._rejected_transaction(policy.name, task, decision, user_state, server_state)
                    records.append(tx)
                    policy.observe_transaction(tx)
                    continue

                actual_delay, actual_energy, server_success = self._execute_task(task, decision.candidate, ctx)
                sla_met = bool(server_success and actual_delay <= task.tau_s)
                settlement = policy.settle(
                    task, decision, ctx, actual_delay, actual_energy, sla_met, server_success
                )
                if decision.candidate.server != LOCAL_SERVER and server_success:
                    server_state.backlog_cycles[decision.candidate.server] += task.cycles
                self._update_reputations(task, decision.candidate.server, user_state, server_state, settlement, sla_met)
                tx = self._completed_transaction(
                    policy.name,
                    task,
                    decision,
                    user_state,
                    server_state,
                    actual_delay,
                    actual_energy,
                    sla_met,
                    settlement,
                )
                records.append(tx)
                policy.observe_transaction(tx)

        df = pd.DataFrame([tx.__dict__ for tx in records])
        if len(df) == 0:
            summary = {}
        else:
            summary = self._summarize(df, policy.name)
        return SimulationResult(policy_name=policy.name, config=cfg, transactions=df, summary=summary)

    def _init_users(self, rng: np.random.Generator) -> UserState:
        cfg = self.cfg
        rep = rng.uniform(cfg.initial_user_rep_range[0], cfg.initial_user_rep_range[1], cfg.num_users)
        low_count = int(round(cfg.low_reputation_ratio * cfg.num_users))
        if low_count > 0:
            low_idx = rng.choice(cfg.num_users, size=low_count, replace=False)
            rep[low_idx] = rng.uniform(cfg.low_reputation_range[0], cfg.low_reputation_range[1], low_count)
        return UserState(
            local_cpu_hz=rng.uniform(cfg.local_cpu_hz_range[0], cfg.local_cpu_hz_range[1], cfg.num_users),
            tx_power_w=rng.uniform(cfg.user_tx_power_w_range[0], cfg.user_tx_power_w_range[1], cfg.num_users),
            kappa=rng.uniform(cfg.kappa_range[0], cfg.kappa_range[1], cfg.num_users),
            credit_limit=rng.uniform(cfg.credit_limit_range[0], cfg.credit_limit_range[1], cfg.num_users),
            reputation=rep,
        )

    def _init_servers(self, rng: np.random.Generator) -> ServerState:
        cfg = self.cfg
        return ServerState(
            cpu_hz=rng.uniform(cfg.server_cpu_hz_range[0], cfg.server_cpu_hz_range[1], cfg.num_servers),
            reputation=rng.uniform(cfg.initial_server_rep_range[0], cfg.initial_server_rep_range[1], cfg.num_servers),
            backlog_cycles=np.zeros(cfg.num_servers, dtype=float),
            cost_per_gcycle=rng.uniform(0.85, 1.15, cfg.num_servers) * cfg.server_cost_per_gcycle,
        )

    def _drain_server_backlogs(self, server_state: ServerState) -> None:
        processed = server_state.cpu_hz * self.cfg.slot_duration
        server_state.backlog_cycles = np.maximum(0.0, server_state.backlog_cycles - processed)

    def _sample_rates(
        self, rng: np.random.Generator, user_state: UserState, distances: np.ndarray
    ) -> np.ndarray:
        cfg = self.cfg
        fading = rng.exponential(scale=1.0, size=distances.shape)
        path_gain = 2.0e-7 * np.power(np.maximum(distances, 1.0) / 100.0, -3.5)
        h = path_gain * fading
        snr = user_state.tx_power_w[:, None] * h / cfg.noise_power_w
        rates = cfg.bandwidth_hz * np.log2(1.0 + np.maximum(snr, 1e-9))
        return np.clip(rates, 1.0e6, 2.5e8)

    def _sample_task(self, rng: np.random.Generator, slot: int, user: int) -> Task:
        cfg = self.cfg
        data_mb = float(rng.uniform(cfg.data_size_mb_range[0], cfg.data_size_mb_range[1]))
        cycles = float(rng.uniform(cfg.cpu_cycles_range[0], cfg.cpu_cycles_range[1]))
        tau_s = float(rng.uniform(cfg.tau_ms_range[0], cfg.tau_ms_range[1]) / 1000.0)
        priority = int(rng.integers(1, cfg.priority_levels + 1))
        value = (
            cfg.value_base
            + cfg.value_priority_step * priority
            + 0.18 * max(0.0, 0.5 - tau_s)
            + float(rng.normal(0.0, cfg.value_noise_sigma))
        )
        return Task(
            slot=slot,
            user=user,
            data_mb=data_mb,
            data_bits=data_mb * 8.0e6,
            cycles=cycles,
            tau_s=tau_s,
            priority=priority,
            value=max(0.05, value),
        )

    def _execute_task(self, task: Task, cand, ctx: SlotContext) -> tuple:
        cfg = self.cfg
        server = cand.server
        if server == LOCAL_SERVER:
            delay = task.cycles / ctx.user_state.local_cpu_hz[task.user]
            energy = ctx.user_state.kappa[task.user] * (ctx.user_state.local_cpu_hz[task.user] ** 2) * task.cycles
            jitter = ctx.rng.lognormal(mean=0.0, sigma=cfg.actual_delay_jitter_sigma)
            return float(delay * jitter), float(energy), True

        rate = max(float(ctx.actual_rates_bps[task.user, server]), 1.0)
        tx_delay = task.data_bits / rate
        queue_delay = min(
            float(ctx.server_state.backlog_cycles[server] / ctx.server_state.cpu_hz[server]),
            cfg.max_server_queue_s,
        )
        exec_delay = task.cycles / ctx.server_state.cpu_hz[server]
        psi = float(ctx.server_state.reputation[server])
        failure_prob = cfg.server_failure_base_prob + cfg.server_failure_rep_sensitivity * (1.0 - psi)
        server_success = bool(ctx.rng.random() >= failure_prob)
        instability = 1.0 + 0.55 * (1.0 - psi)
        jitter = ctx.rng.lognormal(mean=0.0, sigma=cfg.actual_delay_jitter_sigma)
        if not server_success:
            instability += 1.0 + 1.5 * ctx.rng.random()
        reservation_factor = float(getattr(cand, "reservation_delay_factor", 1.0))
        reserved_edge_delay = reservation_factor * (queue_delay + exec_delay)
        delay = (tx_delay + reserved_edge_delay) * instability * jitter
        energy = ctx.user_state.tx_power_w[task.user] * tx_delay
        return float(delay), float(energy), server_success

    def _update_reputations(
        self,
        task: Task,
        server: int,
        user_state: UserState,
        server_state: ServerState,
        settlement: Dict[str, float],
        sla_met: bool,
    ) -> None:
        defaulted = bool(settlement.get("defaulted", 0.0))
        user_score = 0.0 if defaulted else 1.0
        u = task.user
        user_state.reputation[u] = np.clip(
            (1.0 - self.cfg.user_rep_eta) * user_state.reputation[u] + self.cfg.user_rep_eta * user_score,
            0.0,
            1.0,
        )
        if server != LOCAL_SERVER:
            server_score = 1.0 if sla_met else 0.0
            server_state.reputation[server] = np.clip(
                (1.0 - self.cfg.server_rep_eta) * server_state.reputation[server]
                + self.cfg.server_rep_eta * server_score,
                0.0,
                1.0,
            )

    def _rejected_transaction(
        self, policy_name: str, task: Task, decision, user_state: UserState, server_state: ServerState
    ) -> Transaction:
        cand = decision.candidate
        server_rep = 1.0 if cand.server == LOCAL_SERVER else float(server_state.reputation[cand.server])
        return Transaction(
            policy=policy_name,
            slot=task.slot,
            user=task.user,
            server=cand.server,
            arrived=1,
            accepted=0,
            completed=0,
            delay=0.0,
            energy=0.0,
            sla_met=0,
            est_price=cand.est_price,
            final_price=0.0,
            deposit=cand.deposit,
            collected_payment=0.0,
            server_payment=0.0,
            broker_utility=0.0,
            bad_debt_loss=0.0,
            settlement_deviation=0.0,
            user_reputation=float(user_state.reputation[task.user]),
            server_reputation=server_rep,
            user_utility=0.0,
            individual_rational=0,
            settlement_bounded=1,
            risk_bounded=1,
            budget_balanced=1,
            metadata={"acceptance_prob": decision.acceptance_prob},
        )

    def _completed_transaction(
        self,
        policy_name: str,
        task: Task,
        decision,
        user_state: UserState,
        server_state: ServerState,
        actual_delay: float,
        actual_energy: float,
        sla_met: bool,
        settlement: Dict[str, float],
    ) -> Transaction:
        cand = decision.candidate
        server_rep = 1.0 if cand.server == LOCAL_SERVER else float(server_state.reputation[cand.server])
        user_utility = (
            task.value
            - settlement["final_price"]
            - self.cfg.user_delay_cost * actual_delay
            - self.cfg.user_energy_cost * actual_energy
        )
        bound_slack = 1e-8
        settlement_bounded = cand.server == LOCAL_SERVER or (
            settlement["final_price"] >= cand.lower_price - bound_slack
            and settlement["final_price"] <= cand.upper_price + bound_slack
        )
        risk_upper = max(0.0, settlement["final_price"] - cand.deposit)
        risk_bounded = settlement["bad_debt_loss"] <= risk_upper + bound_slack
        return Transaction(
            policy=policy_name,
            slot=task.slot,
            user=task.user,
            server=cand.server,
            arrived=1,
            accepted=1,
            completed=1,
            delay=actual_delay,
            energy=actual_energy,
            sla_met=int(sla_met),
            est_price=cand.est_price,
            final_price=settlement["final_price"],
            deposit=cand.deposit,
            collected_payment=settlement["collected_payment"],
            server_payment=settlement["server_payment"],
            broker_utility=settlement["broker_utility"],
            bad_debt_loss=settlement["bad_debt_loss"],
            settlement_deviation=settlement["settlement_deviation"],
            user_reputation=float(user_state.reputation[task.user]),
            server_reputation=server_rep,
            user_utility=float(user_utility),
            individual_rational=int(user_utility >= -1e-8),
            settlement_bounded=int(settlement_bounded),
            risk_bounded=int(risk_bounded),
            budget_balanced=int(settlement["broker_utility"] >= -1e-8),
            metadata={
                "acceptance_prob": decision.acceptance_prob,
                "task_value": task.value,
                "tau_s": task.tau_s,
            },
        )

    def _summarize(self, df: pd.DataFrame, policy_name: str) -> Dict[str, float]:
        cfg = self.cfg
        eval_df = df[df["slot"] >= cfg.warmup_slots]
        completed = eval_df[eval_df["completed"] == 1]
        offloaded = completed[completed["server"] != LOCAL_SERVER]
        total_arrivals = max(float(eval_df["arrived"].sum()), 1.0)
        accepted_count = float(eval_df["accepted"].sum())
        def mean_or_zero(frame: pd.DataFrame, col: str) -> float:
            return float(frame[col].mean()) if len(frame) else 0.0

        broker_utility_sum = float(completed["broker_utility"].sum()) if len(completed) else 0.0
        summary = {
            "policy": policy_name,
            "arrival_rate": cfg.arrival_rate,
            "low_reputation_ratio": cfg.low_reputation_ratio,
            "deposit_price_coeff": cfg.deposit_price_coeff,
            "proposed_reservation_capacity_fraction": cfg.proposed_reservation_capacity_fraction,
            "lyapunov_v": cfg.lyapunov_v,
            "tasks_arrived": total_arrivals,
            "task_acceptance_ratio": accepted_count / total_arrivals,
            "average_delay": mean_or_zero(completed, "delay"),
            "average_energy": mean_or_zero(completed, "energy"),
            "sla_satisfaction_ratio": mean_or_zero(completed, "sla_met"),
            "average_user_payment": mean_or_zero(completed, "final_price"),
            "average_collected_payment": mean_or_zero(completed, "collected_payment"),
            "average_user_utility": mean_or_zero(completed, "user_utility"),
            "broker_utility": broker_utility_sum / max(accepted_count, 1.0),
            "bad_debt_loss": float(completed["bad_debt_loss"].sum()) / max(accepted_count, 1.0),
            "settlement_deviation": mean_or_zero(offloaded, "settlement_deviation"),
            "offloading_ratio": float((completed["server"] != LOCAL_SERVER).mean()) if len(completed) else 0.0,
            "individual_rational_ratio": mean_or_zero(completed, "individual_rational"),
            "budget_balance_ratio": mean_or_zero(completed, "budget_balanced"),
            "risk_bounded_ratio": mean_or_zero(completed, "risk_bounded"),
            "settlement_bounded_ratio": mean_or_zero(completed, "settlement_bounded"),
            "cumulative_broker_utility": broker_utility_sum,
        }
        summary["average_social_welfare"] = summary["average_user_utility"] + summary["broker_utility"]
        return summary


def run_many(cfg: SimConfig, policies: Iterable[Policy], repetitions: int = 3) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for rep in range(repetitions):
        for policy in policies:
            sim = MecSimulator(cfg)
            result = sim.run(policy, seed_offset=1000 * rep + 17)
            row = dict(result.summary)
            row["rep"] = rep
            rows.append(row)
    return pd.DataFrame(rows)
