from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional

import numpy as np

from .config import SimConfig
from .models import Candidate, Decision, LOCAL_SERVER, SlotContext, Task, Transaction


def sigmoid(x: float) -> float:
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))


class Policy(ABC):
    name = "base"

    def reset(self, cfg: SimConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng

    @abstractmethod
    def decide(self, task: Task, ctx: SlotContext) -> Decision:
        raise NotImplementedError

    @abstractmethod
    def settle(
        self,
        task: Task,
        decision: Decision,
        ctx: SlotContext,
        actual_delay: float,
        actual_energy: float,
        sla_met: bool,
        server_success: bool,
    ) -> Dict[str, float]:
        raise NotImplementedError

    def observe_transaction(self, tx: Transaction) -> None:
        return None


class BrokerPolicy(Policy):
    name = "broker"

    def __init__(
        self,
        *,
        use_reputation: bool = True,
        use_deposit: bool = True,
        post_service: bool = True,
        fixed_pricing: bool = False,
        dynamic_pricing: bool = False,
        stackelberg_pricing: bool = False,
        truthful_auction: bool = False,
        double_auction: bool = False,
        drl_offloading: bool = False,
        delay_energy_only: bool = False,
        prepaid: bool = False,
        score_label: Optional[str] = None,
    ) -> None:
        self.use_reputation = use_reputation
        self.use_deposit = use_deposit
        self.post_service = post_service
        self.fixed_pricing = fixed_pricing
        self.dynamic_pricing = dynamic_pricing
        self.stackelberg_pricing = stackelberg_pricing
        self.truthful_auction = truthful_auction
        self.double_auction = double_auction
        self.drl_offloading = drl_offloading
        self.delay_energy_only = delay_energy_only
        self.prepaid = prepaid
        if score_label:
            self.name = score_label

    def reset(self, cfg: SimConfig, rng: np.random.Generator) -> None:
        super().reset(cfg, rng)

    def decide(self, task: Task, ctx: SlotContext) -> Decision:
        candidates = self._build_candidates(task, ctx)
        best = min(candidates, key=lambda item: item.score)
        accepted, prob = self._accepts(task, best, ctx)
        return Decision(policy_name=self.name, candidate=best, accepted=accepted, acceptance_prob=prob)

    def _effective_reps(self, task: Task, server: int, ctx: SlotContext) -> tuple:
        if self.use_reputation:
            rho = float(ctx.user_state.reputation[task.user])
            psi = 1.0 if server == LOCAL_SERVER else float(ctx.server_state.reputation[server])
        else:
            rho = 0.75
            psi = 0.80
        return rho, psi

    def _build_candidates(self, task: Task, ctx: SlotContext) -> List[Candidate]:
        cfg = self.cfg
        candidates: List[Candidate] = []
        local_delay = task.cycles / ctx.user_state.local_cpu_hz[task.user]
        local_energy = ctx.user_state.kappa[task.user] * (ctx.user_state.local_cpu_hz[task.user] ** 2) * task.cycles
        local = Candidate(
            server=LOCAL_SERVER,
            est_delay=float(local_delay),
            est_energy=float(local_energy),
            est_price=0.0,
            deposit=0.0,
            expected_risk=0.0,
            expected_settlement_dev=0.0,
            expected_broker_utility=0.0,
            lower_price=0.0,
            upper_price=0.0,
            score=self._score_candidate(
                task,
                LOCAL_SERVER,
                float(local_delay),
                float(local_energy),
                0.0,
                0.0,
                0.0,
                0.0,
                ctx,
            ),
        )
        candidates.append(local)

        for s in range(cfg.num_servers):
            rate = max(float(ctx.est_rates_bps[task.user, s]), 1.0)
            tx_delay = task.data_bits / rate
            queue_delay = ctx.server_state.backlog_cycles[s] / ctx.server_state.cpu_hz[s]
            queue_delay = min(float(queue_delay), cfg.max_server_queue_s)
            exec_delay = task.cycles / ctx.server_state.cpu_hz[s]
            est_delay = float(tx_delay + queue_delay + exec_delay)
            if self.use_reputation:
                _, psi = self._effective_reps(task, s, ctx)
                est_delay *= 1.0 + 0.45 * (1.0 - psi)
            est_energy = float(ctx.user_state.tx_power_w[task.user] * tx_delay)
            est_price = self._estimate_price(task, s, est_delay, est_energy, ctx)
            deposit = self._deposit(task, s, est_price, ctx)
            expected_risk = self._expected_risk(task, s, est_price, deposit, ctx)
            expected_settlement_dev = self._expected_settlement_dev(task, s, est_price, ctx)
            server_payment = self._server_payment(task, s, ctx, estimated=True)
            expected_utility = est_price - server_payment - expected_risk
            lower = cfg.settlement_lower_mult * est_price
            upper = cfg.settlement_upper_mult * est_price
            score = self._score_candidate(
                task,
                s,
                est_delay,
                est_energy,
                est_price,
                expected_risk,
                expected_settlement_dev,
                expected_utility,
                ctx,
            )
            candidates.append(
                Candidate(
                    server=s,
                    est_delay=est_delay,
                    est_energy=est_energy,
                    est_price=est_price,
                    deposit=deposit,
                    expected_risk=expected_risk,
                    expected_settlement_dev=expected_settlement_dev,
                    expected_broker_utility=expected_utility,
                    lower_price=lower,
                    upper_price=upper,
                    score=score,
                )
            )
        return candidates

    def _estimate_price(
        self, task: Task, server: int, est_delay: float, est_energy: float, ctx: SlotContext
    ) -> float:
        cfg = self.cfg
        rho, psi = self._effective_reps(task, server, ctx)
        data_component = cfg.fixed_price_per_mb * task.data_mb
        resource_component = cfg.price_resource * (task.cycles / 1.0e9)
        if self.fixed_pricing or self.drl_offloading:
            return max(0.001, cfg.fixed_price_per_gcycle * (task.cycles / 1.0e9) + data_component)
        load_factor = ctx.server_state.backlog_cycles[server] / max(ctx.server_state.cpu_hz[server], 1.0)
        normalized_load = min(load_factor / 1.5, 1.0)
        if self.dynamic_pricing:
            congestion_markup = 1.0 + 0.75 * normalized_load
            urgency_markup = 1.0 + 0.20 * max(0.0, est_delay - task.tau_s) / max(task.tau_s, 1e-9)
            return max(0.001, (resource_component + data_component + 0.04 * est_delay) * congestion_markup * urgency_markup)
        if self.stackelberg_pricing:
            # A compact Stackelberg-style proxy: edge servers set a revenue-maximizing
            # markup that increases with load and user urgency, while users still choose
            # the option with acceptable utility.
            marginal_cost = ctx.server_state.cost_per_gcycle[server] * (task.cycles / 1.0e9) + data_component
            urgency = max(0.0, task.value - cfg.user_delay_cost * est_delay - cfg.user_energy_cost * est_energy)
            markup = 0.35 + 0.45 * normalized_load + 0.18 * min(urgency, 1.0)
            return max(0.001, marginal_cost * (1.0 + markup))
        if self.truthful_auction:
            load_factor = ctx.server_state.backlog_cycles[server] / max(ctx.server_state.cpu_hz[server], 1.0)
            bid = (
                ctx.server_state.cost_per_gcycle[server]
                * (task.cycles / 1.0e9)
                * (1.0 + 0.25 * load_factor)
                * (1.0 + 0.15 * (1.0 - psi))
            )
            return max(0.001, bid * (1.0 + cfg.broker_commission_rate))
        if self.double_auction:
            ask = self._server_payment(task, server, ctx, estimated=True)
            user_bid = max(0.001, task.value - cfg.user_delay_cost * est_delay - cfg.user_energy_cost * est_energy)
            if user_bid < ask:
                return user_bid
            return max(0.001, 0.5 * (ask + user_bid))
        quote = (
            cfg.price_delay * est_delay
            + cfg.price_energy * est_energy
            + resource_component
            + data_component
            + cfg.price_risk * (1.0 - rho)
            - cfg.price_server_discount * psi
        )
        return max(0.001, float(quote))

    def _deposit(self, task: Task, server: int, est_price: float, ctx: SlotContext) -> float:
        if server == LOCAL_SERVER or not self.use_deposit:
            return 0.0
        cfg = self.cfg
        rho, _ = self._effective_reps(task, server, ctx)
        if rho >= cfg.high_rep_threshold:
            deposit = cfg.min_deposit + cfg.deposit_price_coeff * est_price * 0.50
        else:
            deposit = cfg.min_deposit + cfg.deposit_price_coeff * est_price + cfg.deposit_risk_coeff * (1.0 - rho)
        return float(np.clip(deposit, 0.0, ctx.user_state.credit_limit[task.user]))

    def _expected_risk(
        self, task: Task, server: int, est_price: float, deposit: float, ctx: SlotContext
    ) -> float:
        if server == LOCAL_SERVER or self.prepaid:
            return 0.0
        cfg = self.cfg
        rho, _ = self._effective_reps(task, server, ctx)
        uncovered = max(0.0, est_price - deposit)
        default_prob = cfg.default_base_prob + cfg.default_rep_sensitivity * (1.0 - rho)
        return float(default_prob * uncovered)

    def _expected_settlement_dev(
        self, task: Task, server: int, est_price: float, ctx: SlotContext
    ) -> float:
        if server == LOCAL_SERVER or not self.post_service:
            return 0.0
        _, psi = self._effective_reps(task, server, ctx)
        queue_s = ctx.server_state.backlog_cycles[server] / max(ctx.server_state.cpu_hz[server], 1.0)
        uncertainty = 0.05 + 0.12 * (1.0 - psi) + 0.03 * min(queue_s, 2.0)
        return float(est_price * uncertainty)

    def _server_payment(self, task: Task, server: int, ctx: SlotContext, estimated: bool) -> float:
        if server == LOCAL_SERVER:
            return 0.0
        queue_s = ctx.server_state.backlog_cycles[server] / max(ctx.server_state.cpu_hz[server], 1.0)
        load_cost = self.cfg.server_load_cost * min(queue_s, 2.5)
        base = ctx.server_state.cost_per_gcycle[server] * (task.cycles / 1.0e9)
        return float(base + load_cost)

    def _score_candidate(
        self,
        task: Task,
        server: int,
        delay: float,
        energy: float,
        price: float,
        risk: float,
        settle_dev: float,
        expected_utility: float,
        ctx: SlotContext,
    ) -> float:
        if self.delay_energy_only or self.drl_offloading:
            return float(delay / task.tau_s + 0.25 * energy)
        if self.truthful_auction:
            ask = 0.0 if server == LOCAL_SERVER else self._server_payment(task, server, ctx, estimated=True)
            surplus = task.value - ask - self.cfg.user_delay_cost * delay - self.cfg.user_energy_cost * energy
            infeasible_penalty = 5.0 if surplus < 0 else 0.0
            return float(-surplus + 0.35 * delay / max(task.tau_s, 1e-9) + infeasible_penalty)
        if self.double_auction:
            ask = 0.0 if server == LOCAL_SERVER else self._server_payment(task, server, ctx, estimated=True)
            user_bid = task.value - self.cfg.user_delay_cost * delay - self.cfg.user_energy_cost * energy
            surplus = user_bid - ask
            infeasible_penalty = 8.0 if surplus < 0 else 0.0
            return float(-surplus + 0.25 * delay / max(task.tau_s, 1e-9) + infeasible_penalty)
        cfg = self.cfg
        rho, psi = self._effective_reps(task, server, ctx)
        reputation_penalty = 0.0 if server == LOCAL_SERVER else (1.0 - rho) + 0.5 * (1.0 - psi)
        sla_pressure = max(0.0, delay - task.tau_s) / max(task.tau_s, 1e-9)
        score = (
            cfg.objective_delay_weight * delay
            + cfg.objective_energy_weight * energy
            + cfg.objective_sla_weight * sla_pressure
            + 0.35 * price
            + cfg.objective_settlement_weight * settle_dev
            + cfg.objective_risk_weight * risk
            - cfg.objective_utility_weight * expected_utility
            + cfg.objective_reputation_weight * reputation_penalty
        )
        return float(score)

    def _accepts(self, task: Task, cand: Candidate, ctx: SlotContext) -> tuple:
        if cand.server == LOCAL_SERVER:
            utility = task.value - self.cfg.user_delay_cost * cand.est_delay - self.cfg.user_energy_cost * cand.est_energy
        else:
            utility = (
                task.value
                - cand.est_price
                - self.cfg.user_delay_cost * cand.est_delay
                - self.cfg.user_energy_cost * cand.est_energy
            )
        feasible = cand.deposit <= float(ctx.user_state.credit_limit[task.user]) + 1e-12
        prob = sigmoid(utility / self.cfg.acceptance_temperature) if feasible else 0.0
        return bool(ctx.rng.random() < prob), float(prob)

    def settle(
        self,
        task: Task,
        decision: Decision,
        ctx: SlotContext,
        actual_delay: float,
        actual_energy: float,
        sla_met: bool,
        server_success: bool,
    ) -> Dict[str, float]:
        cand = decision.candidate
        if cand.server == LOCAL_SERVER:
            return {
                "final_price": 0.0,
                "collected_payment": 0.0,
                "server_payment": 0.0,
                "bad_debt_loss": 0.0,
                "broker_utility": 0.0,
                "settlement_deviation": 0.0,
                "subsidy_loss": 0.0,
            }

        if self.fixed_pricing or self.drl_offloading or not self.post_service:
            raw_price = cand.est_price
        else:
            d_delta = (actual_delay - cand.est_delay) / (cand.est_delay + 1e-9)
            e_delta = (actual_energy - cand.est_energy) / (cand.est_energy + 1e-9)
            sla_discount = 0.0 if sla_met else self.cfg.settlement_sla_penalty_rate * cand.est_price
            raw_price = cand.est_price * (
                1.0 + self.cfg.settlement_alpha_delay * d_delta + self.cfg.settlement_alpha_energy * e_delta
            ) - sla_discount

        lower = cand.lower_price
        upper = cand.upper_price
        final_price = float(np.clip(raw_price, lower, upper))
        subsidy_loss = max(0.0, raw_price - final_price)
        server_payment = self._server_payment(task, cand.server, ctx, estimated=False)

        if self.prepaid:
            return {
                "final_price": final_price,
                "collected_payment": final_price,
                "server_payment": server_payment,
                "bad_debt_loss": 0.0,
                "broker_utility": float(final_price - server_payment - subsidy_loss),
                "settlement_deviation": 0.0,
                "subsidy_loss": float(subsidy_loss),
                "defaulted": 0.0,
            }

        rho, _ = self._effective_reps(task, cand.server, ctx)
        uncovered = max(0.0, final_price - cand.deposit)
        default_prob = (
            self.cfg.default_base_prob
            + self.cfg.default_rep_sensitivity * (1.0 - rho)
            + self.cfg.default_uncovered_sensitivity * uncovered
        )
        default_prob = float(np.clip(default_prob, 0.0, 0.95))
        defaulted = ctx.rng.random() < default_prob
        collected = final_price if not defaulted else min(cand.deposit, final_price)
        bad_debt = max(0.0, final_price - collected)
        broker_utility = collected - server_payment - subsidy_loss
        return {
            "final_price": final_price,
            "collected_payment": float(collected),
            "server_payment": float(server_payment),
            "bad_debt_loss": float(bad_debt),
            "broker_utility": float(broker_utility),
            "settlement_deviation": float(abs(final_price - cand.est_price)),
            "subsidy_loss": float(subsidy_loss),
            "defaulted": float(defaulted),
        }


class ProposedLyapunovPolicy(BrokerPolicy):
    name = "QCP"

    def __init__(
        self,
        *,
        use_reservation: bool = True,
        use_virtual_queues: bool = True,
        score_label: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(score_label=score_label, **kwargs)
        self.use_reservation = use_reservation
        self.use_virtual_queues = use_virtual_queues

    def reset(self, cfg: SimConfig, rng: np.random.Generator) -> None:
        super().reset(cfg, rng)
        self.sla_queue = np.zeros(cfg.num_users, dtype=float)
        self.risk_queue = 0.0
        self.settlement_queue = 0.0
        self.reserved_cycles = np.zeros(cfg.num_servers, dtype=float)
        self.current_reservation_slot = -1

    def _ensure_reservation_slot(self, slot: int) -> None:
        if slot != self.current_reservation_slot:
            self.current_reservation_slot = slot
            self.reserved_cycles.fill(0.0)

    def _estimate_price(
        self, task: Task, server: int, est_delay: float, est_energy: float, ctx: SlotContext
    ) -> float:
        price = super()._estimate_price(task, server, est_delay, est_energy, ctx)
        if server == LOCAL_SERVER:
            return price
        return max(0.001, self.cfg.proposed_price_multiplier * price)

    def _build_candidates(self, task: Task, ctx: SlotContext) -> List[Candidate]:
        self._ensure_reservation_slot(ctx.slot)
        candidates = super()._build_candidates(task, ctx)
        for cand in candidates:
            if cand.server == LOCAL_SERVER or not self.use_reservation:
                continue
            factor = self._adaptive_reservation_factor(task, cand, ctx)
            rate = max(float(ctx.est_rates_bps[task.user, cand.server]), 1.0)
            tx_delay = task.data_bits / rate
            queue_delay = ctx.server_state.backlog_cycles[cand.server] / ctx.server_state.cpu_hz[cand.server]
            queue_delay = min(float(queue_delay), self.cfg.max_server_queue_s)
            exec_delay = task.cycles / ctx.server_state.cpu_hz[cand.server]
            _, psi = self._effective_reps(task, cand.server, ctx)
            reputation_multiplier = 1.0 + 0.45 * (1.0 - psi)
            cand.reservation_delay_factor = factor
            cand.est_delay = float((tx_delay + factor * (queue_delay + exec_delay)) * reputation_multiplier)
            cand.est_price = self._estimate_price(task, cand.server, cand.est_delay, cand.est_energy, ctx)
            cand.deposit = self._deposit(task, cand.server, cand.est_price, ctx)
            cand.expected_risk = self._expected_risk(task, cand.server, cand.est_price, cand.deposit, ctx)
            cand.expected_settlement_dev = self._expected_settlement_dev(task, cand.server, cand.est_price, ctx)
            server_payment = self._server_payment(task, cand.server, ctx, estimated=True)
            cand.expected_broker_utility = cand.est_price - server_payment - cand.expected_risk
            cand.lower_price = self.cfg.settlement_lower_mult * cand.est_price
            cand.upper_price = self.cfg.settlement_upper_mult * cand.est_price
            cand.score = self._score_candidate(
                task,
                cand.server,
                cand.est_delay,
                cand.est_energy,
                cand.est_price,
                cand.expected_risk,
                cand.expected_settlement_dev,
                cand.expected_broker_utility,
                ctx,
            )
        return candidates

    def _adaptive_reservation_factor(self, task: Task, cand: Candidate, ctx: SlotContext) -> float:
        cfg = self.cfg
        queue_delay = min(
            float(ctx.server_state.backlog_cycles[cand.server] / ctx.server_state.cpu_hz[cand.server]),
            cfg.max_server_queue_s,
        )
        load = queue_delay / max(cfg.max_server_queue_s, 1e-9)
        urgency = max(0.0, task.tau_s - cand.est_delay) / max(task.tau_s, 1e-9)
        sla_backlog = min(self.sla_queue[task.user] / 5.0, 1.0)
        rho, psi = self._effective_reps(task, cand.server, ctx)
        reliability = 0.5 * rho + 0.5 * psi
        pressure = np.clip(
            0.35 * sla_backlog + 0.25 * urgency + 0.20 * load + 0.20 * reliability,
            0.0,
            1.0,
        )
        raw_factor = cfg.proposed_reservation_max_factor - pressure * (
            cfg.proposed_reservation_max_factor - cfg.proposed_reservation_min_factor
        )

        budget = cfg.proposed_reservation_capacity_fraction * ctx.server_state.cpu_hz[cand.server] * cfg.slot_duration
        remaining = max(0.0, budget - self.reserved_cycles[cand.server])
        requested = max(0.0, (1.0 - raw_factor) * task.cycles)
        if requested > remaining and task.cycles > 0.0:
            raw_factor = max(raw_factor, 1.0 - remaining / task.cycles)
        return float(np.clip(raw_factor, cfg.proposed_reservation_min_factor, 1.0))

    def decide(self, task: Task, ctx: SlotContext) -> Decision:
        decision = super().decide(task, ctx)
        cand = decision.candidate
        if decision.accepted and cand.server != LOCAL_SERVER and self.use_reservation:
            self._ensure_reservation_slot(ctx.slot)
            reserved = max(0.0, (1.0 - cand.reservation_delay_factor) * task.cycles)
            self.reserved_cycles[cand.server] += reserved
        return decision

    def _server_payment(self, task: Task, server: int, ctx: SlotContext, estimated: bool) -> float:
        base_payment = super()._server_payment(task, server, ctx, estimated)
        if server == LOCAL_SERVER:
            return base_payment
        reserved_share = 1.0 - self.cfg.proposed_reservation_delay_factor
        reservation_cost = (
            self.cfg.proposed_reservation_cost_per_gcycle
            * reserved_share
            * (task.cycles / 1.0e9)
        )
        return float(base_payment + reservation_cost)

    def _score_candidate(
        self,
        task: Task,
        server: int,
        delay: float,
        energy: float,
        price: float,
        risk: float,
        settle_dev: float,
        expected_utility: float,
        ctx: SlotContext,
    ) -> float:
        cfg = self.cfg
        violation = 1.0 if delay > task.tau_s else 0.0
        immediate_penalty = (
            cfg.objective_delay_weight * delay
            + cfg.proposed_energy_weight_multiplier * cfg.objective_energy_weight * energy
            + cfg.proposed_sla_weight_multiplier
            * cfg.objective_sla_weight
            * max(0.0, delay - task.tau_s)
            / max(task.tau_s, 1e-9)
            + cfg.objective_settlement_weight * settle_dev
            + cfg.objective_risk_weight * risk
            - cfg.proposed_utility_weight_multiplier * cfg.objective_utility_weight * expected_utility
        )
        drift_proxy = 0.0
        if self.use_virtual_queues:
            drift_proxy = (
                self.sla_queue[task.user] * (violation - cfg.sla_violation_budget)
                + self.risk_queue * (risk - cfg.risk_budget_per_task)
                + self.settlement_queue * (settle_dev - cfg.settlement_budget_per_task)
            )
        return float(drift_proxy + cfg.lyapunov_v * immediate_penalty)

    def observe_transaction(self, tx: Transaction) -> None:
        if not self.use_virtual_queues:
            return
        cfg = self.cfg
        if tx.arrived == 0 or tx.accepted == 0:
            return
        self.sla_queue[tx.user] = max(
            0.0, self.sla_queue[tx.user] + (1.0 - tx.sla_met) - cfg.sla_violation_budget
        )
        self.risk_queue = max(0.0, self.risk_queue + tx.bad_debt_loss - cfg.risk_budget_per_task)
        self.settlement_queue = max(
            0.0, self.settlement_queue + tx.settlement_deviation - cfg.settlement_budget_per_task
        )


def make_default_policies() -> Iterable[Policy]:
    return [
        ProposedLyapunovPolicy(),
        BrokerPolicy(
            dynamic_pricing=True,
            post_service=False,
            use_deposit=False,
            use_reputation=False,
            prepaid=True,
            score_label="DP-MEC",
        ),
        BrokerPolicy(
            stackelberg_pricing=True,
            post_service=False,
            use_deposit=False,
            use_reputation=False,
            prepaid=True,
            score_label="Stackelberg-EPRA",
        ),
        BrokerPolicy(
            truthful_auction=True,
            post_service=False,
            use_deposit=False,
            use_reputation=False,
            prepaid=True,
            score_label="TARFO",
        ),
        BrokerPolicy(
            double_auction=True,
            use_deposit=False,
            post_service=False,
            use_reputation=False,
            prepaid=True,
            score_label="DoubleAuction",
        ),
        BrokerPolicy(
            drl_offloading=True,
            delay_energy_only=True,
            use_deposit=False,
            post_service=False,
            use_reputation=False,
            prepaid=True,
            score_label="DRL-Offloading",
        ),
        ProposedLyapunovPolicy(use_reputation=False, score_label="NoReputation"),
    ]


def make_ablation_policies() -> Iterable[Policy]:
    return [
        ProposedLyapunovPolicy(score_label="QCP"),
        ProposedLyapunovPolicy(use_reservation=False, score_label="Ablation-NoReservation"),
        ProposedLyapunovPolicy(use_reputation=False, score_label="NoReputation"),
        ProposedLyapunovPolicy(use_deposit=False, score_label="Ablation-NoDeposit"),
        ProposedLyapunovPolicy(post_service=False, prepaid=True, use_deposit=False, score_label="Ablation-NoPostSettlement"),
        ProposedLyapunovPolicy(use_virtual_queues=False, score_label="Ablation-NoVirtualQueues"),
    ]
