from __future__ import annotations

from dataclasses import asdict

from paper4_sim.config import SimConfig
from paper4_sim.policies import BrokerPolicy, ProposedLyapunovPolicy, make_ablation_policies, make_default_policies


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def policy_map():
    policies = list(make_default_policies())
    return {policy.name: policy for policy in policies}


def check_proposed(policies: dict[str, object]) -> None:
    proposed = policies["QCP"]
    require(isinstance(proposed, ProposedLyapunovPolicy), "QCP must use ProposedLyapunovPolicy")
    require(proposed.use_reputation, "QCP must enable reputation")
    require(proposed.use_deposit, "QCP must enable deposit reservation")
    require(proposed.post_service, "QCP must enable post-service settlement")
    require(not proposed.prepaid, "QCP must not be prepaid")
    require(proposed.use_reservation, "QCP must enable adaptive reservation")
    require(proposed.use_virtual_queues, "QCP must enable virtual queues")


def check_pre_service_baseline(policy: BrokerPolicy, active_flag: str) -> None:
    require(getattr(policy, active_flag), f"{policy.name} must enable {active_flag}")
    require(not policy.post_service, f"{policy.name} must use pre-service settlement")
    require(policy.prepaid, f"{policy.name} must be prepaid")
    require(not policy.use_deposit, f"{policy.name} must not use QCP deposit reservation")
    require(not policy.use_reputation, f"{policy.name} must not use QCP reputation terms")


def check_baselines(policies: dict[str, object]) -> None:
    expected_flags = {
        "DP-MEC": "dynamic_pricing",
        "Stackelberg-EPRA": "stackelberg_pricing",
        "TARFO": "truthful_auction",
        "DoubleAuction": "double_auction",
        "DRL-Offloading": "drl_offloading",
    }
    for name, flag in expected_flags.items():
        policy = policies[name]
        require(isinstance(policy, BrokerPolicy), f"{name} must use BrokerPolicy")
        check_pre_service_baseline(policy, flag)
    doda = policies["DRL-Offloading"]
    require(doda.delay_energy_only, "DRL-Offloading must use delay-energy-oriented scoring")

    no_rep = policies["NoReputation"]
    require(isinstance(no_rep, ProposedLyapunovPolicy), "NoReputation must keep QCP policy structure")
    require(not no_rep.use_reputation, "NoReputation must disable reputation")
    require(no_rep.post_service and no_rep.use_deposit, "NoReputation must keep post-service deposit structure")


def check_ablation_suite() -> None:
    ablations = {policy.name: policy for policy in make_ablation_policies()}
    expected = {
        "QCP",
        "Ablation-NoReservation",
        "NoReputation",
        "Ablation-NoDeposit",
        "Ablation-NoPostSettlement",
        "Ablation-NoVirtualQueues",
    }
    require(set(ablations) == expected, "Ablation policy set is inconsistent with the paper")
    require(not ablations["Ablation-NoReservation"].use_reservation, "NoReservation ablation must disable reservation")
    require(not ablations["NoReputation"].use_reputation, "NoReputation ablation must disable reputation")
    require(not ablations["Ablation-NoDeposit"].use_deposit, "NoDeposit ablation must disable deposit")
    require(not ablations["Ablation-NoPostSettlement"].post_service, "NoPostSettlement must disable post-service settlement")
    require(ablations["Ablation-NoPostSettlement"].prepaid, "NoPostSettlement must be prepaid")
    require(not ablations["Ablation-NoVirtualQueues"].use_virtual_queues, "NoVirtualQueues must disable virtual queues")


def check_config() -> None:
    cfg = SimConfig()
    values = asdict(cfg)
    require(0.0 < values["settlement_lower_mult"] < 1.0, "Settlement lower multiplier should be below 1")
    require(values["settlement_upper_mult"] > 1.0, "Settlement upper multiplier should be above 1")
    require(values["deposit_price_coeff"] > 0.0, "Deposit coefficient should be positive")
    require(values["default_rep_sensitivity"] > 0.0, "Default probability should depend on reputation")
    require(values["proposed_reservation_capacity_fraction"] > 0.0, "QCP reservation budget should be positive")
    require(values["lyapunov_v"] > 0.0, "Lyapunov V should be positive")


def main() -> None:
    policies = policy_map()
    expected_names = {
        "QCP",
        "DP-MEC",
        "Stackelberg-EPRA",
        "TARFO",
        "DoubleAuction",
        "DRL-Offloading",
        "NoReputation",
    }
    require(set(policies) == expected_names, "Default policy set is inconsistent with the paper")
    check_proposed(policies)
    check_baselines(policies)
    check_ablation_suite()
    check_config()
    print("Configuration audit passed: policy flags and selected configuration checks passed.")
    print("This checks configuration only; run the experiments and verify_outputs.py for numerical reproduction.")


if __name__ == "__main__":
    main()
