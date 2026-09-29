"""Aggregation for the paper's payment-discipline figure (no simulation decisions)."""
from __future__ import annotations

import pandas as pd


def summarize_payment_discipline(transactions: pd.DataFrame) -> pd.DataFrame:
    """Pool accepted edge transactions in 40-slot bins, including the transient.

    `nonpayment` is the fraction with positive residual bad debt, not the
    fraction of all default events. A default fully covered by the deposit
    has zero bad debt. `user_reputation` records the post-transaction value.
    """
    required = {"policy", "slot", "arrived", "accepted", "completed",
                "bad_debt_loss", "user_reputation", "est_price", "deposit"}
    missing = required.difference(transactions.columns)
    if missing:
        raise ValueError(f"Payment-discipline trace is missing columns: {sorted(missing)}")
    data = transactions[transactions["policy"].isin(["QCP", "NoReputation"])].copy()
    mask = (
        data["arrived"].astype(bool)
        & data["accepted"].astype(bool)
        & data["completed"].astype(bool)
        & (data["est_price"].astype(float) > 1e-12)
    )
    data = data[mask].copy()
    data["slot_bin"] = (data["slot"].astype(int) // 40) * 40
    data["nonpayment"] = (data["bad_debt_loss"].astype(float) > 1e-12).astype(float)
    data["deposit_quote_ratio"] = data["deposit"].astype(float) / data["est_price"].astype(float)
    return (
        data.groupby(["policy", "slot_bin"], as_index=False)
        .agg(nonpayment=("nonpayment", "mean"),
             user_reputation=("user_reputation", "mean"),
             deposit_quote_ratio=("deposit_quote_ratio", "mean"))
        .sort_values(["policy", "slot_bin"])
        .reset_index(drop=True)
    )
