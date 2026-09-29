from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare generated CSV outputs with reference outputs.")
    parser.add_argument("--results", default="results", help="Directory containing generated summary CSVs.")
    parser.add_argument("--reference", default="reference_results", help="Directory containing reference summary CSVs.")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance for numeric comparisons.")
    parser.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance for numeric comparisons.")
    return parser.parse_args()


def sort_frame(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = [
        col
        for col in ["policy", "arrival_rate", "low_reputation_ratio", "deposit_price_coeff",
                    "proposed_reservation_capacity_fraction", "lyapunov_v", "slot_bin", "rep"]
        if col in df.columns
    ]
    return df.sort_values(key_cols, kind="stable").reset_index(drop=True) if key_cols else df.reset_index(drop=True)


def compare_file(result_path: Path, reference_path: Path, atol: float, rtol: float) -> list[str]:
    errors: list[str] = []
    if not result_path.exists():
        return [f"Missing generated file: {result_path}"]
    if not reference_path.exists():
        return [f"Missing reference file: {reference_path}"]

    result = sort_frame(pd.read_csv(result_path))
    reference = sort_frame(pd.read_csv(reference_path))

    if list(result.columns) != list(reference.columns):
        errors.append(f"{result_path.name}: column mismatch")
        errors.append(f"  generated: {list(result.columns)}")
        errors.append(f"  reference: {list(reference.columns)}")
        return errors

    if result.shape != reference.shape:
        errors.append(f"{result_path.name}: shape mismatch {result.shape} != {reference.shape}")
        return errors

    for col in result.columns:
        if pd.api.types.is_numeric_dtype(reference[col]):
            if not np.allclose(result[col].to_numpy(), reference[col].to_numpy(), atol=atol, rtol=rtol):
                diff = np.nanmax(np.abs(result[col].to_numpy() - reference[col].to_numpy()))
                errors.append(f"{result_path.name}: numeric mismatch in {col}, max abs diff={diff:.6g}")
        elif not result[col].astype(str).equals(reference[col].astype(str)):
            errors.append(f"{result_path.name}: value mismatch in {col}")
    return errors


def main() -> None:
    args = parse_args()
    result_dir = Path(args.results)
    reference_dir = Path(args.reference)
    all_errors: list[str] = []
    reference_files = sorted(path.name for path in reference_dir.glob("*.csv"))
    if not reference_files:
        raise SystemExit(f"No reference CSV files found in {reference_dir}")

    for filename in reference_files:
        all_errors.extend(
            compare_file(result_dir / filename, reference_dir / filename, args.atol, args.rtol)
        )

    if all_errors:
        print("Verification failed:")
        for error in all_errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(f"Verification passed: {len(reference_files)} generated CSV files match the reference outputs.")


if __name__ == "__main__":
    main()
