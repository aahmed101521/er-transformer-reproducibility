#!/usr/bin/env python3
"""
analysis14_counting_audit.py

Audit the saved June per-record predictions under Zhang's corrected counting
rule. This script does NOT train, embed, block, or rescore any model.

It compares three evaluations:

1. OLD
   - correctness is taken from the saved *_correct column, which used the
     single retained true_idb;
   - psi = 1 - correct / n_true_saved, so a wrong declared link is also
     counted as a missing match.

2. ANY-VALID, OVERLAPPING
   - a declared link is correct when the predicted B id belongs to the full
     many-match ground-truth set for that A record;
   - psi still uses 1 - correct / n_true_raw. This isolates the effect of
     accepting alternative valid matches, before changing the counting rule.

3. CORRECTED, MUTUALLY EXCLUSIVE
   - correct link: a link is declared to any valid B match;
   - false link: a link is declared, but not to a valid B match;
   - missing match: the A record has at least one true B match and no link is
     declared;
   - lambda = false links / declared links;
   - psi = missing matches / A records with at least one true match.

For matched A records, the corrected categories satisfy exactly:

    correct + false_link_on_matched_record + missing = n_true_records

Records with no raw true match and no declared link are reported separately as
true negatives; they are outside both lambda and psi.

Default server layout:

    ~/er_paper/
      data/dblp/{DBLP1.csv, Scholar.csv,
                 DBLP-Scholar_perfectMapping.csv}
      data/abt_buy/{Abt.csv, Buy.csv,
                    abt_buy_perfectMapping.csv}
      results/analysis11_predictions.csv
      scripts/analysis14_counting_audit.py

Run:

    python ~/er_paper/scripts/analysis14_counting_audit.py

Optional:

    python analysis14_counting_audit.py --root /path/to/er_paper

Outputs:

    results/analysis14_counting_audit.csv
    results/analysis14_alternative_valid_links.csv
    results/analysis14_truth_mismatches.csv
"""

from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


DATASETS = {
    "DBLP": {
        "a_rel": Path("dblp/DBLP1.csv"),
        "b_rel": Path("dblp/Scholar.csv"),
        "map_rel": Path("dblp/DBLP-Scholar_perfectMapping.csv"),
        "map_a": "idDBLP",
        "map_b": "idScholar",
        "expected_a": 2616,
        "expected_b": 64263,
        "expected_pairs": 5347,
    },
    "ECOM": {
        "a_rel": Path("abt_buy/Abt.csv"),
        "b_rel": Path("abt_buy/Buy.csv"),
        "map_rel": Path("abt_buy/abt_buy_perfectMapping.csv"),
        "map_a": "idAbt",
        "map_b": "idBuy",
        "expected_a": 1081,
        "expected_b": 1092,
        "expected_pairs": 1097,
    },
}

MODELS = ("lr", "ce")
TRUE_STRINGS = {"true", "1", "yes", "y", "t"}
FALSE_STRINGS = {"false", "0", "no", "n", "f", ""}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit old versus corrected entity-resolution counting."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("ER_PROJECT_ROOT", Path(__file__).resolve().parents[1])).expanduser(),
        help="Project root. Default: package root",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help=(
            "Saved per-record predictions. Default: "
            "<root>/results/analysis11_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <root>/results",
    )
    return parser.parse_args()


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove BOM artefacts, surrounding quotes, and whitespace."""
    df = df.copy()
    df.columns = [
        str(col)
        .replace("\ufeff", "")
        .replace("ï»¿", "")
        .strip()
        .strip('"')
        .strip()
        for col in df.columns
    ]
    return df


def read_csv_robust(path: Path, *, usecols: Iterable[str] | None = None) -> pd.DataFrame:
    """Read a CSV as strings, trying UTF-8-with-BOM before latin-1."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(
                path,
                dtype=str,
                encoding=encoding,
                keep_default_na=False,
                usecols=usecols,
            )
            return clean_columns(df)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise RuntimeError(f"Could not decode {path}: {last_error}")


def find_id_column(df: pd.DataFrame, path: Path) -> str:
    for col in df.columns:
        if col.lower() == "id":
            return col
    raise KeyError(f"No id column found in {path}. Columns: {list(df.columns)}")


def normalise_ids(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def parse_bool(series: pd.Series, column_name: str) -> pd.Series:
    """Parse a saved boolean column without relying on pandas inference."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    values = series.astype(str).str.strip().str.lower()
    unknown = sorted(set(values) - TRUE_STRINGS - FALSE_STRINGS)
    if unknown:
        raise ValueError(
            f"Unexpected boolean values in {column_name}: {unknown[:10]}"
        )
    return values.isin(TRUE_STRINGS)


def safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else math.nan


def build_truth(
    mapping: pd.DataFrame,
    a_col: str,
    b_col: str,
) -> tuple[dict[str, set[str]], dict[str, str]]:
    """Return full many-match truth and first-listed retained match."""
    truth: dict[str, set[str]] = defaultdict(set)
    retained: dict[str, str] = {}

    a_values = normalise_ids(mapping[a_col])
    b_values = normalise_ids(mapping[b_col])

    for ida, idb in zip(a_values, b_values):
        if not ida or not idb:
            continue
        truth[ida].add(idb)
        retained.setdefault(ida, idb)

    return dict(truth), retained


def load_and_validate_dataset(
    name: str,
    cfg: dict,
    data_dir: Path,
) -> tuple[dict[str, set[str]], dict[str, str], dict[str, int]]:
    a_path = data_dir / cfg["a_rel"]
    b_path = data_dir / cfg["b_rel"]
    map_path = data_dir / cfg["map_rel"]

    df_a = read_csv_robust(a_path)
    df_b = read_csv_robust(b_path)
    df_m = read_csv_robust(map_path)

    for required in (cfg["map_a"], cfg["map_b"]):
        if required not in df_m.columns:
            raise KeyError(
                f"{map_path} is missing {required}. Columns: {list(df_m.columns)}"
            )

    a_id_col = find_id_column(df_a, a_path)
    b_id_col = find_id_column(df_b, b_path)
    a_ids = set(normalise_ids(df_a[a_id_col]))
    b_ids = set(normalise_ids(df_b[b_id_col]))

    truth, retained = build_truth(df_m, cfg["map_a"], cfg["map_b"])

    mapped_a = normalise_ids(df_m[cfg["map_a"]])
    mapped_b = normalise_ids(df_m[cfg["map_b"]])
    reachable_mask = mapped_a.isin(a_ids) & mapped_b.isin(b_ids)

    a_counts = mapped_a.value_counts()
    b_counts = mapped_b.value_counts()

    diagnostics = {
        "A": len(df_a),
        "B": len(df_b),
        "pairs": len(df_m),
        "distinct_A_in_map": int(mapped_a.nunique()),
        "distinct_B_in_map": int(mapped_b.nunique()),
        "A_with_multiple_matches": int((a_counts > 1).sum()),
        "B_claimed_by_multiple_A": int((b_counts > 1).sum()),
        "max_matches_per_A": int(a_counts.max()),
        "reachable_pairs": int(reachable_mask.sum()),
        "unreachable_pairs": int((~reachable_mask).sum()),
        "mapped_A_absent_from_A": int((~mapped_a.isin(a_ids)).sum()),
        "mapped_B_absent_from_B": int((~mapped_b.isin(b_ids)).sum()),
    }

    expected = {
        "A": cfg["expected_a"],
        "B": cfg["expected_b"],
        "pairs": cfg["expected_pairs"],
    }
    mismatches = {
        key: (diagnostics[key], expected_value)
        for key, expected_value in expected.items()
        if diagnostics[key] != expected_value
    }
    if mismatches:
        details = ", ".join(
            f"{key}={actual} (expected {expected_value})"
            for key, (actual, expected_value) in mismatches.items()
        )
        raise AssertionError(f"{name} raw-count validation failed: {details}")

    return truth, retained, diagnostics


def validate_prediction_columns(pred: pd.DataFrame) -> None:
    required = {"dataset", "seed", "ida", "true_idb", "has_true_match"}
    for model in MODELS:
        required.update(
            {
                f"{model}_top1_idb",
                f"{model}_declared",
                f"{model}_correct",
            }
        )

    missing = sorted(required - set(pred.columns))
    if missing:
        raise KeyError(f"Predictions file is missing columns: {missing}")


def audit_group(
    group: pd.DataFrame,
    *,
    dataset: str,
    model: str,
    truth: dict[str, set[str]],
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    ida = normalise_ids(group["ida"])
    predicted_b = normalise_ids(group[f"{model}_top1_idb"])
    saved_true_b = normalise_ids(group["true_idb"])

    declared = parse_bool(group[f"{model}_declared"], f"{model}_declared")
    saved_correct = parse_bool(group[f"{model}_correct"], f"{model}_correct")
    saved_has_true = parse_bool(group["has_true_match"], "has_true_match")

    raw_has_true = ida.map(lambda value: value in truth)
    any_valid_correct = pd.Series(
        [
            bool(is_declared and pred_b in truth.get(record_id, set()))
            for record_id, pred_b, is_declared in zip(ida, predicted_b, declared)
        ],
        index=group.index,
        dtype=bool,
    )

    # Verify that the saved correctness flag really is the historical
    # retained-single-match rule encoded by true_idb.
    recomputed_saved_correct = declared & (predicted_b == saved_true_b)
    saved_flag_disagreements = int((saved_correct != recomputed_saved_correct).sum())
    if saved_flag_disagreements:
        raise AssertionError(
            f"{dataset}/{model}: {saved_flag_disagreements} rows disagree "
            "between the saved *_correct flag and declared & top1==true_idb."
        )

    n_rows = len(group)
    n_true_saved = int(saved_has_true.sum())
    n_true_raw = int(raw_has_true.sum())
    n_declared = int(declared.sum())

    old_correct = int(saved_correct.sum())
    old_false = n_declared - old_correct
    old_missing_overlap = n_true_saved - old_correct

    any_correct = int(any_valid_correct.sum())
    any_false = n_declared - any_correct
    any_missing_overlap = n_true_raw - any_correct

    false_on_matched = raw_has_true & declared & ~any_valid_correct
    false_on_unmatched = ~raw_has_true & declared
    missing_exclusive = raw_has_true & ~declared
    true_negative = ~raw_has_true & ~declared

    n_false_on_matched = int(false_on_matched.sum())
    n_false_on_unmatched = int(false_on_unmatched.sum())
    n_missing_exclusive = int(missing_exclusive.sum())
    n_true_negative = int(true_negative.sum())

    partition_total = (
        any_correct
        + n_false_on_matched
        + n_false_on_unmatched
        + n_missing_exclusive
        + n_true_negative
    )
    if partition_total != n_rows:
        raise AssertionError(
            f"{dataset}/{model}: corrected categories sum to "
            f"{partition_total}, expected {n_rows}."
        )

    matched_partition = any_correct + n_false_on_matched + n_missing_exclusive
    if matched_partition != n_true_raw:
        raise AssertionError(
            f"{dataset}/{model}: matched-record categories sum to "
            f"{matched_partition}, expected {n_true_raw}."
        )

    alternative_valid = any_valid_correct & ~saved_correct
    alt_rows = group.loc[alternative_valid].copy()
    if not alt_rows.empty:
        alt_rows.insert(0, "model", model)
        alt_rows["raw_valid_matches"] = ida.loc[alternative_valid].map(
            lambda value: "|".join(sorted(truth.get(value, set())))
        )

    truth_mismatch = saved_has_true != raw_has_true
    mismatch_rows = group.loc[truth_mismatch].copy()
    if not mismatch_rows.empty:
        mismatch_rows.insert(0, "model", model)
        mismatch_rows["raw_has_true_match"] = raw_has_true.loc[truth_mismatch].values
        mismatch_rows["raw_valid_matches"] = ida.loc[truth_mismatch].map(
            lambda value: "|".join(sorted(truth.get(value, set())))
        )

    row = {
        "dataset": dataset,
        "model": model.upper(),
        "n_rows": n_rows,
        "n_true_saved": n_true_saved,
        "n_true_raw": n_true_raw,
        "saved_vs_raw_truth_mismatches": int(truth_mismatch.sum()),
        "declared": n_declared,
        "old_correct": old_correct,
        "old_false": old_false,
        "old_missing_overlap": old_missing_overlap,
        "any_valid_correct": any_correct,
        "alternative_valid_links": int(alternative_valid.sum()),
        "corrected_false_total": any_false,
        "corrected_false_on_matched": n_false_on_matched,
        "corrected_false_on_unmatched": n_false_on_unmatched,
        "corrected_missing_exclusive": n_missing_exclusive,
        "true_negative": n_true_negative,
        "lambda_old": safe_rate(old_false, n_declared),
        "psi_old_overlap": safe_rate(old_missing_overlap, n_true_saved),
        "lambda_any_valid": safe_rate(any_false, n_declared),
        "psi_any_valid_overlap": safe_rate(any_missing_overlap, n_true_raw),
        "lambda_corrected": safe_rate(any_false, n_declared),
        "psi_corrected_exclusive": safe_rate(n_missing_exclusive, n_true_raw),
    }

    return row, alt_rows, mismatch_rows


def add_pooled_rows(summary: pd.DataFrame) -> pd.DataFrame:
    pooled_rows = []

    count_columns = [
        "n_rows",
        "n_true_saved",
        "n_true_raw",
        "saved_vs_raw_truth_mismatches",
        "declared",
        "old_correct",
        "old_false",
        "old_missing_overlap",
        "any_valid_correct",
        "alternative_valid_links",
        "corrected_false_total",
        "corrected_false_on_matched",
        "corrected_false_on_unmatched",
        "corrected_missing_exclusive",
        "true_negative",
    ]

    for model, model_rows in summary.groupby("model", sort=False):
        counts = model_rows[count_columns].sum(numeric_only=True)
        pooled = {
            "dataset": "POOLED",
            "model": model,
            **{column: int(counts[column]) for column in count_columns},
        }
        pooled.update(
            {
                "lambda_old": safe_rate(pooled["old_false"], pooled["declared"]),
                "psi_old_overlap": safe_rate(
                    pooled["old_missing_overlap"], pooled["n_true_saved"]
                ),
                "lambda_any_valid": safe_rate(
                    pooled["corrected_false_total"], pooled["declared"]
                ),
                "psi_any_valid_overlap": safe_rate(
                    pooled["n_true_raw"] - pooled["any_valid_correct"],
                    pooled["n_true_raw"],
                ),
                "lambda_corrected": safe_rate(
                    pooled["corrected_false_total"], pooled["declared"]
                ),
                "psi_corrected_exclusive": safe_rate(
                    pooled["corrected_missing_exclusive"], pooled["n_true_raw"]
                ),
            }
        )
        pooled_rows.append(pooled)

    return pd.concat([summary, pd.DataFrame(pooled_rows)], ignore_index=True)


def print_dataset_validation(name: str, diagnostics: dict[str, int]) -> None:
    print(f"\n{name} raw-data validation")
    print("-" * 72)
    print(
        f"|A|={diagnostics['A']:,}  |B|={diagnostics['B']:,}  "
        f"raw pairs={diagnostics['pairs']:,}"
    )
    print(
        f"distinct mapped A={diagnostics['distinct_A_in_map']:,}  "
        f"distinct mapped B={diagnostics['distinct_B_in_map']:,}"
    )
    print(
        f"A with >1 match={diagnostics['A_with_multiple_matches']:,}  "
        f"max matches/A={diagnostics['max_matches_per_A']:,}  "
        f"B claimed by >1 A={diagnostics['B_claimed_by_multiple_A']:,}"
    )
    print(
        f"reachable pairs={diagnostics['reachable_pairs']:,}  "
        f"unreachable pairs={diagnostics['unreachable_pairs']:,}"
    )


def print_summary(summary: pd.DataFrame) -> None:
    columns = [
        "dataset",
        "model",
        "n_rows",
        "n_true_saved",
        "n_true_raw",
        "declared",
        "old_correct",
        "any_valid_correct",
        "alternative_valid_links",
        "corrected_false_total",
        "corrected_missing_exclusive",
        "lambda_old",
        "lambda_corrected",
        "psi_old_overlap",
        "psi_corrected_exclusive",
    ]
    display = summary[columns].copy()
    for col in (
        "lambda_old",
        "lambda_corrected",
        "psi_old_overlap",
        "psi_corrected_exclusive",
    ):
        display[col] = display[col].map(
            lambda value: "NA" if pd.isna(value) else f"{value:.6f}"
        )

    print("\nCounting audit")
    print("=" * 72)
    print(display.to_string(index=False))

    alt_total = int(summary.loc[summary["dataset"] != "POOLED", "alternative_valid_links"].sum())
    truth_mismatch_total = int(
        summary.loc[
            summary["dataset"] != "POOLED", "saved_vs_raw_truth_mismatches"
        ].sum()
    )
    # Truth mismatch rows repeat once per model, so divide by number of models
    # for the number of distinct prediction rows affected.
    truth_mismatch_rows = truth_mismatch_total // len(MODELS)

    print("\nInterpretation checks")
    print("-" * 72)
    print(f"Alternative valid links found: {alt_total}")
    print(
        "Saved has_true_match disagrees with raw many-match truth on "
        f"{truth_mismatch_rows} prediction rows."
    )
    if alt_total == 0:
        print(
            "Result: accepting any valid duplicate match does not change "
            "the saved LR or CE correct-link counts."
        )
    else:
        print(
            "Result: accepting alternative valid matches changes at least "
            "one saved result; inspect the alternative-links CSV."
        )
    print(
        "The difference between psi_old_overlap and "
        "psi_corrected_exclusive is the effect of making false links and "
        "missing matches mutually exclusive."
    )


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    data_dir = root / "data"
    predictions_path = (
        args.predictions.expanduser().resolve()
        if args.predictions is not None
        else root / "results" / "analysis11_predictions.csv"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Analysis 14: counting and many-match truth audit")
    print("=" * 72)
    print(f"Project root: {root}")
    print(f"Predictions: {predictions_path}")

    truths: dict[str, dict[str, set[str]]] = {}
    diagnostics_by_dataset: dict[str, dict[str, int]] = {}

    for name, cfg in DATASETS.items():
        truth, _retained, diagnostics = load_and_validate_dataset(
            name, cfg, data_dir
        )
        truths[name] = truth
        diagnostics_by_dataset[name] = diagnostics
        print_dataset_validation(name, diagnostics)

    pred = read_csv_robust(predictions_path)
    validate_prediction_columns(pred)
    pred["dataset"] = pred["dataset"].astype(str).str.strip().str.upper()

    unexpected_datasets = sorted(set(pred["dataset"]) - set(DATASETS))
    if unexpected_datasets:
        raise ValueError(
            f"Unexpected dataset labels in predictions: {unexpected_datasets}"
        )

    summary_rows: list[dict] = []
    alternative_rows: list[pd.DataFrame] = []
    mismatch_rows: list[pd.DataFrame] = []

    for dataset in DATASETS:
        group = pred.loc[pred["dataset"] == dataset].copy()
        if group.empty:
            raise ValueError(f"No prediction rows found for {dataset}")

        for model in MODELS:
            row, alternatives, mismatches = audit_group(
                group,
                dataset=dataset,
                model=model,
                truth=truths[dataset],
            )
            summary_rows.append(row)
            if not alternatives.empty:
                alternative_rows.append(alternatives)
            if not mismatches.empty:
                mismatch_rows.append(mismatches)

    summary = pd.DataFrame(summary_rows)
    summary = add_pooled_rows(summary)

    summary_path = output_dir / "analysis14_counting_audit.csv"
    alternative_path = output_dir / "analysis14_alternative_valid_links.csv"
    mismatch_path = output_dir / "analysis14_truth_mismatches.csv"

    summary.to_csv(summary_path, index=False)

    if alternative_rows:
        pd.concat(alternative_rows, ignore_index=True).to_csv(
            alternative_path, index=False
        )
    else:
        pd.DataFrame(
            columns=[
                "model",
                "dataset",
                "seed",
                "ida",
                "true_idb",
                "raw_valid_matches",
            ]
        ).to_csv(alternative_path, index=False)

    if mismatch_rows:
        # The same truth mismatch is found for LR and CE. Keep both model
        # views because their declaration/prediction columns may differ.
        pd.concat(mismatch_rows, ignore_index=True).to_csv(
            mismatch_path, index=False
        )
    else:
        pd.DataFrame(
            columns=[
                "model",
                "dataset",
                "seed",
                "ida",
                "has_true_match",
                "raw_has_true_match",
                "raw_valid_matches",
            ]
        ).to_csv(mismatch_path, index=False)

    print_summary(summary)

    print("\nSaved outputs")
    print("-" * 72)
    print(summary_path)
    print(alternative_path)
    print(mismatch_path)


if __name__ == "__main__":
    main()
