"""
analysis18_section7_direct.py
=============================

Final direct-score reconstruction for the Section 4.3.1 training-fraction
experiment in the ER Transformer paper.

Purpose
-------
The original Analysis 18 fitted the required cross-encoder checkpoints but
applied an additional sigmoid to CrossEncoder.predict() before threshold
calibration and declaration.

The final manuscript uses the direct CrossEncoder.predict() score path.

This script therefore:

1. NEVER trains a cross-encoder.
2. Reuses the exact fitted Analysis 18 checkpoints.
3. Reconstructs the exact outer A-record split and pair-level 80/20
   calibration split.
4. Recalibrates the threshold using direct CrossEncoder.predict().
5. Reclassifies the already-saved holdout predictions using the
   mathematically equivalent sigmoid-space threshold.
6. Recomputes the final paper metrics using Zhang's definitions:

       FLR = 1 - correct / declared
       MMR = 1 - correct / n_true_records

The old missing-only quantity

       missing_matches / n_true_records

is retained only as ``legacy_psi`` for auditability.  It is NOT labelled
as MMR.

No model fitting occurs anywhere in this script.

Prerequisites
-------------
Run analysis18_section7.py first.  It creates:

- fitted cross-encoder checkpoints;
- fixed top-50 blocking cache;
- original per-record prediction files.

Those objects are then reused here without modification.

Outputs
-------
results/analysis18_section7_direct/cells/
results/analysis18_section7_direct/predictions/

results/analysis18_fraction_sweep_direct_perseed.csv
results/analysis18_fraction_sweep_direct_primary3.csv
results/analysis18_fraction_sweep_direct_mc10.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from sklearn.model_selection import train_test_split

import analysis18_section7 as a18


STATE_DIR = a18.RESULTS_DIR / "analysis18_section7_direct"
CELL_DIR = STATE_DIR / "cells"
PREDICTION_DIR = STATE_DIR / "predictions"

for directory in (STATE_DIR, CELL_DIR, PREDICTION_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct final direct-score Analysis 18 results "
            "without cross-encoder training."
        )
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Use primary seeds 42, 43, 44 only.",
    )
    parser.add_argument(
        "--datasets",
        default="DBLP,ECOM",
        help="Comma-separated subset of DBLP,ECOM.",
    )
    parser.add_argument(
        "--fractions",
        default="0.50,0.60,0.70,0.80,0.90",
        help="Comma-separated subset of the Analysis 18 fractions.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
def direct_cell_path(
    dataset_name: str,
    fraction: float,
    seed: int,
) -> Path:
    return CELL_DIR / (
        f"direct_{dataset_name}_"
        f"f{a18.fraction_tag(fraction)}_seed{seed}.json"
    )


def direct_prediction_path(
    dataset_name: str,
    fraction: float,
    seed: int,
) -> Path:
    return PREDICTION_DIR / (
        f"direct_{dataset_name}_"
        f"f{a18.fraction_tag(fraction)}_seed{seed}.csv"
    )


# ---------------------------------------------------------------------
# FINAL PAPER METRICS
# ---------------------------------------------------------------------
def zhang_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    declared = int(frame["direct_declared"].sum())
    correct = int((frame["direct_category"] == "correct").sum())
    false_links = int(
        (frame["direct_category"] == "false_link").sum()
    )
    missing_matches = int(
        (frame["direct_category"] == "missing_match").sum()
    )
    n_true_records = int(frame["has_true_match"].sum())

    if declared != correct + false_links:
        raise AssertionError(
            "Declared links do not equal correct + false links."
        )

    if declared <= 0:
        raise ValueError(
            "FLR is undefined because no links were declared."
        )

    if n_true_records <= 0:
        raise ValueError(
            "MMR is undefined because the holdout has no matchable "
            "source records."
        )

    flr = 1.0 - (correct / declared)

    # Zhang's MMR:
    # a matchable source record is correct only when it is linked
    # to one of its valid partners.  Both a wrong declared link and
    # a missing declaration therefore contribute to MMR.
    mmr = 1.0 - (correct / n_true_records)

    # Historical diagnostic only.  This is not Zhang's MMR.
    legacy_psi = missing_matches / n_true_records

    return {
        "FLR": float(flr),
        "MMR": float(mmr),
        "legacy_psi": float(legacy_psi),
        "declared": declared,
        "correct": correct,
        "false_links": false_links,
        "missing_matches": missing_matches,
        "n_true_records": n_true_records,
    }


# ---------------------------------------------------------------------
# ONE DIRECT-SCORE CELL
# ---------------------------------------------------------------------
def run_cell(
    dataset: dict[str, Any],
    fraction: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    output_cell = direct_cell_path(
        dataset["name"],
        fraction,
        seed,
    )
    output_predictions = direct_prediction_path(
        dataset["name"],
        fraction,
        seed,
    )

    if output_cell.exists() and output_predictions.exists():
        print(
            f"    SKIP completed direct result: "
            f"{dataset['name']} f={fraction:.2f} seed={seed}"
        )
        return a18.read_json(output_cell)

    # --------------------------------------------------------------
    # Reconstruct the original outer A-record split exactly.
    # --------------------------------------------------------------
    labelled_ids, holdout_ids = train_test_split(
        dataset["a_ids"],
        train_size=fraction,
        random_state=seed,
    )
    labelled_ids = list(labelled_ids)
    holdout_ids = list(holdout_ids)

    # --------------------------------------------------------------
    # Reconstruct the original supervised pair collection exactly.
    # One retained valid positive per matchable A; all other known
    # valid partners are excluded from negatives.
    # --------------------------------------------------------------
    train_pairs = a18.build_train_pairs(
        labelled_ids,
        dataset["retained"],
        dataset["truth"],
        dataset["top50"],
    )

    labels = [label for _, _, label in train_pairs]
    if len(set(labels)) < 2:
        raise ValueError(
            "Training-pair collection does not contain both classes."
        )

    fit_indices, validation_indices = train_test_split(
        list(range(len(train_pairs))),
        test_size=a18.VALIDATION_FRAC,
        random_state=seed,
        stratify=labels,
    )

    validation_only = [
        train_pairs[index]
        for index in validation_indices
    ]

    # We deliberately reconstruct fit_indices even though this script
    # performs no fitting.  This checks that the calibration partition
    # follows exactly the original Analysis 18 design.
    n_fit_pairs = len(fit_indices)

    # --------------------------------------------------------------
    # Require the original fitted model.
    # --------------------------------------------------------------
    checkpoint = a18.model_path(
        dataset["name"],
        fraction,
        seed,
    )

    if not a18.model_complete(checkpoint):
        raise FileNotFoundError(
            "Required fitted Analysis 18 checkpoint is missing:\n"
            f"  {checkpoint}\n\n"
            "Run scripts/analysis18_section7.py first. "
            "This direct reconstruction never trains a model."
        )

    original_prediction_file = a18.prediction_path(
        dataset["name"],
        fraction,
        seed,
    )
    original_cell_file = a18.cell_path(
        dataset["name"],
        fraction,
        seed,
    )

    if not original_prediction_file.exists():
        raise FileNotFoundError(
            "Required Analysis 18 prediction file is missing:\n"
            f"  {original_prediction_file}"
        )

    if not original_cell_file.exists():
        raise FileNotFoundError(
            "Required Analysis 18 cell file is missing:\n"
            f"  {original_cell_file}"
        )

    print(f"      loading checkpoint: {checkpoint}")

    model = CrossEncoder(
        str(checkpoint),
        max_length=a18.CE_MAX_LEN,
        device=device,
    )

    # --------------------------------------------------------------
    # DIRECT validation score path.
    # --------------------------------------------------------------
    fields = dataset["cfg"]["fields"]

    validation_text = [
        (
            a18.serialise(dataset["df_a"].loc[ida], fields),
            a18.serialise(dataset["df_b"].loc[idb], fields),
        )
        for ida, idb, _ in validation_only
    ]
    validation_labels = [
        label
        for _, _, label in validation_only
    ]

    validation_direct = a18.ce_predict(
        model,
        validation_text,
    )

    direct_threshold, validation_f1 = a18.choose_threshold(
        validation_direct,
        validation_labels,
    )

    # The original holdout file stores sigmoid(direct score).
    # Because sigmoid is strictly monotone:
    #
    # direct_score > t
    #
    # is equivalent to:
    #
    # sigmoid(direct_score) > sigmoid(t)
    #
    # Therefore the original holdout scores can be reclassified
    # exactly without another holdout model pass.
    sigmoid_equivalent = float(
        a18.sigmoid(
            np.asarray(
                [direct_threshold],
                dtype=np.float64,
            )
        )[0]
    )

    original_cell = a18.read_json(original_cell_file)

    predictions = pd.read_csv(
        original_prediction_file,
        dtype={
            "ida": str,
            "top1_idb": str,
            "retained_idb": str,
        },
    )

    observed_ids = predictions["ida"].astype(str).tolist()
    expected_ids = [str(value) for value in holdout_ids]

    if observed_ids != expected_ids:
        raise AssertionError(
            "Saved Analysis 18 prediction rows do not match the "
            "deterministic outer holdout."
        )

    if len(predictions) != len(holdout_ids):
        raise AssertionError(
            "Saved Analysis 18 prediction count does not match "
            "the deterministic holdout size."
        )

    predictions["has_true_match"] = [
        bool(dataset["truth"].get(str(ida), set()))
        for ida in predictions["ida"]
    ]

    predictions["legacy_sigmoid_threshold"] = float(
        original_cell["threshold"]
    )
    predictions["direct_threshold"] = float(
        direct_threshold
    )
    predictions[
        "direct_threshold_sigmoid_equivalent"
    ] = sigmoid_equivalent

    predictions["direct_declared"] = (
        predictions["top1_score"].astype(float)
        > sigmoid_equivalent
    )

    direct_categories: list[str] = []

    for row in predictions.itertuples(index=False):
        ida = str(row.ida)
        top1_idb = str(row.top1_idb)
        true_set = dataset["truth"].get(ida, set())

        direct_categories.append(
            a18.category_for(
                bool(row.direct_declared),
                top1_idb,
                true_set,
            )
        )

    predictions["direct_category"] = direct_categories

    metrics = zhang_metrics(predictions)

    a18.atomic_write_csv(
        predictions,
        output_predictions,
    )

    result: dict[str, Any] = {
        "analysis": "analysis18_section7_direct",
        "dataset": dataset["name"],
        "fraction": float(fraction),
        "seed": int(seed),
        "score_path": "CrossEncoder.predict",
        "training_performed": False,
        "threshold": float(direct_threshold),
        "threshold_sigmoid_equivalent": sigmoid_equivalent,
        "validation_f1": float(validation_f1),
        "n_labelled_a": len(labelled_ids),
        "n_holdout": len(holdout_ids),
        "n_train_pairs_total": len(train_pairs),
        "n_fit_pairs": n_fit_pairs,
        "n_validation_pairs": len(validation_only),
        "FLR": metrics["FLR"],
        "MMR": metrics["MMR"],
        "legacy_psi": metrics["legacy_psi"],
        "declared": metrics["declared"],
        "correct": metrics["correct"],
        "false_links": metrics["false_links"],
        "missing_matches": metrics["missing_matches"],
        "n_true_records": metrics["n_true_records"],
        "model_path": str(checkpoint),
        "source_cell_path": str(original_cell_file),
        "source_prediction_path": str(
            original_prediction_file
        ),
        "prediction_path": str(output_predictions),
    }

    a18.atomic_write_json(
        output_cell,
        result,
    )

    print(
        f"      direct threshold={direct_threshold:.2f}; "
        f"sigmoid-equivalent={sigmoid_equivalent:.6f}; "
        f"validation F1={validation_f1:.4f}"
    )
    print(
        f"      result: "
        f"FLR={metrics['FLR']:.6f}; "
        f"Zhang MMR={metrics['MMR']:.6f}; "
        f"legacy psi={metrics['legacy_psi']:.6f}; "
        f"declared={metrics['declared']}; "
        f"false={metrics['false_links']}; "
        f"missing={metrics['missing_matches']}"
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------
# COLLECTION AND AGGREGATION
# ---------------------------------------------------------------------
def load_direct_cells() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for path in sorted(CELL_DIR.glob("direct_*_seed*.json")):
        rows.append(a18.read_json(path))

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["dataset", "fraction", "seed"]
        )
        .reset_index(drop=True)
    )


def aggregate(
    frame: pd.DataFrame,
    seeds: list[int],
    label: str,
    require_complete: bool,
) -> pd.DataFrame:
    subset = frame[
        frame["seed"].astype(int).isin(seeds)
    ].copy()

    if subset.empty:
        return pd.DataFrame()

    output = (
        subset.groupby(
            ["dataset", "fraction"],
            sort=True,
        )
        .agg(
            n_splits=("seed", "nunique"),
            FLR_mean=("FLR", "mean"),
            FLR_sd=("FLR", "std"),
            MMR_mean=("MMR", "mean"),
            MMR_sd=("MMR", "std"),
            legacy_psi_mean=("legacy_psi", "mean"),
            declared_mean=("declared", "mean"),
            correct_mean=("correct", "mean"),
            false_links_mean=("false_links", "mean"),
            missing_matches_mean=("missing_matches", "mean"),
            n_true_records_mean=("n_true_records", "mean"),
        )
        .reset_index()
    )

    output["FLR_se"] = (
        output["FLR_sd"]
        / np.sqrt(output["n_splits"])
    )
    output["MMR_se"] = (
        output["MMR_sd"]
        / np.sqrt(output["n_splits"])
    )

    output.insert(0, "aggregation", label)

    if require_complete:
        output = output[
            output["n_splits"] == len(seeds)
        ].copy()

    return output


def save_outputs() -> None:
    frame = load_direct_cells()

    if frame.empty:
        print("No direct-score cells are available.")
        return

    perseed_path = (
        a18.RESULTS_DIR
        / "analysis18_fraction_sweep_direct_perseed.csv"
    )
    primary_path = (
        a18.RESULTS_DIR
        / "analysis18_fraction_sweep_direct_primary3.csv"
    )
    mc10_path = (
        a18.RESULTS_DIR
        / "analysis18_fraction_sweep_direct_mc10.csv"
    )

    a18.atomic_write_csv(
        frame,
        perseed_path,
    )

    primary = aggregate(
        frame,
        list(a18.PRIMARY_SEEDS),
        "primary_3_seeds",
        require_complete=True,
    )

    if not primary.empty:
        a18.atomic_write_csv(
            primary,
            primary_path,
        )

    mc10 = aggregate(
        frame,
        list(a18.ALL_SEEDS),
        "all_10_seeds",
        require_complete=True,
    )

    expected_mc10_groups = (
        len(a18.CONFIGS)
        * len(a18.FRACTIONS)
    )

    if len(mc10) == expected_mc10_groups:
        a18.atomic_write_csv(
            mc10,
            mc10_path,
        )
    elif mc10_path.exists():
        mc10_path.unlink()

    print("\n" + "=" * 78)
    print("DIRECT-SCORE SUMMARY")
    print("=" * 78)

    if not primary.empty:
        print("\nPrimary three seeds:")
        print(
            primary[
                [
                    "dataset",
                    "fraction",
                    "FLR_mean",
                    "MMR_mean",
                ]
            ].to_string(index=False)
        )

    if not mc10.empty:
        print("\nComplete ten-seed groups:")
        print(
            mc10[
                [
                    "dataset",
                    "fraction",
                    "FLR_mean",
                    "MMR_mean",
                ]
            ].to_string(index=False)
        )

    print("\nSaved:")
    print(f"  {perseed_path}")
    if primary_path.exists():
        print(f"  {primary_path}")
    if mc10_path.exists():
        print(f"  {mc10_path}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    dataset_names = a18.parse_csv_values(
        args.datasets,
        set(a18.CONFIGS),
        "dataset",
    )
    fractions = a18.parse_fractions(
        args.fractions
    )

    seeds = (
        list(a18.PRIMARY_SEEDS)
        if args.primary_only
        else list(a18.ALL_SEEDS)
    )

    print("=" * 78)
    print("Analysis 18 direct-score reconstruction")
    print("=" * 78)
    print("NO CROSS-ENCODER TRAINING")
    print("Validation: direct CrossEncoder.predict()")
    print(
        "Holdout: saved sigmoid scores reclassified using "
        "the mathematically equivalent sigmoid threshold"
    )
    print(
        "Final metrics: Zhang FLR and Zhang MMR"
    )
    print(f"Seeds: {seeds}")
    print(f"Fractions: {fractions}")

    device = a18.require_single_visible_gpu()

    datasets = [
        a18.load_dataset(name)
        for name in dataset_names
    ]

    for dataset in datasets:
        dataset["blocking_fingerprint"] = (
            a18.blocking_fingerprint(dataset)
        )

    # In a normal reproduction this cache already exists because
    # analysis18_section7.py has just run.  The helper is retained
    # so the exact blocking object is attached and validated.
    a18.attach_top50(
        datasets,
        device,
    )

    for dataset in datasets:
        for fraction in fractions:
            for seed in seeds:
                print(
                    f"\n    {dataset['name']} "
                    f"f={fraction:.2f} seed={seed}"
                )
                run_cell(
                    dataset,
                    fraction,
                    seed,
                    device,
                )

    save_outputs()

    print(
        "\nAnalysis 18 direct-score reconstruction complete."
    )


if __name__ == "__main__":
    main()