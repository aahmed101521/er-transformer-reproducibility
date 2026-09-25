"""
analysis22_section332_bias_direct.py
====================================

Final direct-score reconstruction for the Section 4.3.2 nested
subsample bias experiment in the ER Transformer paper.

Purpose
-------
The original Analysis 22 fitted the required cross-encoder checkpoints
using the intended training design, but its evaluation path applied an
additional sigmoid to CrossEncoder.predict() before threshold
calibration.

The final manuscript uses the direct CrossEncoder.predict() score path.

This script therefore:

1. NEVER trains a cross-encoder.
2. Reuses the fitted Analysis 22 checkpoints.
3. Reconstructs the deterministic outer and inner source-record splits.
4. Reconstructs the exact pair-level 80/20 calibration split.
5. Calibrates thresholds on direct CrossEncoder.predict() scores.
6. Reclassifies the saved holdout predictions using the mathematically
   equivalent sigmoid-space threshold.
7. Uses the corrected direct-score Analysis 18 result as the outer
   target.
8. Calculates final FLR and MMR using the definitions in the paper:

       FLR = 1 - correct / declared
       MMR = 1 - correct / n_true_records

The historical missing-only quantity

       missing_matches / n_true_records

is retained only as legacy_psi for auditability. It is not MMR.

No model fitting occurs anywhere in this script.

Prerequisites
-------------
The following must have been run first in a full reproduction:

    analysis18_section7.py
    analysis18_section7_direct.py
    analysis22_section332_bias.py

They generate the blocking cache, fitted checkpoints, and saved
prediction files reused here.

Outputs
-------
results/analysis22_section332_bias_direct/cells/
results/analysis22_section332_bias_direct/predictions/

results/analysis22_section332_bias_direct_perseed.csv
results/analysis22_section332_bias_direct_primary3.csv
results/analysis22_section332_bias_direct_mc10.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder

import analysis22_section332_bias as a22


ANALYSIS_NAME = "analysis22_section332_bias_direct"

STATE_DIR = a22.RESULTS_DIR / ANALYSIS_NAME
CELL_DIR = STATE_DIR / "cells"
PREDICTION_DIR = STATE_DIR / "predictions"

ANALYSIS18_DIRECT_STATE_DIR = (
    a22.RESULTS_DIR / "analysis18_section7_direct"
)
ANALYSIS18_DIRECT_CELL_DIR = (
    ANALYSIS18_DIRECT_STATE_DIR / "cells"
)

for directory in (
    STATE_DIR,
    CELL_DIR,
    PREDICTION_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


PRIMARY_SEEDS = [42, 43, 44]


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct final direct-score Analysis 22 bias results "
            "without cross-encoder training."
        )
    )
    parser.add_argument(
        "--phase",
        choices=["run", "aggregate"],
        default="run",
        help=(
            "run: reconstruct requested cells and aggregate; "
            "aggregate: rebuild summaries from completed direct cells only."
        ),
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Use primary seeds 42, 43, and 44 only.",
    )
    parser.add_argument(
        "--datasets",
        default="DBLP,ECOM",
        help="Comma-separated subset of DBLP,ECOM.",
    )
    parser.add_argument(
        "--fractions",
        default=",".join(
            f"{value:.2f}"
            for value in a22.FRACTIONS
        ),
        help="Comma-separated subset of 0.50,0.60,0.70,0.80,0.90.",
    )
    parser.add_argument(
        "--schemes",
        default=",".join(a22.SCHEMES),
        help=(
            "Comma-separated subset of "
            "SRSWOR_65_35,bootstrap_OOB."
        ),
    )
    return parser.parse_args()


# =============================================================================
# PATHS
# =============================================================================
def analysis18_direct_cell_path(
    dataset_name: str,
    fraction: float,
    seed: int,
) -> Path:
    return ANALYSIS18_DIRECT_CELL_DIR / (
        f"direct_{dataset_name}_"
        f"f{a22.fraction_tag(fraction)}_seed{seed}.json"
    )


def direct_cell_path(
    dataset_name: str,
    fraction: float,
    seed: int,
    scheme: str,
) -> Path:
    return CELL_DIR / (
        f"direct_{dataset_name}_"
        f"f{a22.fraction_tag(fraction)}_"
        f"seed{seed}_{scheme}.json"
    )


def direct_prediction_path(
    dataset_name: str,
    fraction: float,
    seed: int,
    scheme: str,
) -> Path:
    return PREDICTION_DIR / (
        f"direct_{dataset_name}_"
        f"f{a22.fraction_tag(fraction)}_"
        f"seed{seed}_{scheme}.csv"
    )


# =============================================================================
# FINAL ZHANG METRICS
# =============================================================================
def zhang_metrics(
    frame: pd.DataFrame,
) -> dict[str, float | int]:
    required = {
        "direct_declared",
        "direct_category",
        "has_true_match",
    }
    missing = required.difference(frame.columns)

    if missing:
        raise KeyError(
            f"Direct prediction frame missing columns: "
            f"{sorted(missing)}"
        )

    declared_values = a22.to_bool_series(
        frame["direct_declared"]
    )
    truth_values = a22.to_bool_series(
        frame["has_true_match"]
    )

    declared = int(declared_values.sum())
    correct = int(
        (frame["direct_category"] == "correct").sum()
    )
    false_links = int(
        (frame["direct_category"] == "false_link").sum()
    )
    missing_matches = int(
        (frame["direct_category"] == "missing_match").sum()
    )
    unmatched_abstentions = int(
        (
            frame["direct_category"]
            == "unmatched_abstention"
        ).sum()
    )
    n_true_records = int(truth_values.sum())

    if declared != correct + false_links:
        raise AssertionError(
            "Declared links do not equal correct + false links."
        )

    if len(frame) != (
        correct
        + false_links
        + missing_matches
        + unmatched_abstentions
    ):
        raise AssertionError(
            "Direct prediction categories are not exhaustive."
        )

    if declared <= 0:
        raise ValueError(
            "FLR is undefined because no links were declared."
        )

    if n_true_records <= 0:
        raise ValueError(
            "MMR is undefined because no evaluated source records "
            "are matchable."
        )

    # Final paper / Zhang definitions.
    flr = 1.0 - (correct / declared)
    mmr = 1.0 - (correct / n_true_records)

    # Historical diagnostic only.
    legacy_psi = missing_matches / n_true_records

    return {
        "FLR": float(flr),
        "MMR": float(mmr),
        "legacy_psi": float(legacy_psi),
        "declared": declared,
        "correct": correct,
        "false_links": false_links,
        "missing_matches": missing_matches,
        "unmatched_abstentions": unmatched_abstentions,
        "n_true_records": n_true_records,
    }


# =============================================================================
# TARGET
# =============================================================================
def load_direct_target(
    dataset_name: str,
    fraction: float,
    seed: int,
) -> dict[str, Any]:
    path = analysis18_direct_cell_path(
        dataset_name,
        fraction,
        seed,
    )

    if not path.exists():
        raise FileNotFoundError(
            "Required direct-score Analysis 18 target is missing:\n"
            f"  {path}\n\n"
            "Run scripts/analysis18_section7.py followed by "
            "scripts/analysis18_section7_direct.py first."
        )

    target = a22.read_json(path)

    if target.get("dataset") != dataset_name:
        raise AssertionError(
            f"Analysis 18 direct target dataset mismatch: {path}"
        )

    if not math.isclose(
        float(target.get("fraction", -1.0)),
        float(fraction),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError(
            f"Analysis 18 direct target fraction mismatch: {path}"
        )

    if int(target.get("seed", -1)) != int(seed):
        raise AssertionError(
            f"Analysis 18 direct target seed mismatch: {path}"
        )

    for field in ("FLR", "MMR"):
        if field not in target:
            raise KeyError(
                f"Analysis 18 direct target lacks {field}: {path}"
            )

    return target


# =============================================================================
# CHECKPOINT AUDIT
# =============================================================================
def require_original_checkpoint(
    dataset_name: str,
    fraction: float,
    seed: int,
    scheme: str,
) -> Path:
    path = a22.model_path(
        "inner",
        dataset_name,
        fraction,
        seed,
        scheme,
    )

    marker_path = a22.model_marker(path)

    if not path.is_dir() or not marker_path.exists():
        raise FileNotFoundError(
            "Required fitted Analysis 22 checkpoint is missing:\n"
            f"  {path}\n\n"
            "Run scripts/analysis22_section332_bias.py first. "
            "This direct reconstruction never trains a model."
        )

    marker = a22.read_json(marker_path)

    expected = {
        "analysis": a22.ANALYSIS_NAME,
        "role": "inner",
        "scheme": scheme,
        "dataset": dataset_name,
        "fraction": fraction,
        "seed": seed,
        "model": a22.CE_MODEL_NAME,
        "score_path": a22.SCORE_PATH,
    }

    mismatches: list[str] = []

    for key, expected_value in expected.items():
        observed = marker.get(key)

        if key == "fraction":
            if not math.isclose(
                float(observed),
                float(expected_value),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                mismatches.append(
                    f"{key}: observed={observed!r}, "
                    f"expected={expected_value!r}"
                )
        elif observed != expected_value:
            mismatches.append(
                f"{key}: observed={observed!r}, "
                f"expected={expected_value!r}"
            )

    if mismatches:
        raise AssertionError(
            "Analysis 22 checkpoint marker is incompatible:\n"
            f"  {marker_path}\n  - "
            + "\n  - ".join(mismatches)
        )

    return path


# =============================================================================
# ONE DIRECT-SCORE INNER CELL
# =============================================================================
def run_cell(
    dataset: dict[str, Any],
    fraction: float,
    seed: int,
    scheme: str,
    device: str,
) -> dict[str, Any]:
    output_cell = direct_cell_path(
        dataset["name"],
        fraction,
        seed,
        scheme,
    )
    output_predictions = direct_prediction_path(
        dataset["name"],
        fraction,
        seed,
        scheme,
    )

    if output_cell.exists() and output_predictions.exists():
        print(
            "      SKIP completed direct result: "
            f"{dataset['name']} f={fraction:.2f} "
            f"seed={seed} {scheme}"
        )
        return a22.read_json(output_cell)

    # -----------------------------------------------------------------
    # Corrected direct Analysis 18 outer target.
    # -----------------------------------------------------------------
    target = load_direct_target(
        dataset["name"],
        fraction,
        seed,
    )

    # -----------------------------------------------------------------
    # Reconstruct deterministic outer split exactly.
    # -----------------------------------------------------------------
    labelled_ids, holdout_ids = a22.make_outer_split(
        dataset,
        fraction,
        seed,
    )

    if "n_labelled_a" in target:
        if int(target["n_labelled_a"]) != len(labelled_ids):
            raise AssertionError(
                "Direct Analysis 18 target labelled size disagrees "
                "with deterministic outer split."
            )

    if "n_holdout" in target:
        if int(target["n_holdout"]) != len(holdout_ids):
            raise AssertionError(
                "Direct Analysis 18 target holdout size disagrees "
                "with deterministic outer split."
            )

    # -----------------------------------------------------------------
    # Reconstruct the Analysis 22 inner sampling scheme exactly.
    # -----------------------------------------------------------------
    train_draw, inner_test_ids, _ = a22.make_inner_split(
        labelled_ids,
        scheme,
        seed,
    )

    # -----------------------------------------------------------------
    # Reconstruct exact supervised pair collection and 80/20
    # calibration partition.
    # -----------------------------------------------------------------
    (
        train_pairs,
        fit_pairs,
        validation_pairs,
    ) = a22.build_pair_split(
        dataset,
        train_draw,
        seed,
    )

    # -----------------------------------------------------------------
    # Require the already-fitted Analysis 22 model.
    # NO TRAINING is permitted here.
    # -----------------------------------------------------------------
    checkpoint = require_original_checkpoint(
        dataset["name"],
        fraction,
        seed,
        scheme,
    )

    original_prediction_file = (
        a22.inner_prediction_path(
            dataset["name"],
            fraction,
            seed,
            scheme,
        )
    )

    original_cell_file = a22.inner_cell_path(
        dataset["name"],
        fraction,
        seed,
        scheme,
    )

    if not original_prediction_file.exists():
        raise FileNotFoundError(
            "Required Analysis 22 prediction file is missing:\n"
            f"  {original_prediction_file}"
        )

    if not original_cell_file.exists():
        raise FileNotFoundError(
            "Required Analysis 22 cell file is missing:\n"
            f"  {original_cell_file}"
        )

    print(f"      loading checkpoint: {checkpoint}")

    model = CrossEncoder(
        str(checkpoint),
        max_length=a22.CE_MAX_LEN,
        device=device,
    )

    # -----------------------------------------------------------------
    # Direct-score validation.
    # -----------------------------------------------------------------
    fields = dataset["cfg"]["fields"]

    validation_text = [
        (
            a22.serialise(
                dataset["df_a"].loc[ida],
                fields,
            ),
            a22.serialise(
                dataset["df_b"].loc[idb],
                fields,
            ),
        )
        for ida, idb, _ in validation_pairs
    ]

    validation_labels = [
        label
        for _, _, label in validation_pairs
    ]

    validation_direct_scores = a22.ce_predict(
        model,
        validation_text,
    )

    direct_threshold, validation_f1 = (
        a22.choose_threshold(
            validation_direct_scores,
            validation_labels,
        )
    )

    # The saved Analysis 22 holdout top1 scores are
    # sigmoid(CrossEncoder.predict()).
    #
    # Since sigmoid is strictly monotone:
    #
    #     raw_score > t
    #
    # iff
    #
    #     sigmoid(raw_score) > sigmoid(t)
    #
    # we can reclassify the already-saved holdout predictions exactly
    # without another model pass over the inner test set.
    sigmoid_equivalent = float(
        a22.sigmoid(
            np.asarray(
                [direct_threshold],
                dtype=np.float64,
            )
        )[0]
    )

    original_cell = a22.read_json(
        original_cell_file
    )

    predictions = pd.read_csv(
        original_prediction_file,
        dtype={
            "ida": str,
            "top1_idb": str,
            "retained_idb": str,
        },
    )

    observed_ids = (
        predictions["ida"]
        .astype(str)
        .tolist()
    )
    expected_ids = [
        str(value)
        for value in inner_test_ids
    ]

    if observed_ids != expected_ids:
        raise AssertionError(
            "Saved Analysis 22 prediction rows do not match "
            "the deterministic inner test split."
        )

    if len(predictions) != len(inner_test_ids):
        raise AssertionError(
            "Saved Analysis 22 prediction count does not match "
            "the deterministic inner test size."
        )

    # Reconstruct truth status directly from the complete mapping.
    predictions["has_true_match"] = [
        bool(dataset["truth"].get(str(ida), set()))
        for ida in predictions["ida"]
    ]

    # Preserve the historical sigmoid-path decision for auditability.
    predictions["legacy_threshold"] = predictions[
        "threshold"
    ]
    predictions["legacy_declared"] = predictions[
        "declared"
    ]
    predictions["legacy_category"] = predictions[
        "category"
    ]

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
        true_set = dataset["truth"].get(
            ida,
            set(),
        )

        direct_categories.append(
            a22.category_for(
                bool(row.direct_declared),
                top1_idb,
                true_set,
            )
        )

    predictions["direct_category"] = (
        direct_categories
    )

    metrics = zhang_metrics(predictions)

    a22.atomic_write_csv(
        predictions,
        output_predictions,
    )

    target_flr = float(target["FLR"])
    target_mmr = float(target["MMR"])

    inner_flr = float(metrics["FLR"])
    inner_mmr = float(metrics["MMR"])

    result: dict[str, Any] = {
        "analysis": ANALYSIS_NAME,
        "role": "inner",
        "dataset": dataset["name"],
        "dataset_display": dataset["display_name"],
        "fraction": float(fraction),
        "seed": int(seed),
        "scheme": scheme,
        "score_path": "CrossEncoder.predict",
        "training_performed": False,

        "target_FLR": target_flr,
        "inner_FLR": inner_flr,
        "bias_FLR": inner_flr - target_flr,

        "target_MMR": target_mmr,
        "inner_MMR": inner_mmr,
        "bias_MMR": inner_mmr - target_mmr,

        "inner_legacy_psi": float(
            metrics["legacy_psi"]
        ),
        "target_legacy_psi": float(
            target.get("legacy_psi", float("nan"))
        ),

        "outer_labelled_size": len(labelled_ids),
        "outer_holdout_size": len(holdout_ids),

        "inner_train_draw_size": len(train_draw),
        "inner_train_unique_size": len(
            set(train_draw)
        ),
        "inner_test_size": len(inner_test_ids),

        "n_train_pairs_total": len(train_pairs),
        "n_fit_pairs": len(fit_pairs),
        "n_validation_pairs": len(
            validation_pairs
        ),

        "direct_threshold": float(
            direct_threshold
        ),
        "direct_threshold_sigmoid_equivalent": (
            sigmoid_equivalent
        ),
        "validation_f1": float(
            validation_f1
        ),

        "inner_declared": metrics["declared"],
        "inner_correct": metrics["correct"],
        "inner_false_links": metrics[
            "false_links"
        ],
        "inner_missing_matches": metrics[
            "missing_matches"
        ],
        "inner_unmatched_abstentions": metrics[
            "unmatched_abstentions"
        ],
        "inner_n_true_records": metrics[
            "n_true_records"
        ],

        "model_path": str(checkpoint),
        "source_analysis22_cell": str(
            original_cell_file
        ),
        "source_analysis22_prediction": str(
            original_prediction_file
        ),
        "source_analysis18_direct_cell": str(
            analysis18_direct_cell_path(
                dataset["name"],
                fraction,
                seed,
            )
        ),
        "prediction_path": str(
            output_predictions
        ),

        "legacy_analysis22_threshold": float(
            original_cell.get(
                "threshold",
                float("nan"),
            )
        ),
        "created_at": a22.ts(),
    }

    a22.atomic_write_json(
        output_cell,
        result,
    )

    print(
        f"      direct threshold={direct_threshold:.2f}; "
        f"sigmoid-equivalent={sigmoid_equivalent:.6f}; "
        f"validation F1={validation_f1:.4f}"
    )
    print(
        f"      target: "
        f"FLR={target_flr:.6f}; "
        f"MMR={target_mmr:.6f}"
    )
    print(
        f"      inner:  "
        f"FLR={inner_flr:.6f}; "
        f"MMR={inner_mmr:.6f}"
    )
    print(
        f"      bias:   "
        f"FLR={inner_flr - target_flr:+.6f}; "
        f"MMR={inner_mmr - target_mmr:+.6f}"
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# =============================================================================
# AGGREGATION
# =============================================================================
def load_direct_frame() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for path in sorted(
        CELL_DIR.glob("direct_*.json")
    ):
        payload = a22.read_json(path)

        required = {
            "dataset",
            "fraction",
            "seed",
            "scheme",
            "target_FLR",
            "inner_FLR",
            "bias_FLR",
            "target_MMR",
            "inner_MMR",
            "bias_MMR",
        }

        if required.issubset(payload):
            rows.append(payload)

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    frame["fraction"] = (
        frame["fraction"]
        .astype(float)
        .round(2)
    )
    frame["seed"] = frame["seed"].astype(int)

    return (
        frame.sort_values(
            [
                "dataset",
                "fraction",
                "scheme",
                "seed",
            ]
        )
        .reset_index(drop=True)
    )


def mc_se(
    values: pd.Series,
) -> float:
    clean = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if len(clean) <= 1:
        return float("nan")

    return float(
        clean.std(ddof=1)
        / math.sqrt(len(clean))
    )


def aggregate_bias(
    frame: pd.DataFrame,
    seeds: list[int],
    label: str,
    require_complete: bool,
) -> pd.DataFrame:
    subset = frame[
        frame["seed"].astype(int).isin(seeds)
    ].copy()

    columns = [
        "aggregation",
        "dataset",
        "f",
        "scheme",
        "target_FLR_mean",
        "inner_FLR_mean",
        "bias_FLR_mean",
        "bias_FLR_SE",
        "target_MMR_mean",
        "inner_MMR_mean",
        "bias_MMR_mean",
        "bias_MMR_SE",
        "n_splits",
    ]

    if subset.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []

    for (
        dataset_name,
        fraction,
        scheme,
    ), group in subset.groupby(
        [
            "dataset",
            "fraction",
            "scheme",
        ],
        sort=True,
    ):
        n_splits = int(
            group["seed"].nunique()
        )

        if require_complete and (
            n_splits != len(seeds)
        ):
            continue

        rows.append(
            {
                "aggregation": label,
                "dataset": dataset_name,
                "f": float(fraction),
                "scheme": scheme,

                "target_FLR_mean": float(
                    group["target_FLR"].mean()
                ),
                "inner_FLR_mean": float(
                    group["inner_FLR"].mean()
                ),
                "bias_FLR_mean": float(
                    group["bias_FLR"].mean()
                ),
                "bias_FLR_SE": mc_se(
                    group["bias_FLR"]
                ),

                "target_MMR_mean": float(
                    group["target_MMR"].mean()
                ),
                "inner_MMR_mean": float(
                    group["inner_MMR"].mean()
                ),
                "bias_MMR_mean": float(
                    group["bias_MMR"].mean()
                ),
                "bias_MMR_SE": mc_se(
                    group["bias_MMR"]
                ),

                "n_splits": n_splits,
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)

    return (
        pd.DataFrame(
            rows,
            columns=columns,
        )
        .sort_values(
            [
                "dataset",
                "f",
                "scheme",
            ]
        )
        .reset_index(drop=True)
    )


def save_outputs() -> None:
    frame = load_direct_frame()

    if frame.empty:
        print(
            "No completed Analysis 22 direct cells "
            "are available."
        )
        return

    perseed_path = (
        a22.RESULTS_DIR
        / "analysis22_section332_bias_direct_perseed.csv"
    )

    primary_path = (
        a22.RESULTS_DIR
        / "analysis22_section332_bias_direct_primary3.csv"
    )

    mc10_path = (
        a22.RESULTS_DIR
        / "analysis22_section332_bias_direct_mc10.csv"
    )

    a22.atomic_write_csv(
        frame,
        perseed_path,
    )

    primary = aggregate_bias(
        frame,
        PRIMARY_SEEDS,
        "primary_3_seeds",
        require_complete=True,
    )

    if not primary.empty:
        a22.atomic_write_csv(
            primary,
            primary_path,
        )

    mc10 = aggregate_bias(
        frame,
        list(a22.SEEDS),
        "all_10_seeds",
        require_complete=True,
    )

    expected_groups = (
        len(a22.CONFIGS)
        * len(a22.FRACTIONS)
        * len(a22.SCHEMES)
    )

    if len(mc10) == expected_groups:
        a22.atomic_write_csv(
            mc10,
            mc10_path,
        )
    elif mc10_path.exists():
        mc10_path.unlink()

    print("\n" + "=" * 100)
    print("ANALYSIS 22 DIRECT-SCORE SUMMARY")
    print("=" * 100)

    if not primary.empty:
        print("\nPrimary three seeds:")
        print(
            primary[
                [
                    "dataset",
                    "f",
                    "scheme",
                    "target_FLR_mean",
                    "bias_FLR_mean",
                    "target_MMR_mean",
                    "bias_MMR_mean",
                ]
            ].to_string(index=False)
        )

    if not mc10.empty:
        print("\nComplete ten-seed groups:")
        print(
            mc10[
                [
                    "dataset",
                    "f",
                    "scheme",
                    "target_FLR_mean",
                    "bias_FLR_mean",
                    "bias_FLR_SE",
                    "target_MMR_mean",
                    "bias_MMR_mean",
                    "bias_MMR_SE",
                ]
            ].to_string(index=False)
        )

    print("\nSaved:")
    print(f"  {perseed_path}")

    if primary_path.exists():
        print(f"  {primary_path}")

    if mc10_path.exists():
        print(f"  {mc10_path}")


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    args = parse_args()

    if args.phase == "aggregate":
        save_outputs()
        return

    dataset_names = a22.parse_csv_values(
        args.datasets,
        set(a22.CONFIGS),
        "dataset",
    )

    fractions = a22.parse_fractions(
        args.fractions
    )

    schemes = a22.parse_csv_values(
        args.schemes,
        set(a22.SCHEMES),
        "scheme",
    )

    seeds = (
        PRIMARY_SEEDS
        if args.primary_only
        else list(a22.SEEDS)
    )

    print("=" * 100)
    print("Analysis 22 direct-score reconstruction")
    print("=" * 100)
    print("NO CROSS-ENCODER TRAINING")
    print(
        "Target: final direct-score Analysis 18 "
        "with Zhang FLR/MMR"
    )
    print(
        "Inner validation: direct CrossEncoder.predict()"
    )
    print(
        "Inner test: saved sigmoid scores reclassified "
        "with equivalent threshold"
    )
    print(
        "Final metrics: "
        "FLR = 1 - correct/declared; "
        "MMR = 1 - correct/matchable"
    )
    print(f"Seeds: {seeds}")
    print(f"Fractions: {fractions}")
    print(f"Schemes: {schemes}")

    device = a22.require_single_visible_gpu()

    datasets = [
        a22.load_dataset(name)
        for name in dataset_names
    ]

    # The full reproduction reaches this script only after Analysis 18
    # and Analysis 22 have created/validated the fixed blocking cache.
    a22.attach_top50(
        datasets,
        device,
    )

    for dataset in datasets:
        for fraction in fractions:
            for seed in seeds:
                for scheme in schemes:
                    print(
                        f"\n    {dataset['name']} "
                        f"f={fraction:.2f} "
                        f"seed={seed} "
                        f"{scheme}"
                    )

                    run_cell(
                        dataset,
                        fraction,
                        seed,
                        scheme,
                        device,
                    )

    save_outputs()

    print(
        "\nAnalysis 22 direct-score reconstruction complete."
    )


if __name__ == "__main__":
    main()