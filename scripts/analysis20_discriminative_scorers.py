"""
analysis20_discriminative_scorers.py
====================================

Corrected ten-seed rerun of the Section 4.5 discriminative-scorer
comparison (paper Table 3).

Paper-facing table
------------------
Four discriminative scorers are compared on per-field bi-encoder agreement
scores for both datasets and all three block regimes:

    LR, RBF SVM, XGBoost, MLP
    forced-1:1, k=5, k=50
    seeds 42--51

The full rerun also evaluates the string representation. Those rows are not
part of Table 3; they are retained solely to audit the current Section 4.5
claim about an Abt-Buy string-side SVM calibration collapse at k=50.

Protocol source
---------------
The corrected data, feature, blocking, pair-construction and evaluation
functions are imported from scripts/analysis16_clean_2x2.py. Runtime
assertions verify the required protocol before any model is fitted:

- complete raw B;
- Latin-1 decoding and the June CSV round-trip;
- legacy JW behaviour in which pandas NaN becomes the literal string "nan";
- one first-listed retained training positive per matched A record;
- every other valid B partner excluded from negatives;
- any valid B partner accepted as correct at test time;
- mutually exclusive correct / false-link / missing-match categories;
- psi denominator = source records with at least one true partner;
- 50/50 A-level split;
- title-only all-MiniLM-L6-v2 blocking;
- forced-1:1, k=5 and k=50 training/test blocks.

Classifier protocol frozen from analysis5_discriminator_choice.py
-----------------------------------------------------------------
- LR: sklearn LogisticRegression(max_iter=500)
- SVM: sklearn SVC(kernel="rbf", probability=True)
- XGB: 200 trees, max_depth=4, learning_rate=0.1, logloss
- MLP: hidden layers (64, 32), max_iter=300
- shared stratified 80/20 pair-level validation split per
  (dataset, seed, block, representation)
- threshold grid 0.10, 0.15, ..., 0.90
- threshold selected by validation F1 using strict probability > threshold
- F1 is calibration-only and is not a paper result metric.

Resumability
------------
Each (dataset, seed, block, representation, classifier) result is saved as an
atomic JSON cell. Completed cells are skipped on restart. Embeddings must be
reconstructed when a partially completed dataset resumes, but completed
classifier fits are not repeated.

Outputs
-------
results/analysis20_discriminative_scorers_perseed.csv
results/analysis20_discriminative_scorers_available.csv
results/analysis20_discriminative_scorers_primary3.csv
results/analysis20_discriminative_scorers_mc10.csv
results/analysis20_table3_biencoder_mc10.csv
results/analysis20_svm_calibration_audit_perseed.csv
results/analysis20_svm_calibration_audit_mc10.csv
results/analysis20_svm_ecom_k50_focus.csv
results/analysis20_discriminative_scorers_manifest.json
results/analysis20_discriminative_scorers/cells/*.json

Run
---
cd ~/er_paper
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python scripts/analysis20_discriminative_scorers.py \\
  2>&1 | tee results/analysis20_discriminative_scorers.log
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

try:
    import xgboost
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - server dependency check
    raise ImportError(
        "Table 3 requires xgboost. Install it inside the project environment "
        "with: python -m pip install xgboost"
    ) from exc


# =============================================================================
# PATHS AND CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path(
    os.environ.get("ER_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).expanduser()
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
RESULTS_DIR = PROJECT_ROOT / "results"
WORK_DIR = RESULTS_DIR / "analysis20_discriminative_scorers"
CELLS_DIR = WORK_DIR / "cells"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CELLS_DIR.mkdir(parents=True, exist_ok=True)

A16_CANDIDATES = [
    SCRIPTS_DIR / "analysis16_clean_2x2.py",
    SCRIPTS_DIR / "analysis16_clean_2x2_final.py",
]

PRIMARY_SEEDS = [42, 43, 44]
DIAGNOSTIC_SEEDS = [45, 46, 47, 48, 49, 50, 51]
ALL_SEEDS = PRIMARY_SEEDS + DIAGNOSTIC_SEEDS

DATASETS = ["DBLP", "ECOM"]
COLUMNS = ["forced-1:1", "k=5", "k=50"]
SIDES = ["string", "bi-encoder"]
CLASSIFIERS = ["LR", "SVM", "XGB", "MLP"]

VAL_FRAC = 0.20
THRESHOLDS = np.arange(0.10, 0.90 + 1e-9, 0.05)

# Provenance of the current paper sentence. These are means of the old June
# three-seed string-side ECOM k=50 SVM rows, not Table 3 rows.
LEGACY_STRING_ECOM_K50 = {
    "lambda_mean": (0.1702 + 0.1455 + 0.1194) / 3.0,
    "psi_mean": (0.9279 + 0.9131 + 0.8909) / 3.0,
    "thresholds": [0.30, 0.10, 0.10],
}


# =============================================================================
# GENERAL UTILITIES
# =============================================================================
def timestamp() -> str:
    return time.strftime("%H:%M:%S")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    os.replace(temporary, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_token(value: str) -> str:
    return (
        value.replace(":", "_")
        .replace("=", "")
        .replace("-", "_")
        .replace(" ", "_")
    )


def cell_path(
    dataset: str,
    seed: int,
    column: str,
    side: str,
    classifier: str,
) -> Path:
    return CELLS_DIR / (
        f"disc_{dataset}_{safe_token(column)}_{safe_token(side)}_"
        f"{classifier}_seed{seed}.json"
    )


def find_analysis16() -> Path:
    for path in A16_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find the corrected Analysis 16 protocol script. Tried:\n"
        + "\n".join(f"  {path}" for path in A16_CANDIDATES)
    )


def import_analysis16(path: Path):
    spec = importlib.util.spec_from_file_location("analysis16_protocol", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


A16_PATH = find_analysis16()
a16 = import_analysis16(A16_PATH)


# =============================================================================
# PROTOCOL ASSERTIONS
# =============================================================================
def assert_protocol() -> None:
    required_functions = [
        "load_dataset",
        "prepare_embeddings_and_candidates",
        "build_train_pairs",
        "build_test_structure",
        "feature_matrix",
        "evaluate",
        "jw",
    ]
    missing = [name for name in required_functions if not hasattr(a16, name)]
    if missing:
        raise AssertionError(
            f"Analysis 16 protocol source lacks functions: {missing}"
        )

    if getattr(a16, "BI_ENCODER_NAME", None) != "all-MiniLM-L6-v2":
        raise AssertionError("Analysis 16 bi-encoder is not all-MiniLM-L6-v2")
    if list(getattr(a16, "COLUMNS", [])) != COLUMNS:
        raise AssertionError(
            f"Analysis 16 columns differ: {getattr(a16, 'COLUMNS', None)}"
        )
    if dict(getattr(a16, "K_VALUES", {})) != {
        "forced-1:1": 1,
        "k=5": 5,
        "k=50": 50,
    }:
        raise AssertionError("Analysis 16 block-size mapping differs")
    if not math.isclose(float(getattr(a16, "LR_VAL_FRAC", -1)), VAL_FRAC):
        raise AssertionError("Analysis 16 validation fraction differs")
    if not np.allclose(np.asarray(a16.LR_THRESHOLDS), THRESHOLDS):
        raise AssertionError("Analysis 16 threshold grid differs")

    # Executed June JW behaviour: pandas NaN is converted to literal "nan".
    if not math.isclose(float(a16.jw(np.nan, np.nan)), 1.0):
        raise AssertionError(
            "Analysis 16 does not preserve JW('nan','nan') = 1.0"
        )

    reader_source = inspect.getsource(a16.read_csv_robust).lower()
    if "latin-1" not in reader_source:
        raise AssertionError("Analysis 16 is not explicitly decoding Latin-1")

    train_source = inspect.getsource(a16.build_train_pairs)
    if "true_set" not in train_source or "not in true_set" not in train_source:
        raise AssertionError(
            "Could not verify exclusion of all valid alternatives from negatives"
        )

    eval_source = inspect.getsource(a16.evaluate)
    required_eval_phrases = ["top_b in true_set", "missing_matches", "true_records"]
    if not all(phrase in eval_source for phrase in required_eval_phrases):
        raise AssertionError(
            "Could not verify corrected many-match mutually exclusive evaluation"
        )


# =============================================================================
# CLASSIFIERS AND CALIBRATION
# =============================================================================
def make_classifier(name: str, seed: int):
    if name == "LR":
        return LogisticRegression(
            max_iter=500,
            random_state=seed,
        )
    if name == "SVM":
        return SVC(
            kernel="rbf",
            probability=True,
            random_state=seed,
        )
    if name == "XGB":
        return XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=seed,
            verbosity=0,
        )
    if name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            random_state=seed,
        )
    raise ValueError(f"Unknown classifier: {name}")


def calibrate_threshold(
    classifier,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> tuple[float, float]:
    probabilities = classifier.predict_proba(x_validation)[:, 1]
    best_threshold = 0.50
    best_f1 = -1.0

    # Preserve the June strict > threshold rule and first-threshold tie break.
    for threshold in THRESHOLDS:
        current_f1 = f1_score(
            y_validation,
            (probabilities > threshold).astype(int),
            zero_division=0,
        )
        if current_f1 > best_f1:
            best_f1 = float(current_f1)
            best_threshold = float(threshold)

    return best_threshold, best_f1


def run_classifier_cell(
    dataset: dict[str, Any],
    train_pairs: list[tuple[str, str, int]],
    test_structure: list[tuple[str, set[str], list[str]]],
    side: str,
    classifier_name: str,
    seed: int,
    column: str,
) -> dict[str, Any]:
    x_train = a16.feature_matrix(
        dataset,
        [(ida, idb) for ida, idb, _ in train_pairs],
        side,
    )
    y_train = np.asarray([label for _, _, label in train_pairs], dtype=int)

    if set(np.unique(y_train)) != {0, 1}:
        raise ValueError(
            f"{dataset['name']} seed={seed} column={column} side={side}: "
            "training data lacks both classes"
        )

    x_fit, x_validation, y_fit, y_validation = train_test_split(
        x_train,
        y_train,
        test_size=VAL_FRAC,
        random_state=seed,
        stratify=y_train,
    )

    test_feature_map: dict[str, np.ndarray] = {}
    for ida, _true_set, candidates in test_structure:
        test_feature_map[ida] = a16.feature_matrix(
            dataset,
            [(ida, idb) for idb in candidates],
            side,
        )

    classifier = make_classifier(classifier_name, seed)
    start = time.time()
    caught_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        classifier.fit(x_fit, y_fit)
        for record in records:
            if issubclass(record.category, ConvergenceWarning) or record.category:
                caught_warnings.append(
                    f"{record.category.__name__}: {record.message}"
                )

    threshold, validation_f1 = calibrate_threshold(
        classifier,
        x_validation,
        y_validation,
    )

    def score_fn(ida: str, _candidates: list[str]) -> np.ndarray:
        return classifier.predict_proba(test_feature_map[ida])[:, 1]

    result = a16.evaluate(
        test_structure,
        score_fn,
        lambda score: score > threshold,
    )

    n_positive = int((y_train == 1).sum())
    n_negative = int((y_train == 0).sum())
    elapsed = time.time() - start

    payload: dict[str, Any] = {
        "dataset": dataset["name"],
        "seed": seed,
        "seed_role": "primary" if seed in PRIMARY_SEEDS else "diagnostic",
        "column": column,
        "side": side,
        "classifier": classifier_name,
        "lambda": float(result["lambda"]),
        "psi": float(result["psi"]),
        "declared": int(result["declared"]),
        "correct": int(result["correct"]),
        "false_links": int(result["false_links"]),
        "missing_matches": int(result["missing_matches"]),
        "n_true_records": int(result["n_true_records"]),
        "n_unmatched_records": int(result["n_unmatched_records"]),
        "abstained_unmatched": int(result["abstained_unmatched"]),
        "threshold": float(threshold),
        "validation_f1": float(validation_f1),
        "threshold_grid_low": bool(np.isclose(threshold, THRESHOLDS[0])),
        "threshold_grid_high": bool(np.isclose(threshold, THRESHOLDS[-1])),
        "threshold_grid_extreme": bool(
            np.isclose(threshold, THRESHOLDS[0])
            or np.isclose(threshold, THRESHOLDS[-1])
        ),
        "n_train_pairs": int(len(train_pairs)),
        "n_train_positive_pairs": n_positive,
        "n_train_negative_pairs": n_negative,
        "negative_to_positive_ratio": (
            float(n_negative / n_positive) if n_positive else None
        ),
        "n_fit_pairs": int(len(x_fit)),
        "n_validation_pairs": int(len(x_validation)),
        "fit_and_eval_seconds": float(elapsed),
        "warnings": caught_warnings,
        "protocol_source": str(A16_PATH),
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
    }
    return payload


# =============================================================================
# CHECKPOINT DISCOVERY AND AGGREGATION
# =============================================================================
def selected_cell_paths(
    datasets: Iterable[str],
    seeds: Iterable[int],
    columns: Iterable[str],
    sides: Iterable[str],
    classifiers: Iterable[str],
) -> list[Path]:
    return [
        cell_path(dataset, seed, column, side, classifier)
        for dataset in datasets
        for seed in seeds
        for column in columns
        for side in sides
        for classifier in classifiers
    ]


def load_completed_cells() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(CELLS_DIR.glob("disc_*.json")):
        try:
            payload = load_json(path)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"WARNING: ignoring unreadable cell {path}: {exc}")
            continue
        rows.append(payload)
    return pd.DataFrame(rows)


def aggregate_results(
    per_seed: pd.DataFrame,
    seeds: list[int],
    label: str,
) -> pd.DataFrame:
    if per_seed.empty:
        return pd.DataFrame()

    subset = per_seed[per_seed["seed"].isin(seeds)].copy()
    if subset.empty:
        return pd.DataFrame()

    group_columns = ["dataset", "column", "side", "classifier"]
    aggregated = (
        subset.groupby(group_columns)
        .agg(
            n_splits=("seed", "nunique"),
            lambda_mean=("lambda", "mean"),
            lambda_sd=("lambda", "std"),
            lambda_min=("lambda", "min"),
            lambda_max=("lambda", "max"),
            psi_mean=("psi", "mean"),
            psi_sd=("psi", "std"),
            psi_min=("psi", "min"),
            psi_max=("psi", "max"),
            threshold_mean=("threshold", "mean"),
            threshold_sd=("threshold", "std"),
            threshold_min=("threshold", "min"),
            threshold_max=("threshold", "max"),
            validation_f1_mean=("validation_f1", "mean"),
            threshold_grid_low_count=("threshold_grid_low", "sum"),
            threshold_grid_high_count=("threshold_grid_high", "sum"),
            threshold_grid_extreme_count=("threshold_grid_extreme", "sum"),
            declared_mean=("declared", "mean"),
            correct_mean=("correct", "mean"),
            false_links_mean=("false_links", "mean"),
            missing_matches_mean=("missing_matches", "mean"),
            n_true_records_mean=("n_true_records", "mean"),
            n_train_pairs_mean=("n_train_pairs", "mean"),
            negative_to_positive_ratio_mean=(
                "negative_to_positive_ratio",
                "mean",
            ),
            fit_and_eval_seconds_mean=("fit_and_eval_seconds", "mean"),
        )
        .reset_index()
    )

    aggregated["lambda_se"] = (
        aggregated["lambda_sd"] / np.sqrt(aggregated["n_splits"])
    )
    aggregated["psi_se"] = (
        aggregated["psi_sd"] / np.sqrt(aggregated["n_splits"])
    )
    aggregated["threshold_grid_extreme_share"] = (
        aggregated["threshold_grid_extreme_count"]
        / aggregated["n_splits"]
    )
    aggregated.insert(0, "aggregation", label)

    numeric = aggregated.select_dtypes(include=[np.number]).columns
    aggregated[numeric] = aggregated[numeric].round(6)
    return aggregated


def expected_group_keys() -> set[tuple[str, str, str, str]]:
    return {
        (dataset, column, side, classifier)
        for dataset in DATASETS
        for column in COLUMNS
        for side in SIDES
        for classifier in CLASSIFIERS
    }


def reorder(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    ordered = frame.copy()
    ordered["dataset"] = pd.Categorical(
        ordered["dataset"], DATASETS, ordered=True
    )
    ordered["column"] = pd.Categorical(
        ordered["column"], COLUMNS, ordered=True
    )
    ordered["side"] = pd.Categorical(
        ordered["side"], SIDES, ordered=True
    )
    ordered["classifier"] = pd.Categorical(
        ordered["classifier"], CLASSIFIERS, ordered=True
    )
    return ordered.sort_values(
        ["dataset", "column", "side", "classifier"]
    ).reset_index(drop=True)


def print_summary(frame: pd.DataFrame, title: str) -> None:
    print("\n" + "=" * 118)
    print(title)
    print("=" * 118)
    if frame.empty:
        print("No rows available.")
        return
    columns = [
        "dataset",
        "column",
        "side",
        "classifier",
        "n_splits",
        "lambda_mean",
        "lambda_sd",
        "lambda_se",
        "psi_mean",
        "psi_sd",
        "psi_se",
        "threshold_mean",
        "threshold_min",
        "threshold_max",
        "threshold_grid_extreme_count",
    ]
    print(frame[columns].to_string(index=False))


def write_analysis16_lr_crosscheck(per_seed: pd.DataFrame) -> None:
    """Compare overlapping LR rows with an existing corrected Analysis 16 run."""
    source = RESULTS_DIR / "analysis16_clean_2x2_perseed.csv"
    if not source.exists():
        print("Analysis 16 LR cross-check unavailable: no per-seed result file.")
        return

    previous = pd.read_csv(source)
    required = {"dataset", "seed", "column", "side", "scorer", "lambda", "psi"}
    if not required.issubset(previous.columns):
        print(
            "Analysis 16 LR cross-check skipped: unexpected columns "
            f"in {source.name}"
        )
        return

    previous = previous[previous["scorer"].astype(str) == "LR"].copy()
    current = per_seed[per_seed["classifier"].astype(str) == "LR"].copy()
    if previous.empty or current.empty:
        return

    previous = previous.rename(
        columns={
            "lambda": "analysis16_lambda",
            "psi": "analysis16_psi",
            "lr_threshold": "analysis16_threshold",
        }
    )
    current = current.rename(
        columns={
            "lambda": "analysis20_lambda",
            "psi": "analysis20_psi",
            "threshold": "analysis20_threshold",
        }
    )
    keys = ["dataset", "seed", "column", "side"]
    prior_columns = keys + ["analysis16_lambda", "analysis16_psi"]
    if "analysis16_threshold" in previous.columns:
        prior_columns.append("analysis16_threshold")
    current_columns = keys + [
        "analysis20_lambda",
        "analysis20_psi",
        "analysis20_threshold",
    ]
    merged = current[current_columns].merge(
        previous[prior_columns], on=keys, how="inner"
    )
    if merged.empty:
        print("Analysis 16 LR cross-check: no overlapping cells.")
        return

    merged["lambda_delta"] = (
        merged["analysis20_lambda"] - merged["analysis16_lambda"]
    )
    merged["psi_delta"] = (
        merged["analysis20_psi"] - merged["analysis16_psi"]
    )
    if "analysis16_threshold" in merged.columns:
        merged["threshold_delta"] = (
            merged["analysis20_threshold"] - merged["analysis16_threshold"]
        )
    atomic_write_csv(
        RESULTS_DIR / "analysis20_lr_analysis16_crosscheck.csv", merged
    )
    print(
        "Analysis 16 LR cross-check: "
        f"n={len(merged)}, max |lambda delta|="
        f"{merged['lambda_delta'].abs().max():.12g}, max |psi delta|="
        f"{merged['psi_delta'].abs().max():.12g}"
    )


def save_outputs() -> None:
    per_seed = load_completed_cells()
    if per_seed.empty:
        print("No completed Analysis 20 cells yet.")
        return

    per_seed = reorder(per_seed)
    write_analysis16_lr_crosscheck(per_seed)
    available = reorder(
        aggregate_results(
            per_seed,
            sorted(per_seed["seed"].unique().tolist()),
            "available_seeds",
        )
    )
    primary = reorder(
        aggregate_results(per_seed, PRIMARY_SEEDS, "primary_3_seeds")
    )
    mc10_all = reorder(
        aggregate_results(per_seed, ALL_SEEDS, "all_10_seeds")
    )

    atomic_write_csv(
        RESULTS_DIR / "analysis20_discriminative_scorers_perseed.csv",
        per_seed,
    )
    atomic_write_csv(
        RESULTS_DIR / "analysis20_discriminative_scorers_available.csv",
        available,
    )
    atomic_write_csv(
        RESULTS_DIR / "analysis20_discriminative_scorers_primary3.csv",
        primary,
    )

    complete_mc10 = mc10_all[mc10_all["n_splits"] == 10].copy()
    complete_keys = {
        tuple(row)
        for row in complete_mc10[
            ["dataset", "column", "side", "classifier"]
        ].astype(str).itertuples(index=False, name=None)
    }
    all_complete = complete_keys == expected_group_keys()

    mc10_path = RESULTS_DIR / "analysis20_discriminative_scorers_mc10.csv"
    table3_path = RESULTS_DIR / "analysis20_table3_biencoder_mc10.csv"

    if all_complete:
        atomic_write_csv(mc10_path, mc10_all)
        table3 = mc10_all[mc10_all["side"].astype(str) == "bi-encoder"].copy()
        atomic_write_csv(table3_path, table3)
    else:
        missing = sorted(expected_group_keys() - complete_keys)
        print("\nMC10 and Table 3 files not written: incomplete groups remain.")
        for key in missing[:30]:
            print(f"  missing/incomplete: {key}")
        if len(missing) > 30:
            print(f"  ... and {len(missing) - 30} more")

    svm_perseed = per_seed[per_seed["classifier"].astype(str) == "SVM"].copy()
    svm_mc10 = mc10_all[mc10_all["classifier"].astype(str) == "SVM"].copy()
    atomic_write_csv(
        RESULTS_DIR / "analysis20_svm_calibration_audit_perseed.csv",
        svm_perseed,
    )
    atomic_write_csv(
        RESULTS_DIR / "analysis20_svm_calibration_audit_mc10.csv",
        svm_mc10,
    )

    focus = svm_mc10[
        (svm_mc10["dataset"].astype(str) == "ECOM")
        & (svm_mc10["column"].astype(str) == "k=50")
    ].copy()
    if not focus.empty:
        focus["legacy_string_lambda_mean"] = LEGACY_STRING_ECOM_K50[
            "lambda_mean"
        ]
        focus["legacy_string_psi_mean"] = LEGACY_STRING_ECOM_K50["psi_mean"]
        focus["lambda_delta_from_legacy_string"] = (
            focus["lambda_mean"] - LEGACY_STRING_ECOM_K50["lambda_mean"]
        )
        focus["psi_delta_from_legacy_string"] = (
            focus["psi_mean"] - LEGACY_STRING_ECOM_K50["psi_mean"]
        )
        atomic_write_csv(
            RESULTS_DIR / "analysis20_svm_ecom_k50_focus.csv",
            focus,
        )

    manifest = {
        "analysis": "analysis20_discriminative_scorers",
        "protocol_source": str(A16_PATH),
        "analysis16_biencoder": a16.BI_ENCODER_NAME,
        "raw_decoding": "latin-1",
        "june_jw_nan": "pandas NaN converted to literal 'nan'",
        "datasets": DATASETS,
        "columns": COLUMNS,
        "sides": SIDES,
        "paper_table_side": "bi-encoder",
        "classifiers": CLASSIFIERS,
        "seeds": ALL_SEEDS,
        "validation_fraction": VAL_FRAC,
        "threshold_grid": [float(value) for value in THRESHOLDS],
        "threshold_rule": "strict probability > threshold",
        "threshold_objective": "validation F1, calibration only",
        "counting": {
            "correct": "any valid B partner",
            "categories": "mutually exclusive correct/false/missing",
            "psi_denominator": "source records with >=1 true partner",
        },
        "legacy_claim_provenance": {
            "dataset": "ECOM",
            "column": "k=50",
            "side": "string",
            "classifier": "SVM",
            **LEGACY_STRING_ECOM_K50,
        },
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "all_expected_groups_complete": all_complete,
    }
    atomic_write_json(
        RESULTS_DIR / "analysis20_discriminative_scorers_manifest.json",
        manifest,
    )

    print_summary(available, "ANALYSIS 20: AVAILABLE DISCRIMINATIVE-SCORER RESULTS")
    if all_complete:
        print_summary(mc10_all, "ANALYSIS 20: FINAL TEN-SEED RESULTS")

    print("\n" + "=" * 118)
    print("FOCUSED SVM CALIBRATION AUDIT: ECOM k=50")
    print("=" * 118)
    print(
        "Legacy June string-side provenance: "
        f"lambda={LEGACY_STRING_ECOM_K50['lambda_mean']:.4f}, "
        f"psi={LEGACY_STRING_ECOM_K50['psi_mean']:.4f}, "
        f"thresholds={LEGACY_STRING_ECOM_K50['thresholds']}"
    )
    if focus.empty:
        print("No corrected focused rows available yet.")
    else:
        focus_columns = [
            "side",
            "n_splits",
            "lambda_mean",
            "lambda_sd",
            "lambda_se",
            "psi_mean",
            "psi_sd",
            "psi_se",
            "threshold_mean",
            "threshold_min",
            "threshold_max",
            "threshold_grid_low_count",
            "threshold_grid_high_count",
            "threshold_grid_extreme_count",
            "declared_mean",
        ]
        print(focus[focus_columns].to_string(index=False))

    print("\nSaved outputs under:")
    print(RESULTS_DIR)
    if all_complete:
        print(f"Final paper Table 3 source: {table3_path}")
        print(f"Full MC10 source: {mc10_path}")


# =============================================================================
# RUNNER
# =============================================================================
def dataset_is_complete(
    dataset: str,
    seeds: list[int],
    columns: list[str],
    sides: list[str],
    classifiers: list[str],
) -> bool:
    return all(
        path.exists()
        for path in selected_cell_paths(
            [dataset], seeds, columns, sides, classifiers
        )
    )


def run_experiment(
    datasets: list[str],
    seeds: list[int],
    columns: list[str],
    sides: list[str],
    classifiers: list[str],
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None

    total = (
        len(datasets)
        * len(seeds)
        * len(columns)
        * len(sides)
        * len(classifiers)
    )
    ordinal = 0

    for dataset_name in datasets:
        if dataset_is_complete(
            dataset_name, seeds, columns, sides, classifiers
        ):
            print(f"\nSKIP fully completed dataset: {dataset_name}")
            ordinal += (
                len(seeds) * len(columns) * len(sides) * len(classifiers)
            )
            continue

        if model is None:
            print(
                f"\nLoading blocking/feature bi-encoder on {device}: "
                f"{a16.BI_ENCODER_NAME}"
            )
            model = a16.SentenceTransformer(a16.BI_ENCODER_NAME, device=device)

        dataset = a16.load_dataset(dataset_name)
        a16.prepare_embeddings_and_candidates(dataset, model)

        for seed in seeds:
            set_seed(seed)
            ids_train, ids_test = train_test_split(
                dataset["a_ids"],
                test_size=0.50,
                random_state=seed,
            )

            for column in columns:
                train_pairs = a16.build_train_pairs(
                    dataset, ids_train, column
                )
                test_structure = a16.build_test_structure(
                    dataset, ids_test, column
                )

                # Runtime supervision assertions for every split and block.
                positive_by_a: dict[str, int] = {}
                for ida, idb, label in train_pairs:
                    if label == 1:
                        positive_by_a[ida] = positive_by_a.get(ida, 0) + 1
                        if idb != dataset["retained"][ida]:
                            raise AssertionError(
                                "Training positive is not the retained first-listed partner"
                            )
                    elif idb in dataset["truth"].get(ida, set()):
                        raise AssertionError(
                            "A valid alternative entered the negative class"
                        )
                if positive_by_a and max(positive_by_a.values()) != 1:
                    raise AssertionError("Training is not one-positive-per-A")

                for side in sides:
                    for classifier_name in classifiers:
                        ordinal += 1
                        path = cell_path(
                            dataset_name,
                            seed,
                            column,
                            side,
                            classifier_name,
                        )
                        if path.exists():
                            print(
                                f"[{ordinal}/{total}] SKIP completed: "
                                f"{dataset_name} seed={seed} {column} "
                                f"{side} {classifier_name}"
                            )
                            continue

                        print("\n" + "-" * 100)
                        print(
                            f"[{ordinal}/{total}] {dataset_name} seed={seed} "
                            f"{column} {side} {classifier_name} [{timestamp()}]"
                        )
                        print("-" * 100)

                        payload = run_classifier_cell(
                            dataset,
                            train_pairs,
                            test_structure,
                            side,
                            classifier_name,
                            seed,
                            column,
                        )
                        atomic_write_json(path, payload)
                        print(
                            f"lambda={payload['lambda']:.4f}; "
                            f"psi={payload['psi']:.4f}; "
                            f"threshold={payload['threshold']:.2f}; "
                            f"val F1={payload['validation_f1']:.4f}; "
                            f"declared={payload['declared']}; "
                            f"saved={path.name}"
                        )

                save_outputs()

        del dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if model is not None:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# CLI
# =============================================================================
def parse_csv_option(raw: str, allowed: list[str], label: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    invalid = set(values).difference(allowed)
    if invalid:
        raise ValueError(
            f"Unknown {label}: {sorted(invalid)}; allowed={allowed}"
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corrected ten-seed discriminative-scorer comparison."
    )
    parser.add_argument(
        "--phase",
        choices=["all", "run", "aggregate"],
        default="all",
        help="Run and aggregate, run cells only, or aggregate existing cells.",
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Run seeds 42-44 only. Default is all ten seeds 42-51.",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help="Comma-separated subset of DBLP,ECOM.",
    )
    parser.add_argument(
        "--columns",
        default=",".join(COLUMNS),
        help="Comma-separated subset of forced-1:1,k=5,k=50.",
    )
    parser.add_argument(
        "--sides",
        default=",".join(SIDES),
        help="Comma-separated subset of string,bi-encoder.",
    )
    parser.add_argument(
        "--classifiers",
        default=",".join(CLASSIFIERS),
        help="Comma-separated subset of LR,SVM,XGB,MLP.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_protocol()

    datasets = parse_csv_option(args.datasets, DATASETS, "datasets")
    columns = parse_csv_option(args.columns, COLUMNS, "columns")
    sides = parse_csv_option(args.sides, SIDES, "sides")
    classifiers = parse_csv_option(
        args.classifiers, CLASSIFIERS, "classifiers"
    )
    seeds = PRIMARY_SEEDS if args.primary_only else ALL_SEEDS

    print("=" * 100)
    print("Analysis 20: corrected discriminative-scorer comparison")
    print("=" * 100)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Analysis 16 protocol source: {A16_PATH}")
    print(f"Phase: {args.phase}")
    print(f"Datasets: {datasets}")
    print(f"Blocks: {columns}")
    print(f"Representations: {sides}")
    print(f"Classifiers: {classifiers}")
    print(f"Seeds: {seeds}")
    print("Paper Table 3 representation: bi-encoder only")
    print("String representation retained for the SVM-collapse audit")
    print("Alternative valid matches excluded from negatives: asserted")
    print("Mutually exclusive corrected counting: asserted")
    print("Latin-1 + June JW NaN behaviour: asserted")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Visible CUDA devices: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Visible device 0: {torch.cuda.get_device_name(0)}")

    if args.phase != "aggregate" and torch.cuda.device_count() > 1:
        raise RuntimeError(
            "More than one GPU is visible. Pin the run explicitly, e.g. "
            "CUDA_VISIBLE_DEVICES=0 python scripts/analysis20_discriminative_scorers.py"
        )

    if args.phase in ("all", "run"):
        run_experiment(datasets, seeds, columns, sides, classifiers)

    if args.phase in ("all", "aggregate"):
        save_outputs()

    print("\nAnalysis 20 completed for the requested cells.")


if __name__ == "__main__":
    main()
