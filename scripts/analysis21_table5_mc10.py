"""
analysis21_table5_mc10.py
=========================

Fresh, checkpointed ten-seed rerun of paper Table 5:

    pre-trained vs fine-tuned all-MiniLM-L6-v2
    MEC and L2 logistic regression
    k=50 only
    DBLP-Scholar and Abt-Buy
    seeds 42--51

Both Table 5 columns are produced inside this script from the same dataset
objects, A-level splits, corrected truth, candidate construction, scorer code,
and aggregation code.

Protocol frozen from corrected Analyses 16 and 17
-------------------------------------------------
- complete raw B candidate universe;
- Latin-1 decoding;
- lowercase + strip preprocessing;
- February-write / June-read pandas CSV round-trip;
- embeddings fill missing text with "";
- one first-listed reachable B partner as the training positive;
- all alternative valid partners excluded from the negative class;
- any valid B partner accepted as correct at test time;
- mutually exclusive correct / false-link / missing-match counting;
- psi denominator = test A records with at least one true partner;
- 50/50 A-level train/test split;
- title-only top-50 blocking with all-MiniLM-L6-v2;
- per-field cosine agreement features;
- MEC KDE bandwidth 0.05 and fixed log-likelihood-ratio threshold 0;
- LR L2, max_iter=500, 80/20 stratified pair-level calibration split;
- threshold grid 0.10, 0.15, ..., 0.90 selected by validation F1;
- strict probability > threshold declaration rule.

Fine-tuning protocol frozen from Analysis 17 / Analysis 10
----------------------------------------------------------
- all-MiniLM-L6-v2;
- SentenceTransformers ContrastiveLoss;
- one epoch;
- 10% warmup;
- learning rate 2e-5;
- batch size 16;
- training pairs are the pre-trained encoder's k=50 pairs;
- after fine-tuning, A and B are re-embedded and top-50 blocking is rebuilt;
- the two-step MEC/LR scorers are then fitted on the fine-tuned k=50 pairs.

Corrected pre-trained cross-check
---------------------------------
The newly computed pre-trained rows are compared seed-by-seed against
results/analysis16_clean_2x2_perseed.csv. Counts must agree exactly and lambda,
psi and LR thresholds must agree within 1e-10. The script stops if they do not.
This establishes that Table 5's pre-trained column is the same quantity used in
corrected Tables 2 and 6.

Outputs
-------
results/analysis21_table5_perseed.csv
results/analysis21_table5_primary3.csv
results/analysis21_table5_mc10.csv
results/analysis21_table5_pretrained_analysis16_crosscheck.csv
results/analysis21_table5_manifest.json
results/analysis21_table5/cells/*.json
models/analysis21_table5/ft_biencoder/<DATASET>/seed_<SEED>/

Run
---
cd ~/er_paper
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python scripts/analysis21_table5_mc10.py \\
  2>&1 | tee results/analysis21_table5_mc10.log
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


# =============================================================================
# PATHS AND CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path(
    os.environ.get("ER_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).expanduser()
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

STATE_DIR = RESULTS_DIR / "analysis21_table5"
CELL_DIR = STATE_DIR / "cells"
MODEL_ROOT = MODELS_DIR / "analysis21_table5"
for directory in (RESULTS_DIR, MODELS_DIR, STATE_DIR, CELL_DIR, MODEL_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

A17_CANDIDATES = [
    SCRIPTS_DIR / "analysis17_section5.py",
    SCRIPTS_DIR / "analysis17_section5_standalone.py",
]
ANALYSIS16_PERSEED = RESULTS_DIR / "analysis16_clean_2x2_perseed.csv"

PRIMARY_SEEDS = [42, 43, 44]
DIAGNOSTIC_SEEDS = [45, 46, 47, 48, 49, 50, 51]
ALL_SEEDS = PRIMARY_SEEDS + DIAGNOSTIC_SEEDS
DATASETS = ["DBLP", "ECOM"]
ENCODERS = ["pre-trained", "fine-tuned"]
SCORERS = ["LR", "MEC"]

BI_ENCODER_NAME = "all-MiniLM-L6-v2"
K = 50
FT_EPOCHS = 1
FT_WARMUP_FRAC = 0.10
FT_LR = 2e-5
FT_BATCH = 16
CROSSCHECK_TOL = 1e-10


# =============================================================================
# BASIC UTILITIES
# =============================================================================
def ts() -> str:
    return time.strftime("%H:%M:%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_analysis17() -> Path:
    for path in A17_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find Analysis 17 protocol source. Tried:\n"
        + "\n".join(f"  {path}" for path in A17_CANDIDATES)
    )


def import_module(path: Path):
    spec = importlib.util.spec_from_file_location("analysis17_protocol", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


A17_PATH = find_analysis17()
a17 = import_module(A17_PATH)


# =============================================================================
# PROTOCOL ASSERTIONS
# =============================================================================
def assert_protocol() -> None:
    required = [
        "load_dataset",
        "prepare_baseline",
        "build_train_pairs",
        "build_test_structure",
        "cosine_feature_matrix",
        "MECScorer",
        "calibrate_lr",
        "evaluate",
        "encode_fields",
        "retrieve_top50",
        "serialise",
        "save_model_atomic",
        "model_complete",
        "safe_remove",
    ]
    missing = [name for name in required if not hasattr(a17, name)]
    if missing:
        raise AssertionError(f"Analysis 17 lacks required objects: {missing}")

    expected_constants = {
        "BI_ENCODER_NAME": BI_ENCODER_NAME,
        "K": K,
        "FT_EPOCHS": FT_EPOCHS,
        "FT_WARMUP_FRAC": FT_WARMUP_FRAC,
        "FT_LR": FT_LR,
        "FT_BATCH": FT_BATCH,
        "KDE_BANDWIDTH": 0.05,
        "LR_MAX_ITER": 500,
        "LR_VAL_FRAC": 0.20,
    }
    for name, expected in expected_constants.items():
        observed = getattr(a17, name, None)
        if isinstance(expected, float):
            if not math.isclose(float(observed), expected):
                raise AssertionError(
                    f"Protocol constant {name}: observed={observed}, expected={expected}"
                )
        elif observed != expected:
            raise AssertionError(
                f"Protocol constant {name}: observed={observed}, expected={expected}"
            )

    expected_thresholds = np.arange(0.10, 0.90 + 1e-9, 0.05)
    if not np.allclose(np.asarray(a17.THRESHOLDS), expected_thresholds):
        raise AssertionError("Analysis 17 LR threshold grid differs")

    load_source = inspect.getsource(a17.load_dataset)
    train_source = inspect.getsource(a17.build_train_pairs)
    eval_source = inspect.getsource(a17.evaluate)
    ft_source = inspect.getsource(a17.run_ft_cell)

    checks = {
        "complete truth set and retained first match": (
            "truth.setdefault" in load_source and "retained.setdefault" in load_source
        ),
        "alternative exclusions": (
            "idb not in true_set" in train_source
        ),
        "any-valid correctness": (
            "top_idb in true_set" in eval_source
        ),
        "mutually exclusive missing counting": (
            "elif true_set" in eval_source
        ),
        "record denominator": (
            "missing_matches / n_true_records" in eval_source
        ),
        "fine-tuning starts from baseline pairs": (
            "baseline_pairs" in ft_source
            and "train_or_load_ft_model" in ft_source
        ),
        "fine-tuned blocking rebuilt": (
            "ft_top50" in ft_source and "retrieve_top50" in ft_source
        ),
    }
    failed = [label for label, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Could not verify protocol items: {failed}")


# =============================================================================
# CHECKPOINT PATHS
# =============================================================================
def cell_path(dataset: str, seed: int, encoder: str) -> Path:
    token = encoder.replace("-", "_")
    return CELL_DIR / f"table5_{dataset}_{token}_seed{seed}.json"


def ft_model_path(dataset: str, seed: int) -> Path:
    return MODEL_ROOT / "ft_biencoder" / dataset / f"seed_{seed}"


# =============================================================================
# FEATURE AND SCORER HELPERS
# =============================================================================
def feature_matrix(
    dataset: dict[str, Any],
    pairs: list[tuple[str, str]],
    emb_a: dict[str, torch.Tensor],
    emb_b: dict[str, torch.Tensor],
) -> np.ndarray:
    return a17.cosine_feature_matrix(
        pairs,
        emb_a,
        emb_b,
        dataset["cfg"]["fields"],
        dataset["a_pos"],
        dataset["b_pos"],
    )


def fit_and_evaluate_two_step(
    dataset: dict[str, Any],
    train_pairs: list[tuple[str, str, int]],
    test_structure: list[tuple[str, set[str], list[str]]],
    emb_a: dict[str, torch.Tensor],
    emb_b: dict[str, torch.Tensor],
    seed: int,
) -> dict[str, Any]:
    pair_ids = [(ida, idb) for ida, idb, _ in train_pairs]
    x_train = feature_matrix(dataset, pair_ids, emb_a, emb_b)
    y_train = np.asarray([label for _, _, label in train_pairs], dtype=int)

    if set(np.unique(y_train)) != {0, 1}:
        raise ValueError(
            f"{dataset['name']} seed={seed}: training pairs lack both classes"
        )

    # Every matched A contributes exactly one retained positive, while every
    # known valid alternative remains outside the negative class.
    positive_counts: dict[str, int] = {}
    for ida, idb, label in train_pairs:
        if label == 1:
            positive_counts[ida] = positive_counts.get(ida, 0) + 1
            if idb != dataset["retained"][ida]:
                raise AssertionError("Positive is not first-listed retained partner")
        elif idb in dataset["truth"].get(ida, set()):
            raise AssertionError("Alternative valid match entered negative class")
    if positive_counts and max(positive_counts.values()) != 1:
        raise AssertionError("Training contains more than one positive for an A record")

    candidate_feature_cache: dict[str, np.ndarray] = {
        ida: feature_matrix(
            dataset,
            [(ida, idb) for idb in candidates],
            emb_a,
            emb_b,
        )
        for ida, _truth, candidates in test_structure
    }

    mec = a17.MECScorer()
    mec.fit(x_train, y_train)
    mec_metrics = a17.evaluate(
        test_structure,
        lambda ida, _candidates: mec.log_ratio(candidate_feature_cache[ida]),
        lambda score: score > 0.0,
    )

    lr_model, lr_threshold, lr_validation_f1 = a17.calibrate_lr(
        x_train,
        y_train,
        seed,
    )
    lr_metrics = a17.evaluate(
        test_structure,
        lambda ida, _candidates: lr_model.predict_proba(
            candidate_feature_cache[ida]
        )[:, 1],
        lambda score: score > lr_threshold,
    )

    return {
        "n_train_pairs": int(len(train_pairs)),
        "n_train_positive_pairs": int((y_train == 1).sum()),
        "n_train_negative_pairs": int((y_train == 0).sum()),
        "lr_threshold": float(lr_threshold),
        "lr_validation_f1": float(lr_validation_f1),
        "MEC": mec_metrics,
        "LR": lr_metrics,
    }


# =============================================================================
# PRE-TRAINED CELL
# =============================================================================
def run_pretrained_cell(
    dataset: dict[str, Any],
    seed: int,
    ids_train: list[str],
    ids_test: list[str],
) -> dict[str, Any]:
    path = cell_path(dataset["name"], seed, "pre-trained")
    if path.exists():
        print(f"    PRE-TRAINED SKIP: {dataset['name']} seed={seed}")
        return read_json(path)

    train_pairs = a17.build_train_pairs(
        ids_train,
        dataset["retained"],
        dataset["truth"],
        dataset["base_top50"],
    )
    test_structure = a17.build_test_structure(
        ids_test,
        dataset["truth"],
        dataset["base_top50"],
    )

    started = time.time()
    result = fit_and_evaluate_two_step(
        dataset,
        train_pairs,
        test_structure,
        dataset["base_emb_a"],
        dataset["base_emb_b"],
        seed,
    )
    payload = {
        "phase": "table5",
        "dataset": dataset["name"],
        "seed": seed,
        "seed_role": "primary" if seed in PRIMARY_SEEDS else "diagnostic",
        "encoder": "pre-trained",
        "n_train_a": len(ids_train),
        "n_test_a": len(ids_test),
        "runtime_seconds": time.time() - started,
        **result,
    }
    atomic_write_json(path, payload)
    print(
        f"    PRE-TRAINED saved: "
        f"LR=({payload['LR']['lambda']:.4f}, {payload['LR']['psi']:.4f}); "
        f"MEC=({payload['MEC']['lambda']:.4f}, {payload['MEC']['psi']:.4f})"
    )
    return payload


# =============================================================================
# FINE-TUNED MODEL AND CELL
# =============================================================================
def train_or_load_ft_model(
    dataset: dict[str, Any],
    baseline_pairs: list[tuple[str, str, int]],
    seed: int,
    device: str,
) -> SentenceTransformer:
    path = ft_model_path(dataset["name"], seed)
    if a17.model_complete(path):
        print(f"      loading saved fine-tuned model: {path}")
        return SentenceTransformer(str(path), device=device)

    if path.exists():
        print(f"      removing incomplete fine-tuned checkpoint: {path}")
        a17.safe_remove(path)

    set_seed(seed)
    fields = dataset["cfg"]["fields"]
    examples = [
        InputExample(
            texts=[
                a17.serialise(dataset["df_a"].loc[ida], fields),
                a17.serialise(dataset["df_b"].loc[idb], fields),
            ],
            label=float(label),
        )
        for ida, idb, label in baseline_pairs
    ]

    model = SentenceTransformer(BI_ENCODER_NAME, device=device)
    loader = DataLoader(examples, shuffle=True, batch_size=FT_BATCH)
    loss = losses.ContrastiveLoss(model)
    total_steps = len(loader) * FT_EPOCHS
    warmup_steps = int(FT_WARMUP_FRAC * total_steps)

    print(
        f"      fine-tuning: pairs={len(examples):,}; epochs={FT_EPOCHS}; "
        f"warmup={warmup_steps}; lr={FT_LR}; batch={FT_BATCH} [{ts()}]"
    )
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=FT_EPOCHS,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": FT_LR},
        show_progress_bar=False,
    )

    a17.save_model_atomic(
        model,
        path,
        {
            "phase": "analysis21_table5_ft_biencoder",
            "dataset": dataset["name"],
            "seed": seed,
            "model": BI_ENCODER_NAME,
            "epochs": FT_EPOCHS,
            "warmup_fraction": FT_WARMUP_FRAC,
            "learning_rate": FT_LR,
            "batch_size": FT_BATCH,
            "training_pairs": len(examples),
            "training_block": "pre-trained encoder k=50",
        },
    )
    print(f"      saved fine-tuned checkpoint: {path}")
    return model


def run_finetuned_cell(
    dataset: dict[str, Any],
    seed: int,
    ids_train: list[str],
    ids_test: list[str],
    device: str,
) -> dict[str, Any]:
    path = cell_path(dataset["name"], seed, "fine-tuned")
    if path.exists():
        print(f"    FINE-TUNED SKIP: {dataset['name']} seed={seed}")
        return read_json(path)

    baseline_pairs = a17.build_train_pairs(
        ids_train,
        dataset["retained"],
        dataset["truth"],
        dataset["base_top50"],
    )

    started = time.time()
    ft_model = train_or_load_ft_model(
        dataset,
        baseline_pairs,
        seed,
        device,
    )

    fields = dataset["cfg"]["fields"]
    print(f"      re-embedding complete A and B [{ts()}]")
    ft_emb_a = a17.encode_fields(ft_model, dataset["df_a"], fields, "FT-A")
    ft_emb_b = a17.encode_fields(ft_model, dataset["df_b"], fields, "FT-B")
    ft_top50 = a17.retrieve_top50(
        ft_emb_a[dataset["cfg"]["retrieval_field"]],
        ft_emb_b[dataset["cfg"]["retrieval_field"]],
        dataset["a_ids"],
        dataset["b_ids"],
    )

    ft_train_pairs = a17.build_train_pairs(
        ids_train,
        dataset["retained"],
        dataset["truth"],
        ft_top50,
    )
    ft_test_structure = a17.build_test_structure(
        ids_test,
        dataset["truth"],
        ft_top50,
    )

    result = fit_and_evaluate_two_step(
        dataset,
        ft_train_pairs,
        ft_test_structure,
        ft_emb_a,
        ft_emb_b,
        seed,
    )
    payload = {
        "phase": "table5",
        "dataset": dataset["name"],
        "seed": seed,
        "seed_role": "primary" if seed in PRIMARY_SEEDS else "diagnostic",
        "encoder": "fine-tuned",
        "n_train_a": len(ids_train),
        "n_test_a": len(ids_test),
        "n_baseline_ft_training_pairs": len(baseline_pairs),
        "runtime_seconds": time.time() - started,
        **result,
    }
    atomic_write_json(path, payload)
    print(
        f"    FINE-TUNED saved: "
        f"LR=({payload['LR']['lambda']:.4f}, {payload['LR']['psi']:.4f}); "
        f"MEC=({payload['MEC']['lambda']:.4f}, {payload['MEC']['psi']:.4f})"
    )

    del ft_model, ft_emb_a, ft_emb_b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


# =============================================================================
# COLLECTION, CROSS-CHECK AND AGGREGATION
# =============================================================================
def collect_perseed() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(CELL_DIR.glob("table5_*_seed*.json")):
        cell = read_json(path)
        for scorer in SCORERS:
            metrics = cell[scorer]
            rows.append(
                {
                    "dataset": cell["dataset"],
                    "seed": int(cell["seed"]),
                    "seed_role": cell["seed_role"],
                    "encoder": cell["encoder"],
                    "scorer": scorer,
                    "lambda": float(metrics["lambda"]),
                    "psi": float(metrics["psi"]),
                    "declared": int(metrics["declared"]),
                    "correct": int(metrics["correct"]),
                    "false_links": int(metrics["false_links"]),
                    "missing_matches": int(metrics["missing_matches"]),
                    "n_true_records": int(metrics["n_true_records"]),
                    "threshold": (
                        float(cell["lr_threshold"])
                        if scorer == "LR"
                        else np.nan
                    ),
                    "validation_f1": (
                        float(cell["lr_validation_f1"])
                        if scorer == "LR"
                        else np.nan
                    ),
                    "n_train_pairs": int(cell["n_train_pairs"]),
                    "n_train_positive_pairs": int(
                        cell["n_train_positive_pairs"]
                    ),
                    "n_train_negative_pairs": int(
                        cell["n_train_negative_pairs"]
                    ),
                    "runtime_seconds": float(cell["runtime_seconds"]),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame["dataset"] = pd.Categorical(frame["dataset"], DATASETS, ordered=True)
    frame["encoder"] = pd.Categorical(frame["encoder"], ENCODERS, ordered=True)
    frame["scorer"] = pd.Categorical(frame["scorer"], SCORERS, ordered=True)
    return frame.sort_values(
        ["dataset", "encoder", "scorer", "seed"]
    ).reset_index(drop=True)


def crosscheck_pretrained(perseed: pd.DataFrame) -> pd.DataFrame:
    if not ANALYSIS16_PERSEED.exists():
        raise FileNotFoundError(
            f"Required corrected Analysis 16 output missing: {ANALYSIS16_PERSEED}"
        )

    reference = pd.read_csv(ANALYSIS16_PERSEED)
    reference = reference[
        (reference["column"] == "k=50")
        & (reference["side"] == "bi-encoder")
        & (reference["scorer"].isin(SCORERS))
        & (reference["seed"].isin(ALL_SEEDS))
    ].copy()

    current = perseed[
        (perseed["encoder"].astype(str) == "pre-trained")
        & (perseed["seed"].isin(ALL_SEEDS))
    ].copy()

    expected_rows = len(DATASETS) * len(ALL_SEEDS) * len(SCORERS)
    if len(reference) != expected_rows:
        raise AssertionError(
            f"Analysis 16 reference has {len(reference)} rows; expected {expected_rows}"
        )
    if len(current) != expected_rows:
        raise AssertionError(
            f"Analysis 21 pre-trained has {len(current)} rows; expected {expected_rows}"
        )

    reference = reference.rename(
        columns={
            "lambda": "lambda_analysis16",
            "psi": "psi_analysis16",
            "declared": "declared_analysis16",
            "correct": "correct_analysis16",
            "false_links": "false_links_analysis16",
            "missing_matches": "missing_matches_analysis16",
            "n_true_records": "n_true_records_analysis16",
            "lr_threshold": "threshold_analysis16",
        }
    )
    keep_reference = [
        "dataset",
        "seed",
        "scorer",
        "lambda_analysis16",
        "psi_analysis16",
        "declared_analysis16",
        "correct_analysis16",
        "false_links_analysis16",
        "missing_matches_analysis16",
        "n_true_records_analysis16",
        "threshold_analysis16",
    ]

    merged = current.merge(
        reference[keep_reference],
        on=["dataset", "seed", "scorer"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise AssertionError("Analysis 16 / 21 pre-trained key mismatch")
    merged = merged.drop(columns="_merge")

    merged["lambda_delta"] = merged["lambda"] - merged["lambda_analysis16"]
    merged["psi_delta"] = merged["psi"] - merged["psi_analysis16"]

    threshold_current = pd.to_numeric(merged["threshold"], errors="coerce")
    threshold_reference = pd.to_numeric(
        merged["threshold_analysis16"], errors="coerce"
    )
    merged["threshold_delta"] = threshold_current - threshold_reference

    count_pairs = [
        ("declared", "declared_analysis16"),
        ("correct", "correct_analysis16"),
        ("false_links", "false_links_analysis16"),
        ("missing_matches", "missing_matches_analysis16"),
        ("n_true_records", "n_true_records_analysis16"),
    ]
    for current_name, reference_name in count_pairs:
        merged[f"{current_name}_delta"] = (
            merged[current_name] - merged[reference_name]
        )

    max_lambda = float(merged["lambda_delta"].abs().max())
    max_psi = float(merged["psi_delta"].abs().max())
    lr_rows = merged[merged["scorer"].astype(str) == "LR"]
    max_threshold = float(lr_rows["threshold_delta"].abs().max())
    max_count_delta = max(
        int(merged[f"{name}_delta"].abs().max())
        for name, _reference in count_pairs
    )

    if max_count_delta != 0:
        raise AssertionError(
            f"Analysis 16 / 21 pre-trained count mismatch: max delta={max_count_delta}"
        )
    if max_lambda > CROSSCHECK_TOL or max_psi > CROSSCHECK_TOL:
        raise AssertionError(
            "Analysis 16 / 21 pre-trained metric mismatch: "
            f"max |lambda delta|={max_lambda}, max |psi delta|={max_psi}"
        )
    if max_threshold > CROSSCHECK_TOL:
        raise AssertionError(
            "Analysis 16 / 21 LR threshold mismatch: "
            f"max |threshold delta|={max_threshold}"
        )

    print(
        "\nPRE-TRAINED CROSS-CHECK PASSED: "
        f"rows={len(merged)}, max |lambda delta|={max_lambda:.3g}, "
        f"max |psi delta|={max_psi:.3g}, "
        f"max |LR threshold delta|={max_threshold:.3g}, "
        f"max count delta={max_count_delta}"
    )
    return merged


def aggregate(
    perseed: pd.DataFrame,
    seeds: list[int],
    label: str,
) -> pd.DataFrame:
    subset = perseed[perseed["seed"].isin(seeds)].copy()
    if subset.empty:
        return pd.DataFrame()

    frame = (
        subset.groupby(["dataset", "encoder", "scorer"], observed=True)
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
            declared_mean=("declared", "mean"),
            correct_mean=("correct", "mean"),
            false_links_mean=("false_links", "mean"),
            missing_matches_mean=("missing_matches", "mean"),
            n_true_records_mean=("n_true_records", "mean"),
            n_train_pairs_mean=("n_train_pairs", "mean"),
            runtime_seconds_mean=("runtime_seconds", "mean"),
        )
        .reset_index()
    )
    frame["lambda_se"] = frame["lambda_sd"] / np.sqrt(frame["n_splits"])
    frame["psi_se"] = frame["psi_sd"] / np.sqrt(frame["n_splits"])
    frame.insert(0, "aggregation", label)

    numeric = frame.select_dtypes(include=[np.number]).columns
    frame[numeric] = frame[numeric].round(6)
    return frame


def save_outputs(require_complete: bool = False) -> None:
    perseed = collect_perseed()
    if perseed.empty:
        print("No completed Analysis 21 cells yet.")
        return

    primary = aggregate(perseed, PRIMARY_SEEDS, "primary_3_seeds")
    mc10 = aggregate(perseed, ALL_SEEDS, "all_10_seeds")

    atomic_write_csv(RESULTS_DIR / "analysis21_table5_perseed.csv", perseed)
    atomic_write_csv(RESULTS_DIR / "analysis21_table5_primary3.csv", primary)

    expected_groups = len(DATASETS) * len(ENCODERS) * len(SCORERS)
    complete = (
        len(mc10) == expected_groups
        and bool((mc10["n_splits"] == 10).all())
    )

    if complete:
        crosscheck = crosscheck_pretrained(perseed)
        atomic_write_csv(
            RESULTS_DIR / "analysis21_table5_pretrained_analysis16_crosscheck.csv",
            crosscheck,
        )
        atomic_write_csv(RESULTS_DIR / "analysis21_table5_mc10.csv", mc10)
    elif require_complete:
        raise AssertionError(
            "Analysis 21 is incomplete: final MC10 file was not written"
        )

    manifest = {
        "analysis": "analysis21_table5_mc10",
        "protocol_source": str(A17_PATH),
        "analysis16_reference": str(ANALYSIS16_PERSEED),
        "datasets": DATASETS,
        "seeds": ALL_SEEDS,
        "encoders": ENCODERS,
        "scorers": SCORERS,
        "block": "k=50",
        "fine_tuning": {
            "model": BI_ENCODER_NAME,
            "loss": "ContrastiveLoss",
            "epochs": FT_EPOCHS,
            "warmup_fraction": FT_WARMUP_FRAC,
            "learning_rate": FT_LR,
            "batch_size": FT_BATCH,
            "training_pairs": "pre-trained encoder k=50 pairs",
            "post_training_blocking": "rebuilt using fine-tuned encoder",
        },
        "counting": {
            "correct": "any valid B partner",
            "categories": "mutually exclusive correct/false/missing",
            "psi_denominator": "test A records with >=1 true partner",
            "negative_exclusion": "all valid alternative partners",
        },
        "all_expected_groups_complete": complete,
    }
    atomic_write_json(
        RESULTS_DIR / "analysis21_table5_manifest.json",
        manifest,
    )

    print("\n" + "=" * 118)
    print("ANALYSIS 21 TABLE 5: AVAILABLE RESULTS")
    print("=" * 118)
    display = [
        "dataset",
        "encoder",
        "scorer",
        "n_splits",
        "lambda_mean",
        "lambda_sd",
        "lambda_se",
        "psi_mean",
        "psi_sd",
        "psi_se",
        "threshold_mean",
    ]
    print(mc10[display].to_string(index=False))

    if complete:
        print("\nPASS: every Table 5 row contains seeds 42-51.")
        print(
            "Final file: "
            f"{RESULTS_DIR / 'analysis21_table5_mc10.csv'}"
        )
    else:
        print("\nFinal MC10 file not yet written; rerun to resume missing cells.")


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================
def run_experiment(datasets: list[str], seeds: list[int]) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model: SentenceTransformer | None = None
    total = len(datasets) * len(seeds)
    ordinal = 0

    for dataset_name in datasets:
        dataset_complete = all(
            cell_path(dataset_name, seed, encoder).exists()
            for seed in seeds
            for encoder in ENCODERS
        )
        if dataset_complete:
            ordinal += len(seeds)
            print(f"\nSKIP fully completed dataset: {dataset_name}")
            continue

        if base_model is None:
            print(f"\nLoading pre-trained bi-encoder on {device}: {BI_ENCODER_NAME}")
            base_model = SentenceTransformer(BI_ENCODER_NAME, device=device)

        dataset = a17.load_dataset(dataset_name)
        a17.prepare_baseline(dataset, base_model)

        for seed in seeds:
            ordinal += 1
            print("\n" + "=" * 100)
            print(
                f"[{ordinal}/{total}] {dataset_name} seed={seed} "
                f"({'primary' if seed in PRIMARY_SEEDS else 'diagnostic'}) [{ts()}]"
            )
            print("=" * 100)

            set_seed(seed)
            ids_train, ids_test = train_test_split(
                dataset["a_ids"],
                test_size=0.50,
                random_state=seed,
            )
            ids_train = list(ids_train)
            ids_test = list(ids_test)

            run_pretrained_cell(dataset, seed, ids_train, ids_test)
            run_finetuned_cell(
                dataset,
                seed,
                ids_train,
                ids_test,
                device,
            )
            save_outputs(require_complete=False)

        del dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if base_model is not None:
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fresh corrected ten-seed rerun of paper Table 5."
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_protocol()

    datasets = [value.strip().upper() for value in args.datasets.split(",") if value.strip()]
    invalid = set(datasets).difference(DATASETS)
    if invalid:
        raise ValueError(f"Unknown datasets: {sorted(invalid)}")
    seeds = PRIMARY_SEEDS if args.primary_only else ALL_SEEDS

    print("=" * 100)
    print("Analysis 21: corrected ten-seed Table 5 rerun")
    print("=" * 100)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Analysis 17 protocol source: {A17_PATH}")
    print(f"Analysis 16 cross-check source: {ANALYSIS16_PERSEED}")
    print(f"Phase: {args.phase}")
    print(f"Datasets: {datasets}")
    print(f"Seeds: {seeds}")
    print("Block: k=50")
    print("Columns computed together: pre-trained and fine-tuned")
    print(
        "Fine-tuning: ContrastiveLoss, one epoch, 10% warmup, "
        "lr=2e-5, batch=16"
    )
    print("Raw B intact; all alternatives excluded from negatives")
    print("Correctness accepts any valid B; counting is mutually exclusive")
    print("psi denominator: truth-bearing test A records")
    print("Latin-1 and June preprocessing: inherited and asserted")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Visible CUDA devices: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Visible device 0: {torch.cuda.get_device_name(0)}")

    if args.phase != "aggregate" and torch.cuda.device_count() > 1:
        raise RuntimeError(
            "More than one GPU is visible. Pin explicitly, e.g. "
            "CUDA_VISIBLE_DEVICES=0 python scripts/analysis21_table5_mc10.py"
        )

    if args.phase in ("all", "run"):
        run_experiment(datasets, seeds)

    if args.phase in ("all", "aggregate"):
        save_outputs(require_complete=not args.primary_only)

    print("\nAnalysis 21 completed for the requested cells.")


if __name__ == "__main__":
    main()
