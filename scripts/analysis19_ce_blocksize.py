"""
analysis19_ce_blocksize.py
==========================

Corrected, checkpointed cross-encoder rerun for the two Section 5 block-size
cells missing from Analysis 17:

    forced-1:1
    k=5

The existing Analysis 17 run supplies k=50. This script retrains the
cross-encoder independently for every (dataset, block size, seed) cell.
It never trains at k=50 and reuses that model for a smaller block.

Protocol inherited directly from analysis17_section5.py
--------------------------------------------------------
- complete raw B candidate universe;
- Latin-1 decoding and June pandas CSV round-trip;
- lowercase + strip preprocessing;
- title-based all-MiniLM-L6-v2 retrieval;
- first-listed reachable B match as the single training positive;
- every known valid alternative excluded from negatives;
- any valid B match accepted as correct at test time;
- mutually exclusive correct / false-link / missing-match categories;
- psi denominator = test records with at least one true match;
- 50/50 A-level train/test split;
- 80/20 stratified pair-level calibration split;
- threshold grid 0.10, 0.15, ..., 0.90 selected by validation F1;
- cross-encoder/ms-marco-MiniLM-L-6-v2;
- 5 epochs, 10% warmup, learning rate 2e-5, batch size 16,
  max_length 128.

Block-specific pair construction
--------------------------------
forced-1:1:
    One retained positive plus the highest-ranked retrieved candidate that
    is outside the complete valid-match set. At test time, a truth-bearing
    record is evaluated on that retained positive and one such negative,
    reproducing Analysis 8's forced one-positive/one-negative design while
    correctly excluding alternative valid links.

k=5:
    One retained positive plus every non-matching candidate in the retrieved
    top five. This gives up to four negatives when a valid match is retrieved,
    and up to five when no valid match is retrieved (fewer if alternative
    valid links occupy retrieved positions).

Strict comparability with Analysis 17 k=50
-------------------------------------------
At startup, this script constructs a generalized k=50 pair set and test
structure and asserts exact equality, including ordering, with
analysis17_section5.build_train_pairs and build_test_structure. It also
asserts all model and calibration constants. The run stops if any check fails.

Outputs
-------
results/analysis19_ce_blocks_perseed.csv
results/analysis19_ce_blocks_primary3.csv
results/analysis19_ce_blocks_mc10.csv
results/analysis19_ce_blocks_available.csv
results/analysis19_ce_blocksize_manifest.json

When results/analysis17_ce_perseed.csv is available, the script additionally
writes combined forced-1:1 / k=5 / k=50 files. A combined MC10 file is written
only if all ten k=50 CE_direct seeds are present for both datasets.

Resumability
------------
Models and atomic cell JSONs are checkpointed by dataset, block size, and seed.
A completed cell is skipped. If model training completed and its checkpoint
was saved, evaluation resumes without retraining that cell.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, InputExample, SentenceTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


# =============================================================================
# LOAD THE EXACT ANALYSIS 17 IMPLEMENTATION
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
A17_PATH = SCRIPT_DIR / "analysis17_section5.py"
if not A17_PATH.exists():
    raise FileNotFoundError(
        f"Required protocol source not found: {A17_PATH}. "
        "Keep analysis17_section5.py in the same scripts directory."
    )

_spec = importlib.util.spec_from_file_location("analysis17_protocol", A17_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not import {A17_PATH}")
a17 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = a17
_spec.loader.exec_module(a17)


# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path(
    os.environ.get("ER_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).expanduser()
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
STATE_DIR = RESULTS_DIR / "analysis19_ce_blocksize"
CELL_DIR = STATE_DIR / "cells"
PREDICTION_DIR = STATE_DIR / "predictions"
MODEL_ROOT = MODELS_DIR / "analysis19_ce_blocksize" / "cross_encoder"

for directory in (
    RESULTS_DIR,
    MODELS_DIR,
    STATE_DIR,
    CELL_DIR,
    PREDICTION_DIR,
    MODEL_ROOT,
):
    directory.mkdir(parents=True, exist_ok=True)

PRIMARY_SEEDS = [42, 43, 44]
DIAGNOSTIC_SEEDS = [45, 46, 47, 48, 49, 50, 51]
ALL_SEEDS = PRIMARY_SEEDS + DIAGNOSTIC_SEEDS

BLOCKS: dict[str, dict[str, Any]] = {
    "forced-1:1": {"k": 1, "slug": "forced_1to1"},
    "k=5": {"k": 5, "slug": "k5"},
}
BLOCK_ORDER = ["forced-1:1", "k=5", "k=50"]

BI_ENCODER_NAME = "all-MiniLM-L6-v2"
CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CE_EPOCHS = 5
CE_WARMUP_FRAC = 0.10
CE_LR = 2e-5
CE_BATCH = 16
CE_MAX_LEN = 128
LR_VAL_FRAC = 0.20
THRESHOLDS = np.arange(0.10, 0.90 + 1e-9, 0.05)
PREDICT_BATCH = 32

MODEL_MARKER = "_analysis19_complete.json"


# =============================================================================
# UTILITIES
# =============================================================================
def ts() -> str:
    return time.strftime("%H:%M:%S")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def model_complete(path: Path) -> bool:
    return path.is_dir() and (path / MODEL_MARKER).exists()


def save_model_atomic(
    model: CrossEncoder,
    path: Path,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    safe_remove(temporary)
    model.save_pretrained(str(temporary))
    atomic_write_json(temporary / MODEL_MARKER, metadata)
    safe_remove(path)
    os.replace(temporary, path)


def choose_threshold(
    scores: np.ndarray,
    labels: Iterable[int],
) -> tuple[float, float]:
    labels_array = np.asarray(list(labels), dtype=int)
    scores_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in THRESHOLDS:
        current_f1 = f1_score(
            labels_array,
            (scores_array > threshold).astype(int),
            zero_division=0,
        )
        if current_f1 > best_f1:
            best_f1 = float(current_f1)
            best_threshold = float(threshold)

    return best_threshold, best_f1


def block_slug(column: str) -> str:
    return BLOCKS[column]["slug"]


def cell_path(dataset_name: str, column: str, seed: int) -> Path:
    return CELL_DIR / f"ce_{dataset_name}_{block_slug(column)}_seed{seed}.json"


def prediction_path(dataset_name: str, column: str, seed: int) -> Path:
    return (
        PREDICTION_DIR
        / f"ce_{dataset_name}_{block_slug(column)}_seed{seed}.csv"
    )


def model_path(dataset_name: str, column: str, seed: int) -> Path:
    return (
        MODEL_ROOT
        / dataset_name
        / block_slug(column)
        / f"seed_{seed}"
    )


# =============================================================================
# BLOCK-SPECIFIC PAIR AND TEST CONSTRUCTION
# =============================================================================
def highest_ranked_nonmatch(
    ranked_candidates: list[str],
    true_set: set[str],
    ida: str,
) -> str:
    for idb in ranked_candidates:
        if idb not in true_set:
            return idb
    raise RuntimeError(
        f"{ida}: no non-matching candidate found in the saved top-50 block."
    )


def build_train_pairs_for_block(
    ids: Iterable[str],
    retained: dict[str, str],
    truth: dict[str, set[str]],
    top50: dict[str, list[str]],
    column: str,
) -> list[tuple[str, str, int]]:
    """
    Build one positive plus block-specific hard negatives.

    The positive is always the first-listed retained valid match. Every valid
    alternative is excluded from negatives.
    """
    if column not in BLOCKS and column != "k=50":
        raise ValueError(f"Unknown block column: {column}")

    pairs: list[tuple[str, str, int]] = []
    for ida in ids:
        positive = retained.get(ida)
        if positive is None:
            continue

        true_set = truth[ida]
        pairs.append((ida, positive, 1))

        if column == "forced-1:1":
            negative = highest_ranked_nonmatch(top50[ida], true_set, ida)
            pairs.append((ida, negative, 0))
            continue

        k = 50 if column == "k=50" else int(BLOCKS[column]["k"])
        retrieved = top50[ida][:k]
        for idb in retrieved:
            if idb not in true_set:
                pairs.append((ida, idb, 0))

    return pairs


def build_test_structure_for_block(
    ids: Iterable[str],
    retained: dict[str, str],
    truth: dict[str, set[str]],
    top50: dict[str, list[str]],
    column: str,
) -> list[tuple[str, set[str], list[str]]]:
    """
    Reproduce Analysis 8's block-specific test design under many-match truth.
    """
    if column not in BLOCKS and column != "k=50":
        raise ValueError(f"Unknown block column: {column}")

    structure: list[tuple[str, set[str], list[str]]] = []
    for ida in ids:
        true_set = truth.get(ida, set())

        if column == "forced-1:1":
            if true_set:
                positive = retained[ida]
                negative = highest_ranked_nonmatch(top50[ida], true_set, ida)
                candidates = [positive, negative]
            else:
                # Exact Analysis 8 unmatched-record behaviour: one retrieved
                # candidate and no inserted positive.
                candidates = top50[ida][:1]
        else:
            k = 50 if column == "k=50" else int(BLOCKS[column]["k"])
            candidates = top50[ida][:k]

        if not candidates:
            raise AssertionError(f"{ida}: empty candidate block for {column}")
        structure.append((ida, true_set, candidates))

    return structure


# =============================================================================
# STRICT ANALYSIS 17 k=50 EQUIVALENCE AUDIT
# =============================================================================
def assert_protocol_constants() -> None:
    expected = {
        "BI_ENCODER_NAME": BI_ENCODER_NAME,
        "CE_MODEL_NAME": CE_MODEL_NAME,
        "CE_EPOCHS": CE_EPOCHS,
        "CE_WARMUP_FRAC": CE_WARMUP_FRAC,
        "CE_LR": CE_LR,
        "CE_BATCH": CE_BATCH,
        "CE_MAX_LEN": CE_MAX_LEN,
        "LR_VAL_FRAC": LR_VAL_FRAC,
    }
    mismatches: list[str] = []
    for name, wanted in expected.items():
        observed = getattr(a17, name)
        if observed != wanted:
            mismatches.append(f"{name}: analysis17={observed!r}, new={wanted!r}")

    if not np.array_equal(np.asarray(a17.THRESHOLDS), THRESHOLDS):
        mismatches.append(
            f"THRESHOLDS: analysis17={a17.THRESHOLDS}, new={THRESHOLDS}"
        )

    if mismatches:
        raise AssertionError(
            "Analysis 17 protocol constants differ:\n  " + "\n  ".join(mismatches)
        )


def assert_k50_equivalence(dataset: dict[str, Any]) -> None:
    ids = dataset["a_ids"]
    generalized_pairs = build_train_pairs_for_block(
        ids,
        dataset["retained"],
        dataset["truth"],
        dataset["base_top50"],
        "k=50",
    )
    analysis17_pairs = a17.build_train_pairs(
        ids,
        dataset["retained"],
        dataset["truth"],
        dataset["base_top50"],
    )
    if generalized_pairs != analysis17_pairs:
        for index, (new_value, old_value) in enumerate(
            zip(generalized_pairs, analysis17_pairs)
        ):
            if new_value != old_value:
                raise AssertionError(
                    "k=50 training-pair mismatch at index "
                    f"{index}: new={new_value}, analysis17={old_value}"
                )
        raise AssertionError(
            "k=50 training-pair lengths differ: "
            f"new={len(generalized_pairs)}, analysis17={len(analysis17_pairs)}"
        )

    generalized_test = build_test_structure_for_block(
        ids,
        dataset["retained"],
        dataset["truth"],
        dataset["base_top50"],
        "k=50",
    )
    analysis17_test = a17.build_test_structure(
        ids,
        dataset["truth"],
        dataset["base_top50"],
    )
    if generalized_test != analysis17_test:
        raise AssertionError("k=50 test-candidate construction differs from Analysis 17")

    print(
        f"  k=50 equivalence audit passed for {dataset['name']}: "
        f"{len(generalized_pairs):,} full training pairs; "
        f"{len(generalized_test):,} test structures"
    )


# =============================================================================
# CROSS-ENCODER TRAINING AND EVALUATION
# =============================================================================
def train_or_load_model(
    dataset: dict[str, Any],
    column: str,
    train_only: list[tuple[str, str, int]],
    seed: int,
    device: str,
) -> CrossEncoder:
    path = model_path(dataset["name"], column, seed)

    if model_complete(path):
        print(f"      loading saved CE: {path}")
        return CrossEncoder(str(path), max_length=CE_MAX_LEN, device=device)

    if path.exists():
        print(f"      removing incomplete CE checkpoint: {path}")
        safe_remove(path)

    fields = dataset["cfg"]["fields"]
    examples = [
        InputExample(
            texts=[
                a17.serialise(dataset["df_a"].loc[ida], fields),
                a17.serialise(dataset["df_b"].loc[idb], fields),
            ],
            label=float(label),
        )
        for ida, idb, label in train_only
    ]

    model = CrossEncoder(
        CE_MODEL_NAME,
        num_labels=1,
        max_length=CE_MAX_LEN,
        device=device,
    )
    loader = DataLoader(examples, shuffle=True, batch_size=CE_BATCH)
    total_steps = len(loader) * CE_EPOCHS
    warmup_steps = int(CE_WARMUP_FRAC * total_steps)

    print(
        f"      training CE: column={column}; pairs={len(examples):,}; "
        f"epochs={CE_EPOCHS}; warmup={warmup_steps}; lr={CE_LR}; "
        f"batch={CE_BATCH}; max_length={CE_MAX_LEN} [{ts()}]"
    )
    model.fit(
        train_dataloader=loader,
        epochs=CE_EPOCHS,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": CE_LR},
        show_progress_bar=False,
    )

    save_model_atomic(
        model,
        path,
        {
            "phase": "ce_blocksize",
            "dataset": dataset["name"],
            "column": column,
            "seed": seed,
            "model": CE_MODEL_NAME,
            "epochs": CE_EPOCHS,
            "warmup_fraction": CE_WARMUP_FRAC,
            "learning_rate": CE_LR,
            "batch_size": CE_BATCH,
            "max_length": CE_MAX_LEN,
            "training_pairs": len(examples),
            "pair_construction": (
                "one retained positive plus non-matching retrieved "
                "candidates from this block; alternatives excluded"
            ),
        },
    )
    print(f"      saved CE checkpoint: {path}")
    return model


def ce_predict(
    model: CrossEncoder,
    pairs: list[tuple[str, str]],
) -> np.ndarray:
    return np.asarray(
        model.predict(
            pairs,
            batch_size=PREDICT_BATCH,
            show_progress_bar=False,
        ),
        dtype=np.float64,
    ).reshape(-1)


def metric_summary(predictions: pd.DataFrame) -> dict[str, float | int]:
    declared = int(predictions["declared"].sum())
    correct = int((predictions["category"] == "correct").sum())
    false_links = int((predictions["category"] == "false_link").sum())
    missing_matches = int((predictions["category"] == "missing_match").sum())
    n_true_records = int(predictions["has_true_match"].sum())

    if declared != correct + false_links:
        raise AssertionError("Declared does not equal correct + false links")

    return {
        "lambda": false_links / declared if declared else 0.0,
        "psi": missing_matches / n_true_records if n_true_records else 0.0,
        "declared": declared,
        "correct": correct,
        "false_links": false_links,
        "missing_matches": missing_matches,
        "n_true_records": n_true_records,
    }


def run_cell(
    dataset: dict[str, Any],
    column: str,
    seed: int,
    device: str,
) -> dict[str, Any]:
    result_path = cell_path(dataset["name"], column, seed)
    predictions_path = prediction_path(dataset["name"], column, seed)

    if result_path.exists() and predictions_path.exists():
        print(
            f"    CE SKIP completed: {dataset['name']} {column} seed={seed}"
        )
        return read_json(result_path)

    seed_everything(seed)
    ids_train, ids_test = train_test_split(
        dataset["a_ids"],
        test_size=0.50,
        random_state=seed,
    )

    train_pairs = build_train_pairs_for_block(
        ids_train,
        dataset["retained"],
        dataset["truth"],
        dataset["base_top50"],
        column,
    )
    test_structure = build_test_structure_for_block(
        ids_test,
        dataset["retained"],
        dataset["truth"],
        dataset["base_top50"],
        column,
    )

    labels = np.asarray([label for _, _, label in train_pairs], dtype=int)
    if set(labels.tolist()) != {0, 1}:
        raise AssertionError(
            f"{dataset['name']} {column} seed={seed}: training pairs "
            "do not contain both classes"
        )

    train_indices, validation_indices = train_test_split(
        np.arange(len(train_pairs)),
        test_size=LR_VAL_FRAC,
        random_state=seed,
        stratify=labels,
    )
    train_only = [train_pairs[int(index)] for index in train_indices]
    validation_only = [train_pairs[int(index)] for index in validation_indices]

    start_time = time.time()
    model = train_or_load_model(
        dataset,
        column,
        train_only,
        seed,
        device,
    )

    fields = dataset["cfg"]["fields"]
    validation_text = [
        (
            a17.serialise(dataset["df_a"].loc[ida], fields),
            a17.serialise(dataset["df_b"].loc[idb], fields),
        )
        for ida, idb, _ in validation_only
    ]
    validation_labels = [label for _, _, label in validation_only]
    validation_scores = ce_predict(model, validation_text)
    threshold, validation_f1 = choose_threshold(
        validation_scores,
        validation_labels,
    )

    rows: list[dict[str, Any]] = []
    print(
        f"      evaluating {len(test_structure):,} A records; "
        f"column={column}; threshold={threshold:.2f}; "
        f"validation F1={validation_f1:.4f} [{ts()}]"
    )

    for record_index, (ida, true_set, candidates) in enumerate(
        test_structure,
        start=1,
    ):
        candidate_text = [
            (
                a17.serialise(dataset["df_a"].loc[ida], fields),
                a17.serialise(dataset["df_b"].loc[idb], fields),
            )
            for idb in candidates
        ]
        scores = ce_predict(model, candidate_text)
        top_index = int(np.argmax(scores))
        top_idb = candidates[top_index]
        top_score = float(scores[top_index])
        declared = bool(top_score > threshold)
        category = a17.category_for(declared, top_idb, true_set)

        rows.append(
            {
                "dataset": dataset["name"],
                "column": column,
                "k": int(BLOCKS[column]["k"]),
                "seed": seed,
                "record_index": record_index - 1,
                "ida": ida,
                "has_true_match": bool(true_set),
                "n_valid_matches": len(true_set),
                "retained_idb": dataset["retained"].get(ida, ""),
                "n_candidates": len(candidates),
                "top_idb": top_idb,
                "top_is_retained": top_idb == dataset["retained"].get(ida),
                "top_is_any_valid": top_idb in true_set,
                "top_score": top_score,
                "threshold": threshold,
                "declared": declared,
                "category": category,
            }
        )

        if record_index % 200 == 0 or record_index == len(test_structure):
            print(
                f"        evaluated {record_index:,}/{len(test_structure):,}",
                end="\r",
                flush=True,
            )
    print()

    predictions = pd.DataFrame(rows)
    atomic_write_csv(predictions, predictions_path)
    metrics = metric_summary(predictions)

    alternative_valid_links = int(
        (
            predictions["declared"]
            & predictions["top_is_any_valid"]
            & ~predictions["top_is_retained"]
        ).sum()
    )
    negative_counter = Counter(
        pair_ida for pair_ida, _, label in train_pairs if label == 0
    )
    negative_counts = pd.Series(
        [
            negative_counter.get(ida, 0)
            for ida in ids_train
            if ida in dataset["retained"]
        ],
        dtype=float,
    )

    payload: dict[str, Any] = {
        "phase": "ce_blocksize",
        "dataset": dataset["name"],
        "column": column,
        "k": int(BLOCKS[column]["k"]),
        "seed": seed,
        "n_train_a": len(ids_train),
        "n_test_a": len(ids_test),
        "n_truth_train_a": sum(ida in dataset["retained"] for ida in ids_train),
        "n_train_pairs_full": len(train_pairs),
        "n_ce_train_pairs": len(train_only),
        "n_ce_validation_pairs": len(validation_only),
        "negative_count_mean_full": float(negative_counts.mean()),
        "negative_count_min_full": int(negative_counts.min()),
        "negative_count_max_full": int(negative_counts.max()),
        "threshold": threshold,
        "validation_f1": validation_f1,
        "alternative_valid_links": alternative_valid_links,
        "runtime_seconds": time.time() - start_time,
        **metrics,
    }
    atomic_write_json(result_path, payload)

    print(
        f"      saved cell: lambda={metrics['lambda']:.4f}; "
        f"psi={metrics['psi']:.4f}; declared={metrics['declared']}; "
        f"false={metrics['false_links']}; missing={metrics['missing_matches']}; "
        f"full training pairs={len(train_pairs):,}; "
        f"negatives/A mean={negative_counts.mean():.2f}"
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return payload


# =============================================================================
# AGGREGATION
# =============================================================================
def collect_results() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(CELL_DIR.glob("ce_*_seed*.json")):
        cell = read_json(path)
        rows.append(
            {
                "dataset": cell["dataset"],
                "column": cell["column"],
                "k": int(cell["k"]),
                "seed": int(cell["seed"]),
                "model": "cross-encoder",
                "lambda": float(cell["lambda"]),
                "psi": float(cell["psi"]),
                "declared": int(cell["declared"]),
                "correct": int(cell["correct"]),
                "false_links": int(cell["false_links"]),
                "missing_matches": int(cell["missing_matches"]),
                "n_true_records": int(cell["n_true_records"]),
                "threshold": float(cell["threshold"]),
                "validation_f1": float(cell["validation_f1"]),
                "n_train_a": int(cell["n_train_a"]),
                "n_test_a": int(cell["n_test_a"]),
                "n_train_pairs_full": int(cell["n_train_pairs_full"]),
                "n_ce_train_pairs": int(cell["n_ce_train_pairs"]),
                "n_ce_validation_pairs": int(cell["n_ce_validation_pairs"]),
                "negative_count_mean_full": float(cell["negative_count_mean_full"]),
                "negative_count_min_full": int(cell["negative_count_min_full"]),
                "negative_count_max_full": int(cell["negative_count_max_full"]),
                "alternative_valid_links": int(cell["alternative_valid_links"]),
                "runtime_seconds": float(cell["runtime_seconds"]),
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["column"] = pd.Categorical(
            frame["column"],
            categories=BLOCK_ORDER,
            ordered=True,
        )
        frame = frame.sort_values(["dataset", "column", "seed"]).reset_index(drop=True)
        frame["column"] = frame["column"].astype(str)
    return frame


def aggregate_metrics(
    frame: pd.DataFrame,
    seeds: list[int],
    aggregation_label: str,
    require_complete: bool,
) -> pd.DataFrame:
    subset = frame[frame["seed"].isin(seeds)].copy()
    rows: list[dict[str, Any]] = []

    for (dataset, column), group in subset.groupby(
        ["dataset", "column"],
        sort=False,
        observed=True,
    ):
        observed_seeds = sorted(group["seed"].unique().tolist())
        if require_complete and observed_seeds != sorted(seeds):
            continue

        n_splits = len(observed_seeds)
        row: dict[str, Any] = {
            "aggregation": aggregation_label,
            "dataset": dataset,
            "column": column,
            "k": int(group["k"].iloc[0]),
            "n_splits": n_splits,
            "seeds": " ".join(map(str, observed_seeds)),
        }

        for metric in ("lambda", "psi"):
            values = group[metric].astype(float)
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if n_splits > 1 else math.nan
            se = sd / math.sqrt(n_splits) if n_splits > 1 else math.nan
            row.update(
                {
                    f"{metric}_mean": mean,
                    f"{metric}_sd": sd,
                    f"{metric}_se": se,
                    f"{metric}_min": float(values.min()),
                    f"{metric}_max": float(values.max()),
                }
            )

        for metric in (
            "declared",
            "correct",
            "false_links",
            "missing_matches",
            "n_true_records",
            "n_train_pairs_full",
            "n_ce_train_pairs",
            "n_ce_validation_pairs",
            "negative_count_mean_full",
            "runtime_seconds",
        ):
            row[f"{metric}_mean"] = float(group[metric].mean())

        rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        result["column"] = pd.Categorical(
            result["column"], categories=BLOCK_ORDER, ordered=True
        )
        result = result.sort_values(["dataset", "column"]).reset_index(drop=True)
        result["column"] = result["column"].astype(str)
    return result


def load_analysis17_k50() -> pd.DataFrame:
    path = RESULTS_DIR / "analysis17_ce_perseed.csv"
    if not path.exists():
        return pd.DataFrame()

    frame = pd.read_csv(path)
    required = {
        "dataset",
        "seed",
        "model",
        "lambda",
        "psi",
        "declared",
        "correct",
        "false_links",
        "missing_matches",
        "n_true_records",
        "threshold",
    }
    missing = required.difference(frame.columns)
    if missing:
        print(
            f"WARNING: cannot merge Analysis 17 k=50; missing columns {sorted(missing)}"
        )
        return pd.DataFrame()

    frame = frame[frame["model"] == "CE_direct"].copy()
    if frame.empty:
        print("WARNING: Analysis 17 has no CE_direct rows for k=50.")
        return pd.DataFrame()

    frame["column"] = "k=50"
    frame["k"] = 50
    frame["model"] = "cross-encoder"
    for column in (
        "validation_f1",
        "n_train_a",
        "n_test_a",
        "n_train_pairs_full",
        "n_ce_train_pairs",
        "n_ce_validation_pairs",
        "negative_count_mean_full",
        "negative_count_min_full",
        "negative_count_max_full",
    ):
        if column not in frame.columns:
            frame[column] = np.nan

    keep = [
        "dataset",
        "column",
        "k",
        "seed",
        "model",
        "lambda",
        "psi",
        "declared",
        "correct",
        "false_links",
        "missing_matches",
        "n_true_records",
        "threshold",
        "validation_f1",
        "n_train_a",
        "n_test_a",
        "n_train_pairs_full",
        "n_ce_train_pairs",
        "n_ce_validation_pairs",
        "negative_count_mean_full",
        "negative_count_min_full",
        "negative_count_max_full",
        "alternative_valid_links",
        "runtime_seconds",
    ]
    for column in keep:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[keep]


def save_outputs() -> None:
    perseed = collect_results()
    atomic_write_csv(
        perseed,
        RESULTS_DIR / "analysis19_ce_blocks_perseed.csv",
    )

    available = aggregate_metrics(
        perseed,
        ALL_SEEDS,
        "available_seeds",
        require_complete=False,
    )
    primary = aggregate_metrics(
        perseed,
        PRIMARY_SEEDS,
        "primary_3_seeds",
        require_complete=True,
    )
    mc10 = aggregate_metrics(
        perseed,
        ALL_SEEDS,
        "all_10_seeds",
        require_complete=True,
    )

    atomic_write_csv(
        available,
        RESULTS_DIR / "analysis19_ce_blocks_available.csv",
    )
    atomic_write_csv(
        primary,
        RESULTS_DIR / "analysis19_ce_blocks_primary3.csv",
    )
    mc10_path = RESULTS_DIR / "analysis19_ce_blocks_mc10.csv"
    expected_mc10_groups = len(a17.CONFIGS) * len(BLOCKS)
    mc10_written = False

    if len(mc10) == expected_mc10_groups:
        atomic_write_csv(mc10, mc10_path)
        mc10_written = True
    else:
        print(
            "\nMC10 block-size summary not written: the current run does "
            "not contain all ten seeds for every dataset/block-size cell. "
            "Any existing complete MC10 file is preserved."
        )

    k50 = load_analysis17_k50()
    combined_mc10_written = False
    k50_missing: dict[str, list[int]] = {}
    if not k50.empty:
        combined = pd.concat([perseed, k50], ignore_index=True, sort=False)
        combined["column"] = pd.Categorical(
            combined["column"], categories=BLOCK_ORDER, ordered=True
        )
        combined = combined.sort_values(
            ["dataset", "column", "seed"]
        ).reset_index(drop=True)
        combined["column"] = combined["column"].astype(str)
        atomic_write_csv(
            combined,
            RESULTS_DIR / "analysis19_ce_blocksize_combined_perseed.csv",
        )

        combined_available = aggregate_metrics(
            combined,
            ALL_SEEDS,
            "available_seeds",
            require_complete=False,
        )
        combined_primary = aggregate_metrics(
            combined,
            PRIMARY_SEEDS,
            "primary_3_seeds",
            require_complete=True,
        )
        combined_mc10 = aggregate_metrics(
            combined,
            ALL_SEEDS,
            "all_10_seeds",
            require_complete=True,
        )
        atomic_write_csv(
            combined_available,
            RESULTS_DIR / "analysis19_ce_blocksize_combined_available.csv",
        )
        atomic_write_csv(
            combined_primary,
            RESULTS_DIR / "analysis19_ce_blocksize_combined_primary3.csv",
        )

        expected_groups = len(a17.CONFIGS) * len(BLOCK_ORDER)
        if len(combined_mc10) == expected_groups:
            atomic_write_csv(
                combined_mc10,
                RESULTS_DIR / "analysis19_ce_blocksize_combined_mc10.csv",
            )
            combined_mc10_written = True

        for dataset in sorted(a17.CONFIGS):
            observed = sorted(
                k50.loc[k50["dataset"] == dataset, "seed"].astype(int).unique()
            )
            missing = sorted(set(ALL_SEEDS).difference(observed))
            if missing:
                k50_missing[dataset] = missing

    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project_root": str(PROJECT_ROOT),
        "analysis17_source": str(A17_PATH),
        "primary_seeds": PRIMARY_SEEDS,
        "diagnostic_seeds": DIAGNOSTIC_SEEDS,
        "blocks_rerun": list(BLOCKS),
        "complete_cells": int(len(perseed)),
        "expected_cells": len(a17.CONFIGS) * len(BLOCKS) * len(ALL_SEEDS),
        "mc10_rows": int(len(mc10)),
        "mc10_written": mc10_written,
        "combined_mc10_written": combined_mc10_written,
        "missing_analysis17_k50_seeds": k50_missing,
        "model": CE_MODEL_NAME,
        "recipe": {
            "epochs": CE_EPOCHS,
            "warmup_fraction": CE_WARMUP_FRAC,
            "learning_rate": CE_LR,
            "batch_size": CE_BATCH,
            "max_length": CE_MAX_LEN,
            "threshold_grid": THRESHOLDS.tolist(),
            "score_path": "direct CrossEncoder.predict output",
        },
        "pair_protocol": {
            "positive": "first-listed valid B",
            "negative_exclusion": "all valid B alternatives",
            "retrained_per_block": True,
            "forced_1to1": "one positive plus highest-ranked nonmatch",
            "k5": "one positive plus all nonmatches in retrieved top five",
        },
        "counting": {
            "categories": "mutually exclusive correct/false/missing",
            "correct": "any valid B",
            "psi_denominator": "test A records with at least one true match",
        },
    }
    atomic_write_json(
        RESULTS_DIR / "analysis19_ce_blocksize_manifest.json",
        manifest,
    )

    print("\n" + "=" * 118)
    print("ANALYSIS 19: AVAILABLE BLOCK-SIZE RESULTS")
    print("=" * 118)
    if available.empty:
        print("No completed cells yet.")
    else:
        columns = [
            "dataset",
            "column",
            "n_splits",
            "lambda_mean",
            "lambda_sd",
            "lambda_se",
            "psi_mean",
            "psi_sd",
            "psi_se",
            "n_train_pairs_full_mean",
            "negative_count_mean_full_mean",
        ]
        print(available[columns].round(6).to_string(index=False))

    print("\nSaved:")
    print(f"  {RESULTS_DIR / 'analysis19_ce_blocks_perseed.csv'}")
    print(f"  {RESULTS_DIR / 'analysis19_ce_blocks_available.csv'}")
    print(f"  {RESULTS_DIR / 'analysis19_ce_blocks_primary3.csv'}")
    if mc10_written:
        print(f"  {RESULTS_DIR / 'analysis19_ce_blocks_mc10.csv'}")
    else:
        print(
            "  MC10 summary not regenerated by this execution "
            "(incomplete ten-seed set; existing file preserved)"
        )
    if not k50.empty:
        print(
            f"  {RESULTS_DIR / 'analysis19_ce_blocksize_combined_perseed.csv'}"
        )
        print(
            f"  {RESULTS_DIR / 'analysis19_ce_blocksize_combined_available.csv'}"
        )
        print(
            f"  {RESULTS_DIR / 'analysis19_ce_blocksize_combined_primary3.csv'}"
        )
        if combined_mc10_written:
            print(
                f"  {RESULTS_DIR / 'analysis19_ce_blocksize_combined_mc10.csv'}"
            )
        elif k50_missing:
            print("\nCombined MC10 not written: Analysis 17 k=50 is missing:")
            for dataset, seeds in k50_missing.items():
                print(f"  {dataset}: seeds {seeds}")


# =============================================================================
# CLI AND MAIN
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Corrected CE rerun for forced-1:1 and k=5, retrained "
            "independently per block size."
        )
    )
    parser.add_argument(
        "--phase",
        choices=["all", "run", "aggregate", "audit"],
        default="all",
        help="Run cells and aggregate, run cells only, aggregate only, or audit protocol only.",
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Run seeds 42, 43, 44 only. Default is all ten seeds.",
    )
    parser.add_argument(
        "--datasets",
        default="DBLP,ECOM",
        help="Comma-separated subset of DBLP,ECOM.",
    )
    parser.add_argument(
        "--blocks",
        default="forced-1:1,k=5",
        help="Comma-separated subset of forced-1:1,k=5.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_protocol_constants()

    datasets = [value.strip().upper() for value in args.datasets.split(",") if value.strip()]
    invalid_datasets = set(datasets).difference(a17.CONFIGS)
    if invalid_datasets:
        raise ValueError(f"Unknown datasets: {sorted(invalid_datasets)}")

    blocks = [value.strip() for value in args.blocks.split(",") if value.strip()]
    invalid_blocks = set(blocks).difference(BLOCKS)
    if invalid_blocks:
        raise ValueError(f"Unknown blocks: {sorted(invalid_blocks)}")

    seeds = PRIMARY_SEEDS if args.primary_only else ALL_SEEDS

    print("=" * 86)
    print("Analysis 19: corrected CE block-size completion")
    print("=" * 86)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Analysis 17 protocol source: {A17_PATH}")
    print(f"Phase: {args.phase}")
    print(f"Datasets: {datasets}")
    print(f"Blocks: {blocks}")
    print(f"Seeds: {seeds}")
    print("Cross-encoder retrained independently for every block-size cell: YES")
    print("k=50 model reused for smaller blocks: NO")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Visible CUDA devices: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Visible device 0: {torch.cuda.get_device_name(0)}")

    if args.phase not in ("aggregate", "audit") and torch.cuda.device_count() > 1:
        raise RuntimeError(
            "More than one GPU is visible. Pin the run explicitly, e.g. "
            "CUDA_VISIBLE_DEVICES=0 python scripts/analysis19_ce_blocksize.py"
        )

    if args.phase == "aggregate":
        save_outputs()
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = SentenceTransformer(BI_ENCODER_NAME, device=device)

    completed_index = 0
    total_requested = len(datasets) * len(blocks) * len(seeds)

    for dataset_name in datasets:
        dataset = a17.load_dataset(dataset_name)
        # Deliberately use Analysis 17's own preprocessing, field encoding,
        # title retrieval and top-50 construction. This also freezes the June
        # string-feature cache path, although the CE itself does not consume JW.
        a17.prepare_baseline(dataset, base_model)
        assert_k50_equivalence(dataset)

        if args.phase == "audit":
            del dataset
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        for column in blocks:
            for seed in seeds:
                completed_index += 1
                print(
                    f"\n{'=' * 86}\n"
                    f"[{completed_index}/{total_requested}] "
                    f"{dataset_name} {column} seed={seed} "
                    f"({'primary' if seed in PRIMARY_SEEDS else 'diagnostic'}) "
                    f"[{ts()}]\n"
                    f"{'=' * 86}"
                )
                run_cell(dataset, column, seed, device)
                save_outputs()

        del dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_outputs()
    if args.phase == "audit":
        print("\nAnalysis 19 protocol audit completed; no CE cells were trained.")
    else:
        print("\nAnalysis 19 completed for the requested cells.")


if __name__ == "__main__":
    main()
