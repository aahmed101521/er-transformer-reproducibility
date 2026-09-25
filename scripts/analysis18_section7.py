"""
analysis18_section7.py
======================

Corrected, resumable Section 7 fraction-sweep rerun for the
entity-resolution paper.

Scope
-----
Section 7 now contains only the gold-standard fraction sweep:

    f in {0.50, 0.60, 0.70, 0.80, 0.90}

For every (dataset, fraction, seed), the MiniLM cross-encoder is trained
on the labelled fraction sM_f and evaluated on the gold holdout A-minus-sM_f.
The old with/without-replacement proxy comparison is deliberately absent.
There are no proxy equations or proxy tables in this script.

Corrections held throughout
---------------------------
1. The complete raw B file remains intact.
2. Test truth is many-match: id_A -> set(all valid id_B).
3. A declared link to any valid B match is correct.
4. Correct links, false links, and missing matches are mutually exclusive.
5. The MMR denominator is the number of holdout A records with at least
   one true match.
6. Training remains one-to-one:
      - the first-listed valid B match is the sole positive;
      - every known valid alternative B match is excluded from negatives.

June / Section 7 protocol held fixed
------------------------------------
- raw files decoded as Latin-1;
- lowercase + strip preprocessing and the February-write / June-read
  pandas CSV round-trip;
- title-only blocking with all-MiniLM-L6-v2, k=50;
- MiniLM cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2;
- CE serialisation: fields joined with " [SEP] ", NaN serialised as "";
- 80/20 pair-level threshold-calibration split within sM_f;
- thresholds 0.10, 0.15, ..., 0.90 selected by validation F1;
- the analysis12 Section 7 score path is preserved: an additional sigmoid
  is applied to CrossEncoder.predict() before threshold calibration and
  holdout declaration. This isolates the data/counting corrections.

Seed design
-----------
Primary:     42, 43, 44
Diagnostic: 45, 46, 47, 48, 49, 50, 51
All:         42-51 (10 Monte Carlo replications per fraction)

Checkpointing
-------------
Each (dataset, fraction, seed) cell has:

- a saved CE model:
    models/analysis18_section7/cross_encoder/<DATASET>/f_<F>/seed_<SEED>/
- an atomic cell JSON:
    results/analysis18_section7/cells/
- an atomic per-record prediction CSV:
    results/analysis18_section7/predictions/

Completed cells are skipped. If model training finished but evaluation did
not, evaluation resumes from the saved model rather than retraining.

GPU pinning
-----------
Training requires exactly one visible CUDA device. Run with, for example:

    CUDA_VISIBLE_DEVICES=0 python scripts/analysis18_section7.py --primary-only

Recommended execution
---------------------
1. Optional: audit Analysis 17 DBLP Q4 assignment and outcomes:

    python scripts/analysis18_section7.py --phase audit-q4

2. Bank primary seeds:

    CUDA_VISIBLE_DEVICES=0 python scripts/analysis18_section7.py \
        --primary-only 2>&1 | tee results/analysis18_section7_primary.log

3. Resume and add diagnostic seeds:

    CUDA_VISIBLE_DEVICES=0 python scripts/analysis18_section7.py \
        2>&1 | tee results/analysis18_section7_diagnostic.log

4. Rebuild summaries without loading a model or requiring CUDA:

    python scripts/analysis18_section7.py --phase aggregate

Final outputs
-------------
results/analysis18_fraction_sweep_perseed.csv
results/analysis18_fraction_sweep_available.csv
results/analysis18_fraction_sweep_primary3.csv
results/analysis18_fraction_sweep_mc10.csv       # written only for complete R=10 groups
results/analysis18_section7_manifest.json
results/analysis18_q4_audit.csv                  # audit mode only

Monte Carlo reporting
---------------------
The aggregate files report mean, SD, SE, and the Section 7 Monte Carlo
coefficient of variation CV = SE/|mean|, matching the prior Section 7
convention. Replication CV = SD/|mean| is also saved as a diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from io import StringIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, InputExample, SentenceTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path(
    os.environ.get("ER_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).expanduser()

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

STATE_DIR = RESULTS_DIR / "analysis18_section7"
CELL_DIR = STATE_DIR / "cells"
PREDICTION_DIR = STATE_DIR / "predictions"
CACHE_DIR = STATE_DIR / "cache"
MODEL_ROOT = MODELS_DIR / "analysis18_section7" / "cross_encoder"

for directory in (
    RESULTS_DIR,
    MODELS_DIR,
    STATE_DIR,
    CELL_DIR,
    PREDICTION_DIR,
    CACHE_DIR,
    MODEL_ROOT,
):
    directory.mkdir(parents=True, exist_ok=True)

PRIMARY_SEEDS = [42, 43, 44]
DIAGNOSTIC_SEEDS = [45, 46, 47, 48, 49, 50, 51]
ALL_SEEDS = PRIMARY_SEEDS + DIAGNOSTIC_SEEDS
FRACTIONS = [0.50, 0.60, 0.70, 0.80, 0.90]

BI_ENCODER_NAME = "all-MiniLM-L6-v2"
CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

K = 50
EMBED_BATCH = 256
RETRIEVAL_CHUNK = 256
PREDICT_BATCH = 32

CE_EPOCHS = 5
CE_WARMUP_FRAC = 0.10
CE_LR = 2e-5
CE_BATCH = 16
CE_MAX_LEN = 128

VALIDATION_FRAC = 0.20
THRESHOLDS = np.arange(0.10, 0.90 + 1e-9, 0.05)

CONFIGS: dict[str, dict[str, Any]] = {
    "DBLP": {
        "a": DATA_DIR / "dblp" / "DBLP1.csv",
        "b": DATA_DIR / "dblp" / "Scholar.csv",
        "mapping": DATA_DIR / "dblp" / "DBLP-Scholar_perfectMapping.csv",
        "id_a": "idDBLP",
        "id_b": "idScholar",
        "rename": {},
        "fields": ["title", "authors", "venue", "year"],
        "retrieval_field": "title",
        "expected": {"A": 2616, "B": 64263, "pairs": 5347},
    },
    "ECOM": {
        "a": DATA_DIR / "abt_buy" / "Abt.csv",
        "b": DATA_DIR / "abt_buy" / "Buy.csv",
        "mapping": DATA_DIR / "abt_buy" / "abt_buy_perfectMapping.csv",
        "id_a": "idAbt",
        "id_b": "idBuy",
        "rename": {"name": "title", "description": "desc"},
        "fields": ["title", "desc"],
        "retrieval_field": "title",
        "expected": {"A": 1081, "B": 1092, "pairs": 1097},
    },
}


# =============================================================================
# GENERAL UTILITIES
# =============================================================================
def ts() -> str:
    return time.strftime("%H:%M:%S")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fraction_tag(fraction: float) -> str:
    return f"{fraction:.2f}".replace(".", "p")


def atomic_write_json(path: Path, payload: Any) -> None:
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


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(-values))


def parse_csv_values(raw: str, allowed: set[str], label: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}; allowed={sorted(allowed)}")
    if not values:
        raise ValueError(f"At least one {label} is required.")
    return values


def parse_fractions(raw: str) -> list[float]:
    values: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = round(float(token), 2)
        if value not in FRACTIONS:
            raise ValueError(
                f"Unsupported fraction {value}; allowed={FRACTIONS}"
            )
        values.append(value)
    if not values:
        raise ValueError("At least one fraction is required.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corrected, resumable Section 7 fraction sweep."
    )
    parser.add_argument(
        "--phase",
        choices=["all", "run", "aggregate", "audit-q4"],
        default="all",
        help="Run cells and aggregate, only cells, only summaries, or audit Analysis 17 Q4.",
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Run seeds 42, 43, and 44 only.",
    )
    parser.add_argument(
        "--datasets",
        default="DBLP,ECOM",
        help="Comma-separated subset of DBLP,ECOM.",
    )
    parser.add_argument(
        "--fractions",
        default=",".join(f"{value:.2f}" for value in FRACTIONS),
        help="Comma-separated subset of 0.50,0.60,0.70,0.80,0.90.",
    )
    parser.add_argument(
        "--analysis17-predictions",
        default=str(RESULTS_DIR / "analysis17_predictions.csv"),
        help="Prediction CSV used by --phase audit-q4.",
    )
    return parser.parse_args()


def require_single_visible_gpu() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Section 7 training is GPU-only; run on the "
            "server with CUDA_VISIBLE_DEVICES=<one GPU id>."
        )
    visible = torch.cuda.device_count()
    if visible != 1:
        raise RuntimeError(
            f"Expected exactly one visible CUDA device, found {visible}. "
            "Pin one GPU, for example CUDA_VISIBLE_DEVICES=0."
        )
    print(
        f"Pinned GPU: cuda:0 ({torch.cuda.get_device_name(0)}); "
        f"visible CUDA devices={visible}"
    )
    return "cuda"


# =============================================================================
# DATA LOADING: JUNE PREPROCESSING + COMPLETE RAW B
# =============================================================================
def clean_header(value: object) -> str:
    return (
        str(value)
        .replace("\ufeff", "")
        .replace("ï»¿", "")
        .strip()
        .strip('"')
        .strip()
    )


def read_latin1(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    frame = pd.read_csv(path, dtype=str, encoding="latin-1")
    frame.columns = [clean_header(column) for column in frame.columns]
    return frame


def find_id_column(frame: pd.DataFrame, path: Path) -> str:
    matches = [column for column in frame.columns if column.lower() == "id"]
    if len(matches) != 1:
        raise KeyError(
            f"Expected one id column in {path}; found {list(frame.columns)}"
        )
    return matches[0]


def normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower().str.strip()


def june_csv_roundtrip(frame: pd.DataFrame, id_column: str) -> pd.DataFrame:
    indexed = frame.set_index(id_column, drop=True)
    indexed.index.name = "id"

    buffer = StringIO()
    indexed.to_csv(buffer)
    buffer.seek(0)

    reloaded = pd.read_csv(buffer, dtype=str).set_index("id")
    reloaded.index = reloaded.index.astype(str)
    reloaded.index.name = "id"
    return reloaded


def serialise(row: pd.Series, fields: list[str]) -> str:
    parts: list[str] = []
    for field in fields:
        value = row.get(field, "")
        parts.append(
            str(value)
            if value is not None and str(value) != "nan"
            else ""
        )
    return " [SEP] ".join(parts)


def load_dataset(name: str) -> dict[str, Any]:
    cfg = CONFIGS[name]
    frame_a = read_latin1(cfg["a"])
    frame_b = read_latin1(cfg["b"])
    mapping = read_latin1(cfg["mapping"])

    id_a_file = find_id_column(frame_a, cfg["a"])
    id_b_file = find_id_column(frame_b, cfg["b"])

    frame_a = frame_a.rename(columns=cfg["rename"])
    frame_b = frame_b.rename(columns=cfg["rename"])

    for frame, path in ((frame_a, cfg["a"]), (frame_b, cfg["b"])):
        missing = set(cfg["fields"]).difference(frame.columns)
        if missing:
            raise KeyError(
                f"{path} is missing fields {sorted(missing)}; "
                f"columns={list(frame.columns)}"
            )
        for field in cfg["fields"]:
            frame[field] = normalize_text(frame[field])

    frame_a[id_a_file] = frame_a[id_a_file].fillna("").astype(str).str.strip()
    frame_b[id_b_file] = frame_b[id_b_file].fillna("").astype(str).str.strip()

    frame_a = june_csv_roundtrip(frame_a, id_a_file)
    frame_b = june_csv_roundtrip(frame_b, id_b_file)

    map_a = cfg["id_a"]
    map_b = cfg["id_b"]
    if map_a not in mapping.columns or map_b not in mapping.columns:
        raise KeyError(
            f"{cfg['mapping']} must contain {map_a!r}, {map_b!r}; "
            f"columns={list(mapping.columns)}"
        )

    mapping[map_a] = mapping[map_a].fillna("").astype(str).str.strip()
    mapping[map_b] = mapping[map_b].fillna("").astype(str).str.strip()

    observed = {"A": len(frame_a), "B": len(frame_b), "pairs": len(mapping)}
    if observed != cfg["expected"]:
        raise AssertionError(
            f"{name} snapshot mismatch: observed={observed}, "
            f"expected={cfg['expected']}"
        )

    if not frame_a.index.is_unique or not frame_b.index.is_unique:
        raise ValueError(f"{name}: A and B ids must be unique.")

    a_ids_set = set(frame_a.index)
    b_ids_set = set(frame_b.index)
    reachable = mapping[map_a].isin(a_ids_set) & mapping[map_b].isin(b_ids_set)
    if not bool(reachable.all()):
        bad = mapping.loc[~reachable, [map_a, map_b]]
        raise AssertionError(
            f"{name}: {len(bad)} raw mapping rows are unreachable. "
            f"Examples={bad.head(10).to_dict('records')}"
        )

    truth: dict[str, set[str]] = {}
    retained: dict[str, str] = {}
    for ida, idb in mapping[[map_a, map_b]].itertuples(index=False, name=None):
        truth.setdefault(ida, set()).add(idb)
        retained.setdefault(ida, idb)

    print(f"\n{name} input validation")
    print("-" * 78)
    print(
        f"|A|={len(frame_a):,}  |B|={len(frame_b):,}  "
        f"raw pairs={len(mapping):,}"
    )
    print(
        f"A records with truth={len(truth):,}  "
        f"A with >1 match={sum(len(value) > 1 for value in truth.values()):,}  "
        f"max matches/A={max(map(len, truth.values()))}"
    )
    print("Raw-file decoding: Latin-1")
    print("B candidate universe: complete raw B, duplicates retained")
    print("Training positive: first-listed valid B only")
    print("Training negatives: all known valid B matches excluded")
    print("Test correctness: any valid B match")
    print("Counting: correct / false / missing are mutually exclusive")

    return {
        "name": name,
        "cfg": cfg,
        "df_a": frame_a,
        "df_b": frame_b,
        "truth": truth,
        "retained": retained,
        "a_ids": list(frame_a.index),
        "b_ids": list(frame_b.index),
    }


# =============================================================================
# TITLE BLOCKING CACHE
# =============================================================================
def blocking_fingerprint(dataset: dict[str, Any]) -> str:
    field = dataset["cfg"]["retrieval_field"]
    digest = hashlib.sha256()
    digest.update(BI_ENCODER_NAME.encode("utf-8"))
    digest.update(str(K).encode("utf-8"))
    for label, frame in (("A", dataset["df_a"]), ("B", dataset["df_b"])):
        digest.update(label.encode("utf-8"))
        for record_id, value in zip(frame.index, frame[field].fillna("")):
            digest.update(str(record_id).encode("utf-8", errors="replace"))
            digest.update(b"\0")
            digest.update(str(value).encode("utf-8", errors="replace"))
            digest.update(b"\n")
    return digest.hexdigest()


def top50_cache_path(dataset_name: str) -> Path:
    return CACHE_DIR / f"{dataset_name}_top50.json"


def top50_cache_valid(payload: dict[str, Any], dataset: dict[str, Any]) -> bool:
    return (
        payload.get("dataset") == dataset["name"]
        and payload.get("bi_encoder") == BI_ENCODER_NAME
        and payload.get("k") == K
        and payload.get("n_a") == len(dataset["a_ids"])
        and payload.get("n_b") == len(dataset["b_ids"])
        and payload.get("fingerprint") == dataset["blocking_fingerprint"]
        and set(payload.get("top50", {})) == set(dataset["a_ids"])
    )


def compute_top50(
    dataset: dict[str, Any],
    model: SentenceTransformer,
) -> dict[str, list[str]]:
    field = dataset["cfg"]["retrieval_field"]
    print(f"\n{dataset['name']} title blocking")
    print("-" * 78)
    print(f"Encoding A.{field} [{ts()}]")
    emb_a = model.encode(
        dataset["df_a"][field].fillna("").tolist(),
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=EMBED_BATCH,
        normalize_embeddings=True,
    )
    print(f"Encoding B.{field} [{ts()}]")
    emb_b = model.encode(
        dataset["df_b"][field].fillna("").tolist(),
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=EMBED_BATCH,
        normalize_embeddings=True,
    )

    top50: dict[str, list[str]] = {}
    transpose_b = emb_b.transpose(0, 1)
    for start in range(0, len(dataset["a_ids"]), RETRIEVAL_CHUNK):
        stop = min(start + RETRIEVAL_CHUNK, len(dataset["a_ids"]))
        scores = torch.matmul(emb_a[start:stop], transpose_b)
        indices = torch.topk(scores, k=K, dim=1).indices.cpu().tolist()
        for offset, row_indices in enumerate(indices):
            ida = dataset["a_ids"][start + offset]
            top50[ida] = [dataset["b_ids"][index] for index in row_indices]
        print(
            f"Retrieved {stop:,}/{len(dataset['a_ids']):,} A records",
            end="\r",
            flush=True,
        )
    print()
    return top50


def attach_top50(
    datasets: list[dict[str, Any]],
    device: str,
) -> None:
    missing: list[dict[str, Any]] = []
    for dataset in datasets:
        path = top50_cache_path(dataset["name"])
        if path.exists():
            payload = read_json(path)
            if top50_cache_valid(payload, dataset):
                dataset["top50"] = payload["top50"]
                print(f"Loaded top-50 cache: {path}")
                continue
            print(f"Ignoring stale top-50 cache: {path}")
        missing.append(dataset)

    if not missing:
        return

    print(f"\nLoading retrieval bi-encoder on {device}: {BI_ENCODER_NAME}")
    model = SentenceTransformer(BI_ENCODER_NAME, device=device)
    for dataset in missing:
        top50 = compute_top50(dataset, model)
        dataset["top50"] = top50
        payload = {
            "dataset": dataset["name"],
            "bi_encoder": BI_ENCODER_NAME,
            "k": K,
            "n_a": len(dataset["a_ids"]),
            "n_b": len(dataset["b_ids"]),
            "fingerprint": dataset["blocking_fingerprint"],
            "top50": top50,
        }
        atomic_write_json(top50_cache_path(dataset["name"]), payload)
        print(f"Saved top-50 cache: {top50_cache_path(dataset['name'])}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# PAIRS, THRESHOLDS, AND COUNTING
# =============================================================================
def build_train_pairs(
    ids: Iterable[str],
    retained: dict[str, str],
    truth: dict[str, set[str]],
    top50: dict[str, list[str]],
) -> list[tuple[str, str, int]]:
    pairs: list[tuple[str, str, int]] = []
    for ida in ids:
        positive = retained.get(ida)
        if positive is None:
            continue
        pairs.append((ida, positive, 1))
        true_set = truth[ida]
        for idb in top50[ida]:
            if idb not in true_set:
                pairs.append((ida, idb, 0))
    return pairs


def choose_threshold(
    scores: np.ndarray,
    labels: Iterable[int],
) -> tuple[float, float]:
    labels_array = np.asarray(list(labels), dtype=int)
    best_threshold = 0.50
    best_f1 = -1.0
    for threshold in THRESHOLDS:
        score = f1_score(
            labels_array,
            (np.asarray(scores) > threshold).astype(int),
            zero_division=0,
        )
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def category_for(
    declared: bool,
    top_idb: str,
    true_set: set[str],
) -> str:
    if declared:
        return "correct" if top_idb in true_set else "false_link"
    return "missing_match" if true_set else "unmatched_abstention"


def metrics_from_predictions(frame: pd.DataFrame) -> dict[str, float | int]:
    declared = int(frame["declared"].sum())
    correct = int((frame["category"] == "correct").sum())
    false_links = int((frame["category"] == "false_link").sum())
    missing_matches = int((frame["category"] == "missing_match").sum())
    unmatched_abstentions = int(
        (frame["category"] == "unmatched_abstention").sum()
    )
    n_true_records = int(frame["has_true_match"].sum())

    if declared != correct + false_links:
        raise AssertionError("Declared does not equal correct + false links.")
    if len(frame) != correct + false_links + missing_matches + unmatched_abstentions:
        raise AssertionError("Prediction categories are not exhaustive.")

    return {
        "lambda": false_links / declared if declared else 0.0,
        "psi": missing_matches / n_true_records if n_true_records else 0.0,
        "declared": declared,
        "correct": correct,
        "false_links": false_links,
        "missing_matches": missing_matches,
        "unmatched_abstentions": unmatched_abstentions,
        "n_true_records": n_true_records,
        "n_holdout": len(frame),
    }


# =============================================================================
# MODEL AND CELL CHECKPOINTS
# =============================================================================
def model_path(dataset_name: str, fraction: float, seed: int) -> Path:
    return (
        MODEL_ROOT
        / dataset_name
        / f"f_{fraction_tag(fraction)}"
        / f"seed_{seed}"
    )


def model_marker(path: Path) -> Path:
    return path / "_analysis18_complete.json"


def model_complete(path: Path) -> bool:
    return path.is_dir() and model_marker(path).exists()


def save_model_atomic(
    model: CrossEncoder,
    path: Path,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    safe_remove(temporary)
    model.save_pretrained(str(temporary))
    atomic_write_json(temporary / "_analysis18_complete.json", metadata)
    safe_remove(path)
    os.replace(temporary, path)


def cell_path(dataset_name: str, fraction: float, seed: int) -> Path:
    return CELL_DIR / (
        f"fraction_{dataset_name}_f{fraction_tag(fraction)}_seed{seed}.json"
    )


def prediction_path(dataset_name: str, fraction: float, seed: int) -> Path:
    return PREDICTION_DIR / (
        f"fraction_{dataset_name}_f{fraction_tag(fraction)}_seed{seed}.csv"
    )


def cell_complete(dataset_name: str, fraction: float, seed: int) -> bool:
    return (
        cell_path(dataset_name, fraction, seed).exists()
        and prediction_path(dataset_name, fraction, seed).exists()
    )


def train_or_load_model(
    dataset: dict[str, Any],
    fraction: float,
    seed: int,
    train_only: list[tuple[str, str, int]],
    device: str,
) -> tuple[CrossEncoder, bool, float]:
    path = model_path(dataset["name"], fraction, seed)
    if model_complete(path):
        print(f"      loading CE checkpoint: {path}")
        return (
            CrossEncoder(str(path), max_length=CE_MAX_LEN, device=device),
            True,
            0.0,
        )

    fields = dataset["cfg"]["fields"]
    examples = [
        InputExample(
            texts=[
                serialise(dataset["df_a"].loc[ida], fields),
                serialise(dataset["df_b"].loc[idb], fields),
            ],
            label=float(label),
        )
        for ida, idb, label in train_only
    ]
    if not examples:
        raise ValueError("No CE training examples were generated.")

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
        f"      training CE: pairs={len(examples):,}, epochs={CE_EPOCHS}, "
        f"warmup={warmup_steps}, lr={CE_LR}, batch={CE_BATCH}, "
        f"max_length={CE_MAX_LEN} [{ts()}]"
    )
    started = time.time()
    model.fit(
        train_dataloader=loader,
        epochs=CE_EPOCHS,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": CE_LR},
        show_progress_bar=False,
    )
    train_seconds = time.time() - started

    save_model_atomic(
        model,
        path,
        {
            "phase": "section7_fraction_sweep",
            "dataset": dataset["name"],
            "fraction": fraction,
            "seed": seed,
            "model": CE_MODEL_NAME,
            "score_path": "sigmoid(CrossEncoder.predict)",
            "epochs": CE_EPOCHS,
            "warmup_fraction": CE_WARMUP_FRAC,
            "learning_rate": CE_LR,
            "batch_size": CE_BATCH,
            "max_length": CE_MAX_LEN,
            "fit_pairs": len(examples),
            "train_seconds": train_seconds,
        },
    )
    print(f"      saved CE checkpoint: {path}")
    return model, False, train_seconds


def ce_predict(model: CrossEncoder, pairs: list[tuple[str, str]]) -> np.ndarray:
    return np.asarray(
        model.predict(
            pairs,
            batch_size=PREDICT_BATCH,
            show_progress_bar=False,
        ),
        dtype=np.float64,
    ).reshape(-1)


# =============================================================================
# ONE FRACTION-SWEEP CELL
# =============================================================================
def run_cell(
    dataset: dict[str, Any],
    fraction: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    result_path = cell_path(dataset["name"], fraction, seed)
    predictions_path = prediction_path(dataset["name"], fraction, seed)
    if cell_complete(dataset["name"], fraction, seed):
        print(
            f"    SKIP completed: {dataset['name']} f={fraction:.2f} "
            f"seed={seed}"
        )
        return read_json(result_path)

    seed_everything(seed)
    labelled_ids, holdout_ids = train_test_split(
        dataset["a_ids"],
        train_size=fraction,
        random_state=seed,
    )

    train_pairs = build_train_pairs(
        labelled_ids,
        dataset["retained"],
        dataset["truth"],
        dataset["top50"],
    )
    labels = [label for _, _, label in train_pairs]
    if len(set(labels)) < 2:
        raise ValueError("Training pairs do not contain both classes.")

    fit_indices, validation_indices = train_test_split(
        list(range(len(train_pairs))),
        test_size=VALIDATION_FRAC,
        random_state=seed,
        stratify=labels,
    )
    train_only = [train_pairs[index] for index in fit_indices]
    validation_only = [train_pairs[index] for index in validation_indices]

    print(
        f"      labelled A={len(labelled_ids):,}; holdout A={len(holdout_ids):,}; "
        f"all pairs={len(train_pairs):,}; fit={len(train_only):,}; "
        f"validation={len(validation_only):,}"
    )

    model, loaded_checkpoint, train_seconds = train_or_load_model(
        dataset,
        fraction,
        seed,
        train_only,
        device,
    )

    fields = dataset["cfg"]["fields"]
    validation_text = [
        (
            serialise(dataset["df_a"].loc[ida], fields),
            serialise(dataset["df_b"].loc[idb], fields),
        )
        for ida, idb, _ in validation_only
    ]
    validation_labels = [label for _, _, label in validation_only]
    validation_scores = sigmoid(ce_predict(model, validation_text))
    threshold, validation_f1 = choose_threshold(
        validation_scores,
        validation_labels,
    )
    print(
        f"      threshold={threshold:.2f}; validation F1={validation_f1:.4f}; "
        f"score path=extra sigmoid"
    )

    rows: list[dict[str, Any]] = []
    evaluation_started = time.time()
    for record_index, ida in enumerate(holdout_ids, start=1):
        true_set = dataset["truth"].get(ida, set())
        candidates = dataset["top50"][ida]
        candidate_text = [
            (
                serialise(dataset["df_a"].loc[ida], fields),
                serialise(dataset["df_b"].loc[idb], fields),
            )
            for idb in candidates
        ]
        direct_scores = ce_predict(model, candidate_text)
        scores = sigmoid(direct_scores)
        order = np.argsort(-scores)
        top1_index = int(order[0])
        top2_index = int(order[1])
        top1_idb = candidates[top1_index]
        top1_score = float(scores[top1_index])
        top2_score = float(scores[top2_index])
        declared = bool(top1_score > threshold)
        category = category_for(declared, top1_idb, true_set)

        rows.append(
            {
                "dataset": dataset["name"],
                "fraction": fraction,
                "seed": seed,
                "record_index": record_index - 1,
                "ida": ida,
                "has_true_match": bool(true_set),
                "n_valid_matches": len(true_set),
                "retained_idb": dataset["retained"].get(ida, ""),
                "top1_idb": top1_idb,
                "top1_is_any_valid": top1_idb in true_set,
                "top1_score": top1_score,
                "top2_score": top2_score,
                "score_gap": top1_score - top2_score,
                "threshold": threshold,
                "declared": declared,
                "category": category,
            }
        )

        if record_index % 100 == 0 or record_index == len(holdout_ids):
            print(
                f"        evaluated {record_index:,}/{len(holdout_ids):,}",
                end="\r",
                flush=True,
            )
    print()

    predictions = pd.DataFrame(rows)
    metrics = metrics_from_predictions(predictions)
    evaluation_seconds = time.time() - evaluation_started

    atomic_write_csv(predictions, predictions_path)
    result: dict[str, Any] = {
        "dataset": dataset["name"],
        "fraction": fraction,
        "seed": seed,
        "lambda": metrics["lambda"],
        "psi": metrics["psi"],
        "declared": metrics["declared"],
        "correct": metrics["correct"],
        "false_links": metrics["false_links"],
        "missing_matches": metrics["missing_matches"],
        "unmatched_abstentions": metrics["unmatched_abstentions"],
        "n_true_records": metrics["n_true_records"],
        "n_holdout": metrics["n_holdout"],
        "n_labelled_a": len(labelled_ids),
        "n_train_pairs_total": len(train_pairs),
        "n_fit_pairs": len(train_only),
        "n_validation_pairs": len(validation_only),
        "threshold": threshold,
        "validation_f1": validation_f1,
        "model_loaded_from_checkpoint": loaded_checkpoint,
        "train_seconds_this_run": train_seconds,
        "evaluation_seconds": evaluation_seconds,
        "score_path": "sigmoid(CrossEncoder.predict)",
        "model_path": str(model_path(dataset["name"], fraction, seed)),
        "prediction_path": str(predictions_path),
    }
    atomic_write_json(result_path, result)

    print(
        f"      result: lambda={metrics['lambda']:.4f}; "
        f"psi={metrics['psi']:.4f}; declared={metrics['declared']:,}; "
        f"false={metrics['false_links']:,}; "
        f"missing={metrics['missing_matches']:,}"
    )
    print(f"      saved cell: {result_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


# =============================================================================
# AGGREGATION AND MONTE CARLO ERROR
# =============================================================================
def load_cell_frame() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(CELL_DIR.glob("fraction_*.json")):
        payload = read_json(path)
        required = {"dataset", "fraction", "seed", "lambda", "psi"}
        if required.issubset(payload):
            rows.append(payload)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["fraction"] = frame["fraction"].astype(float).round(2)
    frame["seed"] = frame["seed"].astype(int)
    return frame.sort_values(["dataset", "fraction", "seed"]).reset_index(drop=True)


def metric_summary(values: pd.Series, prefix: str) -> dict[str, float]:
    array = values.astype(float).to_numpy()
    n = len(array)
    mean = float(np.mean(array)) if n else float("nan")
    sd = float(np.std(array, ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n else float("nan")
    absolute_mean = abs(mean)
    cv = se / absolute_mean if absolute_mean > 0 else float("nan")
    replication_cv = sd / absolute_mean if absolute_mean > 0 else float("nan")
    relative_mc_se = cv
    halfwidth_95 = 1.96 * se
    relative_halfwidth_95 = (
        halfwidth_95 / absolute_mean if absolute_mean > 0 else float("nan")
    )
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_sd": sd,
        f"{prefix}_se": se,
        f"{prefix}_cv": cv,
        f"{prefix}_replication_cv": replication_cv,
        f"{prefix}_relative_mc_se": relative_mc_se,
        f"{prefix}_mc95_halfwidth": halfwidth_95,
        f"{prefix}_relative_mc95_halfwidth": relative_halfwidth_95,
        f"{prefix}_min": float(np.min(array)) if n else float("nan"),
        f"{prefix}_max": float(np.max(array)) if n else float("nan"),
    }


def aggregate_frame(
    frame: pd.DataFrame,
    seeds: list[int] | None,
    label: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    subset = frame.copy()
    if seeds is not None:
        subset = subset[subset["seed"].isin(seeds)].copy()
    if subset.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (dataset_name, fraction), group in subset.groupby(
        ["dataset", "fraction"], sort=True
    ):
        row: dict[str, Any] = {
            "aggregation": label,
            "dataset": dataset_name,
            "fraction": float(fraction),
            "n_splits": len(group),
            "seeds": ",".join(str(value) for value in sorted(group["seed"])),
        }
        row.update(metric_summary(group["lambda"], "lambda"))
        row.update(metric_summary(group["psi"], "psi"))
        for column in (
            "declared",
            "correct",
            "false_links",
            "missing_matches",
            "n_true_records",
            "n_holdout",
            "n_labelled_a",
            "n_fit_pairs",
            "n_validation_pairs",
        ):
            if column in group:
                row[f"{column}_mean"] = float(group[column].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["dataset", "fraction"]
    ).reset_index(drop=True)


def print_aggregate(title: str, frame: pd.DataFrame) -> None:
    print("\n" + "=" * 118)
    print(title)
    print("=" * 118)
    if frame.empty:
        print("No complete cells available.")
        return
    print(
        f"{'data':<6} {'f':>4} {'R':>3}  "
        f"{'lambda':>8} {'SE':>8} {'relSE':>8} {'CV':>8}  "
        f"{'psi':>8} {'SE':>8} {'relSE':>8} {'CV':>8}"
    )
    for row in frame.itertuples(index=False):
        print(
            f"{row.dataset:<6} {row.fraction:>4.2f} {row.n_splits:>3}  "
            f"{row.lambda_mean:>8.4f} {row.lambda_se:>8.4f} "
            f"{row.lambda_relative_mc_se:>8.3f} {row.lambda_cv:>8.3f}  "
            f"{row.psi_mean:>8.4f} {row.psi_se:>8.4f} "
            f"{row.psi_relative_mc_se:>8.3f} {row.psi_cv:>8.3f}"
        )
    print("\nCV = relSE = SE / |mean|, matching the prior Section 7 convention.")
    print("Replication CV = SD / |mean| is retained in the CSV as a diagnostic.")
    print("Large CV or a wide 95% MC half-width is itself a Section 7 finding.")


def save_manifest(perseed: pd.DataFrame) -> None:
    complete_counts: dict[str, dict[str, int]] = {}
    if not perseed.empty:
        for (dataset_name, fraction), group in perseed.groupby(
            ["dataset", "fraction"], sort=True
        ):
            complete_counts.setdefault(dataset_name, {})[f"{fraction:.2f}"] = len(group)

    manifest = {
        "analysis": "analysis18_section7",
        "scope": "gold-standard fraction sweep only; no proxy comparison",
        "fractions": FRACTIONS,
        "primary_seeds": PRIMARY_SEEDS,
        "diagnostic_seeds": DIAGNOSTIC_SEEDS,
        "all_seeds": ALL_SEEDS,
        "k": K,
        "bi_encoder": BI_ENCODER_NAME,
        "cross_encoder": CE_MODEL_NAME,
        "cross_encoder_recipe": {
            "epochs": CE_EPOCHS,
            "warmup_fraction": CE_WARMUP_FRAC,
            "learning_rate": CE_LR,
            "batch_size": CE_BATCH,
            "max_length": CE_MAX_LEN,
            "score_path": "sigmoid(CrossEncoder.predict)",
            "validation_fraction_of_pairs": VALIDATION_FRAC,
            "threshold_grid": [float(value) for value in THRESHOLDS],
        },
        "corrections": {
            "complete_raw_b": True,
            "many_match_truth": True,
            "any_valid_match_is_correct": True,
            "mutually_exclusive_counting": True,
            "psi_denominator": "holdout A records with at least one true match",
            "training_positive": "first-listed valid B",
            "alternative_valid_matches_excluded_from_negatives": True,
        },
        "monte_carlo_columns": {
            "cv": "Monte Carlo SE divided by absolute mean (prior Section 7 convention)",
            "relative_mc_se": "same quantity as CV, named explicitly",
            "replication_cv": "sample SD divided by absolute mean",
            "mc95_halfwidth": "1.96 times Monte Carlo SE",
        },
        "complete_cell_counts": complete_counts,
    }
    atomic_write_json(RESULTS_DIR / "analysis18_section7_manifest.json", manifest)


def aggregate_results() -> None:
    perseed = load_cell_frame()
    perseed_path = RESULTS_DIR / "analysis18_fraction_sweep_perseed.csv"
    available_path = RESULTS_DIR / "analysis18_fraction_sweep_available.csv"
    primary_path = RESULTS_DIR / "analysis18_fraction_sweep_primary3.csv"
    mc10_path = RESULTS_DIR / "analysis18_fraction_sweep_mc10.csv"

    if perseed.empty:
        print("No complete Section 7 cells found.")
        return

    atomic_write_csv(perseed, perseed_path)

    available = aggregate_frame(perseed, None, "available_cells")
    primary = aggregate_frame(perseed, PRIMARY_SEEDS, "primary_3_seeds")
    mc10 = aggregate_frame(perseed, ALL_SEEDS, "all_10_seeds")

    atomic_write_csv(available, available_path)
    atomic_write_csv(primary, primary_path)

    complete_mc10 = mc10[mc10["n_splits"] == len(ALL_SEEDS)].copy()
    if len(complete_mc10) == len(mc10) and not complete_mc10.empty:
        atomic_write_csv(complete_mc10, mc10_path)
        mc10_status = f"written: {mc10_path}"
    else:
        safe_remove(mc10_path)
        incomplete = mc10[mc10["n_splits"] != len(ALL_SEEDS)][
            ["dataset", "fraction", "n_splits"]
        ]
        mc10_status = (
            "not written because R=10 is incomplete; incomplete groups="
            + incomplete.to_dict("records").__repr__()
        )

    save_manifest(perseed)
    print_aggregate("SECTION 7 FRACTION SWEEP: AVAILABLE CELLS", available)
    print_aggregate("SECTION 7 FRACTION SWEEP: PRIMARY SEEDS", primary)
    if not complete_mc10.empty and len(complete_mc10) == len(mc10):
        print_aggregate("SECTION 7 FRACTION SWEEP: ALL TEN SEEDS", complete_mc10)

    print("\nSaved:")
    print(f"  {perseed_path}")
    print(f"  {available_path}")
    print(f"  {primary_path}")
    print(f"  MC10: {mc10_status}")
    print(f"  {RESULTS_DIR / 'analysis18_section7_manifest.json'}")


# =============================================================================
# ANALYSIS 17 Q4 AUDIT
# =============================================================================
def audit_analysis17_q4(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Analysis 17 prediction file not found: {path}")
    frame = pd.read_csv(path)

    if "lr_gap" not in frame.columns:
        required = {"lr_top1_score", "lr_top2_score"}
        if not required.issubset(frame.columns):
            raise KeyError(
                "Analysis 17 predictions require lr_gap or both LR score columns."
            )
        frame["lr_gap"] = frame["lr_top1_score"] - frame["lr_top2_score"]

    required = {"dataset", "seed", "lr_gap", "lr_category"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Analysis 17 prediction file is missing {sorted(missing)}")

    if "ce_legacy_category" in frame.columns:
        ce_column = "ce_legacy_category"
        ce_label = "ce_legacy"
    elif "ce_direct_category" in frame.columns:
        ce_column = "ce_direct_category"
        ce_label = "ce_direct"
    else:
        raise KeyError(
            "Analysis 17 predictions need ce_legacy_category or ce_direct_category."
        )

    rows: list[dict[str, Any]] = []
    for (dataset_name, seed), subset in frame.groupby(
        ["dataset", "seed"], sort=True
    ):
        subset = subset.copy()
        subset["quartile"] = pd.qcut(
            subset["lr_gap"],
            4,
            labels=["Q1", "Q2", "Q3", "Q4"],
            duplicates="drop",
        )
        if subset["quartile"].nunique(dropna=True) != 4:
            raise AssertionError(
                f"{dataset_name} seed={seed}: qcut did not produce four quartiles."
            )

        q3 = subset[subset["quartile"] == "Q3"]
        q4 = subset[subset["quartile"] == "Q4"]
        q4_lr_counts = q4["lr_category"].value_counts()
        q4_ce_counts = q4[ce_column].value_counts()
        rows.append(
            {
                "dataset": dataset_name,
                "seed": int(seed),
                "n_records": len(subset),
                "n_unique_lr_gaps": int(subset["lr_gap"].nunique()),
                "q4_n": len(q4),
                "q3_gap_max": float(q3["lr_gap"].max()),
                "q4_gap_min": float(q4["lr_gap"].min()),
                "q4_gap_max": float(q4["lr_gap"].max()),
                "q4_strictly_above_q3": bool(
                    float(q4["lr_gap"].min()) >= float(q3["lr_gap"].max())
                ),
                "q4_lr_correct": int(q4_lr_counts.get("correct", 0)),
                "q4_lr_false": int(q4_lr_counts.get("false_link", 0)),
                "q4_lr_missing": int(q4_lr_counts.get("missing_match", 0)),
                f"q4_{ce_label}_correct": int(q4_ce_counts.get("correct", 0)),
                f"q4_{ce_label}_false": int(q4_ce_counts.get("false_link", 0)),
                f"q4_{ce_label}_missing": int(q4_ce_counts.get("missing_match", 0)),
                "q4_lr_perfect": bool((q4["lr_category"] == "correct").all()),
                f"q4_{ce_label}_perfect": bool((q4[ce_column] == "correct").all()),
            }
        )

    audit = pd.DataFrame(rows).sort_values(["dataset", "seed"])
    output = RESULTS_DIR / "analysis18_q4_audit.csv"
    atomic_write_csv(audit, output)

    print("\n" + "=" * 122)
    print("ANALYSIS 17 Q4 AUDIT")
    print("=" * 122)
    print(f"Quartiles are recomputed per dataset and seed from lr_gap with Q1=smallest and Q4=largest.")
    print(f"Cross-encoder category column audited: {ce_column}")
    columns = [
        "dataset",
        "seed",
        "n_records",
        "n_unique_lr_gaps",
        "q4_n",
        "q3_gap_max",
        "q4_gap_min",
        "q4_gap_max",
        "q4_lr_correct",
        "q4_lr_false",
        "q4_lr_missing",
        f"q4_{ce_label}_correct",
        f"q4_{ce_label}_false",
        f"q4_{ce_label}_missing",
    ]
    print(audit[columns].to_string(index=False))

    dblp = audit[audit["dataset"] == "DBLP"]
    if not dblp.empty:
        ordered = bool(dblp["q4_strictly_above_q3"].all())
        nondegenerate = bool((dblp["n_unique_lr_gaps"] > 4).all())
        lr_perfect = bool(dblp["q4_lr_perfect"].all())
        ce_perfect = bool(dblp[f"q4_{ce_label}_perfect"].all())
        print("\nDBLP conclusion")
        print(f"  Four ordered, non-overlapping quartiles: {ordered}")
        print(f"  Non-degenerate LR-gap distribution: {nondegenerate}")
        print(f"  Two-step LR perfect in Q4 for every saved seed: {lr_perfect}")
        print(f"  Cross-encoder perfect in Q4 for every saved seed: {ce_perfect}")
        print(
            "  Interpretation: Q4 is the upper decision-margin quartile. "
            "A perfect Q4 is genuine conditional performance on the "
            "highest-margin records; it does not by itself prove that every "
            "record is a literal textual near-duplicate."
        )
    print(f"\nSaved audit: {output}")
    return audit


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    args = parse_args()

    if args.phase == "audit-q4":
        audit_analysis17_q4(Path(args.analysis17_predictions).expanduser())
        return

    if args.phase == "aggregate":
        aggregate_results()
        return

    dataset_names = parse_csv_values(
        args.datasets,
        set(CONFIGS),
        "dataset",
    )
    fractions = parse_fractions(args.fractions)
    seeds = PRIMARY_SEEDS if args.primary_only else ALL_SEEDS

    print("=" * 92)
    print("analysis18_section7: corrected gold-standard fraction sweep")
    print("=" * 92)
    print("Proxy comparison: removed")
    print(f"Datasets: {dataset_names}")
    print(f"Fractions: {fractions}")
    print(f"Seeds: {seeds}")
    print(f"Total requested cells: {len(dataset_names) * len(fractions) * len(seeds)}")
    print(f"Cross-encoder: {CE_MODEL_NAME}")
    print(
        f"Recipe: {CE_EPOCHS} epochs, {CE_WARMUP_FRAC:.0%} warmup, "
        f"lr={CE_LR}, batch={CE_BATCH}, max_length={CE_MAX_LEN}"
    )
    print("Score path: sigmoid(CrossEncoder.predict), preserving analysis12")
    print("Counting: mutually exclusive; psi denominator = truth-bearing holdout records")

    device = require_single_visible_gpu()
    datasets = [load_dataset(name) for name in dataset_names]
    for dataset in datasets:
        dataset["blocking_fingerprint"] = blocking_fingerprint(dataset)
    attach_top50(datasets, device)

    cells = [
        (seed, fraction, dataset)
        for seed in seeds
        for fraction in fractions
        for dataset in datasets
    ]
    started = time.time()
    for index, (seed, fraction, dataset) in enumerate(cells, start=1):
        print("\n" + "=" * 92)
        print(
            f"[{index}/{len(cells)}] {dataset['name']} "
            f"f={fraction:.2f} seed={seed} [{ts()}] "
            f"elapsed={(time.time() - started) / 60:.1f}m"
        )
        print("=" * 92)
        run_cell(dataset, fraction, seed, device)

    if args.phase == "all":
        aggregate_results()

    print("\nAnalysis 18 Section 7 completed for the requested cells.")


if __name__ == "__main__":
    main()
