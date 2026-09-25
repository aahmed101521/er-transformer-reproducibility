r"""
analysis22_section332_bias.py
================================

Checkpointed reproduction of revised-paper Section 3.3.2.

This is a new experiment. It does not import, read, or reuse analysis13 proxy
code or proxy results.

For each dataset, outer labelling fraction f, and seed:

1. Draw sM_f from source-A records by SRSWOR and define the outer holdout
   A \ sM_f.
2. Obtain the target cross-encoder error (FLR, MMR) for the model trained under
   the corrected analysis18 fraction-sweep protocol on sM_f and evaluated on
   A \ sM_f. Exact analysis18 target results/checkpoints are reused only after
   protocol and split validation; otherwise the target is retrained.
3. Treat sM_f as the labelled sample and construct two inner assessments:
   a. SRSWOR 65/35: train on 65% of sM_f and test on the remaining 35%.
   b. Bootstrap/OOB: draw |sM_f| source items with replacement and test on the
      out-of-bag source items.
4. Fine-tune a separate cross-encoder for each inner scheme and report
      bias_FLR = FLR_star - target_FLR
      bias_MMR = MMR_star - target_MMR.

Corrected protocol retained from analyses 17/18
------------------------------------------------
- complete raw B retained;
- complete many-match truth at evaluation;
- any valid B partner counts as correct;
- correct, false-link, and missing-match categories are mutually exclusive;
- MMR denominator is test source items with at least one true partner;
- first-listed retained partner is the sole positive for training;
- all alternative valid partners are excluded from the negative class;
- Latin-1 decoding and the frozen June preprocessing/CSV round-trip;
- title-only scheme-II blocking at k=50 with one fixed all-MiniLM-L6-v2 index;
- cross-encoder/ms-marco-MiniLM-L-6-v2;
- 5 epochs, 10% warmup, lr=2e-5, batch=16, max_length=128;
- the analysis18 threshold protocol: 80/20 pair-level calibration split,
  threshold grid 0.10,...,0.90, and sigmoid(CrossEncoder.predict).

Checkpointing
-------------
Each target and inner-scheme result is written atomically to its own JSON and
prediction CSV. Each trained model is checkpointed separately. A restart loses
at most the currently running model/evaluation unit.

Default paths (relative to ER_PROJECT_ROOT or the repository root)
------------------------------------------------------------------
results/analysis22_section332_bias/cells/
results/analysis22_section332_bias/predictions/
results/analysis22_section332_bias/splits/
models/analysis22_section332_bias/cross_encoder/

Final outputs
-------------
results/analysis22_section332_bias_perseed.csv
results/analysis22_section332_bias_aggregate.csv
results/analysis22_section332_bias_mc10.csv
results/analysis22_section332_sample_sizes.csv
results/analysis22_section332_manifest.json

Recommended commands
--------------------
Audit analysis18 target reuse without training:

    python scripts/analysis22_section332_bias.py --phase audit-targets \
        --target-reuse require

Print deterministic sample sizes without training:

    python scripts/analysis22_section332_bias.py --phase sizes

Run, pinned to exactly one GPU:

    CUDA_VISIBLE_DEVICES=0 python scripts/analysis22_section332_bias.py \
        --target-reuse require 2>&1 | tee results/analysis22_section332_bias.log

Resume with the same command. Rebuild summaries without CUDA:

    python scripts/analysis22_section332_bias.py --phase aggregate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


# =============================================================================
# CONFIGURATION
# =============================================================================
ANALYSIS_NAME = "analysis22_section332_bias"
PROTOCOL_VERSION = "analysis22-v2-seeded-2026-07-28"
PROJECT_ROOT = Path(
    os.environ.get("ER_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).expanduser()

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

STATE_DIR = RESULTS_DIR / ANALYSIS_NAME
CELL_DIR = STATE_DIR / "cells"
PREDICTION_DIR = STATE_DIR / "predictions"
SPLIT_DIR = STATE_DIR / "splits"
CACHE_DIR = STATE_DIR / "cache"
MODEL_ROOT = MODELS_DIR / ANALYSIS_NAME / "cross_encoder"

ANALYSIS18_STATE_DIR = RESULTS_DIR / "analysis18_section7"
ANALYSIS18_CELL_DIR = ANALYSIS18_STATE_DIR / "cells"
ANALYSIS18_PREDICTION_DIR = ANALYSIS18_STATE_DIR / "predictions"
ANALYSIS18_CACHE_DIR = ANALYSIS18_STATE_DIR / "cache"
ANALYSIS18_MODEL_ROOT = MODELS_DIR / "analysis18_section7" / "cross_encoder"
ANALYSIS18_MANIFEST = RESULTS_DIR / "analysis18_section7_manifest.json"
ANALYSIS18_PERSEED = RESULTS_DIR / "analysis18_fraction_sweep_perseed.csv"

for directory in (
    RESULTS_DIR,
    MODELS_DIR,
    STATE_DIR,
    CELL_DIR,
    PREDICTION_DIR,
    SPLIT_DIR,
    CACHE_DIR,
    MODEL_ROOT,
):
    directory.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(42, 52))
FRACTIONS = [0.50, 0.60, 0.70, 0.80, 0.90]

SCHEME_SRS = "SRSWOR_65_35"
SCHEME_BOOT = "bootstrap_OOB"
SCHEMES = [SCHEME_SRS, SCHEME_BOOT]
INNER_SRS_TRAIN_RATIO = 0.65

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
SCORE_PATH = "sigmoid(CrossEncoder.predict)"

CONFIGS: dict[str, dict[str, Any]] = {
    "DBLP": {
        "display_name": "DBLP-Scholar",
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
        "display_name": "Abt-Buy",
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
    return time.strftime("%Y-%m-%d %H:%M:%S")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fraction_tag(fraction: float) -> str:
    return f"{fraction:.2f}".replace(".", "p")


def safe_remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(-values))


def stable_hash(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8", errors="replace"))
        digest.update(b"\n")
    return digest.hexdigest()


def parse_csv_values(raw: str, allowed: set[str], label: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    bad = [value for value in values if value not in allowed]
    if bad:
        raise ValueError(f"Unsupported {label}: {bad}; allowed={sorted(allowed)}")
    if not values:
        raise ValueError(f"At least one {label} is required.")
    return values


def parse_fractions(raw: str) -> list[float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    allowed = {round(value, 2) for value in FRACTIONS}
    parsed: list[float] = []
    for value in values:
        rounded = round(value, 2)
        if rounded not in allowed:
            raise ValueError(f"Unsupported fraction {value}; allowed={FRACTIONS}")
        parsed.append(rounded)
    if not parsed:
        raise ValueError("At least one fraction is required.")
    return parsed


def parse_seeds(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    bad = [value for value in values if value not in SEEDS]
    if bad:
        raise ValueError(f"Unsupported seeds {bad}; allowed={SEEDS}")
    if not values:
        raise ValueError("At least one seed is required.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Revised-paper Section 3.3.2 FLR/MMR bias experiment."
    )
    parser.add_argument(
        "--phase",
        choices=["run", "aggregate", "audit-targets", "sizes"],
        default="run",
    )
    parser.add_argument(
        "--datasets",
        default="DBLP,ECOM",
        help="Comma-separated subset of DBLP,ECOM.",
    )
    parser.add_argument(
        "--fractions",
        default=",".join(f"{value:.2f}" for value in FRACTIONS),
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(value) for value in SEEDS),
    )
    parser.add_argument(
        "--schemes",
        default=",".join(SCHEMES),
        help=f"Comma-separated subset of {SCHEMES}.",
    )
    parser.add_argument(
        "--target-reuse",
        choices=["auto", "require", "never"],
        default="auto",
        help=(
            "auto: reuse only verified analysis18 targets, else retrain; "
            "require: fail if verified reuse is unavailable; "
            "never: retrain every target."
        ),
    )
    parser.add_argument(
        "--prune-completed-models",
        action="store_true",
        help="Delete analysis22 model checkpoints after their result JSON/CSV is complete.",
    )
    return parser.parse_args()


def require_single_visible_gpu() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Training is GPU-only; run on the server with "
            "CUDA_VISIBLE_DEVICES=<one GPU id>."
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
# DATA LOADING: EXACT ANALYSIS18 SNAPSHOT AND PREPROCESSING
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
        raise KeyError(f"Expected one id column in {path}; found {list(frame.columns)}")
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
    # Exact analysis18 CE serialisation. The broader June JW feature path is not
    # invoked by this cross-encoder-only experiment.
    parts: list[str] = []
    for field in fields:
        value = row.get(field, "")
        parts.append(str(value) if value is not None and str(value) != "nan" else "")
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
                f"{path} is missing fields {sorted(missing)}; columns={list(frame.columns)}"
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
            f"{name} snapshot mismatch: observed={observed}, expected={cfg['expected']}"
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

    print(f"\n{name} ({cfg['display_name']}) input validation")
    print("-" * 78)
    print(f"|A|={len(frame_a):,}  |B|={len(frame_b):,}  raw pairs={len(mapping):,}")
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
        "display_name": cfg["display_name"],
        "cfg": cfg,
        "df_a": frame_a,
        "df_b": frame_b,
        "truth": truth,
        "retained": retained,
        "a_ids": list(frame_a.index),
        "b_ids": list(frame_b.index),
        "data_fingerprint": stable_hash(
            [name, len(frame_a), len(frame_b), len(mapping)]
            + list(frame_a.index)
            + list(frame_b.index)
            + [f"{a}\0{b}" for a, b in mapping[[map_a, map_b]].itertuples(index=False, name=None)]
        ),
    }


# =============================================================================
# FIXED TITLE BLOCKING CACHE
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


def analysis22_top50_cache_path(dataset_name: str) -> Path:
    return CACHE_DIR / f"{dataset_name}_top50.json"


def analysis18_top50_cache_path(dataset_name: str) -> Path:
    return ANALYSIS18_CACHE_DIR / f"{dataset_name}_top50.json"


def top50_cache_valid(payload: dict[str, Any], dataset: dict[str, Any]) -> bool:
    return (
        payload.get("dataset") == dataset["name"]
        and payload.get("bi_encoder") == BI_ENCODER_NAME
        and int(payload.get("k", -1)) == K
        and int(payload.get("n_a", -1)) == len(dataset["a_ids"])
        and int(payload.get("n_b", -1)) == len(dataset["b_ids"])
        and payload.get("fingerprint") == dataset["blocking_fingerprint"]
        and set(payload.get("top50", {})) == set(dataset["a_ids"])
        and all(len(value) == K for value in payload.get("top50", {}).values())
    )


def compute_top50(dataset: dict[str, Any], device: str) -> dict[str, list[str]]:
    from sentence_transformers import SentenceTransformer

    field = dataset["cfg"]["retrieval_field"]
    print(f"\n{dataset['name']} fixed title blocking")
    print("-" * 78)
    model = SentenceTransformer(BI_ENCODER_NAME, device=device)
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
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return top50


def attach_top50(datasets: list[dict[str, Any]], device: str | None) -> None:
    for dataset in datasets:
        dataset["blocking_fingerprint"] = blocking_fingerprint(dataset)
        loaded = False
        for source, path in (
            ("analysis18", analysis18_top50_cache_path(dataset["name"])),
            ("analysis22", analysis22_top50_cache_path(dataset["name"])),
        ):
            if not path.exists():
                continue
            payload = read_json(path)
            if top50_cache_valid(payload, dataset):
                dataset["top50"] = payload["top50"]
                dataset["top50_source"] = source
                print(f"Loaded fixed top-50 cache ({source}): {path}")
                loaded = True
                break
            print(f"Ignoring stale top-50 cache: {path}")
        if loaded:
            continue
        if device is None:
            raise RuntimeError(
                f"No valid top-50 cache for {dataset['name']}; GPU is required to build it."
            )
        top50 = compute_top50(dataset, device)
        dataset["top50"] = top50
        dataset["top50_source"] = "analysis22_computed"
        payload = {
            "dataset": dataset["name"],
            "bi_encoder": BI_ENCODER_NAME,
            "k": K,
            "n_a": len(dataset["a_ids"]),
            "n_b": len(dataset["b_ids"]),
            "fingerprint": dataset["blocking_fingerprint"],
            "top50": top50,
        }
        atomic_write_json(analysis22_top50_cache_path(dataset["name"]), payload)
        print(f"Saved fixed top-50 cache: {analysis22_top50_cache_path(dataset['name'])}")


# =============================================================================
# PAIRS, THRESHOLDS, EVALUATION, AND COUNTING
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


def choose_threshold(scores: np.ndarray, labels: Iterable[int]) -> tuple[float, float]:
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


def category_for(declared: bool, top_idb: str, true_set: set[str]) -> str:
    if declared:
        return "correct" if top_idb in true_set else "false_link"
    return "missing_match" if true_set else "unmatched_abstention"


def to_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.astype(str).str.lower().map({"true": True, "false": False}).fillna(False)


def metrics_from_predictions(frame: pd.DataFrame) -> dict[str, float | int]:
    required = {"declared", "category", "has_true_match"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Prediction frame missing columns: {sorted(missing)}")

    declared_values = to_bool_series(frame["declared"])
    truth_values = to_bool_series(frame["has_true_match"])
    declared = int(declared_values.sum())
    correct = int((frame["category"] == "correct").sum())
    false_links = int((frame["category"] == "false_link").sum())
    missing_matches = int((frame["category"] == "missing_match").sum())
    unmatched_abstentions = int((frame["category"] == "unmatched_abstention").sum())
    n_true_records = int(truth_values.sum())

    if declared != correct + false_links:
        raise AssertionError("Declared does not equal correct + false links.")
    if len(frame) != correct + false_links + missing_matches + unmatched_abstentions:
        raise AssertionError("Prediction categories are not exhaustive.")

    return {
        "FLR": false_links / declared if declared else 0.0,
        "MMR": missing_matches / n_true_records if n_true_records else 0.0,
        "declared": declared,
        "correct": correct,
        "false_links": false_links,
        "missing_matches": missing_matches,
        "unmatched_abstentions": unmatched_abstentions,
        "n_true_records": n_true_records,
        "n_test": len(frame),
    }


def ce_predict(model: Any, pairs: list[tuple[str, str]]) -> np.ndarray:
    return np.asarray(
        model.predict(pairs, batch_size=PREDICT_BATCH, show_progress_bar=False),
        dtype=np.float64,
    ).reshape(-1)


def evaluate_model(
    dataset: dict[str, Any],
    model: Any,
    threshold: float,
    test_ids: Sequence[str],
    fraction: float,
    seed: int,
    role: str,
    scheme: str,
) -> tuple[pd.DataFrame, dict[str, float | int], float]:
    fields = dataset["cfg"]["fields"]
    rows: list[dict[str, Any]] = []
    started = time.time()

    for record_index, ida in enumerate(test_ids, start=1):
        true_set = dataset["truth"].get(ida, set())
        candidates = dataset["top50"][ida]
        candidate_text = [
            (
                serialise(dataset["df_a"].loc[ida], fields),
                serialise(dataset["df_b"].loc[idb], fields),
            )
            for idb in candidates
        ]
        scores = sigmoid(ce_predict(model, candidate_text))
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
                "role": role,
                "scheme": scheme,
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
        if record_index % 100 == 0 or record_index == len(test_ids):
            print(
                f"        evaluated {record_index:,}/{len(test_ids):,}",
                end="\r",
                flush=True,
            )
    print()

    predictions = pd.DataFrame(rows)
    metrics = metrics_from_predictions(predictions)
    return predictions, metrics, time.time() - started


# =============================================================================
# DETERMINISTIC OUTER AND INNER SPLITS
# =============================================================================
def make_outer_split(
    dataset: dict[str, Any], fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    labelled_ids, holdout_ids = train_test_split(
        dataset["a_ids"], train_size=fraction, random_state=seed
    )
    return list(labelled_ids), list(holdout_ids)


def make_inner_split(
    labelled_ids: Sequence[str], scheme: str, seed: int
) -> tuple[list[str], list[str], dict[str, Any]]:
    labelled_ids = list(labelled_ids)
    if scheme == SCHEME_SRS:
        train_ids, test_ids = train_test_split(
            labelled_ids,
            train_size=INNER_SRS_TRAIN_RATIO,
            random_state=seed,
        )
        train_draw = list(train_ids)
        test_ids = list(test_ids)
        metadata = {
            "scheme": SCHEME_SRS,
            "sampling": "SRSWOR over outer-labelled source records",
            "train_ratio": INNER_SRS_TRAIN_RATIO,
            "test_ratio": 1.0 - INNER_SRS_TRAIN_RATIO,
            "draw_size": len(train_draw),
            "unique_train_size": len(set(train_draw)),
            "test_size": len(test_ids),
        }
    elif scheme == SCHEME_BOOT:
        rng = np.random.RandomState(seed)
        indices = rng.randint(0, len(labelled_ids), size=len(labelled_ids))
        train_draw = [labelled_ids[int(index)] for index in indices]
        selected = set(train_draw)
        test_ids = [ida for ida in labelled_ids if ida not in selected]
        if not test_ids:
            raise RuntimeError(
                f"Bootstrap OOB set is empty for seed={seed}, n={len(labelled_ids)}."
            )
        metadata = {
            "scheme": SCHEME_BOOT,
            "sampling": "n-out-of-n bootstrap over outer-labelled source records",
            "draw_size": len(train_draw),
            "unique_train_size": len(selected),
            "test_size": len(test_ids),
            "distinct_fraction": len(selected) / len(labelled_ids),
            "oob_fraction": len(test_ids) / len(labelled_ids),
        }
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    train_unique = set(train_draw)
    test_set = set(test_ids)
    if train_unique & test_set:
        raise AssertionError(f"{scheme}: train and inner test sets overlap.")
    if not test_set.issubset(set(labelled_ids)):
        raise AssertionError(f"{scheme}: inner test is not a subset of sM_f.")

    metadata.update(
        {
            "outer_labelled_size": len(labelled_ids),
            "seed": seed,
            "train_draw_hash": stable_hash(train_draw),
            "train_unique_hash": stable_hash(sorted(train_unique)),
            "test_ids_hash": stable_hash(test_ids),
        }
    )
    return train_draw, test_ids, metadata


def split_path(dataset_name: str, fraction: float, seed: int, scheme: str) -> Path:
    return SPLIT_DIR / (
        f"split_{dataset_name}_f{fraction_tag(fraction)}_seed{seed}_{scheme}.json"
    )


def save_or_validate_split(
    dataset_name: str,
    fraction: float,
    seed: int,
    scheme: str,
    train_draw: Sequence[str],
    test_ids: Sequence[str],
    metadata: dict[str, Any],
) -> None:
    path = split_path(dataset_name, fraction, seed, scheme)
    payload = {
        "analysis": ANALYSIS_NAME,
        "dataset": dataset_name,
        "fraction": fraction,
        "seed": seed,
        "scheme": scheme,
        "metadata": metadata,
        "train_draw": list(train_draw),
        "inner_test_ids": list(test_ids),
    }
    if path.exists():
        existing = read_json(path)
        if existing != payload:
            raise AssertionError(f"Existing split file disagrees with deterministic split: {path}")
        return
    atomic_write_json(path, payload)


# =============================================================================
# MODEL CHECKPOINTS AND FIT/CALIBRATION
# =============================================================================
def model_path(
    role: str, dataset_name: str, fraction: float, seed: int, scheme: str
) -> Path:
    role_tag = "target" if role == "target" else scheme
    return (
        MODEL_ROOT
        / role_tag
        / dataset_name
        / f"f_{fraction_tag(fraction)}"
        / f"seed_{seed}"
    )


def model_marker(path: Path) -> Path:
    return path / f"_{ANALYSIS_NAME}_complete.json"


def analysis18_model_path(dataset_name: str, fraction: float, seed: int) -> Path:
    return (
        ANALYSIS18_MODEL_ROOT
        / dataset_name
        / f"f_{fraction_tag(fraction)}"
        / f"seed_{seed}"
    )


def analysis18_model_marker(path: Path) -> Path:
    return path / "_analysis18_complete.json"


def model_marker_compatible(marker: dict[str, Any], expected: dict[str, Any]) -> bool:
    exact_keys = [
        "analysis",
        "protocol_version",
        "training_seed_applied",
        "role",
        "scheme",
        "dataset",
        "fraction",
        "seed",
        "model",
        "score_path",
        "epochs",
        "warmup_fraction",
        "learning_rate",
        "batch_size",
        "max_length",
        "train_draw_hash",
        "n_train_draw",
    ]
    for key in exact_keys:
        if marker.get(key) != expected.get(key):
            return False
    return True


def save_model_atomic(model: Any, path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    safe_remove(temporary)
    model.save_pretrained(str(temporary))
    atomic_write_json(temporary / f"_{ANALYSIS_NAME}_complete.json", metadata)
    safe_remove(path)
    os.replace(temporary, path)


def build_pair_split(
    dataset: dict[str, Any], train_draw: Sequence[str], seed: int
) -> tuple[
    list[tuple[str, str, int]],
    list[tuple[str, str, int]],
    list[tuple[str, str, int]],
]:
    train_pairs = build_train_pairs(
        train_draw,
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
    fit_pairs = [train_pairs[index] for index in fit_indices]
    validation_pairs = [train_pairs[index] for index in validation_indices]
    return train_pairs, fit_pairs, validation_pairs


def load_or_train_model(
    dataset: dict[str, Any],
    fraction: float,
    seed: int,
    role: str,
    scheme: str,
    train_draw: Sequence[str],
    device: str,
) -> tuple[Any, dict[str, Any]]:
    from sentence_transformers import CrossEncoder, InputExample

    path = model_path(role, dataset["name"], fraction, seed, scheme)
    expected_marker = {
        "analysis": ANALYSIS_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "training_seed_applied": True,
        "role": role,
        "scheme": scheme,
        "dataset": dataset["name"],
        "fraction": fraction,
        "seed": seed,
        "model": CE_MODEL_NAME,
        "score_path": SCORE_PATH,
        "epochs": CE_EPOCHS,
        "warmup_fraction": CE_WARMUP_FRAC,
        "learning_rate": CE_LR,
        "batch_size": CE_BATCH,
        "max_length": CE_MAX_LEN,
        "train_draw_hash": stable_hash(train_draw),
        "n_train_draw": len(train_draw),
    }

    # Reset Python, NumPy and Torch RNGs before every independent CE fit.
    # This makes the target, SRSWOR and bootstrap models reproducible within
    # each Monte Carlo seed and prevents execution order from changing results.
    seed_everything(seed)

    train_pairs, fit_pairs, validation_pairs = build_pair_split(
        dataset, train_draw, seed
    )

    loaded_checkpoint = False
    train_seconds = 0.0
    marker = model_marker(path)
    if path.is_dir() and marker.exists():
        existing_marker = read_json(marker)
        if model_marker_compatible(existing_marker, expected_marker):
            print(f"      loading analysis22 CE checkpoint: {path}")
            model = CrossEncoder(str(path), max_length=CE_MAX_LEN, device=device)
            loaded_checkpoint = True
        else:
            print(f"      removing stale analysis22 CE checkpoint: {path}")
            safe_remove(path)

    if not loaded_checkpoint:
        fields = dataset["cfg"]["fields"]
        examples = [
            InputExample(
                texts=[
                    serialise(dataset["df_a"].loc[ida], fields),
                    serialise(dataset["df_b"].loc[idb], fields),
                ],
                label=float(label),
            )
            for ida, idb, label in fit_pairs
        ]
        if not examples:
            raise ValueError("No CE fitting examples were generated.")

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
            f"      training CE: role={role}, scheme={scheme}, "
            f"draw_items={len(train_draw):,}, all_pairs={len(train_pairs):,}, "
            f"fit_pairs={len(fit_pairs):,}, epochs={CE_EPOCHS}, "
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
        marker_payload = {
            **expected_marker,
            "phase": "section_3_3_2_bias",
            "n_train_pairs_total": len(train_pairs),
            "n_fit_pairs": len(fit_pairs),
            "n_validation_pairs": len(validation_pairs),
            "train_seconds": train_seconds,
        }
        save_model_atomic(model, path, marker_payload)
        print(f"      saved analysis22 CE checkpoint: {path}")

    fields = dataset["cfg"]["fields"]
    validation_text = [
        (
            serialise(dataset["df_a"].loc[ida], fields),
            serialise(dataset["df_b"].loc[idb], fields),
        )
        for ida, idb, _ in validation_pairs
    ]
    validation_labels = [label for _, _, label in validation_pairs]
    validation_scores = sigmoid(ce_predict(model, validation_text))
    threshold, validation_f1 = choose_threshold(validation_scores, validation_labels)
    print(
        f"      threshold={threshold:.2f}; validation F1={validation_f1:.4f}; "
        f"score path={SCORE_PATH}"
    )

    fit_info = {
        "model_path": str(path),
        "model_loaded_from_checkpoint": loaded_checkpoint,
        "train_seconds_this_run": train_seconds,
        "n_train_pairs_total": len(train_pairs),
        "n_fit_pairs": len(fit_pairs),
        "n_validation_pairs": len(validation_pairs),
        "threshold": threshold,
        "validation_f1": validation_f1,
    }
    return model, fit_info


# =============================================================================
# ANALYSIS18 TARGET REUSE AUDIT
# =============================================================================
def analysis18_manifest_errors() -> list[str]:
    if not ANALYSIS18_MANIFEST.exists():
        return [f"missing manifest: {ANALYSIS18_MANIFEST}"]
    manifest = read_json(ANALYSIS18_MANIFEST)
    errors: list[str] = []

    expected_top = {
        "analysis": "analysis18_section7",
        "scope": "gold-standard fraction sweep only; no proxy comparison",
        "k": K,
        "bi_encoder": BI_ENCODER_NAME,
        "cross_encoder": CE_MODEL_NAME,
    }
    for key, expected in expected_top.items():
        if manifest.get(key) != expected:
            errors.append(f"manifest {key}: observed={manifest.get(key)!r}, expected={expected!r}")

    recipe = manifest.get("cross_encoder_recipe", {})
    expected_recipe = {
        "epochs": CE_EPOCHS,
        "warmup_fraction": CE_WARMUP_FRAC,
        "learning_rate": CE_LR,
        "batch_size": CE_BATCH,
        "max_length": CE_MAX_LEN,
        "score_path": SCORE_PATH,
        "validation_fraction_of_pairs": VALIDATION_FRAC,
        "threshold_grid": [float(value) for value in THRESHOLDS],
    }
    for key, expected in expected_recipe.items():
        if recipe.get(key) != expected:
            errors.append(
                f"manifest cross_encoder_recipe.{key}: "
                f"observed={recipe.get(key)!r}, expected={expected!r}"
            )

    corrections = manifest.get("corrections", {})
    expected_corrections = {
        "complete_raw_b": True,
        "many_match_truth": True,
        "any_valid_match_is_correct": True,
        "mutually_exclusive_counting": True,
        "psi_denominator": "holdout A records with at least one true match",
        "training_positive": "first-listed valid B",
        "alternative_valid_matches_excluded_from_negatives": True,
    }
    for key, expected in expected_corrections.items():
        if corrections.get(key) != expected:
            errors.append(
                f"manifest corrections.{key}: observed={corrections.get(key)!r}, "
                f"expected={expected!r}"
            )
    return errors


def analysis18_cell_path(dataset_name: str, fraction: float, seed: int) -> Path:
    return ANALYSIS18_CELL_DIR / (
        f"fraction_{dataset_name}_f{fraction_tag(fraction)}_seed{seed}.json"
    )


def analysis18_prediction_path(dataset_name: str, fraction: float, seed: int) -> Path:
    return ANALYSIS18_PREDICTION_DIR / (
        f"fraction_{dataset_name}_f{fraction_tag(fraction)}_seed{seed}.csv"
    )


def validate_analysis18_checkpoint(
    dataset_name: str, fraction: float, seed: int
) -> tuple[bool, list[str]]:
    path = analysis18_model_path(dataset_name, fraction, seed)
    marker_path = analysis18_model_marker(path)
    errors: list[str] = []
    if not path.is_dir() or not marker_path.exists():
        return False, [f"analysis18 checkpoint/marker missing: {path}"]
    marker = read_json(marker_path)
    expected = {
        "phase": "section7_fraction_sweep",
        "dataset": dataset_name,
        "fraction": fraction,
        "seed": seed,
        "model": CE_MODEL_NAME,
        "score_path": SCORE_PATH,
        "epochs": CE_EPOCHS,
        "warmup_fraction": CE_WARMUP_FRAC,
        "learning_rate": CE_LR,
        "batch_size": CE_BATCH,
        "max_length": CE_MAX_LEN,
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            errors.append(
                f"analysis18 checkpoint marker {key}: observed={marker.get(key)!r}, expected={value!r}"
            )
    return not errors, errors


def target_metrics_from_analysis18_cell(
    dataset: dict[str, Any], fraction: float, seed: int
) -> tuple[dict[str, Any] | None, list[str]]:
    errors = analysis18_manifest_errors()
    if errors:
        return None, errors

    labelled_ids, holdout_ids = make_outer_split(dataset, fraction, seed)
    cell_path = analysis18_cell_path(dataset["name"], fraction, seed)
    prediction_path = analysis18_prediction_path(dataset["name"], fraction, seed)
    if not cell_path.exists() or not prediction_path.exists():
        return None, [
            f"analysis18 cell/prediction missing: {cell_path} / {prediction_path}"
        ]

    cell = read_json(cell_path)
    expected_scalar = {
        "dataset": dataset["name"],
        "fraction": fraction,
        "seed": seed,
        "n_holdout": len(holdout_ids),
        "n_labelled_a": len(labelled_ids),
        "score_path": SCORE_PATH,
    }
    for key, value in expected_scalar.items():
        if cell.get(key) != value:
            errors.append(
                f"analysis18 cell {key}: observed={cell.get(key)!r}, expected={value!r}"
            )

    predictions = pd.read_csv(prediction_path, dtype={"ida": str, "top1_idb": str})
    observed_ids = predictions["ida"].astype(str).tolist()
    if observed_ids != list(holdout_ids):
        errors.append("analysis18 prediction ida order/content does not match deterministic outer holdout")
    if len(predictions) != len(holdout_ids):
        errors.append(
            f"analysis18 prediction rows={len(predictions)}, expected={len(holdout_ids)}"
        )

    if not errors:
        metrics = metrics_from_predictions(predictions)
        if not math.isclose(float(cell["lambda"]), float(metrics["FLR"]), rel_tol=0.0, abs_tol=1e-12):
            errors.append("analysis18 cell lambda disagrees with recomputed predictions")
        if not math.isclose(float(cell["psi"]), float(metrics["MMR"]), rel_tol=0.0, abs_tol=1e-12):
            errors.append("analysis18 cell psi disagrees with recomputed predictions")

    checkpoint_ok, checkpoint_errors = validate_analysis18_checkpoint(
        dataset["name"], fraction, seed
    )
    # A verified cell + prediction + manifest is sufficient result reuse, as
    # requested. Checkpoint status is recorded but does not invalidate metrics.

    if errors:
        return None, errors

    return {
        "target_FLR": float(cell["lambda"]),
        "target_MMR": float(cell["psi"]),
        "target_threshold": float(cell["threshold"]),
        "target_validation_f1": float(cell["validation_f1"]),
        "target_declared": int(cell["declared"]),
        "target_correct": int(cell["correct"]),
        "target_false_links": int(cell["false_links"]),
        "target_missing_matches": int(cell["missing_matches"]),
        "target_n_true_records": int(cell["n_true_records"]),
        "outer_labelled_size": len(labelled_ids),
        "outer_holdout_size": len(holdout_ids),
        "target_source": "analysis18_cell_prediction_manifest",
        "target_reused": True,
        "analysis18_checkpoint_present_and_compatible": checkpoint_ok,
        "analysis18_checkpoint_audit_errors": checkpoint_errors,
        "analysis18_cell_path": str(cell_path),
        "analysis18_prediction_path": str(prediction_path),
        "analysis18_model_path": str(analysis18_model_path(dataset["name"], fraction, seed)),
    }, []


def target_metrics_from_analysis18_perseed(
    dataset: dict[str, Any], fraction: float, seed: int
) -> tuple[dict[str, Any] | None, list[str]]:
    errors = analysis18_manifest_errors()
    if errors:
        return None, errors
    if not ANALYSIS18_PERSEED.exists():
        return None, [f"missing analysis18 per-seed CSV: {ANALYSIS18_PERSEED}"]

    frame = pd.read_csv(ANALYSIS18_PERSEED)
    subset = frame[
        (frame["dataset"].astype(str) == dataset["name"])
        & (frame["fraction"].astype(float).round(2) == round(fraction, 2))
        & (frame["seed"].astype(int) == seed)
    ]
    if len(subset) != 1:
        return None, [
            f"analysis18 per-seed row count={len(subset)} for {dataset['name']} f={fraction} seed={seed}"
        ]
    row = subset.iloc[0]
    labelled_ids, holdout_ids = make_outer_split(dataset, fraction, seed)
    for column, expected in (
        ("n_holdout", len(holdout_ids)),
        ("n_labelled_a", len(labelled_ids)),
    ):
        if column not in row.index or int(row[column]) != expected:
            errors.append(
                f"analysis18 per-seed {column}: observed={row.get(column)!r}, expected={expected}"
            )
    if errors:
        return None, errors

    checkpoint_ok, checkpoint_errors = validate_analysis18_checkpoint(
        dataset["name"], fraction, seed
    )
    return {
        "target_FLR": float(row["lambda"]),
        "target_MMR": float(row["psi"]),
        "target_threshold": float(row.get("threshold", np.nan)),
        "target_validation_f1": float(row.get("validation_f1", np.nan)),
        "target_declared": int(row.get("declared", -1)),
        "target_correct": int(row.get("correct", -1)),
        "target_false_links": int(row.get("false_links", -1)),
        "target_missing_matches": int(row.get("missing_matches", -1)),
        "target_n_true_records": int(row.get("n_true_records", -1)),
        "outer_labelled_size": len(labelled_ids),
        "outer_holdout_size": len(holdout_ids),
        "target_source": "analysis18_perseed_manifest",
        "target_reused": True,
        "analysis18_checkpoint_present_and_compatible": checkpoint_ok,
        "analysis18_checkpoint_audit_errors": checkpoint_errors,
        "analysis18_perseed_path": str(ANALYSIS18_PERSEED),
        "analysis18_model_path": str(analysis18_model_path(dataset["name"], fraction, seed)),
    }, []


def find_verified_analysis18_target(
    dataset: dict[str, Any], fraction: float, seed: int
) -> tuple[dict[str, Any] | None, list[str]]:
    cell_result, cell_errors = target_metrics_from_analysis18_cell(
        dataset, fraction, seed
    )
    if cell_result is not None:
        return cell_result, []
    csv_result, csv_errors = target_metrics_from_analysis18_perseed(
        dataset, fraction, seed
    )
    if csv_result is not None:
        return csv_result, []
    return None, cell_errors + csv_errors


# =============================================================================
# RESULT PATHS AND ONE UNIT OF WORK
# =============================================================================
def target_cell_path(dataset_name: str, fraction: float, seed: int) -> Path:
    return CELL_DIR / f"target_{dataset_name}_f{fraction_tag(fraction)}_seed{seed}.json"


def target_prediction_path(dataset_name: str, fraction: float, seed: int) -> Path:
    return PREDICTION_DIR / f"target_{dataset_name}_f{fraction_tag(fraction)}_seed{seed}.csv"


def inner_cell_path(
    dataset_name: str, fraction: float, seed: int, scheme: str
) -> Path:
    return CELL_DIR / (
        f"inner_{dataset_name}_f{fraction_tag(fraction)}_seed{seed}_{scheme}.json"
    )


def inner_prediction_path(
    dataset_name: str, fraction: float, seed: int, scheme: str
) -> Path:
    return PREDICTION_DIR / (
        f"inner_{dataset_name}_f{fraction_tag(fraction)}_seed{seed}_{scheme}.csv"
    )


def result_complete(cell: Path, predictions: Path | None = None) -> bool:
    if not cell.exists():
        return False
    if predictions is not None and not predictions.exists():
        return False
    return True


def ensure_target(
    dataset: dict[str, Any],
    fraction: float,
    seed: int,
    target_reuse: str,
    device: str,
    prune_completed_models: bool,
) -> dict[str, Any]:
    cell_path = target_cell_path(dataset["name"], fraction, seed)
    if cell_path.exists():
        existing = read_json(cell_path)
        if (
            existing.get("analysis") == ANALYSIS_NAME
            and existing.get("protocol_version") == PROTOCOL_VERSION
            and existing.get("dataset") == dataset["name"]
            and existing.get("fraction") == fraction
            and existing.get("seed") == seed
        ):
            print(f"      SKIP completed target record: {cell_path}")
            return existing
        raise AssertionError(f"Stale/incompatible target cell: {cell_path}")

    labelled_ids, holdout_ids = make_outer_split(dataset, fraction, seed)

    if target_reuse != "never":
        reused, errors = find_verified_analysis18_target(dataset, fraction, seed)
        if reused is not None:
            result = {
                "analysis": ANALYSIS_NAME,
                "protocol_version": PROTOCOL_VERSION,
                "role": "target",
                "dataset": dataset["name"],
                "dataset_display": dataset["display_name"],
                "fraction": fraction,
                "seed": seed,
                **reused,
                "outer_labelled_ids_hash": stable_hash(labelled_ids),
                "outer_holdout_ids_hash": stable_hash(holdout_ids),
                "created_at": ts(),
            }
            atomic_write_json(cell_path, result)
            print(
                f"      REUSED verified analysis18 target: FLR={result['target_FLR']:.6f}; "
                f"MMR={result['target_MMR']:.6f}; source={result['target_source']}"
            )
            return result
        if target_reuse == "require":
            raise RuntimeError(
                "Verified analysis18 target reuse was required but failed for "
                f"{dataset['name']} f={fraction:.2f} seed={seed}:\n  - "
                + "\n  - ".join(errors)
            )
        print(
            "      analysis18 target reuse unavailable/incompatible; retraining. "
            "Reasons:\n        - " + "\n        - ".join(errors)
        )

    model, fit_info = load_or_train_model(
        dataset=dataset,
        fraction=fraction,
        seed=seed,
        role="target",
        scheme="outer_target",
        train_draw=labelled_ids,
        device=device,
    )
    predictions, metrics, evaluation_seconds = evaluate_model(
        dataset,
        model,
        float(fit_info["threshold"]),
        holdout_ids,
        fraction,
        seed,
        role="target",
        scheme="outer_target",
    )
    prediction_path = target_prediction_path(dataset["name"], fraction, seed)
    atomic_write_csv(predictions, prediction_path)

    result = {
        "analysis": ANALYSIS_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "role": "target",
        "dataset": dataset["name"],
        "dataset_display": dataset["display_name"],
        "fraction": fraction,
        "seed": seed,
        "target_FLR": metrics["FLR"],
        "target_MMR": metrics["MMR"],
        "target_threshold": fit_info["threshold"],
        "target_validation_f1": fit_info["validation_f1"],
        "target_declared": metrics["declared"],
        "target_correct": metrics["correct"],
        "target_false_links": metrics["false_links"],
        "target_missing_matches": metrics["missing_matches"],
        "target_n_true_records": metrics["n_true_records"],
        "outer_labelled_size": len(labelled_ids),
        "outer_holdout_size": len(holdout_ids),
        "outer_labelled_ids_hash": stable_hash(labelled_ids),
        "outer_holdout_ids_hash": stable_hash(holdout_ids),
        "target_source": "analysis22_retrained_exact_analysis18_protocol",
        "target_reused": False,
        "analysis18_checkpoint_present_and_compatible": False,
        "evaluation_seconds": evaluation_seconds,
        "prediction_path": str(prediction_path),
        **fit_info,
        "created_at": ts(),
    }
    atomic_write_json(cell_path, result)
    print(
        f"      saved target: FLR={metrics['FLR']:.6f}; MMR={metrics['MMR']:.6f}; "
        f"cell={cell_path}"
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if prune_completed_models:
        safe_remove(model_path("target", dataset["name"], fraction, seed, "outer_target"))
    return result


def run_inner_scheme(
    dataset: dict[str, Any],
    fraction: float,
    seed: int,
    scheme: str,
    target: dict[str, Any],
    device: str,
    prune_completed_models: bool,
) -> dict[str, Any]:
    cell_path = inner_cell_path(dataset["name"], fraction, seed, scheme)
    prediction_path = inner_prediction_path(dataset["name"], fraction, seed, scheme)
    if result_complete(cell_path, prediction_path):
        existing = read_json(cell_path)
        compatible = (
            existing.get("analysis") == ANALYSIS_NAME
            and existing.get("protocol_version") == PROTOCOL_VERSION
            and existing.get("dataset") == dataset["name"]
            and existing.get("fraction") == fraction
            and existing.get("seed") == seed
            and existing.get("scheme") == scheme
        )
        if not compatible:
            raise AssertionError(
                f"Stale/incompatible inner cell found: {cell_path}. "
                "Remove the old analysis22 cell and checkpoint before resuming."
            )
        print(f"      SKIP completed inner result: {cell_path}")
        return existing

    labelled_ids, holdout_ids = make_outer_split(dataset, fraction, seed)
    if len(labelled_ids) != int(target["outer_labelled_size"]):
        raise AssertionError("Target outer-labelled size disagrees with deterministic split.")
    if len(holdout_ids) != int(target["outer_holdout_size"]):
        raise AssertionError("Target outer-holdout size disagrees with deterministic split.")

    train_draw, inner_test_ids, split_meta = make_inner_split(
        labelled_ids, scheme, seed
    )
    save_or_validate_split(
        dataset["name"],
        fraction,
        seed,
        scheme,
        train_draw,
        inner_test_ids,
        split_meta,
    )

    model, fit_info = load_or_train_model(
        dataset=dataset,
        fraction=fraction,
        seed=seed,
        role="inner",
        scheme=scheme,
        train_draw=train_draw,
        device=device,
    )
    predictions, metrics, evaluation_seconds = evaluate_model(
        dataset,
        model,
        float(fit_info["threshold"]),
        inner_test_ids,
        fraction,
        seed,
        role="inner",
        scheme=scheme,
    )
    atomic_write_csv(predictions, prediction_path)

    result = {
        "analysis": ANALYSIS_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "role": "inner",
        "dataset": dataset["name"],
        "dataset_display": dataset["display_name"],
        "fraction": fraction,
        "seed": seed,
        "scheme": scheme,
        "target_FLR": float(target["target_FLR"]),
        "inner_FLR": float(metrics["FLR"]),
        "bias_FLR": float(metrics["FLR"]) - float(target["target_FLR"]),
        "target_MMR": float(target["target_MMR"]),
        "inner_MMR": float(metrics["MMR"]),
        "bias_MMR": float(metrics["MMR"]) - float(target["target_MMR"]),
        "target_source": target["target_source"],
        "target_reused": bool(target["target_reused"]),
        "analysis18_checkpoint_present_and_compatible": bool(
            target.get("analysis18_checkpoint_present_and_compatible", False)
        ),
        "outer_labelled_size": len(labelled_ids),
        "outer_holdout_size": len(holdout_ids),
        "inner_train_draw_size": len(train_draw),
        "inner_train_unique_size": len(set(train_draw)),
        "inner_test_size": len(inner_test_ids),
        "inner_train_draw_hash": stable_hash(train_draw),
        "inner_test_ids_hash": stable_hash(inner_test_ids),
        "inner_declared": metrics["declared"],
        "inner_correct": metrics["correct"],
        "inner_false_links": metrics["false_links"],
        "inner_missing_matches": metrics["missing_matches"],
        "inner_n_true_records": metrics["n_true_records"],
        "evaluation_seconds": evaluation_seconds,
        "prediction_path": str(prediction_path),
        "split_path": str(split_path(dataset["name"], fraction, seed, scheme)),
        **fit_info,
        "created_at": ts(),
    }
    atomic_write_json(cell_path, result)
    print(
        f"      saved {scheme}: FLR*={result['inner_FLR']:.6f}; "
        f"bias_FLR={result['bias_FLR']:+.6f}; MMR*={result['inner_MMR']:.6f}; "
        f"bias_MMR={result['bias_MMR']:+.6f}; inner_test={len(inner_test_ids)}"
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if prune_completed_models:
        safe_remove(model_path("inner", dataset["name"], fraction, seed, scheme))
    return result


# =============================================================================
# AGGREGATION AND SAMPLE-SIZE REPORTING
# =============================================================================
def load_inner_frame() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(CELL_DIR.glob("inner_*.json")):
        payload = read_json(path)
        if payload.get("protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError(
                f"Stale analysis22 result uses a different protocol version: {path}. "
                "Remove or archive the old analysis22 output directory before aggregating."
            )
        required = {
            "dataset",
            "fraction",
            "seed",
            "scheme",
            "target_FLR",
            "bias_FLR",
            "target_MMR",
            "bias_MMR",
        }
        if required.issubset(payload):
            rows.append(payload)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["fraction"] = frame["fraction"].astype(float).round(2)
    frame["seed"] = frame["seed"].astype(int)
    return frame.sort_values(["dataset", "fraction", "scheme", "seed"]).reset_index(drop=True)


def mc_se(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) <= 1:
        return float("nan")
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def aggregate_bias(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset",
        "f",
        "scheme",
        "target_FLR_mean",
        "bias_FLR_mean",
        "bias_FLR_SE",
        "target_MMR_mean",
        "bias_MMR_mean",
        "bias_MMR_SE",
        "n_splits",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for (dataset_name, fraction, scheme), group in frame.groupby(
        ["dataset", "fraction", "scheme"], sort=True
    ):
        rows.append(
            {
                "dataset": dataset_name,
                "f": float(fraction),
                "scheme": scheme,
                "target_FLR_mean": float(group["target_FLR"].mean()),
                "bias_FLR_mean": float(group["bias_FLR"].mean()),
                "bias_FLR_SE": mc_se(group["bias_FLR"]),
                "target_MMR_mean": float(group["target_MMR"].mean()),
                "bias_MMR_mean": float(group["bias_MMR"].mean()),
                "bias_MMR_SE": mc_se(group["bias_MMR"]),
                "n_splits": int(len(group)),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["dataset", "f", "scheme"]
    )


def sample_size_summary(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset",
        "f",
        "scheme",
        "outer_labelled_size",
        "outer_holdout_size",
        "inner_train_draw_mean",
        "inner_train_unique_mean",
        "inner_test_mean",
        "inner_test_sd",
        "inner_test_min",
        "inner_test_max",
        "n_splits",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (dataset_name, fraction, scheme), group in frame.groupby(
        ["dataset", "fraction", "scheme"], sort=True
    ):
        if group["outer_holdout_size"].nunique() != 1:
            raise AssertionError("Outer holdout size varies within a dataset/fraction group.")
        if group["outer_labelled_size"].nunique() != 1:
            raise AssertionError("Outer labelled size varies within a dataset/fraction group.")
        rows.append(
            {
                "dataset": dataset_name,
                "f": float(fraction),
                "scheme": scheme,
                "outer_labelled_size": int(group["outer_labelled_size"].iloc[0]),
                "outer_holdout_size": int(group["outer_holdout_size"].iloc[0]),
                "inner_train_draw_mean": float(group["inner_train_draw_size"].mean()),
                "inner_train_unique_mean": float(group["inner_train_unique_size"].mean()),
                "inner_test_mean": float(group["inner_test_size"].mean()),
                "inner_test_sd": float(group["inner_test_size"].std(ddof=1)) if len(group) > 1 else float("nan"),
                "inner_test_min": int(group["inner_test_size"].min()),
                "inner_test_max": int(group["inner_test_size"].max()),
                "n_splits": int(len(group)),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["dataset", "f", "scheme"]
    )


def save_manifest(frame: pd.DataFrame) -> None:
    target_source_counts: dict[str, int] = {}
    if not frame.empty and "target_source" in frame.columns:
        target_source_counts = {
            str(key): int(value)
            for key, value in frame["target_source"].value_counts().to_dict().items()
        }
    complete_counts: dict[str, dict[str, dict[str, int]]] = {}
    if not frame.empty:
        for (dataset_name, fraction, scheme), group in frame.groupby(
            ["dataset", "fraction", "scheme"], sort=True
        ):
            complete_counts.setdefault(dataset_name, {}).setdefault(
                f"{float(fraction):.2f}", {}
            )[scheme] = int(len(group))

    manifest = {
        "analysis": ANALYSIS_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "scope": "revised-paper Section 3.3.2 FLR/MMR bias; not analysis13 proxy",
        "datasets": list(CONFIGS),
        "fractions": FRACTIONS,
        "seeds": SEEDS,
        "schemes": {
            SCHEME_SRS: {
                "sampling": "SRSWOR within sM_f",
                "train_ratio": INNER_SRS_TRAIN_RATIO,
                "test_ratio": 1.0 - INNER_SRS_TRAIN_RATIO,
            },
            SCHEME_BOOT: {
                "sampling": "n-out-of-n bootstrap within sM_f",
                "inner_test": "out-of-bag source items",
                "expected_distinct_fraction": 1.0 - math.exp(-1.0),
            },
        },
        "target": {
            "definition": "cross-encoder trained under exact analysis18 protocol on sM_f and evaluated on A\\sM_f",
            "reuse_source": "analysis18 only after manifest/split/result audit",
            "target_source_counts": target_source_counts,
        },
        "blocking": {
            "scheme": "scheme-II title-only bi-encoder",
            "k": K,
            "bi_encoder": BI_ENCODER_NAME,
            "fixed_across_all_fractions": True,
        },
        "cross_encoder": CE_MODEL_NAME,
        "cross_encoder_recipe": {
            "epochs": CE_EPOCHS,
            "warmup_fraction": CE_WARMUP_FRAC,
            "learning_rate": CE_LR,
            "batch_size": CE_BATCH,
            "max_length": CE_MAX_LEN,
            "score_path": SCORE_PATH,
            "validation_fraction_of_pairs": VALIDATION_FRAC,
            "threshold_grid": [float(value) for value in THRESHOLDS],
            "rng_reset_before_each_fit": True,
            "rng_sources": ["python", "numpy", "torch", "torch_cuda"],
        },
        "corrections": {
            "complete_raw_b": True,
            "many_match_truth": True,
            "any_valid_match_is_correct": True,
            "mutually_exclusive_counting": True,
            "MMR_denominator": "test source items with at least one true partner",
            "training_positive": "first-listed retained valid B partner",
            "alternative_valid_matches_excluded_from_negatives": True,
            "raw_decoding": "Latin-1",
            "preprocessing": "frozen June analysis18 CSV round-trip and CE serialisation",
        },
        "bias": {
            "FLR": "inner_FLR - target_FLR",
            "MMR": "inner_MMR - target_MMR",
            "Monte_Carlo_SE": "sample SD of the ten seed-level biases divided by sqrt(10)",
        },
        "checkpointing": "atomic JSON/CSV per target and per inner scheme; model checkpoints retained unless explicitly pruned",
        "complete_counts": complete_counts,
        "generated_at": ts(),
    }
    atomic_write_json(RESULTS_DIR / "analysis22_section332_manifest.json", manifest)


def aggregate_results() -> None:
    frame = load_inner_frame()
    perseed_path = RESULTS_DIR / "analysis22_section332_bias_perseed.csv"
    aggregate_path = RESULTS_DIR / "analysis22_section332_bias_aggregate.csv"
    mc10_path = RESULTS_DIR / "analysis22_section332_bias_mc10.csv"
    sizes_path = RESULTS_DIR / "analysis22_section332_sample_sizes.csv"

    if frame.empty:
        print("No completed inner cells found; no aggregate outputs written.")
        return

    preferred_columns = [
        "dataset",
        "dataset_display",
        "fraction",
        "seed",
        "scheme",
        "target_FLR",
        "inner_FLR",
        "bias_FLR",
        "target_MMR",
        "inner_MMR",
        "bias_MMR",
        "outer_labelled_size",
        "outer_holdout_size",
        "inner_train_draw_size",
        "inner_train_unique_size",
        "inner_test_size",
        "target_reused",
        "target_source",
        "analysis18_checkpoint_present_and_compatible",
        "threshold",
        "validation_f1",
        "n_train_pairs_total",
        "n_fit_pairs",
        "n_validation_pairs",
        "inner_declared",
        "inner_false_links",
        "inner_missing_matches",
        "inner_n_true_records",
        "train_seconds_this_run",
        "evaluation_seconds",
        "model_path",
        "prediction_path",
        "split_path",
    ]
    remaining = [column for column in frame.columns if column not in preferred_columns]
    atomic_write_csv(frame[[c for c in preferred_columns if c in frame] + remaining], perseed_path)

    aggregate = aggregate_bias(frame)
    atomic_write_csv(aggregate, aggregate_path)
    complete = aggregate[aggregate["n_splits"] == len(SEEDS)].copy()
    if not complete.empty:
        atomic_write_csv(complete, mc10_path)
    elif mc10_path.exists():
        mc10_path.unlink()

    sizes = sample_size_summary(frame)
    atomic_write_csv(sizes, sizes_path)
    save_manifest(frame)

    print("\nSection 3.3.2 bias aggregate")
    print("=" * 110)
    print(aggregate.to_string(index=False))
    print("\nSample sizes")
    print("=" * 110)
    print(sizes.to_string(index=False))
    print("\nSaved:")
    print(f"  {perseed_path}")
    print(f"  {aggregate_path}")
    print(
        f"  MC10: {'written' if not complete.empty else 'not yet complete'}: {mc10_path}"
    )
    print(f"  {sizes_path}")
    print(f"  {RESULTS_DIR / 'analysis22_section332_manifest.json'}")


def print_sizes(
    datasets: list[dict[str, Any]], fractions: list[float], seeds: list[int]
) -> None:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for fraction in fractions:
            for seed in seeds:
                labelled_ids, holdout_ids = make_outer_split(dataset, fraction, seed)
                for scheme in SCHEMES:
                    train_draw, inner_test_ids, _ = make_inner_split(
                        labelled_ids, scheme, seed
                    )
                    rows.append(
                        {
                            "dataset": dataset["name"],
                            "f": fraction,
                            "seed": seed,
                            "scheme": scheme,
                            "outer_labelled_size": len(labelled_ids),
                            "outer_holdout_size": len(holdout_ids),
                            "inner_train_draw_size": len(train_draw),
                            "inner_train_unique_size": len(set(train_draw)),
                            "inner_test_size": len(inner_test_ids),
                        }
                    )
    frame = pd.DataFrame(rows)
    summary = sample_size_summary(frame.rename(columns={"f": "fraction"}))
    print("\nDeterministic/seed-specific sample-size design")
    print("=" * 110)
    print(summary.to_string(index=False))


# =============================================================================
# AUDIT AND MAIN RUNNER
# =============================================================================
def audit_targets(
    datasets: list[dict[str, Any]], fractions: list[float], seeds: list[int]
) -> None:
    manifest_errors = analysis18_manifest_errors()
    if manifest_errors:
        print("Analysis18 manifest audit FAILED:")
        for error in manifest_errors:
            print(f"  - {error}")
    else:
        print("Analysis18 manifest audit: PASS")

    passed = 0
    failed = 0
    checkpoint_compatible = 0
    for dataset in datasets:
        for fraction in fractions:
            for seed in seeds:
                result, errors = find_verified_analysis18_target(
                    dataset, fraction, seed
                )
                label = f"{dataset['name']} f={fraction:.2f} seed={seed}"
                if result is None:
                    failed += 1
                    print(f"FAIL {label}")
                    for error in errors:
                        print(f"    - {error}")
                else:
                    passed += 1
                    if result["analysis18_checkpoint_present_and_compatible"]:
                        checkpoint_compatible += 1
                    print(
                        f"PASS {label}: FLR={result['target_FLR']:.6f}; "
                        f"MMR={result['target_MMR']:.6f}; "
                        f"checkpoint={'yes' if result['analysis18_checkpoint_present_and_compatible'] else 'no/result-only'}"
                    )
    print("\nAnalysis18 target-reuse audit summary")
    print("=" * 78)
    print(f"Verified reusable target results: {passed}")
    print(f"Failed target results: {failed}")
    print(f"Compatible checkpoints present: {checkpoint_compatible}")
    if failed:
        raise RuntimeError(f"Target reuse audit failed for {failed} requested cells.")


def run_experiment(
    datasets: list[dict[str, Any]],
    fractions: list[float],
    seeds: list[int],
    schemes: list[str],
    target_reuse: str,
    prune_completed_models: bool,
) -> None:
    device = require_single_visible_gpu()
    attach_top50(datasets, device)

    total_outer = len(datasets) * len(fractions) * len(seeds)
    total_inner = total_outer * len(schemes)
    print("\nanalysis22: revised-paper Section 3.3.2 bias experiment")
    print("=" * 96)
    print("This is not analysis13 and uses no analysis13 proxy code or results.")
    print(f"Datasets: {[dataset['name'] for dataset in datasets]}")
    print(f"Fractions: {fractions}")
    print(f"Seeds: {seeds}")
    print(f"Schemes: {schemes}")
    print(f"Target reuse mode: {target_reuse}")
    print(f"Requested outer cells: {total_outer}")
    print(f"Requested inner model cells: {total_inner}")
    print(
        f"Maximum fine-tunes if no target reuse: {total_outer + total_inner}; "
        f"with full target reuse: {total_inner}"
    )
    print(
        f"CE recipe: {CE_EPOCHS} epochs, {CE_WARMUP_FRAC:.0%} warmup, "
        f"lr={CE_LR}, batch={CE_BATCH}, max_length={CE_MAX_LEN}"
    )

    cells = [
        (dataset, fraction, seed)
        for dataset in datasets
        for fraction in fractions
        for seed in seeds
    ]
    started = time.time()
    for index, (dataset, fraction, seed) in enumerate(cells, start=1):
        print("\n" + "=" * 96)
        print(
            f"[{index}/{len(cells)}] {dataset['name']} f={fraction:.2f} seed={seed} "
            f"[{ts()}] elapsed={(time.time() - started) / 3600:.2f}h"
        )
        print("=" * 96)
        target = ensure_target(
            dataset,
            fraction,
            seed,
            target_reuse,
            device,
            prune_completed_models,
        )
        for scheme in schemes:
            run_inner_scheme(
                dataset,
                fraction,
                seed,
                scheme,
                target,
                device,
                prune_completed_models,
            )
        aggregate_results()

    aggregate_results()
    print("\nAnalysis22 requested cells complete.")


def main() -> None:
    args = parse_args()
    if args.phase == "aggregate":
        aggregate_results()
        return

    dataset_names = parse_csv_values(
        args.datasets, set(CONFIGS), "dataset names"
    )
    fractions = parse_fractions(args.fractions)
    seeds = parse_seeds(args.seeds)
    schemes = parse_csv_values(args.schemes, set(SCHEMES), "schemes")
    datasets = [load_dataset(name) for name in dataset_names]

    if args.phase == "sizes":
        print_sizes(datasets, fractions, seeds)
        return
    if args.phase == "audit-targets":
        # The experiment is intentionally GPU-dependent. Require the same
        # single pinned CUDA device even for the pre-run audit so the launch
        # path cannot proceed on a CPU-only host by mistake.
        require_single_visible_gpu()
        audit_targets(datasets, fractions, seeds)
        return

    run_experiment(
        datasets=datasets,
        fractions=fractions,
        seeds=seeds,
        schemes=schemes,
        target_reuse=args.target_reuse,
        prune_completed_models=args.prune_completed_models,
    )


if __name__ == "__main__":
    main()
