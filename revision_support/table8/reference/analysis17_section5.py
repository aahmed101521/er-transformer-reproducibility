"""
analysis17_section5.py
======================

Corrected, resumable rerun for Section 5 of the entity-resolution paper.

Covers
------
Table 5:
    Pre-trained versus fine-tuned bi-encoder at k=50, with MEC and LR.

Table 6:
    The four corrected two-step cells recomputed internally at k=50,
    plus the corrected cross-encoder row.

This file is standalone for Section 5. It does not require Analysis 16.
It recomputes only the corrected k=50 two-step baselines needed by
Tables 5 and 6; it does not regenerate the full Section 4 block-size table.

Table 7:
    False-link and missing-match rates by two-step decision-gap quartile,
    using the corrected record-level counting rule.

Corrections held throughout
---------------------------
1. The complete raw B file remains intact.
2. Test truth is many-match: id_A -> set(all valid id_B).
3. A declared link to any valid B match is correct.
4. False links and missing matches are mutually exclusive.
5. The MMR denominator is the number of A records with at least one
   true match.
6. Training remains one-to-one:
      - first-listed reachable B match is the sole positive;
      - all known valid alternative B matches are excluded from negatives.

June protocol held fixed
------------------------
- raw files decoded as Latin-1;
- lowercase + strip preprocessing and the February-write / June-read
  pandas CSV round-trip;
- pre-trained bi-encoder: all-MiniLM-L6-v2;
- fine-tuned bi-encoder recipe from analysis10_finetuned_biencoder.py;
- cross-encoder recipe from analysis8_crossencoder.py;
- CE serialisation: fields joined with " [SEP] ", NaN serialised as "";
- 50/50 A split and 80/20 pair-level threshold-calibration split;
- thresholds selected over 0.10, 0.15, ..., 0.90 by validation F1.

Cross-encoder recipe read directly from analysis8_crossencoder.py
-----------------------------------------------------------------
    model       cross-encoder/ms-marco-MiniLM-L-6-v2
    epochs      5
    warmup      10% of total training steps
    learning rate 2e-5
    batch size  16
    max_length  128

Fine-tuned bi-encoder recipe read directly from analysis10
-----------------------------------------------------------
    model       all-MiniLM-L6-v2
    loss        ContrastiveLoss
    epochs      1
    warmup      10%
    learning rate 2e-5
    batch size  16

Important Table 7 fidelity note
-------------------------------
The current draft's Table 7 matches analysis11_hardness.py exactly and
therefore comes from the MiniLM cross-encoder, not the older RoBERTa
hardest-20% branch.

analysis11 applied an additional sigmoid to CrossEncoder.predict() before
threshold calibration. The current draft's quartile values were produced
that way. To isolate the counting and candidate-universe corrections:

- Table 6 uses the analysis8 score path: CrossEncoder.predict() directly.
- Table 7 preserves the analysis11 legacy path: sigmoid(predict_output).

The top-candidate ranking is unchanged because sigmoid is monotone, but
the coarse threshold grid can change declarations. Both thresholds and
both metric sets are saved so this legacy difference is explicit.

Resumability
------------
Work is checkpointed by (phase, dataset, seed).

- Fine-tuned bi-encoder models:
    models/analysis17_section5/ft_biencoder/<DATASET>/seed_<SEED>/

- Cross-encoder models:
    models/analysis17_section5/cross_encoder/<DATASET>/seed_<SEED>/

- Atomic cell result JSON:
    results/analysis17_section5/cells/

- Per-cell prediction partitions:
    results/analysis17_section5/predictions/

If the process is interrupted:
- a completed cell is skipped on restart;
- if training completed and its model checkpoint was saved, evaluation
  resumes from the saved model rather than retraining;
- a cell interrupted during model training restarts that one cell only.

Seed design
-----------
Primary:     42, 43, 44
Diagnostic: 45, 46, 47, 48, 49, 50, 51

Primary cells are always processed first.

Recommended execution
---------------------
First bank the three primary seeds:

    cd ~/er_paper
    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis17_section5.py \
        --primary-only 2>&1 | tee results/analysis17_primary_console.log

Then resume and add the seven diagnostic seeds:

    CUDA_VISIBLE_DEVICES=0 python scripts/analysis17_section5.py \
        2>&1 | tee results/analysis17_diagnostic_console.log

To run only one phase:

    --phase baseline
    --phase ft
    --phase ce
    --phase aggregate

Final outputs
-------------
results/analysis17_baseline_k50_perseed.csv
results/analysis17_ft_biencoder_perseed.csv
results/analysis17_ce_perseed.csv
results/analysis17_predictions.csv

results/analysis17_table5_primary3.csv
results/analysis17_table5_mc10.csv
results/analysis17_table6_primary3.csv
results/analysis17_table6_mc10.csv
results/analysis17_table7_primary3.csv
results/analysis17_table7_mc10.csv

results/analysis17_table7_direct_primary3.csv
results/analysis17_table7_direct_mc10.csv
    Diagnostic Table 7 using the analysis8 direct CE score path.

results/analysis17_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Iterable

import jellyfish
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import (
    CrossEncoder,
    InputExample,
    SentenceTransformer,
    losses,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
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

STATE_DIR = RESULTS_DIR / "analysis17_section5"
CELL_DIR = STATE_DIR / "cells"
PRED_PART_DIR = STATE_DIR / "predictions"
MODEL_ROOT = MODELS_DIR / "analysis17_section5"

for directory in (
    RESULTS_DIR,
    MODELS_DIR,
    STATE_DIR,
    CELL_DIR,
    PRED_PART_DIR,
    MODEL_ROOT,
):
    directory.mkdir(parents=True, exist_ok=True)

PRIMARY_SEEDS = [42, 43, 44]
DIAGNOSTIC_SEEDS = [45, 46, 47, 48, 49, 50, 51]
ALL_SEEDS = PRIMARY_SEEDS + DIAGNOSTIC_SEEDS

BI_ENCODER_NAME = "all-MiniLM-L6-v2"
CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

K = 50
KDE_BANDWIDTH = 0.05
LR_MAX_ITER = 500
LR_VAL_FRAC = 0.20
THRESHOLDS = np.arange(0.10, 0.90 + 1e-9, 0.05)

# analysis10_finetuned_biencoder.py
FT_EPOCHS = 1
FT_WARMUP_FRAC = 0.10
FT_LR = 2e-5
FT_BATCH = 16

# analysis8_crossencoder.py
CE_EPOCHS = 5
CE_WARMUP_FRAC = 0.10
CE_LR = 2e-5
CE_BATCH = 16
CE_MAX_LEN = 128

EMBED_BATCH = 256
RETRIEVAL_CHUNK = 256
PREDICT_BATCH = 32
FEATURE_CHUNK = 8192

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


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temp_path, path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp_path, index=False)
    os.replace(temp_path, path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-values))


def model_complete(model_path: Path) -> bool:
    return model_path.is_dir() and (model_path / "_analysis17_complete.json").exists()


def mark_model_complete(model_path: Path, metadata: dict[str, Any]) -> None:
    atomic_write_json(model_path / "_analysis17_complete.json", metadata)


def save_model_atomic(
    model: Any,
    model_path: Path,
    metadata: dict[str, Any],
) -> None:
    """
    Save a SentenceTransformer or CrossEncoder directory atomically enough
    for cell-level recovery: save to a sibling temporary directory, add a
    completion marker, then rename into place.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = model_path.with_name(model_path.name + ".tmp")
    safe_remove(temp_path)

    if isinstance(model, CrossEncoder):
        model.save_pretrained(str(temp_path))
    else:
        model.save(str(temp_path))
    mark_model_complete(temp_path, metadata)

    safe_remove(model_path)
    os.replace(temp_path, model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corrected, resumable Section 5 rerun."
    )
    parser.add_argument(
        "--phase",
        choices=["all", "baseline", "ft", "ce", "aggregate"],
        default="all",
        help="Run all phases, only k=50 baselines, only fine-tuned bi-encoder, only CE, or aggregate.",
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Run seeds 42, 43, 44 only.",
    )
    parser.add_argument(
        "--datasets",
        default="DBLP,ECOM",
        help="Comma-separated subset of DBLP,ECOM.",
    )
    return parser.parse_args()


# =============================================================================
# DATA LOADING: EXACT JUNE PREPROCESSING + COMPLETE RAW B
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
    frame.columns = [clean_header(c) for c in frame.columns]
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
    """
    Reproduce the state seen by the June scripts after clean CSVs were
    written and read with pandas' default NA parsing.
    """
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
    """
    Exact analysis8/analysis11 serialisation: pandas NaN becomes blank.
    """
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
    df_a = read_latin1(cfg["a"])
    df_b = read_latin1(cfg["b"])
    mapping = read_latin1(cfg["mapping"])

    id_a_file = find_id_column(df_a, cfg["a"])
    id_b_file = find_id_column(df_b, cfg["b"])

    df_a = df_a.rename(columns=cfg["rename"])
    df_b = df_b.rename(columns=cfg["rename"])

    for frame, path in ((df_a, cfg["a"]), (df_b, cfg["b"])):
        missing = set(cfg["fields"]).difference(frame.columns)
        if missing:
            raise KeyError(
                f"{path} is missing fields {sorted(missing)}; "
                f"columns={list(frame.columns)}"
            )
        for field in cfg["fields"]:
            frame[field] = normalize_text(frame[field])

    df_a[id_a_file] = df_a[id_a_file].fillna("").astype(str).str.strip()
    df_b[id_b_file] = df_b[id_b_file].fillna("").astype(str).str.strip()

    df_a = june_csv_roundtrip(df_a, id_a_file)
    df_b = june_csv_roundtrip(df_b, id_b_file)

    map_a = cfg["id_a"]
    map_b = cfg["id_b"]
    if map_a not in mapping.columns or map_b not in mapping.columns:
        raise KeyError(
            f"{cfg['mapping']} must contain {map_a!r}, {map_b!r}; "
            f"columns={list(mapping.columns)}"
        )

    mapping[map_a] = mapping[map_a].fillna("").astype(str).str.strip()
    mapping[map_b] = mapping[map_b].fillna("").astype(str).str.strip()

    observed = {"A": len(df_a), "B": len(df_b), "pairs": len(mapping)}
    if observed != cfg["expected"]:
        raise AssertionError(
            f"{name} snapshot mismatch: observed={observed}, "
            f"expected={cfg['expected']}"
        )

    if not df_a.index.is_unique or not df_b.index.is_unique:
        raise ValueError(f"{name}: A and B ids must be unique.")

    a_ids_set = set(df_a.index)
    b_ids_set = set(df_b.index)
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
        f"|A|={len(df_a):,}  |B|={len(df_b):,}  "
        f"raw pairs={len(mapping):,}"
    )
    print(
        f"A records with truth={len(truth):,}  "
        f"A with >1 match={sum(len(v) > 1 for v in truth.values()):,}  "
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
        "df_a": df_a,
        "df_b": df_b,
        "truth": truth,
        "retained": retained,
        "a_ids": list(df_a.index),
        "b_ids": list(df_b.index),
    }


# =============================================================================
# BASELINE EMBEDDINGS, TOP-50 BLOCKS, FEATURES
# =============================================================================
def encode_fields(
    model: SentenceTransformer,
    frame: pd.DataFrame,
    fields: list[str],
    label: str,
) -> dict[str, torch.Tensor]:
    embeddings: dict[str, torch.Tensor] = {}
    for field in fields:
        print(f"  encoding {label}.{field} [{ts()}]")
        embeddings[field] = model.encode(
            frame[field].fillna("").tolist(),
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=EMBED_BATCH,
            normalize_embeddings=True,
        )
    return embeddings


def retrieve_top50(
    emb_a_title: torch.Tensor,
    emb_b_title: torch.Tensor,
    a_ids: list[str],
    b_ids: list[str],
) -> dict[str, list[str]]:
    top50: dict[str, list[str]] = {}

    for start in range(0, len(a_ids), RETRIEVAL_CHUNK):
        stop = min(start + RETRIEVAL_CHUNK, len(a_ids))
        scores = torch.matmul(
            emb_a_title[start:stop],
            emb_b_title.transpose(0, 1),
        )
        indices = torch.topk(scores, k=K, dim=1).indices.cpu().tolist()

        for offset, row_indices in enumerate(indices):
            top50[a_ids[start + offset]] = [b_ids[i] for i in row_indices]

        print(
            f"  retrieved {stop:,}/{len(a_ids):,} A records",
            end="\r",
            flush=True,
        )
    print()
    return top50


def prepare_baseline(
    dataset: dict[str, Any],
    base_model: SentenceTransformer,
) -> None:
    fields = dataset["cfg"]["fields"]
    print(f"\n{dataset['name']} baseline embeddings and blocking")
    print("-" * 78)

    emb_a = encode_fields(base_model, dataset["df_a"], fields, "A")
    emb_b = encode_fields(base_model, dataset["df_b"], fields, "B")

    top50 = retrieve_top50(
        emb_a[dataset["cfg"]["retrieval_field"]],
        emb_b[dataset["cfg"]["retrieval_field"]],
        dataset["a_ids"],
        dataset["b_ids"],
    )

    dataset["base_emb_a"] = emb_a
    dataset["base_emb_b"] = emb_b
    dataset["base_top50"] = top50
    dataset["a_pos"] = {
        record_id: index for index, record_id in enumerate(dataset["a_ids"])
    }
    dataset["b_pos"] = {
        record_id: index for index, record_id in enumerate(dataset["b_ids"])
    }

    # Cache the June string-side features once. The June JW helper only
    # checked ``is None`` before converting values with ``str``; pandas
    # NaN therefore became the literal string "nan". Preserve that
    # executed behaviour to isolate only the B/truth/counting corrections.
    print(f"  caching June string features for k=50 pairs [{ts()}]")
    string_cache: dict[tuple[str, str], np.ndarray] = {}
    fields = dataset["cfg"]["fields"]

    for record_index, ida in enumerate(dataset["a_ids"], start=1):
        candidate_ids = set(top50[ida])
        retained_idb = dataset["retained"].get(ida)
        if retained_idb is not None:
            candidate_ids.add(retained_idb)

        row_a = dataset["df_a"].loc[ida]
        for idb in candidate_ids:
            row_b = dataset["df_b"].loc[idb]
            string_cache[(ida, idb)] = np.asarray(
                [
                    june_jw(row_a.get(field), row_b.get(field))
                    for field in fields
                ],
                dtype=np.float64,
            )

        if record_index % 250 == 0 or record_index == len(dataset["a_ids"]):
            print(
                f"    cached {record_index:,}/{len(dataset['a_ids']):,} A records",
                end="\r",
                flush=True,
            )
    print()
    dataset["string_feature_cache"] = string_cache


def june_jw(value_a: object, value_b: object) -> float:
    """
    Exact Jaro-Winkler path executed by analysis1_clean_2x2.py.

    The June function's docstring said NaN mapped to zero, but the code
    only tested ``is None`` and then called ``str``. Thus np.nan was
    compared as the literal text "nan". This is retained deliberately.
    """
    if value_a is None or value_b is None:
        return 0.0
    text_a = str(value_a)
    text_b = str(value_b)
    if not text_a or not text_b:
        return 0.0
    return float(jellyfish.jaro_winkler_similarity(text_a, text_b))


def string_feature_matrix(
    dataset: dict[str, Any],
    pairs: list[tuple[str, str]],
) -> np.ndarray:
    if not pairs:
        return np.empty((0, len(dataset["cfg"]["fields"])), dtype=np.float64)
    return np.vstack(
        [dataset["string_feature_cache"][(ida, idb)] for ida, idb in pairs]
    )


def build_train_pairs(
    ids: Iterable[str],
    retained: dict[str, str],
    truth: dict[str, set[str]],
    top50: dict[str, list[str]],
) -> list[tuple[str, str, int]]:
    """
    One positive per matched A; all valid alternatives excluded from negatives.
    """
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


def build_test_structure(
    ids: Iterable[str],
    truth: dict[str, set[str]],
    top50: dict[str, list[str]],
) -> list[tuple[str, set[str], list[str]]]:
    return [
        (ida, truth.get(ida, set()), top50[ida])
        for ida in ids
    ]


def cosine_feature_matrix(
    pairs: list[tuple[str, str]],
    emb_a: dict[str, torch.Tensor],
    emb_b: dict[str, torch.Tensor],
    fields: list[str],
    a_pos: dict[str, int],
    b_pos: dict[str, int],
) -> np.ndarray:
    """
    Vectorised per-field cosine features. Embeddings are already L2-normalised,
    so row-wise dot products are cosine similarities. Chunking avoids the
    per-pair GPU synchronisation that would otherwise dominate the cheap
    two-step phases.
    """
    if not pairs:
        return np.empty((0, len(fields)), dtype=np.float64)

    matrix = np.empty((len(pairs), len(fields)), dtype=np.float64)
    device = emb_a[fields[0]].device

    a_indices = np.fromiter(
        (a_pos[ida] for ida, _ in pairs),
        dtype=np.int64,
        count=len(pairs),
    )
    b_indices = np.fromiter(
        (b_pos[idb] for _, idb in pairs),
        dtype=np.int64,
        count=len(pairs),
    )

    for start in range(0, len(pairs), FEATURE_CHUNK):
        stop = min(start + FEATURE_CHUNK, len(pairs))
        ia = torch.as_tensor(
            a_indices[start:stop], dtype=torch.long, device=device
        )
        ib = torch.as_tensor(
            b_indices[start:stop], dtype=torch.long, device=device
        )

        for field_index, field in enumerate(fields):
            similarities = (
                emb_a[field].index_select(0, ia)
                * emb_b[field].index_select(0, ib)
            ).sum(dim=1)
            matrix[start:stop, field_index] = (
                similarities.detach().cpu().numpy().astype(np.float64)
            )

    return matrix


# =============================================================================
# SCORERS AND COUNTING
# =============================================================================
class MECScorer:
    def __init__(self, bandwidth: float = KDE_BANDWIDTH):
        self.bandwidth = bandwidth
        self.match_kdes: list[KernelDensity] = []
        self.nonmatch_kdes: list[KernelDensity] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.match_kdes = []
        self.nonmatch_kdes = []

        for field_index in range(x.shape[1]):
            match_values = x[y == 1, field_index].reshape(-1, 1)
            nonmatch_values = x[y == 0, field_index].reshape(-1, 1)
            self.match_kdes.append(
                KernelDensity(bandwidth=self.bandwidth).fit(match_values)
            )
            self.nonmatch_kdes.append(
                KernelDensity(bandwidth=self.bandwidth).fit(nonmatch_values)
            )

    def log_ratio(self, x: np.ndarray) -> np.ndarray:
        total = np.zeros(len(x), dtype=np.float64)
        for index, (match_kde, nonmatch_kde) in enumerate(
            zip(self.match_kdes, self.nonmatch_kdes)
        ):
            values = x[:, index].reshape(-1, 1)
            total += (
                match_kde.score_samples(values)
                - nonmatch_kde.score_samples(values)
            )
        return total


def calibrate_lr(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[LogisticRegression, float, float]:
    x_fit, x_val, y_fit, y_val = train_test_split(
        x,
        y,
        test_size=LR_VAL_FRAC,
        random_state=seed,
        stratify=y,
    )
    model = LogisticRegression(
        penalty="l2",
        max_iter=LR_MAX_ITER,
        random_state=seed,
    ).fit(x_fit, y_fit)

    probabilities = model.predict_proba(x_val)[:, 1]
    threshold, validation_f1 = choose_threshold(probabilities, y_val)
    return model, threshold, validation_f1


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


def evaluate(
    test_structure: list[tuple[str, set[str], list[str]]],
    score_fn: Callable[[str, list[str]], np.ndarray],
    declare_fn: Callable[[float], bool],
) -> dict[str, float | int]:
    """
    Zhang's mutually exclusive record-level rule.
    """
    declared = 0
    correct = 0
    false_links = 0
    missing_matches = 0
    n_true_records = 0

    for ida, true_set, candidates in test_structure:
        if true_set:
            n_true_records += 1

        scores = np.asarray(score_fn(ida, candidates), dtype=float)
        top_index = int(np.argmax(scores))
        top_score = float(scores[top_index])
        top_idb = candidates[top_index]

        if declare_fn(top_score):
            declared += 1
            if top_idb in true_set:
                correct += 1
            else:
                false_links += 1
        elif true_set:
            missing_matches += 1

    if declared != correct + false_links:
        raise AssertionError("Declared != correct + false links")

    return {
        "lambda": false_links / declared if declared else 0.0,
        "psi": (
            missing_matches / n_true_records
            if n_true_records
            else 0.0
        ),
        "declared": declared,
        "correct": correct,
        "false_links": false_links,
        "missing_matches": missing_matches,
        "n_true_records": n_true_records,
    }


def category_for(
    declared: bool,
    top_idb: str,
    true_set: set[str],
) -> str:
    if declared:
        return "correct" if top_idb in true_set else "false_link"
    return "missing_match" if true_set else "unmatched_abstention"


# =============================================================================
# CORRECTED PRE-TRAINED k=50 BASELINES: TABLES 5 AND 6
# =============================================================================
def baseline_cell_path(dataset_name: str, seed: int) -> Path:
    return CELL_DIR / f"baseline_{dataset_name}_seed{seed}.json"


def run_baseline_cell(
    dataset: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """
    Recompute only the four corrected k=50 two-step cells required by
    Section 5: string/MEC, string/LR, bi-encoder/MEC, bi-encoder/LR.
    """
    result_path = baseline_cell_path(dataset["name"], seed)
    if result_path.exists():
        print(f"    BASELINE SKIP completed: {dataset['name']} seed={seed}")
        return read_json(result_path)

    seed_everything(seed)
    ids_train, ids_test = train_test_split(
        dataset["a_ids"],
        test_size=0.50,
        random_state=seed,
    )

    train_pairs = build_train_pairs(
        ids_train,
        dataset["retained"],
        dataset["truth"],
        dataset["base_top50"],
    )
    test_structure = build_test_structure(
        ids_test,
        dataset["truth"],
        dataset["base_top50"],
    )
    pair_ids = [(ida, idb) for ida, idb, _ in train_pairs]
    y_train = np.asarray(
        [label for _, _, label in train_pairs],
        dtype=int,
    )

    fields = dataset["cfg"]["fields"]
    start_time = time.time()
    results: dict[str, Any] = {}

    for side in ("string", "bi-encoder"):
        if side == "string":
            x_train = string_feature_matrix(dataset, pair_ids)

            def candidate_features(
                ida: str,
                candidates: list[str],
            ) -> np.ndarray:
                return string_feature_matrix(
                    dataset,
                    [(ida, idb) for idb in candidates],
                )
        else:
            x_train = cosine_feature_matrix(
                pair_ids,
                dataset["base_emb_a"],
                dataset["base_emb_b"],
                fields,
                dataset["a_pos"],
                dataset["b_pos"],
            )

            def candidate_features(
                ida: str,
                candidates: list[str],
            ) -> np.ndarray:
                return cosine_feature_matrix(
                    [(ida, idb) for idb in candidates],
                    dataset["base_emb_a"],
                    dataset["base_emb_b"],
                    fields,
                    dataset["a_pos"],
                    dataset["b_pos"],
                )

        mec = MECScorer()
        mec.fit(x_train, y_train)
        mec_metrics = evaluate(
            test_structure,
            lambda ida, cands: mec.log_ratio(
                candidate_features(ida, cands)
            ),
            lambda score: score > 0.0,
        )

        lr_model, lr_threshold, lr_validation_f1 = calibrate_lr(
            x_train,
            y_train,
            seed,
        )
        lr_metrics = evaluate(
            test_structure,
            lambda ida, cands: lr_model.predict_proba(
                candidate_features(ida, cands)
            )[:, 1],
            lambda score: score > lr_threshold,
        )

        results[side] = {
            "MEC": mec_metrics,
            "LR": lr_metrics,
            "lr_threshold": lr_threshold,
            "lr_validation_f1": lr_validation_f1,
        }

        print(
            f"      baseline {side}: "
            f"MEC lambda={mec_metrics['lambda']:.4f}, "
            f"psi={mec_metrics['psi']:.4f}; "
            f"LR lambda={lr_metrics['lambda']:.4f}, "
            f"psi={lr_metrics['psi']:.4f}, "
            f"threshold={lr_threshold:.2f}"
        )

    payload: dict[str, Any] = {
        "phase": "baseline_k50",
        "dataset": dataset["name"],
        "seed": seed,
        "n_train_a": len(ids_train),
        "n_test_a": len(ids_test),
        "n_train_pairs": len(train_pairs),
        "runtime_seconds": time.time() - start_time,
        "results": results,
    }
    atomic_write_json(result_path, payload)
    return payload


# =============================================================================
# FINE-TUNED BI-ENCODER: TABLE 5
# =============================================================================
def ft_cell_path(dataset_name: str, seed: int) -> Path:
    return CELL_DIR / f"ft_{dataset_name}_seed{seed}.json"


def ft_model_path(dataset_name: str, seed: int) -> Path:
    return (
        MODEL_ROOT
        / "ft_biencoder"
        / dataset_name
        / f"seed_{seed}"
    )


def train_or_load_ft_model(
    dataset: dict[str, Any],
    train_pairs: list[tuple[str, str, int]],
    seed: int,
    device: str,
) -> SentenceTransformer:
    path = ft_model_path(dataset["name"], seed)

    if model_complete(path):
        print(f"      loading saved fine-tuned bi-encoder: {path}")
        return SentenceTransformer(str(path), device=device)

    if path.exists():
        print(f"      removing incomplete fine-tuned checkpoint: {path}")
        safe_remove(path)

    fields = dataset["cfg"]["fields"]
    examples = [
        InputExample(
            texts=[
                serialise(dataset["df_a"].loc[ida], fields),
                serialise(dataset["df_b"].loc[idb], fields),
            ],
            label=float(label),
        )
        for ida, idb, label in train_pairs
    ]

    model = SentenceTransformer(BI_ENCODER_NAME, device=device)
    loader = DataLoader(examples, shuffle=True, batch_size=FT_BATCH)
    loss = losses.ContrastiveLoss(model)

    total_steps = len(loader) * FT_EPOCHS
    warmup_steps = int(FT_WARMUP_FRAC * total_steps)

    print(
        f"      fine-tuning bi-encoder: pairs={len(examples):,}, "
        f"epochs={FT_EPOCHS}, warmup={warmup_steps}, "
        f"lr={FT_LR}, batch={FT_BATCH} [{ts()}]"
    )

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=FT_EPOCHS,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": FT_LR},
        show_progress_bar=False,
    )

    save_model_atomic(
        model,
        path,
        {
            "phase": "ft_biencoder",
            "dataset": dataset["name"],
            "seed": seed,
            "model": BI_ENCODER_NAME,
            "epochs": FT_EPOCHS,
            "warmup_fraction": FT_WARMUP_FRAC,
            "learning_rate": FT_LR,
            "batch_size": FT_BATCH,
            "training_pairs": len(examples),
        },
    )
    print(f"      saved fine-tuned checkpoint: {path}")
    return model


def run_ft_cell(
    dataset: dict[str, Any],
    seed: int,
    device: str,
) -> dict[str, Any]:
    result_path = ft_cell_path(dataset["name"], seed)
    if result_path.exists():
        print(f"    FT SKIP completed: {dataset['name']} seed={seed}")
        return read_json(result_path)

    seed_everything(seed)
    ids_train, ids_test = train_test_split(
        dataset["a_ids"],
        test_size=0.50,
        random_state=seed,
    )

    baseline_pairs = build_train_pairs(
        ids_train,
        dataset["retained"],
        dataset["truth"],
        dataset["base_top50"],
    )

    start_time = time.time()
    ft_model = train_or_load_ft_model(
        dataset,
        baseline_pairs,
        seed,
        device,
    )

    fields = dataset["cfg"]["fields"]
    print(f"      re-embedding complete A and B with fine-tuned encoder [{ts()}]")
    ft_emb_a = encode_fields(ft_model, dataset["df_a"], fields, "FT-A")
    ft_emb_b = encode_fields(ft_model, dataset["df_b"], fields, "FT-B")
    ft_top50 = retrieve_top50(
        ft_emb_a[dataset["cfg"]["retrieval_field"]],
        ft_emb_b[dataset["cfg"]["retrieval_field"]],
        dataset["a_ids"],
        dataset["b_ids"],
    )

    ft_train_pairs = build_train_pairs(
        ids_train,
        dataset["retained"],
        dataset["truth"],
        ft_top50,
    )
    test_structure = build_test_structure(
        ids_test,
        dataset["truth"],
        ft_top50,
    )

    pair_ids = [(ida, idb) for ida, idb, _ in ft_train_pairs]
    x_train = cosine_feature_matrix(
        pair_ids,
        ft_emb_a,
        ft_emb_b,
        fields,
        dataset["a_pos"],
        dataset["b_pos"],
    )
    y_train = np.asarray(
        [label for _, _, label in ft_train_pairs],
        dtype=int,
    )

    mec = MECScorer()
    mec.fit(x_train, y_train)

    lr_model, lr_threshold, lr_validation_f1 = calibrate_lr(
        x_train,
        y_train,
        seed,
    )

    def candidate_features(ida: str, candidates: list[str]) -> np.ndarray:
        return cosine_feature_matrix(
            [(ida, idb) for idb in candidates],
            ft_emb_a,
            ft_emb_b,
            fields,
            dataset["a_pos"],
            dataset["b_pos"],
        )

    mec_metrics = evaluate(
        test_structure,
        lambda ida, cands: mec.log_ratio(candidate_features(ida, cands)),
        lambda score: score > 0.0,
    )
    lr_metrics = evaluate(
        test_structure,
        lambda ida, cands: lr_model.predict_proba(
            candidate_features(ida, cands)
        )[:, 1],
        lambda score: score > lr_threshold,
    )

    payload: dict[str, Any] = {
        "phase": "ft_biencoder",
        "dataset": dataset["name"],
        "seed": seed,
        "n_train_a": len(ids_train),
        "n_test_a": len(ids_test),
        "n_baseline_train_pairs": len(baseline_pairs),
        "n_ft_train_pairs": len(ft_train_pairs),
        "lr_threshold": lr_threshold,
        "lr_validation_f1": lr_validation_f1,
        "runtime_seconds": time.time() - start_time,
        "MEC": mec_metrics,
        "LR": lr_metrics,
    }
    atomic_write_json(result_path, payload)
    print(
        f"      FT saved: MEC lambda={mec_metrics['lambda']:.4f}, "
        f"psi={mec_metrics['psi']:.4f}; "
        f"LR lambda={lr_metrics['lambda']:.4f}, "
        f"psi={lr_metrics['psi']:.4f}"
    )

    del ft_model, ft_emb_a, ft_emb_b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return payload


# =============================================================================
# CROSS-ENCODER + HARDNESS: TABLES 6 AND 7
# =============================================================================
def ce_cell_path(dataset_name: str, seed: int) -> Path:
    return CELL_DIR / f"ce_{dataset_name}_seed{seed}.json"


def ce_model_path(dataset_name: str, seed: int) -> Path:
    return (
        MODEL_ROOT
        / "cross_encoder"
        / dataset_name
        / f"seed_{seed}"
    )


def prediction_part_path(dataset_name: str, seed: int) -> Path:
    return PRED_PART_DIR / f"{dataset_name}_seed{seed}.csv"


def train_or_load_ce_model(
    dataset: dict[str, Any],
    train_only: list[tuple[str, str, int]],
    seed: int,
    device: str,
) -> CrossEncoder:
    path = ce_model_path(dataset["name"], seed)

    if model_complete(path):
        print(f"      loading saved cross-encoder: {path}")
        return CrossEncoder(
            str(path),
            max_length=CE_MAX_LEN,
            device=device,
        )

    if path.exists():
        print(f"      removing incomplete CE checkpoint: {path}")
        safe_remove(path)

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
        f"      training CE: pairs={len(examples):,}, "
        f"epochs={CE_EPOCHS}, warmup={warmup_steps}, "
        f"lr={CE_LR}, batch={CE_BATCH}, max_length={CE_MAX_LEN} [{ts()}]"
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
            "phase": "cross_encoder",
            "dataset": dataset["name"],
            "seed": seed,
            "model": CE_MODEL_NAME,
            "epochs": CE_EPOCHS,
            "warmup_fraction": CE_WARMUP_FRAC,
            "learning_rate": CE_LR,
            "batch_size": CE_BATCH,
            "max_length": CE_MAX_LEN,
            "training_pairs": len(examples),
        },
    )
    print(f"      saved CE checkpoint: {path}")
    return model


def fit_baseline_lr_for_seed(
    dataset: dict[str, Any],
    train_pairs: list[tuple[str, str, int]],
    seed: int,
) -> tuple[LogisticRegression, float, float]:
    fields = dataset["cfg"]["fields"]
    x_train = cosine_feature_matrix(
        [(ida, idb) for ida, idb, _ in train_pairs],
        dataset["base_emb_a"],
        dataset["base_emb_b"],
        fields,
        dataset["a_pos"],
        dataset["b_pos"],
    )
    y_train = np.asarray(
        [label for _, _, label in train_pairs],
        dtype=int,
    )
    return calibrate_lr(x_train, y_train, seed)


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


def metrics_from_prediction_rows(
    frame: pd.DataFrame,
    prefix: str,
) -> dict[str, float | int]:
    declared = int(frame[f"{prefix}_declared"].sum())
    correct = int((frame[f"{prefix}_category"] == "correct").sum())
    false_links = int((frame[f"{prefix}_category"] == "false_link").sum())
    missing_matches = int(
        (frame[f"{prefix}_category"] == "missing_match").sum()
    )
    n_true_records = int(frame["has_true_match"].sum())

    if declared != correct + false_links:
        raise AssertionError(
            f"{prefix}: declared does not equal correct + false"
        )

    return {
        "lambda": false_links / declared if declared else 0.0,
        "psi": (
            missing_matches / n_true_records
            if n_true_records
            else 0.0
        ),
        "declared": declared,
        "correct": correct,
        "false_links": false_links,
        "missing_matches": missing_matches,
        "n_true_records": n_true_records,
    }


def run_ce_cell(
    dataset: dict[str, Any],
    seed: int,
    device: str,
) -> dict[str, Any]:
    result_path = ce_cell_path(dataset["name"], seed)
    part_path = prediction_part_path(dataset["name"], seed)

    if result_path.exists() and part_path.exists():
        print(f"    CE SKIP completed: {dataset['name']} seed={seed}")
        return read_json(result_path)

    seed_everything(seed)
    ids_train, ids_test = train_test_split(
        dataset["a_ids"],
        test_size=0.50,
        random_state=seed,
    )

    train_pairs = build_train_pairs(
        ids_train,
        dataset["retained"],
        dataset["truth"],
        dataset["base_top50"],
    )

    labels = [label for _, _, label in train_pairs]
    train_indices, validation_indices = train_test_split(
        list(range(len(train_pairs))),
        test_size=LR_VAL_FRAC,
        random_state=seed,
        stratify=labels,
    )
    train_only = [train_pairs[index] for index in train_indices]
    validation_only = [train_pairs[index] for index in validation_indices]

    start_time = time.time()
    ce_model = train_or_load_ce_model(
        dataset,
        train_only,
        seed,
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

    # Table 6: exact analysis8 score path.
    validation_direct = ce_predict(ce_model, validation_text)
    ce_direct_threshold, ce_direct_validation_f1 = choose_threshold(
        validation_direct,
        validation_labels,
    )

    # Table 7: exact analysis11 legacy extra-sigmoid path.
    validation_legacy = sigmoid(validation_direct)
    ce_legacy_threshold, ce_legacy_validation_f1 = choose_threshold(
        validation_legacy,
        validation_labels,
    )

    # Baseline two-step LR for decision-gap stratification.
    lr_model, lr_threshold, lr_validation_f1 = fit_baseline_lr_for_seed(
        dataset,
        train_pairs,
        seed,
    )

    test_structure = build_test_structure(
        ids_test,
        dataset["truth"],
        dataset["base_top50"],
    )

    rows: list[dict[str, Any]] = []
    print(
        f"      evaluating {len(test_structure):,} test records "
        f"with {K} candidates each [{ts()}]"
    )

    for record_index, (ida, true_set, candidates) in enumerate(
        test_structure,
        start=1,
    ):
        candidate_pairs = [(ida, idb) for idb in candidates]
        x_candidates = cosine_feature_matrix(
            candidate_pairs,
            dataset["base_emb_a"],
            dataset["base_emb_b"],
            fields,
            dataset["a_pos"],
            dataset["b_pos"],
        )

        lr_scores = lr_model.predict_proba(x_candidates)[:, 1]
        lr_order = np.argsort(-lr_scores)
        lr_top1_index = int(lr_order[0])
        lr_top2_index = int(lr_order[1])
        lr_top1_idb = candidates[lr_top1_index]
        lr_top1_score = float(lr_scores[lr_top1_index])
        lr_top2_score = float(lr_scores[lr_top2_index])
        lr_declared = bool(lr_top1_score > lr_threshold)
        lr_category = category_for(
            lr_declared,
            lr_top1_idb,
            true_set,
        )

        ce_text = [
            (
                serialise(dataset["df_a"].loc[ida], fields),
                serialise(dataset["df_b"].loc[idb], fields),
            )
            for idb in candidates
        ]
        ce_direct_scores = ce_predict(ce_model, ce_text)
        ce_order = np.argsort(-ce_direct_scores)
        ce_top1_index = int(ce_order[0])
        ce_top2_index = int(ce_order[1])
        ce_top1_idb = candidates[ce_top1_index]
        ce_direct_top1 = float(ce_direct_scores[ce_top1_index])
        ce_direct_top2 = float(ce_direct_scores[ce_top2_index])

        ce_direct_declared = bool(
            ce_direct_top1 > ce_direct_threshold
        )
        ce_direct_category = category_for(
            ce_direct_declared,
            ce_top1_idb,
            true_set,
        )

        ce_legacy_scores = sigmoid(ce_direct_scores)
        ce_legacy_top1 = float(ce_legacy_scores[ce_top1_index])
        ce_legacy_top2 = float(ce_legacy_scores[ce_top2_index])
        ce_legacy_declared = bool(
            ce_legacy_top1 > ce_legacy_threshold
        )
        ce_legacy_category = category_for(
            ce_legacy_declared,
            ce_top1_idb,
            true_set,
        )

        rows.append(
            {
                "dataset": dataset["name"],
                "seed": seed,
                "record_index": record_index - 1,
                "ida": ida,
                "has_true_match": bool(true_set),
                "n_valid_matches": len(true_set),
                "retained_idb": dataset["retained"].get(ida, ""),
                "lr_top1_idb": lr_top1_idb,
                "lr_top1_score": lr_top1_score,
                "lr_top2_score": lr_top2_score,
                "lr_gap": lr_top1_score - lr_top2_score,
                "lr_threshold": lr_threshold,
                "lr_declared": lr_declared,
                "lr_category": lr_category,
                "ce_top1_idb": ce_top1_idb,
                "ce_top_is_retained": (
                    ce_top1_idb == dataset["retained"].get(ida)
                ),
                "ce_top_is_any_valid": ce_top1_idb in true_set,
                "ce_direct_top1_score": ce_direct_top1,
                "ce_direct_top2_score": ce_direct_top2,
                "ce_direct_threshold": ce_direct_threshold,
                "ce_direct_declared": ce_direct_declared,
                "ce_direct_category": ce_direct_category,
                "ce_legacy_top1_score": ce_legacy_top1,
                "ce_legacy_top2_score": ce_legacy_top2,
                "ce_legacy_threshold": ce_legacy_threshold,
                "ce_legacy_declared": ce_legacy_declared,
                "ce_legacy_category": ce_legacy_category,
            }
        )

        if record_index % 200 == 0 or record_index == len(test_structure):
            print(
                f"        {record_index:,}/{len(test_structure):,}",
                end="\r",
                flush=True,
            )
    print()

    predictions = pd.DataFrame(rows)
    atomic_write_csv(predictions, part_path)

    lr_metrics = metrics_from_prediction_rows(predictions, "lr")
    ce_direct_metrics = metrics_from_prediction_rows(
        predictions,
        "ce_direct",
    )
    ce_legacy_metrics = metrics_from_prediction_rows(
        predictions,
        "ce_legacy",
    )

    alternative_valid_links = int(
        (
            predictions["ce_direct_declared"]
            & predictions["ce_top_is_any_valid"]
            & ~predictions["ce_top_is_retained"]
        ).sum()
    )

    payload: dict[str, Any] = {
        "phase": "cross_encoder",
        "dataset": dataset["name"],
        "seed": seed,
        "n_train_a": len(ids_train),
        "n_test_a": len(ids_test),
        "n_train_pairs_full": len(train_pairs),
        "n_ce_train_pairs": len(train_only),
        "n_ce_validation_pairs": len(validation_only),
        "lr_threshold": lr_threshold,
        "lr_validation_f1": lr_validation_f1,
        "ce_direct_threshold": ce_direct_threshold,
        "ce_direct_validation_f1": ce_direct_validation_f1,
        "ce_legacy_threshold": ce_legacy_threshold,
        "ce_legacy_validation_f1": ce_legacy_validation_f1,
        "alternative_valid_links": alternative_valid_links,
        "runtime_seconds": time.time() - start_time,
        "LR": lr_metrics,
        "CE_direct": ce_direct_metrics,
        "CE_legacy": ce_legacy_metrics,
    }
    atomic_write_json(result_path, payload)

    print(
        f"      CE saved: direct lambda={ce_direct_metrics['lambda']:.4f}, "
        f"psi={ce_direct_metrics['psi']:.4f}; "
        f"legacy lambda={ce_legacy_metrics['lambda']:.4f}, "
        f"psi={ce_legacy_metrics['psi']:.4f}; "
        f"alternative valid links={alternative_valid_links}"
    )

    del ce_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return payload


# =============================================================================
# RESULT COLLECTION AND TABLES
# =============================================================================
def collect_baseline_results() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(CELL_DIR.glob("baseline_*_seed*.json")):
        cell = read_json(path)
        for side in ("string", "bi-encoder"):
            for scorer in ("MEC", "LR"):
                metrics = cell["results"][side][scorer]
                rows.append(
                    {
                        "dataset": cell["dataset"],
                        "seed": int(cell["seed"]),
                        "side": side,
                        "scorer": scorer,
                        "lambda": metrics["lambda"],
                        "psi": metrics["psi"],
                        "declared": metrics["declared"],
                        "correct": metrics["correct"],
                        "false_links": metrics["false_links"],
                        "missing_matches": metrics["missing_matches"],
                        "n_true_records": metrics["n_true_records"],
                        "threshold": (
                            cell["results"][side]["lr_threshold"]
                            if scorer == "LR"
                            else np.nan
                        ),
                        "runtime_seconds": cell["runtime_seconds"],
                    }
                )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            ["dataset", "seed", "side", "scorer"]
        ).reset_index(drop=True)
    return frame


def collect_ft_results() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(CELL_DIR.glob("ft_*_seed*.json")):
        cell = read_json(path)
        for scorer in ("MEC", "LR"):
            metrics = cell[scorer]
            rows.append(
                {
                    "dataset": cell["dataset"],
                    "seed": int(cell["seed"]),
                    "encoder": "fine-tuned",
                    "scorer": scorer,
                    "lambda": metrics["lambda"],
                    "psi": metrics["psi"],
                    "declared": metrics["declared"],
                    "correct": metrics["correct"],
                    "false_links": metrics["false_links"],
                    "missing_matches": metrics["missing_matches"],
                    "n_true_records": metrics["n_true_records"],
                    "threshold": (
                        cell["lr_threshold"]
                        if scorer == "LR"
                        else np.nan
                    ),
                    "runtime_seconds": cell["runtime_seconds"],
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            ["dataset", "seed", "scorer"]
        ).reset_index(drop=True)
    return frame


def collect_ce_results() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(CELL_DIR.glob("ce_*_seed*.json")):
        cell = read_json(path)
        for score_path, label in (
            ("CE_direct", "CE_direct"),
            ("CE_legacy", "CE_legacy"),
            ("LR", "LR_rebuilt"),
        ):
            metrics = cell[score_path]
            rows.append(
                {
                    "dataset": cell["dataset"],
                    "seed": int(cell["seed"]),
                    "model": label,
                    "lambda": metrics["lambda"],
                    "psi": metrics["psi"],
                    "declared": metrics["declared"],
                    "correct": metrics["correct"],
                    "false_links": metrics["false_links"],
                    "missing_matches": metrics["missing_matches"],
                    "n_true_records": metrics["n_true_records"],
                    "threshold": (
                        cell["ce_direct_threshold"]
                        if label == "CE_direct"
                        else (
                            cell["ce_legacy_threshold"]
                            if label == "CE_legacy"
                            else cell["lr_threshold"]
                        )
                    ),
                    "alternative_valid_links": (
                        cell["alternative_valid_links"]
                        if label.startswith("CE")
                        else np.nan
                    ),
                    "runtime_seconds": cell["runtime_seconds"],
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            ["dataset", "seed", "model"]
        ).reset_index(drop=True)
    return frame


def collect_predictions() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    boolean_columns = [
        "has_true_match",
        "lr_declared",
        "ce_top_is_retained",
        "ce_top_is_any_valid",
        "ce_direct_declared",
        "ce_legacy_declared",
    ]

    for path in sorted(PRED_PART_DIR.glob("*_seed*.csv")):
        part = pd.read_csv(path, dtype={"ida": str})
        for column in boolean_columns:
            if column in part.columns and part[column].dtype != bool:
                part[column] = (
                    part[column]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .map({"true": True, "false": False})
                )
                if part[column].isna().any():
                    raise ValueError(
                        f"Could not parse boolean column {column} in {path}"
                    )
        parts.append(part)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(
        ["dataset", "seed", "record_index"]
    ).reset_index(drop=True)


def aggregate_metrics(
    frame: pd.DataFrame,
    group_columns: list[str],
    seeds: list[int],
    aggregation_label: str,
) -> pd.DataFrame:
    if frame.empty or "seed" not in frame.columns:
        return pd.DataFrame()
    subset = frame[frame["seed"].isin(seeds)].copy()
    if subset.empty:
        return pd.DataFrame()

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
            declared_mean=("declared", "mean"),
            correct_mean=("correct", "mean"),
            false_links_mean=("false_links", "mean"),
            missing_matches_mean=("missing_matches", "mean"),
            n_true_records_mean=("n_true_records", "mean"),
        )
        .reset_index()
    )
    aggregated["lambda_se"] = (
        aggregated["lambda_sd"] / np.sqrt(aggregated["n_splits"])
    )
    aggregated["psi_se"] = (
        aggregated["psi_sd"] / np.sqrt(aggregated["n_splits"])
    )
    aggregated.insert(0, "aggregation", aggregation_label)
    return aggregated


def build_table5(
    ft_perseed: pd.DataFrame,
    baseline_perseed: pd.DataFrame,
) -> pd.DataFrame:
    pretrained = baseline_perseed[
        (baseline_perseed["side"] == "bi-encoder")
        & (baseline_perseed["scorer"].isin(["MEC", "LR"]))
    ].copy()

    if not pretrained.empty:
        pretrained = pretrained[
            [
                "dataset",
                "seed",
                "scorer",
                "lambda",
                "psi",
                "declared",
                "correct",
                "false_links",
                "missing_matches",
                "n_true_records",
            ]
        ]
        pretrained["encoder"] = "pre-trained"

    parts: list[pd.DataFrame] = []
    if not pretrained.empty:
        parts.append(pretrained)
    if not ft_perseed.empty:
        parts.append(
            ft_perseed[
                [
                    "dataset",
                    "seed",
                    "scorer",
                    "lambda",
                    "psi",
                    "declared",
                    "correct",
                    "false_links",
                    "missing_matches",
                    "n_true_records",
                    "encoder",
                ]
            ]
        )

    if not parts:
        return pd.DataFrame(
            columns=[
                "dataset", "seed", "scorer", "lambda", "psi",
                "declared", "correct", "false_links",
                "missing_matches", "n_true_records", "encoder",
            ]
        )

    return pd.concat(parts, ignore_index=True).sort_values(
        ["dataset", "encoder", "scorer", "seed"]
    ).reset_index(drop=True)


def build_table6(
    ce_perseed: pd.DataFrame,
    baseline_perseed: pd.DataFrame,
) -> pd.DataFrame:
    two_step = baseline_perseed[
        baseline_perseed["side"].isin(["string", "bi-encoder"])
        & baseline_perseed["scorer"].isin(["MEC", "LR"])
    ].copy()

    if not two_step.empty:
        two_step["model"] = (
            two_step["side"].astype(str)
            + " / "
            + two_step["scorer"].astype(str)
        )
        two_step = two_step[
            [
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
            ]
        ]

    ce_direct = ce_perseed[
        ce_perseed.get("model", pd.Series(dtype=str)) == "CE_direct"
    ].copy() if not ce_perseed.empty else pd.DataFrame()

    if not ce_direct.empty:
        ce_direct["model"] = "cross-encoder"
        ce_direct = ce_direct[
            [
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
            ]
        ]

    parts = [frame for frame in (two_step, ce_direct) if not frame.empty]
    if not parts:
        return pd.DataFrame(
            columns=[
                "dataset", "seed", "model", "lambda", "psi",
                "declared", "correct", "false_links",
                "missing_matches", "n_true_records",
            ]
        )

    return pd.concat(parts, ignore_index=True).sort_values(
        ["dataset", "model", "seed"]
    ).reset_index(drop=True)



def quartile_metric_row(
    subset: pd.DataFrame,
    prefix: str,
) -> dict[str, float | int]:
    declared = int(subset[f"{prefix}_declared"].sum())
    correct = int((subset[f"{prefix}_category"] == "correct").sum())
    false_links = int(
        (subset[f"{prefix}_category"] == "false_link").sum()
    )
    missing_matches = int(
        (subset[f"{prefix}_category"] == "missing_match").sum()
    )
    n_true_records = int(subset["has_true_match"].sum())

    return {
        "declared": declared,
        "correct": correct,
        "false_links": false_links,
        "missing_matches": missing_matches,
        "n_true_records": n_true_records,
        "lambda": false_links / declared if declared else 0.0,
        "psi": (
            missing_matches / n_true_records
            if n_true_records
            else 0.0
        ),
    }


def build_table7_perseed(
    predictions: pd.DataFrame,
    ce_prefix: str,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (dataset_name, seed), subset in predictions.groupby(
        ["dataset", "seed"],
        sort=True,
    ):
        subset = subset.copy()
        subset["quartile"] = pd.qcut(
            subset["lr_gap"],
            4,
            labels=["Q1", "Q2", "Q3", "Q4"],
            duplicates="drop",
        )

        for quartile in ("Q1", "Q2", "Q3", "Q4"):
            quartile_subset = subset[subset["quartile"] == quartile]
            for model_name, prefix in (
                ("two-step LR", "lr"),
                ("cross-encoder", ce_prefix),
            ):
                metrics = quartile_metric_row(quartile_subset, prefix)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "seed": int(seed),
                        "quartile": quartile,
                        "model": model_name,
                        "n_records": len(quartile_subset),
                        **metrics,
                    }
                )

    return pd.DataFrame(rows).sort_values(
        ["dataset", "quartile", "model", "seed"]
    ).reset_index(drop=True)


def save_aggregates(
    frame: pd.DataFrame,
    group_columns: list[str],
    stem: str,
) -> None:
    primary = aggregate_metrics(
        frame,
        group_columns,
        PRIMARY_SEEDS,
        "primary_3_seeds",
    )
    mc10 = aggregate_metrics(
        frame,
        group_columns,
        ALL_SEEDS,
        "all_10_seeds",
    )

    atomic_write_csv(
        primary,
        RESULTS_DIR / f"{stem}_primary3.csv",
    )
    atomic_write_csv(
        mc10,
        RESULTS_DIR / f"{stem}_mc10.csv",
    )


def print_compact_table(
    frame: pd.DataFrame,
    title: str,
    label_columns: list[str],
) -> None:
    if frame.empty:
        print(f"\n{title}: no completed cells yet.")
        return

    print("\n" + "=" * 118)
    print(title)
    print("=" * 118)

    display_columns = (
        label_columns
        + [
            "n_splits",
            "lambda_mean",
            "lambda_sd",
            "lambda_se",
            "psi_mean",
            "psi_sd",
            "psi_se",
        ]
    )
    available = [
        column for column in display_columns if column in frame.columns
    ]
    print(frame[available].round(6).to_string(index=False))


def aggregate_all() -> None:
    baseline_perseed = collect_baseline_results()
    ft_perseed = collect_ft_results()
    ce_perseed = collect_ce_results()
    predictions = collect_predictions()

    atomic_write_csv(
        baseline_perseed,
        RESULTS_DIR / "analysis17_baseline_k50_perseed.csv",
    )
    atomic_write_csv(
        ft_perseed,
        RESULTS_DIR / "analysis17_ft_biencoder_perseed.csv",
    )
    atomic_write_csv(
        ce_perseed,
        RESULTS_DIR / "analysis17_ce_perseed.csv",
    )
    atomic_write_csv(
        predictions,
        RESULTS_DIR / "analysis17_predictions.csv",
    )

    table5 = build_table5(ft_perseed, baseline_perseed)
    table6 = build_table6(ce_perseed, baseline_perseed)
    table7_legacy = build_table7_perseed(
        predictions,
        "ce_legacy",
    )
    table7_direct = build_table7_perseed(
        predictions,
        "ce_direct",
    )

    atomic_write_csv(
        table5,
        RESULTS_DIR / "analysis17_table5_perseed.csv",
    )
    atomic_write_csv(
        table6,
        RESULTS_DIR / "analysis17_table6_perseed.csv",
    )
    atomic_write_csv(
        table7_legacy,
        RESULTS_DIR / "analysis17_table7_perseed.csv",
    )
    atomic_write_csv(
        table7_direct,
        RESULTS_DIR / "analysis17_table7_direct_perseed.csv",
    )

    save_aggregates(
        table5,
        ["dataset", "encoder", "scorer"],
        "analysis17_table5",
    )
    save_aggregates(
        table6,
        ["dataset", "model"],
        "analysis17_table6",
    )
    save_aggregates(
        table7_legacy,
        ["dataset", "quartile", "model"],
        "analysis17_table7",
    )
    save_aggregates(
        table7_direct,
        ["dataset", "quartile", "model"],
        "analysis17_table7_direct",
    )

    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project_root": str(PROJECT_ROOT),
        "primary_seeds": PRIMARY_SEEDS,
        "diagnostic_seeds": DIAGNOSTIC_SEEDS,
        "complete_baseline_cells": int(
            len(
                baseline_perseed[["dataset", "seed"]].drop_duplicates()
            )
            if not baseline_perseed.empty
            else 0
        ),
        "complete_ft_cells": int(
            len(ft_perseed[["dataset", "seed"]].drop_duplicates())
            if not ft_perseed.empty
            else 0
        ),
        "complete_ce_cells": int(
            len(
                ce_perseed[ce_perseed["model"] == "CE_direct"][
                    ["dataset", "seed"]
                ].drop_duplicates()
            )
            if not ce_perseed.empty
            else 0
        ),
        "prediction_rows": int(len(predictions)),
        "models": {
            "bi_encoder": BI_ENCODER_NAME,
            "cross_encoder": CE_MODEL_NAME,
        },
        "ce_recipe": {
            "epochs": CE_EPOCHS,
            "warmup_fraction": CE_WARMUP_FRAC,
            "learning_rate": CE_LR,
            "batch_size": CE_BATCH,
            "max_length": CE_MAX_LEN,
        },
        "ft_biencoder_recipe": {
            "epochs": FT_EPOCHS,
            "warmup_fraction": FT_WARMUP_FRAC,
            "learning_rate": FT_LR,
            "batch_size": FT_BATCH,
            "loss": "ContrastiveLoss",
        },
        "table7_score_paths": {
            "main": (
                "analysis11 legacy: sigmoid(CrossEncoder.predict output)"
            ),
            "diagnostic": "analysis8 direct CrossEncoder.predict output",
        },
    }
    atomic_write_json(
        RESULTS_DIR / "analysis17_manifest.json",
        manifest,
    )

    for stem, labels in (
        (
            "analysis17_table5",
            ["dataset", "encoder", "scorer"],
        ),
        (
            "analysis17_table6",
            ["dataset", "model"],
        ),
        (
            "analysis17_table7",
            ["dataset", "quartile", "model"],
        ),
    ):
        path = RESULTS_DIR / f"{stem}_primary3.csv"
        frame = pd.read_csv(path) if path.exists() else pd.DataFrame()
        print_compact_table(
            frame,
            f"{stem}: primary seeds",
            labels,
        )

    print("\nSaved final outputs under:")
    print(RESULTS_DIR)


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    args = parse_args()

    datasets = [
        value.strip().upper()
        for value in args.datasets.split(",")
        if value.strip()
    ]
    invalid = set(datasets).difference(CONFIGS)
    if invalid:
        raise ValueError(f"Unknown datasets: {sorted(invalid)}")

    seeds = PRIMARY_SEEDS if args.primary_only else ALL_SEEDS

    print("=" * 78)
    print("Analysis 17: standalone corrected, resumable Section 5 rerun")
    print("=" * 78)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Phase: {args.phase}")
    print(f"Datasets: {datasets}")
    print(f"Seeds: {seeds}")
    print("Analysis 16 dependency: none")
    print("Internal baseline work: corrected k=50 two-step cells only")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Visible CUDA devices: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        print(f"Visible device 0: {torch.cuda.get_device_name(0)}")

    if args.phase != "aggregate" and torch.cuda.device_count() > 1:
        raise RuntimeError(
            "More than one GPU is visible. Pin the run explicitly, e.g. "
            "CUDA_VISIBLE_DEVICES=0 python scripts/analysis17_section5.py"
        )

    print("\nConfirmed CE recipe from analysis8_crossencoder.py")
    print(
        f"  {CE_MODEL_NAME}; epochs={CE_EPOCHS}; "
        f"warmup={int(CE_WARMUP_FRAC * 100)}%; lr={CE_LR}; "
        f"batch={CE_BATCH}; max_length={CE_MAX_LEN}"
    )
    print("Confirmed Table 7 lineage")
    print(
        "  Current draft values exactly match analysis11_hardness.py "
        "(MiniLM), not run6_fixed_ce.py (RoBERTa)."
    )
    print(
        "  Main Table 7 preserves analysis11's extra-sigmoid score path; "
        "a direct-score diagnostic table is also saved."
    )

    if args.phase == "aggregate":
        aggregate_all()
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = SentenceTransformer(BI_ENCODER_NAME, device=device)

    for dataset_name in datasets:
        dataset = load_dataset(dataset_name)
        prepare_baseline(dataset, base_model)

        for seed in seeds:
            print(
                f"\n{'=' * 78}\n"
                f"{dataset_name} seed={seed} "
                f"({'primary' if seed in PRIMARY_SEEDS else 'diagnostic'})"
                f"\n{'=' * 78}"
            )

            if args.phase in ("all", "baseline"):
                run_baseline_cell(dataset, seed)

            if args.phase in ("all", "ft"):
                run_ft_cell(dataset, seed, device)

            if args.phase in ("all", "ce"):
                run_ce_cell(dataset, seed, device)

            aggregate_all()

        del dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    aggregate_all()
    print("\nAnalysis 17 complete for the requested cells.")


if __name__ == "__main__":
    main()
