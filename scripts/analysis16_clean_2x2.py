"""
analysis16_clean_2x2.py
=======================

Corrected rerun of the Section 4 representation x scorer comparison.

This script preserves the current paper's modelling protocol:
  - representations: per-field Jaro-Winkler and per-field bi-encoder cosine;
  - scorers: KDE likelihood-ratio MEC and L2 logistic regression;
  - LR threshold selected on an internal 80/20 training split using F1 only
    as a calibration objective;
  - source A split 50/50;
  - title-only all-MiniLM-L6-v2 blocking;
  - block regimes: forced-1:1, k=5, k=50.

It changes only the items required by the counting correction:
  1. the complete raw B files are retained;
  2. the complete many-match ground truth is used at test time;
  3. a prediction to ANY valid B match is correct;
  4. false links and missing matches are mutually exclusive;
  5. the MMR denominator is the number of A records with >=1 true match.

June string-feature fidelity:
  - raw files are decoded as Latin-1, matching the February preprocessing;
  - text fields are lowercased and stripped;
  - the February-write / June-read CSV round-trip is reproduced;
  - pandas NaN values therefore reach the June JW path;
  - the June JW implementation converted NaN to the literal string "nan".
    This legacy behaviour is preserved explicitly here. It is not endorsed
    as ideal missing-value handling; it is held fixed to isolate the
    counting and candidate-universe changes.

Training remains one-to-one:
  - the first-listed reachable B match is the sole positive for each A;
  - alternative valid matches are NOT added as positives;
  - valid alternative matches are excluded from the negative class.

Monte Carlo design:
  - seeds 42, 43, 44 remain the PRIMARY three splits used for the paper table;
  - seeds 45--51 are diagnostic replications;
  - all ten splits are saved so split-to-split SD and SE are known.

Expected server layout
----------------------
  ~/er_paper/data/dblp/DBLP1.csv
  ~/er_paper/data/dblp/Scholar.csv
  ~/er_paper/data/dblp/DBLP-Scholar_perfectMapping.csv
  ~/er_paper/data/abt_buy/Abt.csv
  ~/er_paper/data/abt_buy/Buy.csv
  ~/er_paper/data/abt_buy/abt_buy_perfectMapping.csv

Run
---
  cd ~/er_paper
  source .venv/bin/activate
  CUDA_VISIBLE_DEVICES=0 python scripts/analysis16_clean_2x2.py

Outputs
-------
  results/analysis16_clean_2x2_perseed.csv
  results/analysis16_clean_2x2_primary3.csv
  results/analysis16_clean_2x2_mc10.csv
  results/analysis16_clean_2x2_diagnostics.csv
"""

from __future__ import annotations

import os
import random
import time
from io import StringIO
from pathlib import Path
from typing import Any, Callable

import jellyfish
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity


# ---------------------------------------------------------------------
# PATHS AND CONFIGURATION
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(
    os.environ.get("ER_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).expanduser()
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BI_ENCODER_NAME = "all-MiniLM-L6-v2"
PRIMARY_SEEDS = [42, 43, 44]
DIAGNOSTIC_SEEDS = [45, 46, 47, 48, 49, 50, 51]
ALL_SEEDS = PRIMARY_SEEDS + DIAGNOSTIC_SEEDS

COLUMNS = ["forced-1:1", "k=5", "k=50"]
K_VALUES = {"forced-1:1": 1, "k=5": 5, "k=50": 50}
MAX_RETRIEVAL_K = 50

BATCH_SIZE = 256
RETRIEVAL_CHUNK_SIZE = 256
KDE_BANDWIDTH = 0.05
LR_MAX_ITER = 500
LR_VAL_FRAC = 0.20
LR_THRESHOLDS = np.arange(0.10, 0.90 + 1e-9, 0.05)

# Corrected re-tally of the saved June Analysis 11 predictions.
# These are pooled over seeds 42, 43 and 44 under the old candidate universe.
# Analysis 16 uses the complete B files, so exact equality is not expected;
# the comparison is a faithfulness diagnostic, not a numerical assertion.
AUDIT14_LR_K50_REFERENCE = {
    "DBLP": {"lambda": 0.042772, "psi": 0.121790},
    "ECOM": {"lambda": 0.239220, "psi": 0.399877},
}

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


def timestamp() -> str:
    return time.strftime("%H:%M:%S")


# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------
def clean_header(value: object) -> str:
    return (
        str(value)
        .replace("\ufeff", "")
        .replace("ï»¿", "")
        .strip()
        .strip('"')
        .strip()
    )


def read_csv_robust(path: Path) -> pd.DataFrame:
    """
    Read raw files exactly as the February preprocessing script did.

    Latin-1 is forced for every file. This matters for Scholar.csv:
    accepting UTF-8 would change the decoded form of thousands of titles
    and would therefore create a new retrieval and feature pipeline.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    frame = pd.read_csv(path, dtype=str, encoding="latin-1")
    frame.columns = [clean_header(c) for c in frame.columns]
    return frame


def find_id_column(frame: pd.DataFrame, path: Path) -> str:
    matches = [c for c in frame.columns if c.lower() == "id"]
    if len(matches) != 1:
        raise KeyError(
            f"Expected exactly one 'id' column in {path}; "
            f"found columns {list(frame.columns)}"
        )
    return matches[0]


def normalise_text(series: pd.Series) -> pd.Series:
    """Mirror the February cleaning before the clean CSVs were written."""
    return series.fillna("").astype(str).str.lower().str.strip()


def emulate_june_csv_roundtrip(
    frame: pd.DataFrame,
    id_column: str,
) -> pd.DataFrame:
    """
    Reproduce the exact in-memory state seen by the June scripts.

    February preprocessing filled missing text with an empty string and
    wrote a CSV. The June scripts read that CSV with pandas' default NA
    parsing, so blank cells became NaN again. Embeddings then explicitly
    used fillna(""); the string-side JW helper did not.

    Preserving this round-trip avoids silently changing the string features
    while restoring the complete raw B file.
    """
    indexed = frame.set_index(id_column, drop=True)
    indexed.index.name = "id"

    buffer = StringIO()
    indexed.to_csv(buffer)
    buffer.seek(0)

    roundtripped = pd.read_csv(buffer, dtype=str).set_index("id")
    roundtripped.index = roundtripped.index.astype(str)
    roundtripped.index.name = "id"
    return roundtripped


def load_dataset(name: str) -> dict[str, Any]:
    cfg = CONFIGS[name]

    df_a = read_csv_robust(cfg["a"])
    df_b = read_csv_robust(cfg["b"])
    mapping = read_csv_robust(cfg["mapping"])

    id_a_file = find_id_column(df_a, cfg["a"])
    id_b_file = find_id_column(df_b, cfg["b"])

    df_a = df_a.rename(columns=cfg["rename"])
    df_b = df_b.rename(columns=cfg["rename"])

    for frame, path in ((df_a, cfg["a"]), (df_b, cfg["b"])):
        missing_fields = set(cfg["fields"]).difference(frame.columns)
        if missing_fields:
            raise KeyError(
                f"{path} is missing fields {sorted(missing_fields)}; "
                f"columns are {list(frame.columns)}"
            )
        for field in cfg["fields"]:
            frame[field] = normalise_text(frame[field])

    df_a[id_a_file] = df_a[id_a_file].fillna("").astype(str).str.strip()
    df_b[id_b_file] = df_b[id_b_file].fillna("").astype(str).str.strip()

    # Match the actual February-write / June-read data state.
    df_a = emulate_june_csv_roundtrip(df_a, id_a_file)
    df_b = emulate_june_csv_roundtrip(df_b, id_b_file)

    if not df_a.index.is_unique:
        raise ValueError(f"A ids are not unique in {cfg['a']}")
    if not df_b.index.is_unique:
        raise ValueError(f"B ids are not unique in {cfg['b']}")

    map_a = cfg["id_a"]
    map_b = cfg["id_b"]
    missing_mapping_fields = {map_a, map_b}.difference(mapping.columns)
    if missing_mapping_fields:
        raise KeyError(
            f"{cfg['mapping']} is missing {sorted(missing_mapping_fields)}"
        )

    mapping[map_a] = mapping[map_a].fillna("").astype(str).str.strip()
    mapping[map_b] = mapping[map_b].fillna("").astype(str).str.strip()

    expected = cfg["expected"]
    observed = {"A": len(df_a), "B": len(df_b), "pairs": len(mapping)}
    if observed != expected:
        raise AssertionError(
            f"{name}: observed raw counts {observed}; expected {expected}"
        )

    a_ids = set(df_a.index)
    b_ids = set(df_b.index)
    reachable_mask = mapping[map_a].isin(a_ids) & mapping[map_b].isin(b_ids)
    unreachable = mapping.loc[~reachable_mask, [map_a, map_b]]
    if not unreachable.empty:
        raise AssertionError(
            f"{name}: {len(unreachable)} mapping rows are unreachable. "
            f"Examples: {unreachable.head(10).to_dict('records')}"
        )

    truth: dict[str, set[str]] = {}
    retained: dict[str, str] = {}
    for ida, idb in mapping[[map_a, map_b]].itertuples(index=False, name=None):
        truth.setdefault(ida, set()).add(idb)
        retained.setdefault(ida, idb)

    print(f"\n{name} input validation")
    print("-" * 72)
    print(
        f"|A|={len(df_a):,}  |B|={len(df_b):,}  "
        f"raw pairs={len(mapping):,}"
    )
    print(
        f"A records with truth={len(truth):,}  "
        f"A with >1 match={sum(len(v) > 1 for v in truth.values()):,}  "
        f"max matches/A={max(map(len, truth.values()))}"
    )
    print("Training positives: first-listed reachable match, one per matched A")
    print("Test correctness: any member of the complete truth set")
    print("June JW missing-value semantics: pandas NaN -> literal 'nan'")
    for field in cfg["fields"]:
        print(
            f"  missing after June-style reload, {field}: "
            f"A={int(df_a[field].isna().sum()):,}, "
            f"B={int(df_b[field].isna().sum()):,}"
        )

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


# ---------------------------------------------------------------------
# EMBEDDINGS, RETRIEVAL AND FEATURE CACHE
# ---------------------------------------------------------------------
def prepare_embeddings_and_candidates(
    dataset: dict[str, Any],
    model: SentenceTransformer,
) -> None:
    cfg = dataset["cfg"]
    df_a = dataset["df_a"]
    df_b = dataset["df_b"]
    fields = cfg["fields"]

    print(f"\n{dataset['name']} per-field embeddings [{timestamp()}]")
    print("-" * 72)

    emb_a: dict[str, torch.Tensor] = {}
    emb_b: dict[str, torch.Tensor] = {}

    for field in fields:
        print(f"embedding field: {field}")
        emb_a[field] = model.encode(
            df_a[field].fillna("").tolist(),
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
        )
        emb_b[field] = model.encode(
            df_b[field].fillna("").tolist(),
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
        )

    a_ids = dataset["a_ids"]
    b_ids = dataset["b_ids"]
    a_pos = {record_id: pos for pos, record_id in enumerate(a_ids)}
    b_pos = {record_id: pos for pos, record_id in enumerate(b_ids)}

    retrieval_field = cfg["retrieval_field"]
    top50: dict[str, list[str]] = {}

    print(f"\n{dataset['name']} top-50 title blocking [{timestamp()}]")
    print("-" * 72)

    for start in range(0, len(a_ids), RETRIEVAL_CHUNK_SIZE):
        stop = min(start + RETRIEVAL_CHUNK_SIZE, len(a_ids))
        scores = util.dot_score(
            emb_a[retrieval_field][start:stop],
            emb_b[retrieval_field],
        )
        top_indices = torch.topk(
            scores, k=MAX_RETRIEVAL_K, dim=1
        ).indices.cpu().tolist()

        for offset, indices in enumerate(top_indices):
            ida = a_ids[start + offset]
            top50[ida] = [b_ids[index] for index in indices]

        print(
            f"retrieved {stop:,}/{len(a_ids):,} A records",
            end="\r",
            flush=True,
        )
    print()

    # The maximum truth-set size is 20, so top-50 must contain a real
    # non-match for every record. Assert this before forced-1:1 is built.
    no_nonmatch = [
        ida
        for ida, candidates in top50.items()
        if not any(idb not in dataset["truth"].get(ida, set()) for idb in candidates)
    ]
    if no_nonmatch:
        raise AssertionError(
            f"{dataset['name']}: no top-50 non-match found for "
            f"{len(no_nonmatch)} A records; examples {no_nonmatch[:10]}"
        )

    dataset["emb_a"] = emb_a
    dataset["emb_b"] = emb_b
    dataset["a_pos"] = a_pos
    dataset["b_pos"] = b_pos
    dataset["top50"] = top50

    print(f"\n{dataset['name']} caching pair features [{timestamp()}]")
    print("-" * 72)

    feature_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    pair_count = 0

    for record_number, ida in enumerate(a_ids, start=1):
        required_b = list(top50[ida])
        retained = dataset["retained"].get(ida)
        if retained is not None and retained not in required_b:
            required_b.append(retained)

        row_a = df_a.loc[ida]
        ia = a_pos[ida]

        for idb in required_b:
            key = (ida, idb)
            if key in feature_cache:
                continue

            row_b = df_b.loc[idb]
            ib = b_pos[idb]

            string_features = np.asarray(
                [jw(row_a[field], row_b[field]) for field in fields],
                dtype=np.float64,
            )
            embedding_features = np.asarray(
                [
                    float(torch.dot(emb_a[field][ia], emb_b[field][ib]).item())
                    for field in fields
                ],
                dtype=np.float64,
            )
            feature_cache[key] = (string_features, embedding_features)
            pair_count += 1

        if record_number % 250 == 0 or record_number == len(a_ids):
            print(
                f"cached {record_number:,}/{len(a_ids):,} A records "
                f"({pair_count:,} unique pairs)",
                end="\r",
                flush=True,
            )
    print()

    dataset["feature_cache"] = feature_cache


def jw(value_a: object, value_b: object) -> float:
    """
    Reproduce the June Jaro-Winkler path exactly.

    ``analysis1_clean_2x2.py`` documented NaN as mapping to zero, but its
    actual implementation only checked ``is None`` and then called ``str``.
    A pandas ``np.nan`` therefore became the literal text ``"nan"`` before
    Jaro-Winkler was evaluated. We preserve that executed behaviour
    deliberately so this rerun does not alter the string representation.

    Consequences:
      - NaN versus NaN is JW("nan", "nan") = 1.0;
      - NaN versus text is JW("nan", text);
      - a Python None still maps to 0.0, as in the June function.
    """
    if value_a is None or value_b is None:
        return 0.0

    text_a = "nan" if pd.isna(value_a) else str(value_a)
    text_b = "nan" if pd.isna(value_b) else str(value_b)

    if not text_a or not text_b:
        return 0.0

    return float(jellyfish.jaro_winkler_similarity(text_a, text_b))


# ---------------------------------------------------------------------
# TRAINING PAIRS AND TEST CANDIDATES
# ---------------------------------------------------------------------
def top_nonmatch(dataset: dict[str, Any], ida: str) -> str:
    true_set = dataset["truth"].get(ida, set())
    for idb in dataset["top50"][ida]:
        if idb not in true_set:
            return idb
    raise RuntimeError(f"No non-match available for {dataset['name']}:{ida}")


def build_train_pairs(
    dataset: dict[str, Any],
    ids: list[str],
    column: str,
) -> list[tuple[str, str, int]]:
    """
    One positive per matched A record.

    Alternative true matches are neither extra positives nor negatives.
    This retains one-to-one supervision while preventing known true pairs
    from entering the negative class after B is restored.
    """
    pairs: list[tuple[str, str, int]] = []
    k = K_VALUES[column]

    for ida in ids:
        positive = dataset["retained"].get(ida)
        if positive is None:
            continue

        true_set = dataset["truth"][ida]
        pairs.append((ida, positive, 1))

        if column == "forced-1:1":
            pairs.append((ida, top_nonmatch(dataset, ida), 0))
            continue

        candidates = dataset["top50"][ida][:k]
        for idb in candidates:
            if idb not in true_set:
                pairs.append((ida, idb, 0))

    return pairs


def build_test_structure(
    dataset: dict[str, Any],
    ids: list[str],
    column: str,
) -> list[tuple[str, set[str], list[str]]]:
    structure: list[tuple[str, set[str], list[str]]] = []
    k = K_VALUES[column]

    for ida in ids:
        true_set = dataset["truth"].get(ida, set())

        if column == "forced-1:1":
            if true_set:
                candidates = [
                    dataset["retained"][ida],
                    top_nonmatch(dataset, ida),
                ]
            else:
                candidates = [dataset["top50"][ida][0]]
        else:
            candidates = dataset["top50"][ida][:k]

        structure.append((ida, true_set, candidates))

    return structure


def feature_matrix(
    dataset: dict[str, Any],
    pairs: list[tuple[str, str]],
    side: str,
) -> np.ndarray:
    side_index = 0 if side == "string" else 1
    return np.vstack(
        [dataset["feature_cache"][(ida, idb)][side_index] for ida, idb in pairs]
    )


# ---------------------------------------------------------------------
# SCORERS
# ---------------------------------------------------------------------
class MECScorer:
    """Per-field KDE likelihood ratio, using the current paper's rule."""

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

            if len(match_values) == 0 or len(nonmatch_values) == 0:
                raise ValueError("MEC requires both match and non-match examples")

            self.match_kdes.append(
                KernelDensity(bandwidth=self.bandwidth).fit(match_values)
            )
            self.nonmatch_kdes.append(
                KernelDensity(bandwidth=self.bandwidth).fit(nonmatch_values)
            )

    def log_ratio(self, x: np.ndarray) -> np.ndarray:
        total = np.zeros(x.shape[0], dtype=np.float64)
        for field_index, (match_kde, nonmatch_kde) in enumerate(
            zip(self.match_kdes, self.nonmatch_kdes)
        ):
            values = x[:, field_index].reshape(-1, 1)
            total += (
                match_kde.score_samples(values)
                - nonmatch_kde.score_samples(values)
            )
        return total


# ---------------------------------------------------------------------
# CORRECTED EVALUATION
# ---------------------------------------------------------------------
def evaluate(
    test_structure: list[tuple[str, set[str], list[str]]],
    score_fn: Callable[[str, list[str]], np.ndarray],
    link_decision_fn: Callable[[float], bool],
) -> dict[str, float | int]:
    """
    Apply Zhang's mutually exclusive counting rule.

    For a record with a declared link:
      - linked B in true_set  -> correct;
      - otherwise             -> false link.

    For a record without a declared link:
      - true_set non-empty    -> missing match;
      - true_set empty        -> uncounted true negative.

    Thus a wrong declared link on a matched record is false, not also missing.
    """
    declared = 0
    correct = 0
    false_links = 0
    missing_matches = 0
    true_records = 0
    unmatched_records = 0
    abstained_unmatched = 0

    for ida, true_set, candidates in test_structure:
        if true_set:
            true_records += 1
        else:
            unmatched_records += 1

        scores = np.asarray(score_fn(ida, candidates), dtype=float)
        if scores.ndim != 1 or len(scores) != len(candidates):
            raise ValueError(
                f"Bad score shape for {ida}: {scores.shape}; "
                f"{len(candidates)} candidates"
            )

        top_index = int(np.argmax(scores))
        top_score = float(scores[top_index])
        top_b = candidates[top_index]

        if link_decision_fn(top_score):
            declared += 1
            if top_b in true_set:
                correct += 1
            else:
                false_links += 1
        elif true_set:
            missing_matches += 1
        else:
            abstained_unmatched += 1

    if declared != correct + false_links:
        raise AssertionError("Declared links do not partition into correct + false")
    if true_records < correct + missing_matches:
        raise AssertionError("Impossible corrected record counts")

    lam = false_links / declared if declared > 0 else 0.0
    psi = missing_matches / true_records if true_records > 0 else 0.0

    return {
        "lambda": lam,
        "psi": psi,
        "declared": declared,
        "correct": correct,
        "false_links": false_links,
        "missing_matches": missing_matches,
        "n_true_records": true_records,
        "n_unmatched_records": unmatched_records,
        "abstained_unmatched": abstained_unmatched,
    }


# ---------------------------------------------------------------------
# ONE CONFIGURATION
# ---------------------------------------------------------------------
def run_side(
    dataset: dict[str, Any],
    train_pairs: list[tuple[str, str, int]],
    test_structure: list[tuple[str, set[str], list[str]]],
    side: str,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    x_train = feature_matrix(
        dataset,
        [(ida, idb) for ida, idb, _ in train_pairs],
        side,
    )
    y_train = np.asarray([label for _, _, label in train_pairs], dtype=int)

    if set(np.unique(y_train)) != {0, 1}:
        raise ValueError(
            f"{dataset['name']} seed={seed} side={side}: "
            "training data does not contain both classes"
        )

    def candidate_features(ida: str, candidates: list[str]) -> np.ndarray:
        return feature_matrix(
            dataset,
            [(ida, idb) for idb in candidates],
            side,
        )

    # MEC
    mec = MECScorer()
    mec.fit(x_train, y_train)

    def mec_score(ida: str, candidates: list[str]) -> np.ndarray:
        return mec.log_ratio(candidate_features(ida, candidates))

    mec_result = evaluate(
        test_structure,
        mec_score,
        lambda score: score > 0.0,
    )

    # Logistic regression and internal threshold calibration
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train,
        y_train,
        test_size=LR_VAL_FRAC,
        random_state=seed,
        stratify=y_train,
    )
    lr = LogisticRegression(
        penalty="l2",
        max_iter=LR_MAX_ITER,
        random_state=seed,
    )
    lr.fit(x_fit, y_fit)
    validation_probabilities = lr.predict_proba(x_val)[:, 1]

    best_threshold = 0.50
    best_f1 = -1.0
    for threshold in LR_THRESHOLDS:
        current_f1 = f1_score(
            y_val,
            (validation_probabilities > threshold).astype(int),
            zero_division=0,
        )
        if current_f1 > best_f1:
            best_f1 = float(current_f1)
            best_threshold = float(threshold)

    def lr_score(ida: str, candidates: list[str]) -> np.ndarray:
        return lr.predict_proba(candidate_features(ida, candidates))[:, 1]

    lr_result = evaluate(
        test_structure,
        lr_score,
        lambda score: score > best_threshold,
    )
    lr_result["threshold"] = best_threshold

    return {"MEC": mec_result, "LR": lr_result}


# ---------------------------------------------------------------------
# AGGREGATION
# ---------------------------------------------------------------------
def aggregate_results(
    per_seed: pd.DataFrame,
    seeds: list[int],
    label: str,
) -> pd.DataFrame:
    subset = per_seed[per_seed["seed"].isin(seeds)].copy()
    group_columns = ["dataset", "column", "side", "scorer"]

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
    aggregated.insert(0, "aggregation", label)

    numeric_columns = aggregated.select_dtypes(include=[np.number]).columns
    aggregated[numeric_columns] = aggregated[numeric_columns].round(6)
    return aggregated


def print_aggregate_table(frame: pd.DataFrame, title: str) -> None:
    print("\n" + "=" * 108)
    print(title)
    print("=" * 108)

    for dataset_name in frame["dataset"].unique():
        print(f"\n{dataset_name}")
        print(
            f"{'column':<12} {'side':<12} {'scorer':<7}"
            f"{'lambda mean':>13} {'lambda SD':>12} {'lambda SE':>12}"
            f"{'psi mean':>12} {'psi SD':>11} {'psi SE':>11}"
        )
        subset = frame[frame["dataset"] == dataset_name]
        for row in subset.itertuples(index=False):
            print(
                f"{row.column:<12} {row.side:<12} {row.scorer:<7}"
                f"{row.lambda_mean:>13.6f} {row.lambda_sd:>12.6f}"
                f" {row.lambda_se:>12.6f}"
                f"{row.psi_mean:>12.6f} {row.psi_sd:>11.6f}"
                f" {row.psi_se:>11.6f}"
            )


def print_analysis14_crosscheck(primary: pd.DataFrame) -> None:
    """
    Compare primary k=50 bi-encoder LR means with the corrected re-tally
    of the saved June Analysis 11 predictions.

    The audit values are pooled rates under the old deduplicated B universe.
    The rerun values are means of per-seed rates under complete B, so small
    differences are expected. Large differences are a signal to stop and
    inspect preprocessing, candidate construction, or threshold calibration.
    """
    print("\n" + "=" * 88)
    print("FAITHFULNESS CROSS-CHECK: Analysis 16 LR k=50 vs Analysis 14 audit")
    print("=" * 88)
    print(
        f"{'dataset':<10}"
        f"{'audit lambda':>14}{'rerun lambda':>15}{'delta':>11}"
        f"{'audit psi':>13}{'rerun psi':>12}{'delta':>11}"
    )

    for dataset_name, reference in AUDIT14_LR_K50_REFERENCE.items():
        selected = primary[
            (primary["dataset"] == dataset_name)
            & (primary["column"] == "k=50")
            & (primary["side"] == "bi-encoder")
            & (primary["scorer"] == "LR")
        ]

        if len(selected) != 1:
            print(
                f"{dataset_name:<10} cross-check unavailable: "
                f"found {len(selected)} matching rows"
            )
            continue

        row = selected.iloc[0]
        lambda_delta = float(row["lambda_mean"]) - reference["lambda"]
        psi_delta = float(row["psi_mean"]) - reference["psi"]

        print(
            f"{dataset_name:<10}"
            f"{reference['lambda']:>14.6f}{float(row['lambda_mean']):>15.6f}"
            f"{lambda_delta:>+11.6f}"
            f"{reference['psi']:>13.6f}{float(row['psi_mean']):>12.6f}"
            f"{psi_delta:>+11.6f}"
        )

    print(
        "\nInterpretation: exact equality is not expected because Analysis 16 "
        "restores the complete B files and reports a mean of per-seed rates, "
        "whereas Analysis 14 re-tallied pooled saved predictions from the "
        "deduplicated June candidate universe."
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("Analysis 16: corrected Section 4 representation x scorer comparison")
    print("=" * 72)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Bi-encoder: {BI_ENCODER_NAME}")
    print("String fields: Jaro-Winkler on every field, including authors")
    print(f"MEC KDE bandwidth: {KDE_BANDWIDTH}")
    print("Raw-file decoding: latin-1")
    print("Text preparation: lowercase + strip; June CSV round-trip preserved")
    print("Legacy JW missing handling: pandas NaN is scored as literal 'nan'")
    print(f"Primary seeds: {PRIMARY_SEEDS}")
    print(f"Diagnostic seeds: {DIAGNOSTIC_SEEDS}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    model = SentenceTransformer(BI_ENCODER_NAME, device=device)
    rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for dataset_name in ("DBLP", "ECOM"):
        dataset = load_dataset(dataset_name)
        prepare_embeddings_and_candidates(dataset, model)

        for seed in ALL_SEEDS:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            ids_train, ids_test = train_test_split(
                dataset["a_ids"],
                test_size=0.50,
                random_state=seed,
            )

            print(
                f"\n{dataset_name}, seed={seed} "
                f"({'primary' if seed in PRIMARY_SEEDS else 'diagnostic'}) "
                f"[{timestamp()}]"
            )

            for column in COLUMNS:
                train_pairs = build_train_pairs(dataset, ids_train, column)
                test_structure = build_test_structure(dataset, ids_test, column)

                n_positive = sum(label == 1 for _, _, label in train_pairs)
                n_negative = sum(label == 0 for _, _, label in train_pairs)
                alternatives_in_test_blocks = sum(
                    max(
                        0,
                        sum(idb in true_set for idb in candidates) - 1,
                    )
                    for _, true_set, candidates in test_structure
                    if true_set
                )

                diagnostic_rows.append(
                    {
                        "dataset": dataset_name,
                        "seed": seed,
                        "seed_role": (
                            "primary"
                            if seed in PRIMARY_SEEDS
                            else "diagnostic"
                        ),
                        "column": column,
                        "n_train_a": len(ids_train),
                        "n_test_a": len(ids_test),
                        "n_train_positive_pairs": n_positive,
                        "n_train_negative_pairs": n_negative,
                        "alternative_valid_entries_in_test_blocks": (
                            alternatives_in_test_blocks
                        ),
                    }
                )

                for side in ("string", "bi-encoder"):
                    print(
                        f"  column={column:<12} side={side:<11}",
                        end="  ",
                        flush=True,
                    )

                    results = run_side(
                        dataset,
                        train_pairs,
                        test_structure,
                        side,
                        seed,
                    )

                    print(
                        f"MEC lambda={results['MEC']['lambda']:.4f} "
                        f"psi={results['MEC']['psi']:.4f}; "
                        f"LR lambda={results['LR']['lambda']:.4f} "
                        f"psi={results['LR']['psi']:.4f} "
                        f"(t={results['LR']['threshold']:.2f})"
                    )

                    for scorer in ("MEC", "LR"):
                        result = results[scorer]
                        rows.append(
                            {
                                "dataset": dataset_name,
                                "seed": seed,
                                "seed_role": (
                                    "primary"
                                    if seed in PRIMARY_SEEDS
                                    else "diagnostic"
                                ),
                                "column": column,
                                "side": side,
                                "scorer": scorer,
                                "lambda": result["lambda"],
                                "psi": result["psi"],
                                "declared": result["declared"],
                                "correct": result["correct"],
                                "false_links": result["false_links"],
                                "missing_matches": result["missing_matches"],
                                "n_true_records": result["n_true_records"],
                                "n_unmatched_records": (
                                    result["n_unmatched_records"]
                                ),
                                "abstained_unmatched": (
                                    result["abstained_unmatched"]
                                ),
                                "lr_threshold": (
                                    result.get("threshold", np.nan)
                                    if scorer == "LR"
                                    else np.nan
                                ),
                            }
                        )

        # Release the large DBLP cache before loading the next dataset.
        del dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    per_seed = pd.DataFrame(rows)
    diagnostics = pd.DataFrame(diagnostic_rows)

    primary = aggregate_results(
        per_seed,
        PRIMARY_SEEDS,
        "primary_3_seeds",
    )
    mc10 = aggregate_results(
        per_seed,
        ALL_SEEDS,
        "all_10_seeds",
    )

    per_seed_path = RESULTS_DIR / "analysis16_clean_2x2_perseed.csv"
    primary_path = RESULTS_DIR / "analysis16_clean_2x2_primary3.csv"
    mc10_path = RESULTS_DIR / "analysis16_clean_2x2_mc10.csv"
    diagnostics_path = RESULTS_DIR / "analysis16_clean_2x2_diagnostics.csv"

    per_seed.to_csv(per_seed_path, index=False)
    primary.to_csv(primary_path, index=False)
    mc10.to_csv(mc10_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)

    print_aggregate_table(
        primary,
        "PRIMARY TABLE SUMMARY: seeds 42, 43, 44",
    )
    print_aggregate_table(
        mc10,
        "MONTE CARLO DIAGNOSTIC: all ten 50/50 splits",
    )
    print_analysis14_crosscheck(primary)

    print("\n" + "=" * 72)
    print("Saved outputs")
    print("-" * 72)
    print(per_seed_path)
    print(primary_path)
    print(mc10_path)
    print(diagnostics_path)
    print("=" * 72)


if __name__ == "__main__":
    main()