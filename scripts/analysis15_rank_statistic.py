"""
analysis15_rank_statistic.py
============================

Corrected Stage 0 for the entity-resolution paper.

Purpose
-------
Recompute the title-only bi-encoder blocking statistic using:

  * the COMPLETE raw B files (duplicates remain in the candidate universe);
  * the complete many-match ground truth;
  * one row in the denominator per A record with at least one true match;
  * success when ANY valid true B record appears within the top-k block.

This script does not train or score an entity-resolution model.

Why this rerun is necessary
---------------------------
The June analysis scripts used clean_DBLP_B.csv and clean_ECOM_B.csv.
Those files were produced by deleting secondary matched B records. Zhang's
revision instead requires B to remain intact. Therefore the blocking floor
must be recomputed before rerunning downstream tables.

Server layout
-------------
Expected files:

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
  python scripts/analysis15_rank_statistic.py

Outputs
-------
  results/analysis15_rank_statistic_DBLP.csv
  results/analysis15_rank_statistic_ECOM.csv
  results/analysis15_rank_mmr_by_k.csv
  results/analysis15_rank_summary.csv
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util


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
BATCH_SIZE = 256
A_CHUNK_SIZE = 256

K_GRID = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200]

CONFIGS: dict[str, dict[str, Any]] = {
    "DBLP": {
        "a": DATA_DIR / "dblp" / "DBLP1.csv",
        "b": DATA_DIR / "dblp" / "Scholar.csv",
        "mapping": DATA_DIR / "dblp" / "DBLP-Scholar_perfectMapping.csv",
        "id_a": "idDBLP",
        "id_b": "idScholar",
        "retrieval_source": "title",
        "expected": {"A": 2616, "B": 64263, "pairs": 5347},
    },
    "ECOM": {
        "a": DATA_DIR / "abt_buy" / "Abt.csv",
        "b": DATA_DIR / "abt_buy" / "Buy.csv",
        "mapping": DATA_DIR / "abt_buy" / "abt_buy_perfectMapping.csv",
        "id_a": "idAbt",
        "id_b": "idBuy",
        "retrieval_source": "name",
        "expected": {"A": 1081, "B": 1092, "pairs": 1097},
    },
}


# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------
def clean_header(value: object) -> str:
    """Remove BOM artefacts, surrounding quotes, and whitespace."""
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
    Read a required CSV exactly as the February preprocessing did.

    The June paper-facing experiments were built from files created with
    ``encoding="latin1"``. Latin-1 is therefore forced here rather than
    accepting UTF-8 when a raw file happens to decode successfully.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    frame = pd.read_csv(path, dtype=str, encoding="latin-1")
    frame.columns = [clean_header(c) for c in frame.columns]
    return frame


def find_id_column(frame: pd.DataFrame, path: Path) -> str:
    """Find the record-id column after header cleaning."""
    matches = [c for c in frame.columns if c.lower() == "id"]
    if len(matches) != 1:
        raise KeyError(
            f"Expected exactly one 'id' column in {path}; found {list(frame.columns)}"
        )
    return matches[0]


def normalise_text(series: pd.Series) -> pd.Series:
    """Mirror the text normalisation used by the June cleaned files."""
    return series.fillna("").astype(str).str.lower().str.strip()


def load_dataset(
    name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, set[str]]]:
    """
    Load raw A, raw B and the complete mapping.

    Returns
    -------
    df_a, df_b:
        Indexed by string id and containing a normalised ``retrieval_text``.
    mapping:
        Complete raw mapping after removing rows whose ids do not exist.
    truth:
        id_A -> set of all reachable valid id_B records.
    """
    cfg = CONFIGS[name]

    df_a = read_csv_robust(cfg["a"])
    df_b = read_csv_robust(cfg["b"])
    mapping = read_csv_robust(cfg["mapping"])

    id_a_file = find_id_column(df_a, cfg["a"])
    id_b_file = find_id_column(df_b, cfg["b"])

    for frame, id_column in ((df_a, id_a_file), (df_b, id_b_file)):
        frame[id_column] = frame[id_column].fillna("").astype(str).str.strip()

    map_a = cfg["id_a"]
    map_b = cfg["id_b"]
    missing_map_columns = {map_a, map_b}.difference(mapping.columns)
    if missing_map_columns:
        raise KeyError(
            f"{cfg['mapping']} is missing mapping columns: "
            f"{sorted(missing_map_columns)}"
        )

    mapping[map_a] = mapping[map_a].fillna("").astype(str).str.strip()
    mapping[map_b] = mapping[map_b].fillna("").astype(str).str.strip()

    source_field = cfg["retrieval_source"]
    for frame, path in ((df_a, cfg["a"]), (df_b, cfg["b"])):
        if source_field not in frame.columns:
            raise KeyError(
                f"Retrieval field {source_field!r} not found in {path}; "
                f"columns are {list(frame.columns)}"
            )
        frame["retrieval_text"] = normalise_text(frame[source_field])

    df_a = df_a.set_index(id_a_file, drop=True)
    df_b = df_b.set_index(id_b_file, drop=True)
    df_a.index.name = "id"
    df_b.index.name = "id"

    if not df_a.index.is_unique:
        raise ValueError(f"A ids are not unique in {cfg['a']}")
    if not df_b.index.is_unique:
        raise ValueError(f"B ids are not unique in {cfg['b']}")

    a_ids = set(df_a.index)
    b_ids = set(df_b.index)

    reachable_mask = mapping[map_a].isin(a_ids) & mapping[map_b].isin(b_ids)
    unreachable = mapping.loc[~reachable_mask]

    truth: dict[str, set[str]] = {}
    for ida, idb in mapping.loc[reachable_mask, [map_a, map_b]].itertuples(
        index=False, name=None
    ):
        truth.setdefault(ida, set()).add(idb)

    expected = cfg["expected"]
    observed = {"A": len(df_a), "B": len(df_b), "pairs": len(mapping)}
    if observed != expected:
        raise AssertionError(
            f"{name} raw counts do not match the validated snapshot. "
            f"Observed {observed}; expected {expected}."
        )

    print(f"\n{name} input validation")
    print("-" * 72)
    print(
        f"|A|={len(df_a):,}  |B|={len(df_b):,}  "
        f"raw pairs={len(mapping):,}"
    )
    print(
        f"A records with truth={len(truth):,}  "
        f"reachable pairs={sum(len(v) for v in truth.values()):,}"
    )
    print(
        f"A with >1 reachable match="
        f"{sum(len(v) > 1 for v in truth.values()):,}  "
        f"max matches/A={max(map(len, truth.values()))}"
    )
    print(f"unreachable raw mapping rows={len(unreachable):,}")

    if not unreachable.empty:
        example = unreachable[[map_a, map_b]].head(10).to_dict("records")
        raise AssertionError(
            f"{name} contains unreachable raw mapping rows. Examples: {example}"
        )

    return df_a, df_b, mapping, truth


# ---------------------------------------------------------------------
# CORRECTED BLOCKING RANK
# ---------------------------------------------------------------------
def compute_best_true_ranks(
    name: str,
    model: SentenceTransformer,
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, object]]:
    """
    Compute one rank per A record using the best-ranked valid true match.

    For A record i with truth set T(i), let s* be the maximum cosine
    similarity over b in T(i). The record-level rank is

        1 + number of B records with similarity strictly greater than s*.

    This is the earliest rank at which ANY valid match appears.
    """
    df_a, df_b, _, truth = load_dataset(name)

    print(f"\n{name} embedding")
    print("-" * 72)

    emb_a = model.encode(
        df_a["retrieval_text"].tolist(),
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
    )
    emb_b = model.encode(
        df_b["retrieval_text"].tolist(),
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
    )

    a_ids = list(df_a.index)
    b_ids = list(df_b.index)
    a_pos = {record_id: position for position, record_id in enumerate(a_ids)}
    b_pos = {record_id: position for position, record_id in enumerate(b_ids)}

    rows: list[dict[str, object]] = []
    truth_a_ids = [ida for ida in a_ids if ida in truth]

    print(f"\n{name} rank computation")
    print("-" * 72)
    for start in range(0, len(truth_a_ids), A_CHUNK_SIZE):
        chunk_ids = truth_a_ids[start : start + A_CHUNK_SIZE]
        chunk_positions = [a_pos[ida] for ida in chunk_ids]

        # Embeddings are normalised, so dot product equals cosine similarity.
        similarities = util.dot_score(emb_a[chunk_positions], emb_b)

        for row_number, ida in enumerate(chunk_ids):
            valid_ids = truth[ida]
            valid_columns = [b_pos[idb] for idb in valid_ids]
            valid_scores = similarities[row_number, valid_columns]

            best_local = int(torch.argmax(valid_scores).item())
            best_true_id = b_ids[valid_columns[best_local]]
            best_true_score = float(valid_scores[best_local].item())

            rank = (
                int((similarities[row_number] > best_true_score).sum().item()) + 1
            )

            rows.append(
                {
                    "dataset": name,
                    "ida": ida,
                    "n_valid_matches": len(valid_ids),
                    "best_true_idb": best_true_id,
                    "best_true_similarity": best_true_score,
                    "best_valid_rank": rank,
                }
            )

        completed = min(start + A_CHUNK_SIZE, len(truth_a_ids))
        print(
            f"processed {completed:,}/{len(truth_a_ids):,} matched A records",
            end="\r",
            flush=True,
        )
    print()

    rank_df = pd.DataFrame(rows)
    if len(rank_df) != len(truth):
        raise AssertionError(
            f"{name}: produced {len(rank_df)} ranks for {len(truth)} truth records"
        )

    ranks = rank_df["best_valid_rank"].to_numpy(dtype=int)
    n_records = len(ranks)

    print(f"\n{name} corrected blocking floor")
    print("-" * 72)
    print(f"{'k':>6} {'captured records':>18} {'record-level MMR':>18}")

    mmr_rows: list[dict[str, object]] = []
    for k in K_GRID:
        captured = int((ranks <= k).sum())
        missing = n_records - captured
        mmr = missing / n_records

        mmr_rows.append(
            {
                "dataset": name,
                "k": k,
                "captured_records": captured,
                "missing_records": missing,
                "n_true_records": n_records,
                "blocking_mmr": mmr,
            }
        )
        print(f"{k:>6} {captured:>18,} {mmr:>18.6f}")

    summary: dict[str, object] = {
        "dataset": name,
        "n_true_records": n_records,
        "n_truth_pairs": int(rank_df["n_valid_matches"].sum()),
        "n_multi_match_records": int((rank_df["n_valid_matches"] > 1).sum()),
        "mean_best_rank": float(ranks.mean()),
        "median_best_rank": float(np.median(ranks)),
        "p90_best_rank": float(np.percentile(ranks, 90)),
        "p95_best_rank": float(np.percentile(ranks, 95)),
        "p99_best_rank": float(np.percentile(ranks, 99)),
        "p999_best_rank": float(np.percentile(ranks, 99.9)),
        "max_best_rank": int(ranks.max()),
        "fraction_rank1": float((ranks == 1).mean()),
    }

    print("\nRank summary")
    print(
        f"rank 1={summary['fraction_rank1']:.6f}  "
        f"median={summary['median_best_rank']:.1f}  "
        f"p95={summary['p95_best_rank']:.1f}  "
        f"p99={summary['p99_best_rank']:.1f}  "
        f"max={summary['max_best_rank']}"
    )

    return rank_df, mmr_rows, summary


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("Analysis 15: corrected many-match blocking statistic")
    print("=" * 72)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Bi-encoder: {BI_ENCODER_NAME}")
    print("Raw-file decoding: latin-1 (matches February/June pipeline)")
    print("Text preparation: fill missing, lowercase, strip whitespace")
    print("Truth rule: earliest rank of ANY valid reachable B match")
    print(f"CUDA available: {torch.cuda.is_available()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    model = SentenceTransformer(BI_ENCODER_NAME, device=device)

    all_mmr_rows: list[dict[str, object]] = []
    all_summaries: list[dict[str, object]] = []

    for name in ("DBLP", "ECOM"):
        rank_df, mmr_rows, summary = compute_best_true_ranks(name, model)

        rank_path = RESULTS_DIR / f"analysis15_rank_statistic_{name}.csv"
        rank_df.to_csv(rank_path, index=False)

        all_mmr_rows.extend(mmr_rows)
        all_summaries.append(summary)

    mmr_path = RESULTS_DIR / "analysis15_rank_mmr_by_k.csv"
    summary_path = RESULTS_DIR / "analysis15_rank_summary.csv"

    pd.DataFrame(all_mmr_rows).to_csv(mmr_path, index=False)
    pd.DataFrame(all_summaries).to_csv(summary_path, index=False)

    print("\n" + "=" * 72)
    print("Saved outputs")
    print("-" * 72)
    for name in ("DBLP", "ECOM"):
        print(RESULTS_DIR / f"analysis15_rank_statistic_{name}.csv")
    print(mmr_path)
    print(summary_path)
    print("=" * 72)


if __name__ == "__main__":
    main()