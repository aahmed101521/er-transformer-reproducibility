#!/usr/bin/env python3
"""
Recover the four missing Monte Carlo standard errors for the PRE-TRAINED
cross-encoder row of manuscript Table 7.

This script intentionally does NOT fine-tune the cross-encoder. It evaluates
an untouched:

    cross-encoder/ms-marco-MiniLM-L-6-v2

on the original primary splits (42, 43, 44), using the same corrected k=50
candidate universe, preprocessing, pair construction, and validation-threshold
protocol used in the Section 5 rerun.

The bundle includes the exact raw benchmark snapshots and the saved k=50
all-MiniLM-L6-v2 candidate blocks from the August backup. Reusing those blocks
avoids recomputing the bi-encoder retrieval and isolates this run to the
pre-trained cross-encoder only.

Final manuscript quantities per seed:

    FLR_s = 1 - correct_s / declared_s
    MMR_s = 1 - correct_s / n_true_records_s

Across seeds 42, 43, 44:

    MCSE = sample_SD(seed-level metric) / sqrt(3)

Outputs
-------
results/analysis17_pretrained_ce_perseed.csv
results/analysis17_pretrained_ce_primary3.csv
results/analysis17_pretrained_ce_predictions.csv.gz
results/analysis17_pretrained_ce_manifest.json

Expected manuscript means (three-decimal check only):
    DBLP-Scholar: FLR .061, MMR .026
    Abt-Buy:      FLR .267, MMR .278

If the rounded means do not match, the script prints a warning rather than
silently accepting the run. This can expose environment/model-version drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_SEEDS = [42, 43, 44]
K = 50
VALIDATION_FRAC = 0.20
THRESHOLDS = np.arange(0.10, 0.90 + 1e-9, 0.05)
PREDICT_BATCH = 64
CE_MAX_LEN = 128
CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BI_ENCODER_NAME = "all-MiniLM-L6-v2"

CONFIGS: dict[str, dict[str, Any]] = {
    "DBLP": {
        "paper_name": "DBLP-Scholar",
        "a": DATA_DIR / "dblp" / "DBLP1.csv",
        "b": DATA_DIR / "dblp" / "Scholar.csv",
        "mapping": DATA_DIR / "dblp" / "DBLP-Scholar_perfectMapping.csv",
        "cache": CACHE_DIR / "DBLP_top50.json",
        "id_a": "idDBLP",
        "id_b": "idScholar",
        "rename": {},
        "fields": ["title", "authors", "venue", "year"],
        "expected": {"A": 2616, "B": 64263, "pairs": 5347},
        "expected_paper_mean": {"flr": 0.061, "mmr": 0.026},
    },
    "ECOM": {
        "paper_name": "Abt-Buy",
        "a": DATA_DIR / "abt_buy" / "Abt.csv",
        "b": DATA_DIR / "abt_buy" / "Buy.csv",
        "mapping": DATA_DIR / "abt_buy" / "abt_buy_perfectMapping.csv",
        "cache": CACHE_DIR / "ECOM_top50.json",
        "id_a": "idAbt",
        "id_b": "idBuy",
        "rename": {"name": "title", "description": "desc"},
        "fields": ["title", "desc"],
        "expected": {"A": 1081, "B": 1092, "pairs": 1097},
        "expected_paper_mean": {"flr": 0.267, "mmr": 0.278},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Untouched pre-trained CE rerun for missing Table 7 MCSEs."
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cuda, cpu, cuda:0, etc. Default: auto.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=PREDICT_BATCH,
        help=f"Cross-encoder inference batch size. Default: {PREDICT_BATCH}.",
    )
    parser.add_argument(
        "--model",
        default=CE_MODEL_NAME,
        help="HF model name or local model directory.",
    )
    parser.add_argument(
        "--datasets",
        default="DBLP,ECOM",
        help="Comma-separated subset of DBLP,ECOM.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate data/cache snapshots without loading the cross-encoder.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    matches = [c for c in frame.columns if c.lower() == "id"]
    if len(matches) != 1:
        raise KeyError(
            f"Expected exactly one id column in {path}; columns={list(frame.columns)}"
        )
    return matches[0]


def normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower().str.strip()


def june_csv_roundtrip(frame: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """Reproduce the February-write / June-read pandas CSV state."""
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
    """Exact CE serialisation: fields joined with ' [SEP] '; NaN -> blank."""
    parts: list[str] = []
    for field in fields:
        value = row.get(field, "")
        if value is None or str(value) == "nan":
            parts.append("")
        else:
            parts.append(str(value))
    return " [SEP] ".join(parts)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            raise KeyError(f"{path} missing fields {sorted(missing)}")
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
            f"{name} snapshot mismatch: observed={observed}, expected={cfg['expected']}"
        )

    if not df_a.index.is_unique or not df_b.index.is_unique:
        raise ValueError(f"{name}: A and B ids must be unique")

    a_ids = set(df_a.index)
    b_ids = set(df_b.index)
    reachable = mapping[map_a].isin(a_ids) & mapping[map_b].isin(b_ids)
    if not bool(reachable.all()):
        bad = mapping.loc[~reachable, [map_a, map_b]].head(10)
        raise AssertionError(
            f"{name}: unreachable mapping rows found: {bad.to_dict('records')}"
        )

    # Full many-match truth; first-listed valid match retained as the sole
    # training positive, exactly as in the corrected Section 5 protocol.
    truth: dict[str, set[str]] = {}
    retained: dict[str, str] = {}
    for ida, idb in mapping[[map_a, map_b]].itertuples(index=False, name=None):
        truth.setdefault(ida, set()).add(idb)
        retained.setdefault(ida, idb)

    with cfg["cache"].open("r", encoding="utf-8") as handle:
        cache = json.load(handle)
    if cache.get("dataset") != name:
        raise AssertionError(f"Wrong cache dataset in {cfg['cache']}")
    if cache.get("bi_encoder") != BI_ENCODER_NAME:
        raise AssertionError(f"Wrong bi-encoder in {cfg['cache']}")
    if int(cache.get("k", -1)) != K:
        raise AssertionError(f"Wrong k in {cfg['cache']}")
    if int(cache.get("n_a", -1)) != len(df_a) or int(cache.get("n_b", -1)) != len(df_b):
        raise AssertionError(f"Stale candidate cache: {cfg['cache']}")
    top50 = cache.get("top50", {})
    if set(top50) != set(df_a.index):
        raise AssertionError(f"Candidate cache A-id coverage mismatch: {cfg['cache']}")
    if not all(len(v) == K for v in top50.values()):
        raise AssertionError(f"Candidate cache contains non-{K} blocks")

    print(
        f"{cfg['paper_name']}: |A|={len(df_a):,}, |B|={len(df_b):,}, "
        f"mapping pairs={len(mapping):,}, A with truth={len(truth):,}, "
        f"multi-match A={sum(len(v) > 1 for v in truth.values()):,}"
    )
    print(
        f"  top-{K} cache OK: {cfg['cache'].name}; "
        f"bi-encoder={cache['bi_encoder']}"
    )

    return {
        "name": name,
        "paper_name": cfg["paper_name"],
        "cfg": cfg,
        "df_a": df_a,
        "df_b": df_b,
        "a_ids": list(df_a.index),
        "b_ids": list(df_b.index),
        "truth": truth,
        "retained": retained,
        "top50": top50,
    }


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
        valid = truth[ida]
        for idb in top50[ida]:
            if idb not in valid:
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
        # Strict > means the first threshold wins a tie, matching Analysis 17.
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def unique_pairs_for_dataset(dataset: dict[str, Any]) -> list[tuple[str, str]]:
    """All top-50 pairs plus any retained positive outside the top-50 block."""
    pairs: set[tuple[str, str]] = set()
    for ida in dataset["a_ids"]:
        pairs.update((ida, idb) for idb in dataset["top50"][ida])
        positive = dataset["retained"].get(ida)
        if positive is not None:
            pairs.add((ida, positive))
    return sorted(pairs)


def score_cache_path(dataset_name: str) -> Path:
    return RESULTS_DIR / f"analysis17_pretrained_ce_scores_{dataset_name}.csv.gz"


def load_or_compute_scores(
    model: Any,
    dataset: dict[str, Any],
    batch_size: int,
) -> dict[tuple[str, str], float]:
    path = score_cache_path(dataset["name"])
    if path.exists():
        frame = pd.read_csv(path, dtype={"ida": str, "idb": str})
        expected_cols = {"ida", "idb", "score"}
        if expected_cols.issubset(frame.columns):
            print(f"  loading cached untouched-CE scores: {path}")
            return {
                (str(row.ida), str(row.idb)): float(row.score)
                for row in frame.itertuples(index=False)
            }
        print(f"  ignoring malformed score cache: {path}")

    pairs = unique_pairs_for_dataset(dataset)
    fields = dataset["cfg"]["fields"]
    print(f"  scoring {len(pairs):,} unique pairs with untouched CE...")
    text_pairs = [
        (
            serialise(dataset["df_a"].loc[ida], fields),
            serialise(dataset["df_b"].loc[idb], fields),
        )
        for ida, idb in pairs
    ]
    scores = np.asarray(
        model.predict(
            text_pairs,
            batch_size=batch_size,
            show_progress_bar=True,
        ),
        dtype=np.float64,
    ).reshape(-1)
    if len(scores) != len(pairs):
        raise AssertionError("Cross-encoder returned unexpected number of scores")

    frame = pd.DataFrame(
        {
            "ida": [p[0] for p in pairs],
            "idb": [p[1] for p in pairs],
            "score": scores,
        }
    )
    frame.to_csv(path, index=False, compression="gzip")
    print(f"  saved score cache: {path}")
    return dict(zip(pairs, scores, strict=True))


def run_seed(
    dataset: dict[str, Any],
    seed: int,
    score_lookup: dict[tuple[str, str], float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
        dataset["top50"],
    )
    labels = [label for _, _, label in train_pairs]
    _, validation_indices = train_test_split(
        list(range(len(train_pairs))),
        test_size=VALIDATION_FRAC,
        random_state=seed,
        stratify=labels,
    )
    validation_only = [train_pairs[i] for i in validation_indices]
    validation_scores = np.asarray(
        [score_lookup[(ida, idb)] for ida, idb, _ in validation_only],
        dtype=np.float64,
    )
    validation_labels = [label for _, _, label in validation_only]
    threshold, validation_f1 = choose_threshold(
        validation_scores,
        validation_labels,
    )

    declared = 0
    correct = 0
    false_links = 0
    missing_matches = 0
    n_true_records = 0
    prediction_rows: list[dict[str, Any]] = []

    for ida in ids_test:
        valid = dataset["truth"].get(ida, set())
        if valid:
            n_true_records += 1
        candidates = dataset["top50"][ida]
        candidate_scores = np.asarray(
            [score_lookup[(ida, idb)] for idb in candidates],
            dtype=np.float64,
        )
        top_index = int(np.argmax(candidate_scores))
        top_idb = candidates[top_index]
        top_score = float(candidate_scores[top_index])
        is_declared = bool(top_score > threshold)
        is_correct = bool(is_declared and top_idb in valid)

        if is_declared:
            declared += 1
            if is_correct:
                correct += 1
            else:
                false_links += 1
        elif valid:
            missing_matches += 1

        prediction_rows.append(
            {
                "dataset": dataset["name"],
                "paper_dataset": dataset["paper_name"],
                "seed": seed,
                "ida": ida,
                "has_true_match": bool(valid),
                "n_valid_matches": len(valid),
                "top1_idb": top_idb,
                "top1_score": top_score,
                "threshold": threshold,
                "declared": is_declared,
                "correct": is_correct,
            }
        )

    if declared != correct + false_links:
        raise AssertionError("declared != correct + false_links")

    flr = 1.0 - (correct / declared) if declared else 0.0
    # Final manuscript/Zhang definition requested for the revision.
    mmr = 1.0 - (correct / n_true_records) if n_true_records else 0.0
    # Historical abstention-only quantity retained for audit only.
    psi_historical = (
        missing_matches / n_true_records if n_true_records else 0.0
    )

    result = {
        "dataset": dataset["name"],
        "paper_dataset": dataset["paper_name"],
        "seed": seed,
        "model": "pre-trained cross-encoder",
        "threshold": threshold,
        "validation_f1": validation_f1,
        "n_train_a": len(ids_train),
        "n_test_a": len(ids_test),
        "n_train_pairs_full": len(train_pairs),
        "n_validation_pairs": len(validation_only),
        "declared": declared,
        "correct": correct,
        "false_links": false_links,
        "missing_matches": missing_matches,
        "n_true_records": n_true_records,
        "FLR": flr,
        "MMR": mmr,
        "psi_historical": psi_historical,
    }
    return result, prediction_rows


def aggregate_primary(perseed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, paper_dataset), group in perseed.groupby(
        ["dataset", "paper_dataset"], sort=False
    ):
        if set(group["seed"].astype(int)) != set(PRIMARY_SEEDS):
            raise AssertionError(f"{dataset}: missing primary seed(s)")
        flr = group["FLR"].astype(float)
        mmr = group["MMR"].astype(float)
        rows.append(
            {
                "dataset": dataset,
                "paper_dataset": paper_dataset,
                "n_splits": len(group),
                "FLR_mean": flr.mean(),
                "FLR_sd": flr.std(ddof=1),
                "FLR_MCSE": flr.std(ddof=1) / math.sqrt(len(group)),
                "MMR_mean": mmr.mean(),
                "MMR_sd": mmr.std(ddof=1),
                "MMR_MCSE": mmr.std(ddof=1) / math.sqrt(len(group)),
            }
        )
    return pd.DataFrame(rows)


def rounded_three(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    args = parse_args()
    requested = [x.strip().upper() for x in args.datasets.split(",") if x.strip()]
    unknown = sorted(set(requested).difference(CONFIGS))
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {unknown}; choose from DBLP,ECOM")

    print("=" * 78)
    print("PRE-TRAINED CROSS-ENCODER TABLE 7 MCSE RECOVERY")
    print("=" * 78)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Torch:  {torch.__version__}")
    print(f"Seeds:  {PRIMARY_SEEDS}")
    print(f"Model:  {args.model}")
    print("Fine-tuning: NONE")
    print("Final MMR: 1 - correct / n_true_records")
    print()

    datasets = {name: load_dataset(name) for name in requested}

    if args.validate_only:
        print("\nValidation-only mode complete. No model was loaded.")
        return

    device = resolve_device(args.device)
    print(f"\nInference device: {device}")
    try:
        from sentence_transformers import CrossEncoder
        import sentence_transformers
    except ImportError as exc:
        raise SystemExit(
            "sentence-transformers is required. Install requirements.txt "
            "or run inside the existing ER-paper environment."
        ) from exc

    print(f"sentence-transformers: {sentence_transformers.__version__}")
    print("Loading untouched cross-encoder (no fit() call anywhere in this script)...")
    model = CrossEncoder(
        args.model,
        max_length=CE_MAX_LEN,
        device=device,
    )

    all_results: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    started = time.time()

    for name in requested:
        dataset = datasets[name]
        print("\n" + "-" * 78)
        print(dataset["paper_name"])
        print("-" * 78)
        score_lookup = load_or_compute_scores(model, dataset, args.batch_size)

        for seed in PRIMARY_SEEDS:
            result, predictions = run_seed(dataset, seed, score_lookup)
            all_results.append(result)
            all_predictions.extend(predictions)
            print(
                f"  seed {seed}: threshold={result['threshold']:.2f}, "
                f"declared={result['declared']}, correct={result['correct']}, "
                f"M={result['n_true_records']}, "
                f"FLR={result['FLR']:.6f}, MMR={result['MMR']:.6f}"
            )

    perseed = pd.DataFrame(all_results).sort_values(["dataset", "seed"])
    perseed_path = RESULTS_DIR / "analysis17_pretrained_ce_perseed.csv"
    perseed.to_csv(perseed_path, index=False)

    predictions = pd.DataFrame(all_predictions).sort_values(
        ["dataset", "seed", "ida"]
    )
    predictions_path = RESULTS_DIR / "analysis17_pretrained_ce_predictions.csv.gz"
    predictions.to_csv(predictions_path, index=False, compression="gzip")

    summary = aggregate_primary(perseed)
    summary_path = RESULTS_DIR / "analysis17_pretrained_ce_primary3.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 78)
    print("FINAL THREE-SEED RESULTS: mean (MCSE)")
    print("=" * 78)
    for row in summary.itertuples(index=False):
        print(
            f"{row.paper_dataset:12s}  "
            f"FLR {row.FLR_mean:.3f} ({row.FLR_MCSE:.4f})   "
            f"MMR {row.MMR_mean:.3f} ({row.MMR_MCSE:.4f})"
        )

    print("\nManuscript-mean reproduction check")
    print("-" * 78)
    checks: dict[str, Any] = {}
    for row in summary.itertuples(index=False):
        expected = CONFIGS[row.dataset]["expected_paper_mean"]
        flr_ok = rounded_three(row.FLR_mean) == rounded_three(expected["flr"])
        mmr_ok = rounded_three(row.MMR_mean) == rounded_three(expected["mmr"])
        status = "PASS" if flr_ok and mmr_ok else "WARNING"
        checks[row.dataset] = {
            "status": status,
            "observed_flr_3dp": rounded_three(row.FLR_mean),
            "expected_flr_3dp": rounded_three(expected["flr"]),
            "observed_mmr_3dp": rounded_three(row.MMR_mean),
            "expected_mmr_3dp": rounded_three(expected["mmr"]),
        }
        print(
            f"{status:7s} {row.paper_dataset:12s}: "
            f"observed FLR/MMR={row.FLR_mean:.3f}/{row.MMR_mean:.3f}; "
            f"paper={expected['flr']:.3f}/{expected['mmr']:.3f}"
        )

    manifest = {
        "purpose": "Recover missing MCSEs for pre-trained CE row of manuscript Table 7",
        "model": args.model,
        "fine_tuning": False,
        "max_length": CE_MAX_LEN,
        "candidate_block": f"top-{K} from {BI_ENCODER_NAME}",
        "primary_seeds": PRIMARY_SEEDS,
        "a_split": "50/50 train/test using sklearn train_test_split(random_state=seed)",
        "threshold_calibration": {
            "validation_fraction_of_training_pairs": VALIDATION_FRAC,
            "stratified": True,
            "grid": [float(x) for x in THRESHOLDS],
            "criterion": "pair-level validation F1; strict score > threshold",
        },
        "truth": "many-match id_A -> set(all valid id_B)",
        "training_positive": "first-listed valid id_B only",
        "training_negatives": "top-50 candidates excluding every known valid id_B",
        "correctness": "declared top-1 is correct if it is any valid id_B",
        "FLR": "1 - correct/declared",
        "MMR": "1 - correct/n_true_records",
        "runtime_seconds": time.time() - started,
        "versions": {
            "python": sys.version,
            "torch": torch.__version__,
            "sentence_transformers": sentence_transformers.__version__,
        },
        "data_sha256": {
            str(CONFIGS[name][key].relative_to(ROOT)): file_sha256(CONFIGS[name][key])
            for name in requested
            for key in ("a", "b", "mapping", "cache")
        },
        "reproduction_checks": checks,
    }
    manifest_path = RESULTS_DIR / "analysis17_pretrained_ce_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(f"  {perseed_path}")
    print(f"  {summary_path}")
    print(f"  {predictions_path}")
    print(f"  {manifest_path}")
    print("\nPlease paste the FINAL THREE-SEED RESULTS block and the reproduction check back into ChatGPT.")


if __name__ == "__main__":
    main()
