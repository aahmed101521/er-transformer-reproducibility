#!/usr/bin/env python3
"""
Recover the manuscript Table 8 discrimination-gap results for the primary
three seeds (42, 43, 44), then compute Monte Carlo standard errors using the
final record-level definitions:

    FLR_s = 1 - correct_s / declared_s
    MMR_s = 1 - correct_s / n_true_records_s
    MCSE  = sample_SD(seed values) / sqrt(3)

This script deliberately reconstructs only the missing Table 8 lineage.
It does NOT retrain the cross-encoder. Historical cross-encoder test decisions
for seeds 42--44 are bundled from the August backup and are merged with newly
reconstructed fine-tuned-bi-encoder logistic-regression discrimination gaps.

The fine-tuned bi-encoder recipe is frozen to the historical Analysis 17:
  - all-MiniLM-L6-v2
  - ContrastiveLoss
  - one epoch
  - 10% warmup
  - learning rate 2e-5
  - batch size 16
  - 50/50 A-level train/test split
  - pretrained top-50 candidate blocks for the final Table 8 comparison
  - LR calibration on an 80/20 stratified pair-level validation split
  - thresholds 0.10, 0.15, ..., 0.90 selected by validation F1
  - strict score > threshold declaration rule

For auditability, the script first reproduces the historical whole-test
fine-tuned LR results (both reblocked and fixed-pretrained-block versions),
and then reproduces the August-30 quartile means before reporting the new
mean(MCSE) values under the final MMR definition.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from io import StringIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
HIST_CE_DIR = ROOT / "historical_ce_predictions"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models" / "ft_biencoder"
PARTS_DIR = RESULTS_DIR / "parts"

for p in (RESULTS_DIR, MODELS_DIR, PARTS_DIR):
    p.mkdir(parents=True, exist_ok=True)

PRIMARY_SEEDS = [42, 43, 44]
BI_ENCODER_NAME = "all-MiniLM-L6-v2"
K = 50
LR_MAX_ITER = 500
LR_VAL_FRAC = 0.20
THRESHOLDS = np.arange(0.10, 0.90 + 1e-9, 0.05)
FT_EPOCHS = 1
FT_WARMUP_FRAC = 0.10
FT_LR = 2e-5
FT_BATCH = 16
EMBED_BATCH = 256
RETRIEVAL_CHUNK = 256
FEATURE_CHUNK = 8192

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
        "retrieval_field": "title",
        "expected": {"A": 2616, "B": 64263, "pairs": 5347},
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
        "retrieval_field": "title",
        "expected": {"A": 1081, "B": 1092, "pairs": 1097},
    },
}

# Exact whole-test checkpoints from the historical Analysis 17 / Aug-30
# reconstruction. These are used as guardrails: if they do not reproduce,
# the script refuses to present the new MCSEs as recovered historical results.
EXPECTED_WHOLE: dict[tuple[str, int], dict[str, float | int]] = {
    ("DBLP", 42): {
        "reblocked_lambda": 0.01929530201342282,
        "reblocked_psi": 0.0256198347107438,
        "reblocked_threshold": 0.50,
        "reblocked_pairs": 58427,
        "fixed_lambda": 0.015228,
        "fixed_psi": 0.032231,
        "fixed_threshold": 0.55,
        "fixed_pairs": 58465,
    },
    ("DBLP", 43): {
        "reblocked_lambda": 0.03297609233305853,
        "reblocked_psi": 0.014214046822742474,
        "reblocked_threshold": 0.40,
        "reblocked_pairs": 59102,
        "fixed_lambda": 0.026029,
        "fixed_psi": 0.025920,
        "fixed_threshold": 0.50,
        "fixed_pairs": 59133,
    },
    ("DBLP", 44): {
        "reblocked_lambda": 0.026359143327841845,
        "reblocked_psi": 0.01728395061728395,
        "reblocked_threshold": 0.50,
        "reblocked_pairs": 58203,
        "fixed_lambda": 0.011036,
        "fixed_psi": 0.033745,
        "fixed_threshold": 0.60,
        "fixed_pairs": 58235,
    },
    ("ECOM", 42): {
        "reblocked_lambda": 0.048426150121065374,
        "reblocked_psi": 0.2365988909426987,
        "reblocked_threshold": 0.40,
        "reblocked_pairs": 26996,
        "fixed_lambda": 0.050360,
        "fixed_psi": 0.229205,
        "fixed_threshold": 0.40,
        "fixed_pairs": 26997,
    },
    ("ECOM", 43): {
        "reblocked_lambda": 0.05314009661835749,
        "reblocked_psi": 0.23475046210720887,
        "reblocked_threshold": 0.35,
        "reblocked_pairs": 26995,
        "fixed_lambda": 0.059770,
        "fixed_psi": 0.195933,
        "fixed_threshold": 0.30,
        "fixed_pairs": 26997,
    },
    ("ECOM", 44): {
        "reblocked_lambda": 0.056338028169014086,
        "reblocked_psi": 0.21256931608133087,
        "reblocked_threshold": 0.35,
        "reblocked_pairs": 26992,
        "fixed_lambda": 0.066964,
        "fixed_psi": 0.171904,
        "fixed_threshold": 0.30,
        "fixed_pairs": 26994,
    },
}

# Historical Aug-30 quartile means (lambda and the then-stored psi =
# missing_matches / n_true_records). These are another independent lineage check.
EXPECTED_QUARTILE_OLD: dict[tuple[str, str, str], tuple[float, float]] = {
    ("DBLP", "Q1", "Cross-encoder"): (0.033214, 0.018663),
    ("DBLP", "Q1", "Logistic regression"): (0.037495, 0.033247),
    ("DBLP", "Q2", "Cross-encoder"): (0.034252, 0.011940),
    ("DBLP", "Q2", "Logistic regression"): (0.031464, 0.050051),
    ("DBLP", "Q3", "Cross-encoder"): (0.008250, 0.010294),
    ("DBLP", "Q3", "Logistic regression"): (0.006414, 0.041181),
    ("DBLP", "Q4", "Cross-encoder"): (0.0, 0.0),
    ("DBLP", "Q4", "Logistic regression"): (0.0, 0.0),
    ("ECOM", "Q1", "Cross-encoder"): (0.120995, 0.316176),
    ("ECOM", "Q1", "Logistic regression"): (0.230193, 0.617647),
    ("ECOM", "Q2", "Cross-encoder"): (0.051143, 0.083951),
    ("ECOM", "Q2", "Logistic regression"): (0.101332, 0.175309),
    ("ECOM", "Q3", "Cross-encoder"): (0.015730, 0.056790),
    ("ECOM", "Q3", "Logistic regression"): (0.017284, 0.0),
    ("ECOM", "Q4", "Cross-encoder"): (0.0, 0.004938),
    ("ECOM", "Q4", "Logistic regression"): (0.0, 0.0),
}


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
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype=str, encoding="latin-1")
    df.columns = [clean_header(c) for c in df.columns]
    return df


def find_id_column(frame: pd.DataFrame, path: Path) -> str:
    matches = [c for c in frame.columns if c.lower() == "id"]
    if len(matches) != 1:
        raise KeyError(f"Expected exactly one ID column in {path}; got {list(frame.columns)}")
    return matches[0]


def normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower().str.strip()


def june_csv_roundtrip(frame: pd.DataFrame, id_column: str) -> pd.DataFrame:
    indexed = frame.set_index(id_column, drop=True)
    indexed.index.name = "id"
    buf = StringIO()
    indexed.to_csv(buf)
    buf.seek(0)
    out = pd.read_csv(buf, dtype=str).set_index("id")
    out.index = out.index.astype(str)
    out.index.name = "id"
    return out


def serialise(row: pd.Series, fields: list[str]) -> str:
    parts: list[str] = []
    for field in fields:
        value = row.get(field, "")
        parts.append(str(value) if value is not None and str(value) != "nan" else "")
    return " [SEP] ".join(parts)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_csv(df: pd.DataFrame, path: Path, *, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, compression=compression)
    os.replace(tmp, path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def load_top50(path: Path, a_ids: list[str], b_ids: list[str]) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    top50 = payload.get("top50")
    if not isinstance(top50, dict):
        raise ValueError(f"{path}: missing top50 dictionary")
    if set(top50) != set(a_ids):
        missing = set(a_ids) - set(top50)
        extra = set(top50) - set(a_ids)
        raise AssertionError(f"{path}: A-ID mismatch; missing={len(missing)}, extra={len(extra)}")
    bset = set(b_ids)
    for ida, vals in top50.items():
        if len(vals) != K:
            raise AssertionError(f"{path}: {ida} has {len(vals)} candidates, expected {K}")
        if any(v not in bset for v in vals):
            raise AssertionError(f"{path}: unknown B candidate for {ida}")
    return {str(k): [str(v) for v in vals] for k, vals in top50.items()}


def load_dataset(name: str) -> dict[str, Any]:
    cfg = CONFIGS[name]
    a = read_latin1(cfg["a"])
    b = read_latin1(cfg["b"])
    mapping = read_latin1(cfg["mapping"])
    aid_file = find_id_column(a, cfg["a"])
    bid_file = find_id_column(b, cfg["b"])
    a = a.rename(columns=cfg["rename"])
    b = b.rename(columns=cfg["rename"])
    for frame, path in ((a, cfg["a"]), (b, cfg["b"])):
        missing = set(cfg["fields"]) - set(frame.columns)
        if missing:
            raise KeyError(f"{path}: missing fields {sorted(missing)}")
        for field in cfg["fields"]:
            frame[field] = normalize_text(frame[field])
    a[aid_file] = a[aid_file].fillna("").astype(str).str.strip()
    b[bid_file] = b[bid_file].fillna("").astype(str).str.strip()
    a = june_csv_roundtrip(a, aid_file)
    b = june_csv_roundtrip(b, bid_file)

    ma, mb = cfg["id_a"], cfg["id_b"]
    if ma not in mapping.columns or mb not in mapping.columns:
        raise KeyError(f"{cfg['mapping']}: missing mapping columns {ma},{mb}")
    mapping[ma] = mapping[ma].fillna("").astype(str).str.strip()
    mapping[mb] = mapping[mb].fillna("").astype(str).str.strip()

    observed = {"A": len(a), "B": len(b), "pairs": len(mapping)}
    if observed != cfg["expected"]:
        raise AssertionError(f"{name} snapshot mismatch: {observed} != {cfg['expected']}")
    if not a.index.is_unique or not b.index.is_unique:
        raise AssertionError(f"{name}: non-unique A/B IDs")
    aset, bset = set(a.index), set(b.index)
    reachable = mapping[ma].isin(aset) & mapping[mb].isin(bset)
    if not bool(reachable.all()):
        raise AssertionError(f"{name}: unreachable mapping rows={int((~reachable).sum())}")

    truth: dict[str, set[str]] = {}
    retained: dict[str, str] = {}
    for ida, idb in mapping[[ma, mb]].itertuples(index=False, name=None):
        truth.setdefault(ida, set()).add(idb)
        retained.setdefault(ida, idb)

    a_ids = list(a.index)
    b_ids = list(b.index)
    top50 = load_top50(cfg["cache"], a_ids, b_ids)

    ds = {
        "name": name,
        "cfg": cfg,
        "df_a": a,
        "df_b": b,
        "truth": truth,
        "retained": retained,
        "a_ids": a_ids,
        "b_ids": b_ids,
        "a_pos": {v: i for i, v in enumerate(a_ids)},
        "b_pos": {v: i for i, v in enumerate(b_ids)},
        "base_top50": top50,
    }
    print(
        f"{cfg['paper_name']}: |A|={len(a):,}, |B|={len(b):,}, "
        f"mapping={len(mapping):,}, A-with-truth={len(truth):,}, "
        f"multi-match-A={sum(len(v)>1 for v in truth.values()):,}; top-50 cache OK"
    )
    return ds


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


def encode_fields(
    model: SentenceTransformer,
    frame: pd.DataFrame,
    fields: list[str],
    label: str,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for field in fields:
        print(f"    encoding {label}.{field}")
        out[field] = model.encode(
            frame[field].fillna("").tolist(),
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=EMBED_BATCH,
            normalize_embeddings=True,
        )
    return out


def retrieve_top50(
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
    a_ids: list[str],
    b_ids: list[str],
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for start in range(0, len(a_ids), RETRIEVAL_CHUNK):
        stop = min(start + RETRIEVAL_CHUNK, len(a_ids))
        scores = torch.matmul(emb_a[start:stop], emb_b.transpose(0, 1))
        idx = torch.topk(scores, k=K, dim=1).indices.cpu().tolist()
        for offset, row_idx in enumerate(idx):
            out[a_ids[start + offset]] = [b_ids[j] for j in row_idx]
    return out


def cosine_feature_matrix(
    pairs: list[tuple[str, str]],
    emb_a: dict[str, torch.Tensor],
    emb_b: dict[str, torch.Tensor],
    fields: list[str],
    a_pos: dict[str, int],
    b_pos: dict[str, int],
) -> np.ndarray:
    if not pairs:
        return np.empty((0, len(fields)), dtype=np.float64)
    matrix = np.empty((len(pairs), len(fields)), dtype=np.float64)
    device = emb_a[fields[0]].device
    ai = np.fromiter((a_pos[a] for a, _ in pairs), dtype=np.int64, count=len(pairs))
    bi = np.fromiter((b_pos[b] for _, b in pairs), dtype=np.int64, count=len(pairs))
    for start in range(0, len(pairs), FEATURE_CHUNK):
        stop = min(start + FEATURE_CHUNK, len(pairs))
        ia = torch.as_tensor(ai[start:stop], dtype=torch.long, device=device)
        ib = torch.as_tensor(bi[start:stop], dtype=torch.long, device=device)
        for j, field in enumerate(fields):
            sim = (emb_a[field].index_select(0, ia) * emb_b[field].index_select(0, ib)).sum(dim=1)
            matrix[start:stop, j] = sim.detach().cpu().numpy().astype(np.float64)
    return matrix


def choose_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    best_t = 0.50
    best_f1 = -1.0
    for t in THRESHOLDS:
        f = f1_score(labels, (np.asarray(scores) > t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1 = float(f)
            best_t = float(t)
    return best_t, best_f1


def calibrate_lr(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[LogisticRegression, float, float]:
    x_fit, x_val, y_fit, y_val = train_test_split(
        x, y, test_size=LR_VAL_FRAC, random_state=seed, stratify=y
    )
    model = LogisticRegression(penalty="l2", max_iter=LR_MAX_ITER, random_state=seed).fit(x_fit, y_fit)
    probs = model.predict_proba(x_val)[:, 1]
    t, f = choose_threshold(probs, y_val)
    return model, t, f


def model_path(dataset: str, seed: int) -> Path:
    return MODELS_DIR / dataset / f"seed_{seed}"


def marker_path(path: Path) -> Path:
    return path / "_table8_recovery_complete.json"


def save_model_atomic(model: SentenceTransformer, path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save(str(tmp))
    atomic_json(metadata, tmp / "_table8_recovery_complete.json")
    if path.exists():
        shutil.rmtree(path)
    os.replace(tmp, path)


def train_or_load_model(
    ds: dict[str, Any],
    train_pairs: list[tuple[str, str, int]],
    seed: int,
    device: str,
    force_retrain: bool,
) -> SentenceTransformer:
    from sentence_transformers import InputExample, SentenceTransformer, losses

    path = model_path(ds["name"], seed)
    if force_retrain and path.exists():
        shutil.rmtree(path)
    if marker_path(path).exists():
        print(f"    loading recovered FT model: {path}")
        return SentenceTransformer(str(path), device=device)

    if path.exists():
        shutil.rmtree(path)
    fields = ds["cfg"]["fields"]
    examples = [
        InputExample(
            texts=[serialise(ds["df_a"].loc[ida], fields), serialise(ds["df_b"].loc[idb], fields)],
            label=float(label),
        )
        for ida, idb, label in train_pairs
    ]
    seed_everything(seed)
    model = SentenceTransformer(BI_ENCODER_NAME, device=device)
    loader = DataLoader(examples, shuffle=True, batch_size=FT_BATCH)
    loss = losses.ContrastiveLoss(model)
    total_steps = len(loader) * FT_EPOCHS
    warmup = int(FT_WARMUP_FRAC * total_steps)
    print(
        f"    fine-tuning {BI_ENCODER_NAME}: pairs={len(examples):,}, "
        f"epochs=1, warmup={warmup}, lr=2e-5, batch=16"
    )
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=FT_EPOCHS,
        warmup_steps=warmup,
        optimizer_params={"lr": FT_LR},
        show_progress_bar=False,
    )
    save_model_atomic(
        model,
        path,
        {
            "dataset": ds["name"],
            "seed": seed,
            "model": BI_ENCODER_NAME,
            "epochs": FT_EPOCHS,
            "warmup_fraction": FT_WARMUP_FRAC,
            "learning_rate": FT_LR,
            "batch_size": FT_BATCH,
            "training_pairs": len(examples),
        },
    )
    print(f"    saved model: {path}")
    return model


def lr_predictions(
    ds: dict[str, Any],
    ids_train: list[str],
    ids_test: list[str],
    top50: dict[str, list[str]],
    emb_a: dict[str, torch.Tensor],
    emb_b: dict[str, torch.Tensor],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    train_pairs = build_train_pairs(ids_train, ds["retained"], ds["truth"], top50)
    pair_ids = [(a, b) for a, b, _ in train_pairs]
    x_train = cosine_feature_matrix(
        pair_ids, emb_a, emb_b, ds["cfg"]["fields"], ds["a_pos"], ds["b_pos"]
    )
    y_train = np.asarray([y for _, _, y in train_pairs], dtype=int)
    lr, threshold, validation_f1 = calibrate_lr(x_train, y_train, seed)

    rows: list[dict[str, Any]] = []
    declared = correct = false_links = missing_matches = n_true = 0
    for record_index, ida in enumerate(ids_test):
        candidates = top50[ida]
        features = cosine_feature_matrix(
            [(ida, idb) for idb in candidates],
            emb_a,
            emb_b,
            ds["cfg"]["fields"],
            ds["a_pos"],
            ds["b_pos"],
        )
        scores = lr.predict_proba(features)[:, 1]
        order = np.argsort(-scores, kind="stable")
        i1 = int(order[0])
        i2 = int(order[1])
        top1 = candidates[i1]
        s1, s2 = float(scores[i1]), float(scores[i2])
        gap = s1 - s2
        dec = bool(s1 > threshold)
        true_set = ds["truth"].get(ida, set())
        if true_set:
            n_true += 1
        if dec:
            declared += 1
            if top1 in true_set:
                correct += 1
                category = "correct"
            else:
                false_links += 1
                category = "false_link"
        else:
            if true_set:
                missing_matches += 1
                category = "missing_match"
            else:
                category = "unmatched_abstention"
        rows.append(
            {
                "record_index": record_index,
                "ida": ida,
                "has_true_match": bool(true_set),
                "n_valid_matches": len(true_set),
                "ft_lr_top1_idb": top1,
                "ft_lr_top1_score": s1,
                "ft_lr_top2_score": s2,
                "ft_lr_gap": gap,
                "ft_lr_threshold": threshold,
                "ft_lr_declared": dec,
                "ft_lr_category": category,
                "dataset": ds["name"],
                "seed": seed,
            }
        )
    metrics = {
        "lambda": false_links / declared if declared else 0.0,
        "psi": missing_matches / n_true if n_true else 0.0,
        "declared": declared,
        "correct": correct,
        "false_links": false_links,
        "missing_matches": missing_matches,
        "n_true_records": n_true,
        "threshold": threshold,
        "validation_f1": validation_f1,
        "n_train_pairs": len(train_pairs),
    }
    return pd.DataFrame(rows), metrics


def as_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
    vals = series.astype(str).str.strip().str.lower().map(mapping)
    if vals.isna().any():
        bad = series[vals.isna()].head().tolist()
        raise ValueError(f"Cannot parse booleans; examples={bad}")
    return vals.astype(bool)


def load_historical_ce(dataset: str, seed: int, ids_test: list[str]) -> pd.DataFrame:
    path = HIST_CE_DIR / f"{dataset}_seed{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={"ida": str})
    required = {"ida", "ce_direct_declared", "ce_direct_category"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{path}: missing {sorted(missing)}")
    df["ida"] = df["ida"].astype(str)
    if set(df["ida"]) != set(ids_test):
        raise AssertionError(
            f"{path}: historical CE test-record set differs from reconstructed split "
            f"({len(set(df['ida']))} vs {len(set(ids_test))})"
        )
    if df["ida"].duplicated().any():
        raise AssertionError(f"{path}: duplicate ida")
    df["ce_direct_declared"] = as_bool_series(df["ce_direct_declared"])
    return df[["ida", "ce_direct_declared", "ce_direct_category"]].copy()


def close(actual: float, expected: float, tol: float = 5e-6) -> bool:
    return abs(float(actual) - float(expected)) <= tol


def check_whole(dataset: str, seed: int, reblocked: dict[str, Any], fixed: dict[str, Any]) -> None:
    exp = EXPECTED_WHOLE[(dataset, seed)]
    checks = [
        ("reblocked lambda", reblocked["lambda"], exp["reblocked_lambda"], 5e-6),
        ("reblocked psi", reblocked["psi"], exp["reblocked_psi"], 5e-6),
        ("reblocked threshold", reblocked["threshold"], exp["reblocked_threshold"], 1e-9),
        ("reblocked pairs", reblocked["n_train_pairs"], exp["reblocked_pairs"], 0),
        ("fixed lambda", fixed["lambda"], exp["fixed_lambda"], 5e-6),
        ("fixed psi", fixed["psi"], exp["fixed_psi"], 5e-6),
        ("fixed threshold", fixed["threshold"], exp["fixed_threshold"], 1e-9),
        ("fixed pairs", fixed["n_train_pairs"], exp["fixed_pairs"], 0),
    ]
    failed: list[str] = []
    for label, actual, expected, tol in checks:
        ok = abs(float(actual) - float(expected)) <= float(tol)
        if not ok:
            failed.append(f"{label}: observed={actual}, expected={expected}")
    if failed:
        raise RuntimeError(
            f"HISTORICAL WHOLE-TEST REPRODUCTION FAILED for {dataset} seed {seed}:\n  "
            + "\n  ".join(failed)
            + "\nDo not use the recovered MCSEs until this mismatch is resolved."
        )
    print(
        f"    WHOLE-TEST CHECK PASS: reblocked λ={reblocked['lambda']:.6f}, "
        f"ψ={reblocked['psi']:.6f}, t={reblocked['threshold']:.2f}; "
        f"fixed λ={fixed['lambda']:.6f}, ψ={fixed['psi']:.6f}, t={fixed['threshold']:.2f}"
    )


def recover_seed(
    ds: dict[str, Any],
    seed: int,
    device: str,
    force_retrain: bool,
    force_recompute: bool,
) -> pd.DataFrame:
    part = PARTS_DIR / f"fixedblock_{ds['name']}_seed{seed}.csv"
    audit_path = PARTS_DIR / f"audit_{ds['name']}_seed{seed}.json"
    if part.exists() and audit_path.exists() and not force_recompute:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("historical_check_passed") is True:
            print(f"    SKIP completed audited seed: {ds['name']} {seed}")
            return pd.read_csv(part, dtype={"ida": str})

    seed_everything(seed)
    ids_train_arr, ids_test_arr = train_test_split(
        ds["a_ids"], test_size=0.50, random_state=seed
    )
    ids_train, ids_test = list(ids_train_arr), list(ids_test_arr)

    baseline_train_pairs = build_train_pairs(
        ids_train, ds["retained"], ds["truth"], ds["base_top50"]
    )
    expected_fixed_pairs = int(EXPECTED_WHOLE[(ds["name"], seed)]["fixed_pairs"])
    if len(baseline_train_pairs) != expected_fixed_pairs:
        raise RuntimeError(
            f"Baseline/fixed train-pair count mismatch {ds['name']} seed {seed}: "
            f"{len(baseline_train_pairs)} != {expected_fixed_pairs}"
        )

    model = train_or_load_model(ds, baseline_train_pairs, seed, device, force_retrain)
    fields = ds["cfg"]["fields"]
    emb_a = encode_fields(model, ds["df_a"], fields, "FT-A")
    emb_b = encode_fields(model, ds["df_b"], fields, "FT-B")

    # Historical Table-6 lineage check: rebuild top-50 after fine-tuning.
    ft_top50 = retrieve_top50(
        emb_a[ds["cfg"]["retrieval_field"]],
        emb_b[ds["cfg"]["retrieval_field"]],
        ds["a_ids"],
        ds["b_ids"],
    )
    _, reblocked_metrics = lr_predictions(
        ds, ids_train, ids_test, ft_top50, emb_a, emb_b, seed
    )

    # Final Table-8 lineage: keep the PRETRAINED top-50 blocks fixed, but score
    # them with logistic regression on FINE-TUNED bi-encoder embeddings.
    fixed_df, fixed_metrics = lr_predictions(
        ds, ids_train, ids_test, ds["base_top50"], emb_a, emb_b, seed
    )
    check_whole(ds["name"], seed, reblocked_metrics, fixed_metrics)

    ce = load_historical_ce(ds["name"], seed, ids_test)
    merged = fixed_df.merge(ce, on="ida", how="left", validate="one_to_one")
    if merged[["ce_direct_declared", "ce_direct_category"]].isna().any().any():
        raise AssertionError("CE merge introduced missing values")

    atomic_csv(merged, part)
    atomic_json(
        {
            "dataset": ds["name"],
            "seed": seed,
            "historical_check_passed": True,
            "reblocked": reblocked_metrics,
            "fixed": fixed_metrics,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        audit_path,
    )

    del model, emb_a, emb_b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return merged


def counts_from_categories(sub: pd.DataFrame, category_col: str, declared_col: str) -> dict[str, int]:
    declared = int(as_bool_series(sub[declared_col]).sum())
    cat = sub[category_col].astype(str)
    correct = int((cat == "correct").sum())
    false_links = int((cat == "false_link").sum())
    missing = int((cat == "missing_match").sum())
    n_true = int(as_bool_series(sub["has_true_match"]).sum())
    if declared != correct + false_links:
        raise AssertionError(
            f"declared={declared}, correct={correct}, false={false_links} for {category_col}"
        )
    return {
        "declared": declared,
        "correct": correct,
        "false_links": false_links,
        "missing_matches": missing,
        "n_true_records": n_true,
    }


def make_quartile_perseed(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, seed), sub0 in predictions.groupby(["dataset", "seed"], sort=True):
        sub = sub0.copy()
        sub["quartile"] = pd.qcut(
            sub["ft_lr_gap"],
            4,
            labels=["Q1", "Q2", "Q3", "Q4"],
            duplicates="drop",
        )
        if sub["quartile"].isna().any() or sub["quartile"].nunique() != 4:
            raise RuntimeError(f"Quartile formation failed for {dataset} seed {seed}")
        for quartile, q in sub.groupby("quartile", observed=False, sort=True):
            for model, category_col, declared_col in (
                ("Logistic regression", "ft_lr_category", "ft_lr_declared"),
                ("Cross-encoder", "ce_direct_category", "ce_direct_declared"),
            ):
                c = counts_from_categories(q, category_col, declared_col)
                d, cor, m = c["declared"], c["correct"], c["n_true_records"]
                flr = 1.0 - cor / d if d else 0.0
                old_psi = c["missing_matches"] / m if m else 0.0
                final_mmr = 1.0 - cor / m if m else 0.0
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "quartile": str(quartile),
                        "model": model,
                        "n_records": len(q),
                        **c,
                        "FLR": flr,
                        "historical_psi": old_psi,
                        "MMR_final": final_mmr,
                    }
                )
    return pd.DataFrame(rows)


def verify_historical_quartile_means(perseed: pd.DataFrame) -> None:
    means = (
        perseed.groupby(["dataset", "quartile", "model"], as_index=False)
        .agg(lambda_mean=("FLR", "mean"), psi_mean=("historical_psi", "mean"))
    )
    failed: list[str] = []
    print("\nHistorical Aug-30 quartile reproduction check")
    print("-" * 78)
    for row in means.itertuples(index=False):
        key = (row.dataset, row.quartile, row.model)
        expected = EXPECTED_QUARTILE_OLD[key]
        ok = close(row.lambda_mean, expected[0]) and close(row.psi_mean, expected[1])
        status = "PASS" if ok else "FAIL"
        print(
            f"{status:4s} {row.dataset:4s} {row.quartile} {row.model:20s} "
            f"observed λ/ψ={row.lambda_mean:.6f}/{row.psi_mean:.6f}; "
            f"historical={expected[0]:.6f}/{expected[1]:.6f}"
        )
        if not ok:
            failed.append(f"{key}: {row.lambda_mean}/{row.psi_mean} vs {expected}")
    if failed:
        raise RuntimeError(
            "HISTORICAL QUARTILE REPRODUCTION FAILED. Do not use MCSE output.\n"
            + "\n".join(failed)
        )


def mcse(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=float)
    if len(arr) <= 1:
        return float("nan")
    return float(np.std(arr, ddof=1) / math.sqrt(len(arr)))


def aggregate_final(perseed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, g in perseed.groupby(["dataset", "quartile", "model"], sort=True):
        dataset, quartile, model = keys
        rows.append(
            {
                "dataset": dataset,
                "paper_dataset": CONFIGS[dataset]["paper_name"],
                "quartile": quartile,
                "model": model,
                "n_splits": len(g),
                "FLR_mean": float(g["FLR"].mean()),
                "FLR_MCSE": mcse(g["FLR"]),
                "MMR_mean": float(g["MMR_final"].mean()),
                "MMR_MCSE": mcse(g["MMR_final"]),
            }
        )
    return pd.DataFrame(rows)


def validate_only() -> None:
    print("=" * 78)
    print("TABLE 8 MCSE RECOVERY — VALIDATION ONLY")
    print("=" * 78)
    for name in ("DBLP", "ECOM"):
        ds = load_dataset(name)
        for seed in PRIMARY_SEEDS:
            _, ids_test = train_test_split(ds["a_ids"], test_size=0.50, random_state=seed)
            ce = load_historical_ce(name, seed, list(ids_test))
            print(f"  {name} seed {seed}: historical CE rows={len(ce):,}; test split aligned")
    print("Validation complete. No model loaded or trained.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recover manuscript Table 8 MCSEs.")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--force-recompute", action="store_true")
    p.add_argument("--datasets", default="DBLP,ECOM")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        validate_only()
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
    datasets = [x.strip().upper() for x in args.datasets.split(",") if x.strip()]
    bad = set(datasets) - set(CONFIGS)
    if bad:
        raise ValueError(f"Unknown datasets: {sorted(bad)}")

    print("=" * 90)
    print("TABLE 8 MCSE RECOVERY")
    print("=" * 90)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Torch: {torch.__version__}; compiled CUDA: {torch.version.cuda}")
    print(f"Device: {args.device}")
    print(f"Seeds: {PRIMARY_SEEDS}")
    print(f"Bi-encoder: {BI_ENCODER_NAME}; one-epoch ContrastiveLoss fine-tuning")
    print("Cross-encoder: NO retraining; historical seed-level decisions are bundled")
    print("Final MMR: 1 - correct / n_true_records")

    all_parts: list[pd.DataFrame] = []
    for name in datasets:
        print("\n" + "=" * 90)
        print(CONFIGS[name]["paper_name"])
        print("=" * 90)
        ds = load_dataset(name)
        for seed in PRIMARY_SEEDS:
            print(f"\n{name} seed={seed}")
            print("-" * 70)
            part = recover_seed(
                ds, seed, args.device, args.force_retrain, args.force_recompute
            )
            all_parts.append(part)
        del ds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    predictions = pd.concat(all_parts, ignore_index=True)
    perseed = make_quartile_perseed(predictions)
    verify_historical_quartile_means(perseed)
    final = aggregate_final(perseed)

    atomic_csv(perseed, RESULTS_DIR / "table8_perseed_final.csv")
    atomic_csv(final, RESULTS_DIR / "table8_primary3_final_mcse.csv")
    # For audit/reuse. gzip is written directly rather than atomically because
    # pandas infers compression from suffix; the file is non-critical cache.
    predictions.to_csv(RESULTS_DIR / "table8_predictions.csv.gz", index=False, compression="gzip")

    try:
        import sentence_transformers
        st_version = sentence_transformers.__version__
    except Exception:
        st_version = "unknown"
    atomic_json(
        {
            "purpose": "Recover manuscript Table 8 primary-three-seed MCSEs",
            "seeds": PRIMARY_SEEDS,
            "final_flr": "1 - correct / declared",
            "final_mmr": "1 - correct / n_true_records",
            "mcse": "sample SD across seeds / sqrt(3)",
            "quartiles": "pd.qcut(ft_lr_gap, 4) separately within each dataset/seed test set",
            "candidate_blocks": "pretrained all-MiniLM-L6-v2 top-50 held fixed for Table 8",
            "ft_biencoder": {
                "model": BI_ENCODER_NAME,
                "loss": "ContrastiveLoss",
                "epochs": 1,
                "warmup_fraction": 0.10,
                "learning_rate": 2e-5,
                "batch_size": 16,
            },
            "cross_encoder": "historical Analysis 17 direct decisions; no retraining in recovery",
            "versions": {
                "python": sys.version,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "sentence_transformers": st_version,
                "pandas": pd.__version__,
            },
            "historical_reproduction_passed": True,
        },
        RESULTS_DIR / "table8_recovery_manifest.json",
    )

    print("\n" + "=" * 104)
    print("TABLE 8 FINAL — mean (MCSE), seeds 42–44")
    print("=" * 104)
    for dataset in datasets:
        print(f"\n{CONFIGS[dataset]['paper_name']}")
        sub = final[final["dataset"] == dataset]
        for quartile in ["Q1", "Q2", "Q3", "Q4"]:
            for model in ["Logistic regression", "Cross-encoder"]:
                r = sub[(sub["quartile"] == quartile) & (sub["model"] == model)].iloc[0]
                print(
                    f"  {quartile} {model:20s}  "
                    f"FLR {r.FLR_mean:.3f} ({r.FLR_MCSE:.4f})   "
                    f"MMR {r.MMR_mean:.3f} ({r.MMR_MCSE:.4f})"
                )

    print("\nSaved:")
    print(f"  {RESULTS_DIR / 'table8_perseed_final.csv'}")
    print(f"  {RESULTS_DIR / 'table8_primary3_final_mcse.csv'}")
    print(f"  {RESULTS_DIR / 'table8_predictions.csv.gz'}")
    print(f"  {RESULTS_DIR / 'table8_recovery_manifest.json'}")
    print("\nPaste the HISTORICAL QUARTILE REPRODUCTION CHECK and TABLE 8 FINAL blocks into ChatGPT.")


if __name__ == "__main__":
    main()
