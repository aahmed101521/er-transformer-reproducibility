"""
build_paper_tables.py
=====================

Final paper-facing reconstruction for:
    Text-based Entity Resolution Using Transformer Models

Authors:
    Abrar Ahmed
    Li-Chun Zhang
    Ralf Münnich
    Martin Vogt

Purpose
-------
The recovered scientific scripts preserve the historical intermediate variable
``psi = missing_matches / n_true_records`` for provenance.  That quantity is
NOT the MMR used in the final manuscript.

This script reconstructs every empirical paper table from the saved record-level
counts using the manuscript definitions

    FLR = 1 - correct / declared
    MMR = 1 - correct / n_true_records

so an incorrect declared link for a matchable file-A record contributes to both
FLR and MMR.  It then checks the reproduced values, rounded to the precision
printed in the LaTeX manuscript, against the manuscript tables.

The script is deliberately a reporting/reconstruction layer.  It does not fit
or score any model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
OUT = ROOT / "reproduced" / "paper_tables"
OUT.mkdir(parents=True, exist_ok=True)

PRIMARY_SEEDS = [42, 43, 44]
DATASET_DISPLAY = {"DBLP": "DBLP-Scholar", "ECOM": "Abt-Buy"}

EXPECTED_DATASETS = {
    "DBLP-Scholar": {
        "nA": 2616,
        "nB": 64263,
        "gold_pairs": 5347,
        "matchable_A": 2408,
        "matched_B": 5218,
        "multi_partner_A": 1238,
        "max_partners_A": 20,
    },
    "Abt-Buy": {
        "nA": 1081,
        "nB": 1092,
        "gold_pairs": 1097,
        "matchable_A": 1081,
        "matched_B": 1092,
        "multi_partner_A": 16,
        "max_partners_A": 2,
    },
}

# Table 4 is generated directly from the preserved Analysis 15 rank-statistic
# implementation.  It is included in the package as a reproducible scientific
# output, but is not used as a hard-coded manuscript-alignment assertion.
# This avoids substituting manuscript constants for the actual computation.

EXPECTED_TABLE5 = {
    ("DBLP-Scholar", "String, Jaro-Winkler", "Generative, kernel density", 5): (0.034, 0.018),
    ("DBLP-Scholar", "String, Jaro-Winkler", "Generative, kernel density", 50): (0.046, 0.013),
    ("DBLP-Scholar", "String, Jaro-Winkler", "Discriminative, logistic", 5): (0.030, 0.036),
    ("DBLP-Scholar", "String, Jaro-Winkler", "Discriminative, logistic", 50): (0.027, 0.050),
    ("DBLP-Scholar", "Bi-encoder, cosine", "Generative, kernel density", 5): (0.023, 0.018),
    ("DBLP-Scholar", "Bi-encoder, cosine", "Generative, kernel density", 50): (0.043, 0.014),
    ("DBLP-Scholar", "Bi-encoder, cosine", "Discriminative, logistic", 5): (0.016, 0.056),
    ("DBLP-Scholar", "Bi-encoder, cosine", "Discriminative, logistic", 50): (0.009, 0.068),
    ("Abt-Buy", "String, Jaro-Winkler", "Generative, kernel density", 5): (0.389, 0.500),
    ("Abt-Buy", "String, Jaro-Winkler", "Generative, kernel density", 50): (0.449, 0.525),
    ("Abt-Buy", "String, Jaro-Winkler", "Discriminative, logistic", 5): (0.343, 0.597),
    ("Abt-Buy", "String, Jaro-Winkler", "Discriminative, logistic", 50): (0.338, 0.662),
    ("Abt-Buy", "Bi-encoder, cosine", "Generative, kernel density", 5): (0.278, 0.425),
    ("Abt-Buy", "Bi-encoder, cosine", "Generative, kernel density", 50): (0.327, 0.356),
    ("Abt-Buy", "Bi-encoder, cosine", "Discriminative, logistic", 5): (0.248, 0.479),
    ("Abt-Buy", "Bi-encoder, cosine", "Discriminative, logistic", 50): (0.233, 0.543),
}

EXPECTED_TABLE6 = {
    ("Logistic", "DBLP-Scholar", 5): (0.016, 0.056),
    ("Logistic", "DBLP-Scholar", 50): (0.009, 0.068),
    ("Logistic", "Abt-Buy", 5): (0.248, 0.479),
    ("Logistic", "Abt-Buy", 50): (0.233, 0.543),
    ("SVM", "DBLP-Scholar", 5): (0.017, 0.040),
    ("SVM", "DBLP-Scholar", 50): (0.018, 0.041),
    ("SVM", "Abt-Buy", 5): (0.252, 0.592),
    ("SVM", "Abt-Buy", 50): (0.211, 0.701),
    ("XGB", "DBLP-Scholar", 5): (0.033, 0.021),
    ("XGB", "DBLP-Scholar", 50): (0.024, 0.041),
    ("XGB", "Abt-Buy", 5): (0.399, 0.538),
    ("XGB", "Abt-Buy", 50): (0.296, 0.551),
    ("MLP", "DBLP-Scholar", 5): (0.030, 0.030),
    ("MLP", "DBLP-Scholar", 50): (0.026, 0.034),
    ("MLP", "Abt-Buy", 5): (0.251, 0.438),
    ("MLP", "Abt-Buy", 50): (0.228, 0.565),
}

EXPECTED_TABLE7 = {
    ("DBLP-Scholar", "Generative, kernel density", "pretrained"): (0.043, 0.014),
    ("DBLP-Scholar", "Generative, kernel density", "fine_tuned"): (0.046, 0.011),
    ("DBLP-Scholar", "Discriminative, logistic", "pretrained"): (0.009, 0.068),
    ("DBLP-Scholar", "Discriminative, logistic", "fine_tuned"): (0.026, 0.027),
    ("DBLP-Scholar", "Cross-encoder", "pretrained"): (0.061, 0.026),
    ("DBLP-Scholar", "Cross-encoder", "fine_tuned"): (0.018, 0.014),
    ("Abt-Buy", "Generative, kernel density", "pretrained"): (0.327, 0.356),
    ("Abt-Buy", "Generative, kernel density", "fine_tuned"): (0.124, 0.145),
    ("Abt-Buy", "Discriminative, logistic", "pretrained"): (0.233, 0.543),
    ("Abt-Buy", "Discriminative, logistic", "fine_tuned"): (0.053, 0.269),
    ("Abt-Buy", "Cross-encoder", "pretrained"): (0.267, 0.278),
    ("Abt-Buy", "Cross-encoder", "fine_tuned"): (0.041, 0.152),
}

EXPECTED_TABLE8 = {
    ("DBLP-Scholar", "Q1", "Logistic regression"): (0.037, 0.045),
    ("DBLP-Scholar", "Q1", "Cross-encoder"): (0.033, 0.029),
    ("DBLP-Scholar", "Q2", "Logistic regression"): (0.031, 0.063),
    ("DBLP-Scholar", "Q2", "Cross-encoder"): (0.034, 0.022),
    ("DBLP-Scholar", "Q3", "Logistic regression"): (0.006, 0.041),
    ("DBLP-Scholar", "Q3", "Cross-encoder"): (0.008, 0.010),
    ("DBLP-Scholar", "Q4", "Logistic regression"): (0.000, 0.000),
    ("DBLP-Scholar", "Q4", "Cross-encoder"): (0.000, 0.000),
    ("Abt-Buy", "Q1", "Logistic regression"): (0.230, 0.706),
    ("Abt-Buy", "Q1", "Cross-encoder"): (0.121, 0.400),
    ("Abt-Buy", "Q2", "Logistic regression"): (0.101, 0.259),
    ("Abt-Buy", "Q2", "Cross-encoder"): (0.051, 0.131),
    ("Abt-Buy", "Q3", "Logistic regression"): (0.017, 0.017),
    ("Abt-Buy", "Q3", "Cross-encoder"): (0.016, 0.072),
    ("Abt-Buy", "Q4", "Logistic regression"): (0.000, 0.000),
    ("Abt-Buy", "Q4", "Cross-encoder"): (0.000, 0.005),
}

EXPECTED_CE_BLOCK = {
    ("DBLP-Scholar", 5): (0.022, 0.013),
    ("DBLP-Scholar", 50): (0.018, 0.014),
    ("Abt-Buy", 5): (0.056, 0.219),
    ("Abt-Buy", 50): (0.041, 0.152),
}

EXPECTED_TABLE9 = {
    ("DBLP-Scholar", 0.5): (0.017, 0.001, 0.058, 0.015, 0.001, 0.077),
    ("DBLP-Scholar", 0.6): (0.019, 0.001, 0.078, 0.016, 0.001, 0.077),
    ("DBLP-Scholar", 0.7): (0.014, 0.002, 0.136, 0.015, 0.002, 0.110),
    ("DBLP-Scholar", 0.8): (0.016, 0.003, 0.171, 0.016, 0.002, 0.102),
    ("DBLP-Scholar", 0.9): (0.015, 0.003, 0.188, 0.019, 0.002, 0.115),
    ("Abt-Buy", 0.5): (0.039, 0.003, 0.064, 0.160, 0.005, 0.033),
    ("Abt-Buy", 0.6): (0.034, 0.003, 0.076, 0.168, 0.009, 0.051),
    ("Abt-Buy", 0.7): (0.029, 0.004, 0.133, 0.164, 0.004, 0.026),
    ("Abt-Buy", 0.8): (0.033, 0.002, 0.066, 0.174, 0.007, 0.041),
    ("Abt-Buy", 0.9): (0.029, 0.004, 0.148, 0.158, 0.013, 0.080),
}

EXPECTED_TABLE10 = {
    ("FLR", "DBLP-Scholar", 0.5): (0.017, 0.003, 0.003, 0.004, 0.002),
    ("FLR", "DBLP-Scholar", 0.6): (0.019, 0.005, 0.003, 0.003, 0.002),
    ("FLR", "DBLP-Scholar", 0.7): (0.014, 0.005, 0.003, 0.010, 0.003),
    ("FLR", "DBLP-Scholar", 0.8): (0.016, 0.003, 0.004, 0.008, 0.004),
    ("FLR", "DBLP-Scholar", 0.9): (0.015, 0.004, 0.003, 0.007, 0.003),
    ("FLR", "Abt-Buy", 0.5): (0.039, 0.001, 0.004, 0.001, 0.006),
    ("FLR", "Abt-Buy", 0.6): (0.034, -0.001, 0.006, 0.004, 0.006),
    ("FLR", "Abt-Buy", 0.7): (0.029, 0.007, 0.007, 0.008, 0.006),
    ("FLR", "Abt-Buy", 0.8): (0.033, 0.001, 0.005, 0.001, 0.004),
    ("FLR", "Abt-Buy", 0.9): (0.029, 0.004, 0.005, 0.006, 0.006),
    ("MMR", "DBLP-Scholar", 0.5): (0.015, -0.002, 0.001, 0.005, 0.002),
    ("MMR", "DBLP-Scholar", 0.6): (0.016, -0.001, 0.002, 0.000, 0.002),
    ("MMR", "DBLP-Scholar", 0.7): (0.015, -0.001, 0.003, 0.003, 0.003),
    ("MMR", "DBLP-Scholar", 0.8): (0.016, 0.001, 0.003, 0.002, 0.003),
    ("MMR", "DBLP-Scholar", 0.9): (0.019, -0.003, 0.003, -0.003, 0.002),
    ("MMR", "Abt-Buy", 0.5): (0.160, 0.015, 0.012, 0.010, 0.009),
    ("MMR", "Abt-Buy", 0.6): (0.168, 0.015, 0.014, -0.004, 0.013),
    ("MMR", "Abt-Buy", 0.7): (0.164, -0.002, 0.008, 0.007, 0.009),
    ("MMR", "Abt-Buy", 0.8): (0.174, -0.007, 0.014, -0.008, 0.008),
    ("MMR", "Abt-Buy", 0.9): (0.158, 0.008, 0.018, 0.007, 0.016),
}


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Required upstream result is missing: {path}\n"
            "Run the full pipeline with: python reproduce.py"
        )
    return path


def clean_header(value: object) -> str:
    return (
        str(value)
        .replace("\ufeff", "")
        .replace("ï»¿", "")
        .replace('"', "")
        .strip()
    )


def read_latin1(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, encoding="latin-1")
    frame.columns = [clean_header(c) for c in frame.columns]
    return frame


def id_column(frame: pd.DataFrame) -> str:
    matches = [c for c in frame.columns if c.lower() == "id"]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one id column, got {matches}")
    return matches[0]


def dataset_summary() -> pd.DataFrame:
    rows = []
    configs = [
        (
            "DBLP-Scholar",
            DATA / "dblp" / "DBLP1.csv",
            DATA / "dblp" / "Scholar.csv",
            DATA / "dblp" / "DBLP-Scholar_perfectMapping.csv",
            "idDBLP",
            "idScholar",
        ),
        (
            "Abt-Buy",
            DATA / "abt_buy" / "Abt.csv",
            DATA / "abt_buy" / "Buy.csv",
            DATA / "abt_buy" / "abt_buy_perfectMapping.csv",
            "idAbt",
            "idBuy",
        ),
    ]
    for display, pa, pb, pm, ida, idb in configs:
        a = read_latin1(require(pa))
        b = read_latin1(require(pb))
        m = read_latin1(require(pm))
        aid = id_column(a)
        bid = id_column(b)
        a_ids = set(a[aid].fillna("").astype(str).str.strip())
        b_ids = set(b[bid].fillna("").astype(str).str.strip())
        m[ida] = m[ida].fillna("").astype(str).str.strip()
        m[idb] = m[idb].fillna("").astype(str).str.strip()
        valid = m[m[ida].isin(a_ids) & m[idb].isin(b_ids)].copy()
        partner_counts = valid.groupby(ida)[idb].nunique()
        rows.append(
            {
                "dataset": display,
                "nA": len(a),
                "nB": len(b),
                "gold_pairs": len(valid),
                "matchable_A": int(partner_counts.size),
                "matched_B": int(valid[idb].nunique()),
                "multi_partner_A": int((partner_counts > 1).sum()),
                "max_partners_A": int(partner_counts.max()),
            }
        )
    return pd.DataFrame(rows)


def with_zhang_rates(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    required = ["declared", "correct", "n_true_records"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing count columns needed for Zhang rates: {missing}")

    for column in required + ["false_links", "missing_matches"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")

    if "false_links" in frame.columns:
        bad = frame["declared"] != frame["correct"] + frame["false_links"]
        if bool(bad.any()):
            raise AssertionError("declared != correct + false_links in upstream results")

    frame["FLR"] = np.where(
        frame["declared"] > 0,
        1.0 - frame["correct"] / frame["declared"],
        np.nan,
    )
    frame["MMR"] = np.where(
        frame["n_true_records"] > 0,
        1.0 - frame["correct"] / frame["n_true_records"],
        np.nan,
    )
    return frame


def assert_three_seeds(frame: pd.DataFrame, group_cols: list[str]) -> None:
    counts = (
        frame.groupby(group_cols, dropna=False)["seed"]
        .nunique()
    )
    if not (counts == len(PRIMARY_SEEDS)).all():
        bad = counts[counts != len(PRIMARY_SEEDS)]
        raise AssertionError(f"Expected seeds 42-44 for every group:\n{bad}")


def mean_rates(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    frame = frame[frame["seed"].isin(PRIMARY_SEEDS)].copy()
    assert_three_seeds(frame, group_cols)
    grouped = (
        frame.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            n_splits=("seed", "nunique"),
            FLR=("FLR", "mean"),
            FLR_sd=("FLR", "std"),
            MMR=("MMR", "mean"),
            MMR_sd=("MMR", "std"),
        )
    )
    grouped["FLR_MCSE"] = grouped["FLR_sd"] / np.sqrt(grouped["n_splits"])
    grouped["MMR_MCSE"] = grouped["MMR_sd"] / np.sqrt(grouped["n_splits"])
    return grouped


def r3(value: float) -> float:
    return round(float(value) + 0.0, 3)


def check_pair(
    checks: list[dict[str, object]],
    table: str,
    key: object,
    actual_flr: float,
    actual_mmr: float,
    expected: tuple[float, float],
) -> None:
    af, am = r3(actual_flr), r3(actual_mmr)
    ef, em = expected
    ok = af == ef and am == em
    checks.append(
        {
            "table": table,
            "key": repr(key),
            "actual": f"{af:.3f}/{am:.3f}",
            "expected": f"{ef:.3f}/{em:.3f}",
            "pass": ok,
        }
    )


def table4(checks: list[dict[str, object]]) -> pd.DataFrame:
    """Build Table 4 directly from the preserved Analysis 15 outputs.

    The blocking-floor calculation is kept as scientific code rather than
    replaced by manuscript constants.  Consequently this stage records Table 4
    as an informational provenance check and does not hard-code expected values.
    """
    floor = pd.read_csv(require(RESULTS / "analysis15_rank_mmr_by_k.csv"))
    summary = pd.read_csv(require(RESULTS / "analysis15_rank_summary.csv"))
    rows = []
    for raw_name in ("DBLP", "ECOM"):
        display = DATASET_DISPLAY[raw_name]
        row = {"dataset": display}
        for k in (1, 5, 50):
            selected = floor[(floor["dataset"] == raw_name) & (floor["k"] == k)]
            if len(selected) != 1:
                raise AssertionError(f"Missing/duplicate floor row: {raw_name}, k={k}")
            value = float(selected.iloc[0]["blocking_mmr"])
            row[f"k{k}_MMR"] = value
            checks.append(
                {
                    "table": "Table 4",
                    "key": repr((display, k)),
                    "actual": f"{r3(value):.3f}",
                    "expected": "generated by analysis15_rank_statistic.py",
                    "pass": True,
                }
            )
        selected_summary = summary[summary["dataset"] == raw_name]
        if len(selected_summary) != 1:
            raise AssertionError(f"Missing/duplicate rank summary: {raw_name}")
        floor_zero = int(selected_summary.iloc[0]["max_best_rank"])
        row["floor_zero_k"] = floor_zero
        checks.append(
            {
                "table": "Table 4",
                "key": repr((display, "floor_zero_k")),
                "actual": str(floor_zero),
                "expected": "generated by analysis15_rank_statistic.py",
                "pass": True,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def table5(checks: list[dict[str, object]]) -> pd.DataFrame:
    source = pd.read_csv(require(RESULTS / "analysis16_clean_2x2_perseed.csv"))
    source = source[
        source["column"].isin(["k=5", "k=50"])
        & source["side"].isin(["string", "bi-encoder"])
        & source["scorer"].isin(["MEC", "LR"])
    ].copy()
    source = with_zhang_rates(source)
    source["dataset_display"] = source["dataset"].map(DATASET_DISPLAY)
    source["comparison"] = source["side"].map(
        {"string": "String, Jaro-Winkler", "bi-encoder": "Bi-encoder, cosine"}
    )
    source["model_display"] = source["scorer"].map(
        {"MEC": "Generative, kernel density", "LR": "Discriminative, logistic"}
    )
    source["k"] = source["column"].str.extract(r"(\d+)").astype(int)
    agg = mean_rates(
        source,
        ["dataset_display", "comparison", "model_display", "k"],
    )

    rows = []
    for (dataset, comparison, model), group in agg.groupby(
        ["dataset_display", "comparison", "model_display"],
        sort=False,
    ):
        out = {
            "dataset": dataset,
            "key_value_comparison": comparison,
            "scoring_classification": model,
        }
        for k in (5, 50):
            selected = group[group["k"] == k]
            if len(selected) != 1:
                raise AssertionError(f"Missing Table 5 row for {(dataset, comparison, model, k)}")
            flr = float(selected.iloc[0]["FLR"])
            mmr = float(selected.iloc[0]["MMR"])
            out[f"k{k}_FLR"] = flr
            out[f"k{k}_FLR_MCSE"] = float(selected.iloc[0]["FLR_MCSE"])
            out[f"k{k}_MMR"] = mmr
            out[f"k{k}_MMR_MCSE"] = float(selected.iloc[0]["MMR_MCSE"])
            check_pair(
                checks,
                "Table 5",
                (dataset, comparison, model, k),
                flr,
                mmr,
                EXPECTED_TABLE5[(dataset, comparison, model, k)],
            )
        rows.append(out)
    return pd.DataFrame(rows)


def table6(checks: list[dict[str, object]]) -> pd.DataFrame:
    source = pd.read_csv(
        require(RESULTS / "analysis20_discriminative_scorers_perseed.csv")
    )
    source = source[
        source["side"].eq("bi-encoder")
        & source["column"].isin(["k=5", "k=50"])
        & source["classifier"].isin(["LR", "SVM", "XGB", "MLP"])
    ].copy()
    source = with_zhang_rates(source)
    source["dataset_display"] = source["dataset"].map(DATASET_DISPLAY)
    source["model"] = source["classifier"].replace({"LR": "Logistic"})
    source["k"] = source["column"].str.extract(r"(\d+)").astype(int)
    agg = mean_rates(source, ["model", "dataset_display", "k"])

    rows = []
    for model in ["Logistic", "SVM", "XGB", "MLP"]:
        out = {"model": model}
        for dataset in ["DBLP-Scholar", "Abt-Buy"]:
            for k in (5, 50):
                selected = agg[
                    (agg["model"] == model)
                    & (agg["dataset_display"] == dataset)
                    & (agg["k"] == k)
                ]
                if len(selected) != 1:
                    raise AssertionError(f"Missing Table 6 row for {(model, dataset, k)}")
                flr = float(selected.iloc[0]["FLR"])
                mmr = float(selected.iloc[0]["MMR"])
                slug = "DBLP" if dataset == "DBLP-Scholar" else "Abt"
                out[f"{slug}_k{k}_FLR"] = flr
                out[f"{slug}_k{k}_FLR_MCSE"] = float(selected.iloc[0]["FLR_MCSE"])
                out[f"{slug}_k{k}_MMR"] = mmr
                out[f"{slug}_k{k}_MMR_MCSE"] = float(selected.iloc[0]["MMR_MCSE"])
                check_pair(
                    checks,
                    "Table 6",
                    (model, dataset, k),
                    flr,
                    mmr,
                    EXPECTED_TABLE6[(model, dataset, k)],
                )
        rows.append(out)
    return pd.DataFrame(rows)


def table7(checks: list[dict[str, object]]) -> pd.DataFrame:
    two_step = pd.read_csv(require(RESULTS / "analysis17_table5_perseed.csv"))
    two_step = with_zhang_rates(two_step)
    two_step["dataset_display"] = two_step["dataset"].map(DATASET_DISPLAY)
    two_step["model_display"] = two_step["scorer"].map(
        {"MEC": "Generative, kernel density", "LR": "Discriminative, logistic"}
    )
    two_step["state"] = two_step["encoder"].map(
        {"pre-trained": "pretrained", "fine-tuned": "fine_tuned"}
    )
    two_agg = mean_rates(
        two_step,
        ["dataset_display", "model_display", "state"],
    )

    pre_ce = pd.read_csv(require(RESULTS / "analysis17_pretrained_ce_perseed.csv"))
    pre_ce = with_zhang_rates(pre_ce)
    pre_ce["dataset_display"] = pre_ce["dataset"].map(DATASET_DISPLAY)
    pre_ce["model_display"] = "Cross-encoder"
    pre_ce["state"] = "pretrained"
    pre_ce_agg = mean_rates(
        pre_ce,
        ["dataset_display", "model_display", "state"],
    )

    ft_ce = pd.read_csv(require(RESULTS / "analysis17_ce_perseed.csv"))
    ft_ce = ft_ce[ft_ce["model"] == "CE_direct"].copy()
    ft_ce = with_zhang_rates(ft_ce)
    ft_ce["dataset_display"] = ft_ce["dataset"].map(DATASET_DISPLAY)
    ft_ce["model_display"] = "Cross-encoder"
    ft_ce["state"] = "fine_tuned"
    ft_ce_agg = mean_rates(
        ft_ce,
        ["dataset_display", "model_display", "state"],
    )

    combined = pd.concat([two_agg, pre_ce_agg, ft_ce_agg], ignore_index=True)
    rows = []
    for dataset in ["DBLP-Scholar", "Abt-Buy"]:
        for model in [
            "Generative, kernel density",
            "Discriminative, logistic",
            "Cross-encoder",
        ]:
            out = {"dataset": dataset, "model": model}
            for state in ["pretrained", "fine_tuned"]:
                selected = combined[
                    (combined["dataset_display"] == dataset)
                    & (combined["model_display"] == model)
                    & (combined["state"] == state)
                ]
                if len(selected) != 1:
                    raise AssertionError(f"Missing Table 7 row for {(dataset, model, state)}")
                flr = float(selected.iloc[0]["FLR"])
                mmr = float(selected.iloc[0]["MMR"])
                out[f"{state}_FLR"] = flr
                out[f"{state}_FLR_MCSE"] = float(selected.iloc[0]["FLR_MCSE"])
                out[f"{state}_MMR"] = mmr
                out[f"{state}_MMR_MCSE"] = float(selected.iloc[0]["MMR_MCSE"])
                check_pair(
                    checks,
                    "Table 7",
                    (dataset, model, state),
                    flr,
                    mmr,
                    EXPECTED_TABLE7[(dataset, model, state)],
                )
            rows.append(out)
    return pd.DataFrame(rows)


def ce_blocksize(checks: list[dict[str, object]]) -> pd.DataFrame:
    source = pd.read_csv(
        require(RESULTS / "analysis19_ce_blocksize_combined_perseed.csv")
    )
    source = source[source["column"].isin(["k=5", "k=50"])].copy()
    source = with_zhang_rates(source)
    source["dataset_display"] = source["dataset"].map(DATASET_DISPLAY)
    source["k"] = source["column"].str.extract(r"(\d+)").astype(int)
    agg = mean_rates(source, ["dataset_display", "k"])

    for row in agg.itertuples(index=False):
        key = (str(row.dataset_display), int(row.k))
        if key in EXPECTED_CE_BLOCK:
            check_pair(
                checks,
                "Section 4.1 CE block-size sensitivity",
                key,
                float(row.FLR),
                float(row.MMR),
                EXPECTED_CE_BLOCK[key],
            )
    return agg.rename(columns={"dataset_display": "dataset"})


def table8(checks: list[dict[str, object]]) -> pd.DataFrame:
    source = pd.read_csv(
        require(RESULTS / "analysis17_table7_ftlr_primary3.csv")
    )
    required = {"dataset", "quartile", "model", "FLR_mean", "MMR_mean"}
    if not required.issubset(source.columns):
        raise ValueError(f"Unexpected Table 8 source columns: {source.columns.tolist()}")
    source = source.copy()
    source["dataset"] = source["dataset"].map(DATASET_DISPLAY).fillna(source["dataset"])

    for row in source.itertuples(index=False):
        key = (str(row.dataset), str(row.quartile), str(row.model))
        if key in EXPECTED_TABLE8:
            check_pair(
                checks,
                "Table 8",
                key,
                float(row.FLR_mean),
                float(row.MMR_mean),
                EXPECTED_TABLE8[key],
            )
    if sum(c["table"] == "Table 8" for c in checks) != len(EXPECTED_TABLE8):
        raise AssertionError("Table 8 verification did not cover every manuscript row.")
    return source


def table9(checks: list[dict[str, object]]) -> pd.DataFrame:
    source = pd.read_csv(
        require(RESULTS / "analysis18_fraction_sweep_direct_mc10.csv")
    ).copy()
    source["dataset"] = source["dataset"].map(DATASET_DISPLAY).fillna(source["dataset"])
    source["RMCSE_FLR"] = source["FLR_se"] / source["FLR_mean"].abs()
    source["RMCSE_MMR"] = source["MMR_se"] / source["MMR_mean"].abs()

    keep = source[
        [
            "dataset",
            "fraction",
            "FLR_mean",
            "FLR_se",
            "RMCSE_FLR",
            "MMR_mean",
            "MMR_se",
            "RMCSE_MMR",
        ]
    ].copy()

    for row in keep.itertuples(index=False):
        key = (str(row.dataset), round(float(row.fraction), 1))
        expected = EXPECTED_TABLE9[key]
        actual = (
            r3(row.FLR_mean),
            r3(row.FLR_se),
            r3(row.RMCSE_FLR),
            r3(row.MMR_mean),
            r3(row.MMR_se),
            r3(row.RMCSE_MMR),
        )
        ok = actual == expected
        checks.append(
            {
                "table": "Table 9",
                "key": repr(key),
                "actual": "/".join(f"{v:.3f}" for v in actual),
                "expected": "/".join(f"{v:.3f}" for v in expected),
                "pass": ok,
            }
        )
    return keep


def table10(checks: list[dict[str, object]]) -> pd.DataFrame:
    source = pd.read_csv(
        require(RESULTS / "analysis22_section332_bias_direct_mc10.csv")
    ).copy()
    source["dataset"] = source["dataset"].map(DATASET_DISPLAY).fillna(source["dataset"])

    rows = []
    for metric in ("FLR", "MMR"):
        for (dataset, fraction), group in source.groupby(["dataset", "f"], sort=True):
            srs = group[group["scheme"] == "SRSWOR_65_35"]
            boot = group[group["scheme"] == "bootstrap_OOB"]
            if len(srs) != 1 or len(boot) != 1:
                raise AssertionError(
                    f"Need one SRSWOR and one bootstrap row for {(dataset, fraction)}"
                )
            target = float(srs.iloc[0][f"target_{metric}_mean"])
            if not math.isclose(
                target,
                float(boot.iloc[0][f"target_{metric}_mean"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise AssertionError("SRSWOR/bootstrap target rates differ.")
            row = {
                "metric": metric,
                "dataset": dataset,
                "f": float(fraction),
                "target": target,
                "SRSWOR_bias": float(srs.iloc[0][f"bias_{metric}_mean"]),
                "SRSWOR_SE": float(srs.iloc[0][f"bias_{metric}_SE"]),
                "bootstrap_bias": float(boot.iloc[0][f"bias_{metric}_mean"]),
                "bootstrap_SE": float(boot.iloc[0][f"bias_{metric}_SE"]),
            }
            rows.append(row)
            key = (metric, str(dataset), round(float(fraction), 1))
            expected = EXPECTED_TABLE10[key]
            actual = tuple(
                r3(row[c])
                for c in [
                    "target",
                    "SRSWOR_bias",
                    "SRSWOR_SE",
                    "bootstrap_bias",
                    "bootstrap_SE",
                ]
            )
            checks.append(
                {
                    "table": "Table 10",
                    "key": repr(key),
                    "actual": "/".join(f"{v:.3f}" for v in actual),
                    "expected": "/".join(f"{v:.3f}" for v in expected),
                    "pass": actual == expected,
                }
            )
    return pd.DataFrame(rows)


def verify_dataset_summary(
    frame: pd.DataFrame,
    checks: list[dict[str, object]],
) -> None:
    for row in frame.to_dict("records"):
        expected = EXPECTED_DATASETS[row["dataset"]]
        for field, value in expected.items():
            actual = int(row[field])
            checks.append(
                {
                    "table": "Table 2 / dataset audit",
                    "key": repr((row["dataset"], field)),
                    "actual": str(actual),
                    "expected": str(value),
                    "pass": actual == int(value),
                }
            )


def main() -> None:
    checks: list[dict[str, object]] = []

    ds = dataset_summary()
    verify_dataset_summary(ds, checks)
    ds.to_csv(OUT / "table2_dataset_summary.csv", index=False)

    outputs = {
        "table4_floor_mmr.csv": table4(checks),
        "table5_representation_scoring.csv": table5(checks),
        "table6_discriminative_sensitivity.csv": table6(checks),
        "table7_pretrained_finetuned.csv": table7(checks),
        "section41_ce_blocksize_sensitivity.csv": ce_blocksize(checks),
        "table8_discrimination_gap.csv": table8(checks),
        "table9_training_fraction.csv": table9(checks),
        "table10_subsample_bias.csv": table10(checks),
    }

    for filename, frame in outputs.items():
        frame.to_csv(OUT / filename, index=False)

    verification = pd.DataFrame(checks)
    verification.to_csv(OUT / "paper_alignment_verification.csv", index=False)

    failed = verification[~verification["pass"]]
    manifest = {
        "paper": "Text-based Entity Resolution Using Transformer Models",
        "authors": [
            "Abrar Ahmed",
            "Li-Chun Zhang",
            "Ralf Münnich",
            "Martin Vogt",
        ],
        "metric_definition": {
            "FLR": "1 - correct / declared",
            "MMR": "1 - correct / n_true_records",
            "legacy_psi": (
                "missing_matches / n_true_records; retained in recovered "
                "intermediate scripts for provenance only; not manuscript MMR"
            ),
        },
        "primary_seeds": PRIMARY_SEEDS,
        "verification_rows": int(len(verification)),
        "failed_rows": int(len(failed)),
        "status": "PASS" if failed.empty else "FAIL",
        "note": (
            "Checks compare values rounded to the precision printed in the "
            "LaTeX manuscript, except Table 4, which is generated directly "
            "from the preserved Analysis 15 scientific code and recorded as "
            "an informational provenance check rather than a hard-coded "
            "manuscript assertion. Table 3 is an illustrative worked example "
            "and is not regenerated by the benchmark experiment pipeline."
        ),
    }
    with (OUT / "paper_alignment_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print("=" * 88)
    print("PAPER-FACING TABLE RECONSTRUCTION")
    print("=" * 88)
    print(f"Output directory : {OUT}")
    print(f"Verification rows: {len(verification)}")
    print(f"Failures         : {len(failed)}")
    print("Metric rule      : FLR=1-C/D; MMR=1-C/M (Zhang definition)")
    if not failed.empty:
        print("\nFAILED ALIGNMENT CHECKS")
        print(failed.to_string(index=False))
        raise SystemExit(1)
    print("\nPASS: paper-facing checks passed. Table 4 is generated directly from Analysis 15 and is informational rather than hard-coded.")


if __name__ == "__main__":
    main()
