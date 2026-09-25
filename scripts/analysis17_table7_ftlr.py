"""
analysis17_table7_ftlr.py
=========================

Final-paper discrimination-gap / routing analysis.

Historical note
---------------
This analysis was originally created after analysis17_section5.py and was
referred to as "corrected Table 7". In the final manuscript it supplies the
record-difficulty / discrimination-gap table.

The script reproduces the following design:

1. Use the primary outer splits, seeds 42, 43, 44.
2. Load the fine-tuned bi-encoder produced by analysis17_section5.py.
3. Re-embed the complete A and B files with that fine-tuned encoder.
4. Rebuild the fine-tuned top-50 block only as a fidelity check against
   the fine-tuned two-step result.
5. For the FINAL difficulty analysis, keep the candidate block fixed at
   the pretrained bi-encoder top-50 block.
6. Refit logistic regression using fine-tuned bi-encoder similarities
   over that fixed block.
7. Define the discrimination gap as

       top1 probability - top2 probability.

8. Divide the test records into within-split quartiles using that gap:
       Q1 = hardest / smallest gaps,
       Q4 = easiest / largest gaps.
9. Evaluate the fine-tuned LR and the already-produced direct-score
   fine-tuned cross-encoder decisions on the SAME records.
10. Report final-paper Zhang metrics:

       FLR = 1 - correct / declared
       MMR = 1 - correct / matchable source records

The historical missing-only quantity

       missing_matches / matchable source records

is retained only as ``legacy_psi`` and is NOT labelled MMR.

This script does not train either transformer model. It requires the
fine-tuned bi-encoder checkpoints and Analysis 17 cross-encoder prediction
files produced by:

    python scripts/analysis17_section5.py --phase all --primary-only

Outputs
-------
results/analysis17_table7_ftlr/
    reblocked_<DATASET>_seed<SEED>.csv
    fixedblock_<DATASET>_seed<SEED>.csv

results/analysis17_table7_ftlr_predictions.csv
results/analysis17_table7_ftlr_perseed.csv
results/analysis17_table7_ftlr_primary3.csv
results/analysis17_table7_ftlr_verification.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

import analysis17_section5 as a17


PRIMARY_SEEDS = [42, 43, 44]
VALID_DATASETS = ["DBLP", "ECOM"]

STATE_DIR = a17.RESULTS_DIR / "analysis17_table7_ftlr"
STATE_DIR.mkdir(parents=True, exist_ok=True)

PERSEED_PATH = (
    a17.RESULTS_DIR / "analysis17_table7_ftlr_perseed.csv"
)
PREDICTIONS_PATH = (
    a17.RESULTS_DIR / "analysis17_table7_ftlr_predictions.csv"
)
PRIMARY3_PATH = (
    a17.RESULTS_DIR / "analysis17_table7_ftlr_primary3.csv"
)
VERIFICATION_PATH = (
    a17.RESULTS_DIR / "analysis17_table7_ftlr_verification.csv"
)


# Final manuscript values, rounded to the three decimals reported in
# the paper. These are verification targets, not inputs to computation.
EXPECTED_TABLE = {
    ("DBLP", "Q1", "Logistic regression"): (0.037, 0.045),
    ("DBLP", "Q1", "Cross-encoder"):       (0.033, 0.029),
    ("DBLP", "Q2", "Logistic regression"): (0.031, 0.063),
    ("DBLP", "Q2", "Cross-encoder"):       (0.034, 0.022),
    ("DBLP", "Q3", "Logistic regression"): (0.006, 0.041),
    ("DBLP", "Q3", "Cross-encoder"):       (0.008, 0.010),
    ("DBLP", "Q4", "Logistic regression"): (0.000, 0.000),
    ("DBLP", "Q4", "Cross-encoder"):       (0.000, 0.000),

    ("ECOM", "Q1", "Logistic regression"): (0.230, 0.706),
    ("ECOM", "Q1", "Cross-encoder"):       (0.121, 0.400),
    ("ECOM", "Q2", "Logistic regression"): (0.101, 0.259),
    ("ECOM", "Q2", "Cross-encoder"):       (0.051, 0.131),
    ("ECOM", "Q3", "Logistic regression"): (0.017, 0.017),
    ("ECOM", "Q3", "Cross-encoder"):       (0.016, 0.072),
    ("ECOM", "Q4", "Logistic regression"): (0.000, 0.000),
    ("ECOM", "Q4", "Cross-encoder"):       (0.000, 0.005),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Final discrimination-gap analysis using fine-tuned "
            "bi-encoder LR on fixed pretrained top-50 blocks."
        )
    )
    parser.add_argument(
        "--datasets",
        default="DBLP,ECOM",
        help="Comma-separated subset of DBLP,ECOM.",
    )
    return parser.parse_args()


def to_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    values = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    unknown = sorted(
        set(values.unique()).difference(mapping)
    )

    if unknown:
        raise ValueError(
            f"Could not interpret boolean values: {unknown}"
        )

    return values.map(mapping).astype(bool)


def metrics_from_rows(
    frame: pd.DataFrame,
    *,
    declared_column: str,
    category_column: str,
) -> dict[str, float | int]:

    declared_series = to_bool_series(
        frame[declared_column]
    )
    truth_series = to_bool_series(
        frame["has_true_match"]
    )

    declared = int(declared_series.sum())
    correct = int(
        (frame[category_column] == "correct").sum()
    )
    false_links = int(
        (frame[category_column] == "false_link").sum()
    )
    missing_matches = int(
        (frame[category_column] == "missing_match").sum()
    )
    unmatched_abstentions = int(
        (
            frame[category_column]
            == "unmatched_abstention"
        ).sum()
    )
    n_true_records = int(truth_series.sum())

    if declared != correct + false_links:
        raise AssertionError(
            "Declared links do not equal correct + false links."
        )

    if len(frame) != (
        correct
        + false_links
        + missing_matches
        + unmatched_abstentions
    ):
        raise AssertionError(
            "Prediction categories are not exhaustive."
        )

    flr = (
        1.0 - correct / declared
        if declared
        else 0.0
    )

    mmr = (
        1.0 - correct / n_true_records
        if n_true_records
        else 0.0
    )

    legacy_psi = (
        missing_matches / n_true_records
        if n_true_records
        else 0.0
    )

    return {
        "FLR": float(flr),
        "MMR": float(mmr),
        "legacy_psi": float(legacy_psi),
        "declared": declared,
        "correct": correct,
        "false_links": false_links,
        "missing_matches": missing_matches,
        "unmatched_abstentions": unmatched_abstentions,
        "n_true_records": n_true_records,
    }


def prediction_rows(
    dataset: dict[str, Any],
    test_structure: list[
        tuple[str, set[str], list[str]]
    ],
    lr_model: Any,
    threshold: float,
    emb_a: dict[str, torch.Tensor],
    emb_b: dict[str, torch.Tensor],
    seed: int,
) -> pd.DataFrame:

    fields = dataset["cfg"]["fields"]

    rows: list[dict[str, Any]] = []

    for record_index, (
        ida,
        true_set,
        candidates,
    ) in enumerate(test_structure):

        features = a17.cosine_feature_matrix(
            [(ida, idb) for idb in candidates],
            emb_a,
            emb_b,
            fields,
            dataset["a_pos"],
            dataset["b_pos"],
        )

        probabilities = np.asarray(
            lr_model.predict_proba(features)[:, 1],
            dtype=float,
        )

        if len(probabilities) < 2:
            raise AssertionError(
                "Difficulty analysis requires at least two candidates."
            )

        order = np.argsort(probabilities)[::-1]

        top1_index = int(order[0])
        top2_index = int(order[1])

        top1_score = float(
            probabilities[top1_index]
        )
        top2_score = float(
            probabilities[top2_index]
        )

        top1_idb = str(
            candidates[top1_index]
        )

        declared = bool(
            top1_score > threshold
        )

        rows.append(
            {
                "record_index": record_index,
                "ida": str(ida),
                "has_true_match": bool(true_set),
                "n_valid_matches": len(true_set),
                "ft_lr_top1_idb": top1_idb,
                "ft_lr_top1_score": top1_score,
                "ft_lr_top2_score": top2_score,
                "ft_lr_gap": (
                    top1_score - top2_score
                ),
                "ft_lr_threshold": float(
                    threshold
                ),
                "ft_lr_declared": declared,
                "ft_lr_category": (
                    a17.category_for(
                        declared,
                        top1_idb,
                        true_set,
                    )
                ),
                "dataset": dataset["name"],
                "seed": int(seed),
            }
        )

    return pd.DataFrame(rows)


def fit_lr_for_block(
    dataset: dict[str, Any],
    ids_train: list[str],
    ids_test: list[str],
    top50: dict[str, list[str]],
    emb_a: dict[str, torch.Tensor],
    emb_b: dict[str, torch.Tensor],
    seed: int,
) -> tuple[
    pd.DataFrame,
    dict[str, float | int],
    int,
    float,
    float,
]:

    train_pairs = a17.build_train_pairs(
        ids_train,
        dataset["retained"],
        dataset["truth"],
        top50,
    )

    pair_ids = [
        (ida, idb)
        for ida, idb, _ in train_pairs
    ]

    fields = dataset["cfg"]["fields"]

    x_train = a17.cosine_feature_matrix(
        pair_ids,
        emb_a,
        emb_b,
        fields,
        dataset["a_pos"],
        dataset["b_pos"],
    )

    y_train = np.asarray(
        [
            label
            for _, _, label in train_pairs
        ],
        dtype=int,
    )

    if set(np.unique(y_train)) != {0, 1}:
        raise AssertionError(
            "LR training pairs do not contain both classes."
        )

    lr_model, threshold, validation_f1 = (
        a17.calibrate_lr(
            x_train,
            y_train,
            seed,
        )
    )

    test_structure = a17.build_test_structure(
        ids_test,
        dataset["truth"],
        top50,
    )

    rows = prediction_rows(
        dataset,
        test_structure,
        lr_model,
        threshold,
        emb_a,
        emb_b,
        seed,
    )

    metrics = metrics_from_rows(
        rows,
        declared_column="ft_lr_declared",
        category_column="ft_lr_category",
    )

    return (
        rows,
        metrics,
        len(train_pairs),
        float(threshold),
        float(validation_f1),
    )


def require_ft_model(
    dataset_name: str,
    seed: int,
    device: str,
) -> SentenceTransformer:

    path = a17.ft_model_path(
        dataset_name,
        seed,
    )

    if not a17.model_complete(path):
        raise FileNotFoundError(
            "Required fine-tuned bi-encoder is missing:\n"
            f"  {path}\n\n"
            "Run analysis17_section5.py --phase all "
            "--primary-only first."
        )

    print(f"  loading FT model: {path}")

    return SentenceTransformer(
        str(path),
        device=device,
    )


def require_ce_predictions(
    dataset_name: str,
    seed: int,
) -> pd.DataFrame:

    path = a17.prediction_part_path(
        dataset_name,
        seed,
    )

    if not path.exists():
        raise FileNotFoundError(
            "Required Analysis 17 cross-encoder "
            "prediction file is missing:\n"
            f"  {path}\n\n"
            "Run analysis17_section5.py --phase all "
            "--primary-only first."
        )

    frame = pd.read_csv(
        path,
        dtype={"ida": str},
    )

    required = {
        "ida",
        "ce_direct_declared",
        "ce_direct_category",
    }

    missing = required.difference(
        frame.columns
    )

    if missing:
        raise KeyError(
            f"{path} lacks required direct-score "
            f"columns: {sorted(missing)}"
        )

    frame = frame[
        [
            "ida",
            "ce_direct_declared",
            "ce_direct_category",
        ]
    ].copy()

    frame["ida"] = (
        frame["ida"]
        .astype(str)
    )

    frame["ce_direct_declared"] = (
        to_bool_series(
            frame["ce_direct_declared"]
        )
    )

    if frame["ida"].duplicated().any():
        raise AssertionError(
            f"Duplicate source IDs in {path}"
        )

    return frame


def crosscheck_reblocked(
    dataset_name: str,
    seed: int,
    metrics: dict[str, float | int],
) -> None:

    cell_path = a17.ft_cell_path(
        dataset_name,
        seed,
    )

    if not cell_path.exists():
        raise FileNotFoundError(
            "Analysis 17 fine-tuned result cell "
            f"is missing: {cell_path}"
        )

    payload = a17.read_json(
        cell_path
    )

    reference = payload["LR"]

    count_fields = [
        "declared",
        "correct",
        "false_links",
        "missing_matches",
        "n_true_records",
    ]

    mismatches = []

    for field in count_fields:
        observed = int(metrics[field])
        expected = int(reference[field])

        if observed != expected:
            mismatches.append(
                f"{field}: observed={observed}, "
                f"expected={expected}"
            )

    if mismatches:
        raise AssertionError(
            "Reblocked fine-tuned LR check failed:\n  "
            + "\n  ".join(mismatches)
        )


def run_seed(
    dataset: dict[str, Any],
    seed: int,
    device: str,
) -> pd.DataFrame:

    print(
        f"\n{dataset['name']} seed={seed}"
    )
    print("-" * 60)

    a17.seed_everything(seed)

    ids_train, ids_test = (
        train_test_split(
            dataset["a_ids"],
            test_size=0.50,
            random_state=seed,
        )
    )

    ids_train = list(ids_train)
    ids_test = list(ids_test)

    ft_model = require_ft_model(
        dataset["name"],
        seed,
        device,
    )

    fields = dataset["cfg"]["fields"]

    print(
        "  encoding fine-tuned per-field embeddings..."
    )

    ft_emb_a = a17.encode_fields(
        ft_model,
        dataset["df_a"],
        fields,
        "FT-A",
    )

    ft_emb_b = a17.encode_fields(
        ft_model,
        dataset["df_b"],
        fields,
        "FT-B",
    )

    # --------------------------------------------------------------
    # Historical fidelity check:
    # rebuild the candidate block using the adapted encoder and ensure
    # it reproduces the Analysis 17 fine-tuned LR cell.
    # --------------------------------------------------------------
    print(
        "  rebuilding FT top-50 for "
        "fine-tuned-cell verification..."
    )

    ft_top50 = a17.retrieve_top50(
        ft_emb_a[
            dataset["cfg"]["retrieval_field"]
        ],
        ft_emb_b[
            dataset["cfg"]["retrieval_field"]
        ],
        dataset["a_ids"],
        dataset["b_ids"],
    )

    (
        reblocked_rows,
        reblocked_metrics,
        reblocked_pairs,
        reblocked_threshold,
        _reblocked_validation_f1,
    ) = fit_lr_for_block(
        dataset,
        ids_train,
        ids_test,
        ft_top50,
        ft_emb_a,
        ft_emb_b,
        seed,
    )

    crosscheck_reblocked(
        dataset["name"],
        seed,
        reblocked_metrics,
    )

    reblocked_path = STATE_DIR / (
        f"reblocked_{dataset['name']}_"
        f"seed{seed}.csv"
    )

    a17.atomic_write_csv(
        reblocked_rows,
        reblocked_path,
    )

    print(
        "  REBLOCKED CHECK PASS: "
        f"FLR={reblocked_metrics['FLR']:.6f}, "
        f"MMR={reblocked_metrics['MMR']:.6f}, "
        f"legacy psi="
        f"{reblocked_metrics['legacy_psi']:.6f}, "
        f"threshold={reblocked_threshold:.2f}, "
        f"pairs={reblocked_pairs:,}"
    )

    # --------------------------------------------------------------
    # FINAL PAPER DIFFICULTY MODEL:
    # keep pretrained bi-encoder top-50 candidates fixed, but calculate
    # LR features using fine-tuned bi-encoder embeddings.
    # --------------------------------------------------------------
    (
        fixed_rows,
        fixed_metrics,
        fixed_pairs,
        fixed_threshold,
        fixed_validation_f1,
    ) = fit_lr_for_block(
        dataset,
        ids_train,
        ids_test,
        dataset["base_top50"],
        ft_emb_a,
        ft_emb_b,
        seed,
    )

    ce_rows = require_ce_predictions(
        dataset["name"],
        seed,
    )

    combined = fixed_rows.merge(
        ce_rows,
        on="ida",
        how="left",
        validate="one_to_one",
    )

    if len(combined) != len(fixed_rows):
        raise AssertionError(
            "Cross-encoder merge changed the number "
            "of test records."
        )

    if combined[
        "ce_direct_category"
    ].isna().any():
        raise AssertionError(
            "Cross-encoder predictions are missing "
            "for some fixed-block test records."
        )

    combined["quartile"] = pd.qcut(
        combined["ft_lr_gap"],
        4,
        labels=[
            "Q1",
            "Q2",
            "Q3",
            "Q4",
        ],
        duplicates="drop",
    )

    if (
        combined["quartile"]
        .nunique(dropna=True)
        != 4
    ):
        raise AssertionError(
            f"{dataset['name']} seed={seed}: "
            "qcut did not produce four quartiles."
        )

    fixed_path = STATE_DIR / (
        f"fixedblock_{dataset['name']}_"
        f"seed{seed}.csv"
    )

    a17.atomic_write_csv(
        combined,
        fixed_path,
    )

    print(
        "  FIXED-BLOCK FT-LR: "
        f"FLR={fixed_metrics['FLR']:.6f}, "
        f"MMR={fixed_metrics['MMR']:.6f}, "
        f"legacy psi={fixed_metrics['legacy_psi']:.6f}, "
        f"threshold={fixed_threshold:.2f}, "
        f"validation F1={fixed_validation_f1:.4f}, "
        f"pairs={fixed_pairs:,}"
    )

    del ft_model
    del ft_emb_a
    del ft_emb_b

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return combined


def quartile_rows(
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:

    rows: list[dict[str, Any]] = []

    for quartile in [
        "Q1",
        "Q2",
        "Q3",
        "Q4",
    ]:
        subset = frame[
            frame["quartile"].astype(str)
            == quartile
        ].copy()

        if subset.empty:
            raise AssertionError(
                f"Empty quartile: {quartile}"
            )

        lr = metrics_from_rows(
            subset,
            declared_column="ft_lr_declared",
            category_column="ft_lr_category",
        )

        ce = metrics_from_rows(
            subset,
            declared_column="ce_direct_declared",
            category_column="ce_direct_category",
        )

        common = {
            "dataset": str(
                subset["dataset"].iloc[0]
            ),
            "seed": int(
                subset["seed"].iloc[0]
            ),
            "quartile": quartile,
            "n_records": len(subset),
        }

        rows.append(
            {
                **common,
                "model": "Logistic regression",
                **lr,
            }
        )

        rows.append(
            {
                **common,
                "model": "Cross-encoder",
                **ce,
            }
        )

    return rows


def aggregate_primary(
    perseed: pd.DataFrame,
) -> pd.DataFrame:

    aggregate = (
        perseed.groupby(
            [
                "dataset",
                "quartile",
                "model",
            ],
            sort=True,
        )
        .agg(
            n_splits=("seed", "nunique"),
            FLR_mean=("FLR", "mean"),
            FLR_sd=("FLR", "std"),
            MMR_mean=("MMR", "mean"),
            MMR_sd=("MMR", "std"),
            legacy_psi_mean=(
                "legacy_psi",
                "mean",
            ),
            declared_mean=("declared", "mean"),
            correct_mean=("correct", "mean"),
            false_links_mean=(
                "false_links",
                "mean",
            ),
            missing_matches_mean=(
                "missing_matches",
                "mean",
            ),
            n_true_records_mean=(
                "n_true_records",
                "mean",
            ),
        )
        .reset_index()
    )

    aggregate["FLR_se"] = (
        aggregate["FLR_sd"]
        / np.sqrt(
            aggregate["n_splits"]
        )
    )

    aggregate["MMR_se"] = (
        aggregate["MMR_sd"]
        / np.sqrt(
            aggregate["n_splits"]
        )
    )

    if not (
        aggregate["n_splits"]
        == len(PRIMARY_SEEDS)
    ).all():
        raise AssertionError(
            "Primary table does not contain "
            "all three seeds in every row."
        )

    return aggregate


def verify_final_table(
    aggregate: pd.DataFrame,
) -> pd.DataFrame:

    rows: list[dict[str, Any]] = []

    for row in aggregate.itertuples(
        index=False
    ):
        key = (
            str(row.dataset),
            str(row.quartile),
            str(row.model),
        )

        if key not in EXPECTED_TABLE:
            continue

        expected_flr, expected_mmr = (
            EXPECTED_TABLE[key]
        )

        observed_flr = float(
            row.FLR_mean
        )
        observed_mmr = float(
            row.MMR_mean
        )

        flr_pass = (
            round(observed_flr, 3)
            == expected_flr
        )
        mmr_pass = (
            round(observed_mmr, 3)
            == expected_mmr
        )

        rows.append(
            {
                "dataset": key[0],
                "quartile": key[1],
                "model": key[2],
                "observed_FLR": observed_flr,
                "expected_FLR_3dp": expected_flr,
                "FLR_pass": flr_pass,
                "observed_MMR": observed_mmr,
                "expected_MMR_3dp": expected_mmr,
                "MMR_pass": mmr_pass,
                "row_pass": (
                    flr_pass and mmr_pass
                ),
            }
        )

    verification = pd.DataFrame(rows)

    if len(verification) != len(
        EXPECTED_TABLE
    ):
        raise AssertionError(
            "Verification did not cover every "
            "final-paper Table 8 row."
        )

    return verification


def main() -> None:
    args = parse_args()

    datasets = [
        value.strip().upper()
        for value in args.datasets.split(",")
        if value.strip()
    ]

    invalid = set(datasets).difference(
        VALID_DATASETS
    )

    if invalid:
        raise ValueError(
            f"Unknown datasets: "
            f"{sorted(invalid)}"
        )

    if (
        torch.cuda.is_available()
        and torch.cuda.device_count() > 1
    ):
        raise RuntimeError(
            "More than one GPU is visible. "
            "Pin one GPU using CUDA_VISIBLE_DEVICES."
        )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 78)
    print(
        "Final discrimination-gap analysis: "
        "fine-tuned LR on fixed pretrained block"
    )
    print("=" * 78)
    print(f"Project root: {a17.PROJECT_ROOT}")
    print(f"Device: {device}")
    print(f"Datasets: {datasets}")
    print(f"Seeds: {PRIMARY_SEEDS}")
    print(
        "Transformer training performed here: NO"
    )
    print(
        "Difficulty block: pretrained bi-encoder top-50"
    )
    print(
        "Difficulty features: fine-tuned "
        "bi-encoder field similarities"
    )
    print(
        "Comparison CE path: direct "
        "CrossEncoder.predict() decisions"
    )
    print(
        "Metrics: Zhang FLR/MMR"
    )

    base_model = SentenceTransformer(
        a17.BI_ENCODER_NAME,
        device=device,
    )

    all_predictions: list[pd.DataFrame] = []
    all_metrics: list[dict[str, Any]] = []

    for dataset_name in datasets:

        print("\n" + "=" * 78)
        print(dataset_name)
        print("=" * 78)

        dataset = a17.load_dataset(
            dataset_name
        )

        print(
            "Building common pretrained "
            "top-50 candidate blocks..."
        )

        a17.prepare_baseline(
            dataset,
            base_model,
        )

        for seed in PRIMARY_SEEDS:
            predictions = run_seed(
                dataset,
                seed,
                device,
            )

            all_predictions.append(
                predictions
            )

            all_metrics.extend(
                quartile_rows(
                    predictions
                )
            )

        del dataset

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del base_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    prediction_frame = (
        pd.concat(
            all_predictions,
            ignore_index=True,
        )
        .sort_values(
            [
                "dataset",
                "seed",
                "record_index",
            ]
        )
        .reset_index(drop=True)
    )

    perseed = (
        pd.DataFrame(all_metrics)
        .sort_values(
            [
                "dataset",
                "seed",
                "quartile",
                "model",
            ]
        )
        .reset_index(drop=True)
    )

    primary = aggregate_primary(
        perseed
    )

    verification = verify_final_table(
        primary
    )

    a17.atomic_write_csv(
        prediction_frame,
        PREDICTIONS_PATH,
    )

    a17.atomic_write_csv(
        perseed,
        PERSEED_PATH,
    )

    a17.atomic_write_csv(
        primary,
        PRIMARY3_PATH,
    )

    a17.atomic_write_csv(
        verification,
        VERIFICATION_PATH,
    )

    print("\n" + "=" * 92)
    print("FINAL DISCRIMINATION-GAP TABLE")
    print("=" * 92)

    print(
        primary[
            [
                "dataset",
                "quartile",
                "model",
                "FLR_mean",
                "MMR_mean",
            ]
        ].to_string(index=False)
    )

    print("\nFinal-paper 3-decimal verification:")
    print(
        verification[
            [
                "dataset",
                "quartile",
                "model",
                "FLR_pass",
                "MMR_pass",
            ]
        ].to_string(index=False)
    )

    if verification["row_pass"].all():
        print(
            "\nPASS: every discrimination-gap "
            "row reproduces the final-paper "
            "FLR/MMR values to three decimals."
        )
    else:
        print(
            "\nWARNING: at least one row differs "
            "from the final manuscript at three "
            "decimal places. Inspect:"
        )
        print(f"  {VERIFICATION_PATH}")

    print("\nSaved:")
    print(f"  {PREDICTIONS_PATH}")
    print(f"  {PERSEED_PATH}")
    print(f"  {PRIMARY3_PATH}")
    print(f"  {VERIFICATION_PATH}")


if __name__ == "__main__":
    main()