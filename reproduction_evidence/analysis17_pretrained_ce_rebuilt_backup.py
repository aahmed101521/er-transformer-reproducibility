"""
analysis17_pretrained_ce.py
===========================

Reproduction of the pretrained cross-encoder row in the final paper.

The cross-encoder is NEVER fitted in this script.

Protocol
--------
- pretrained all-MiniLM-L6-v2 top-50 candidate blocks;
- 50/50 source-A outer split;
- same Analysis 17 supervised-pair construction;
- same 80/20 pair-level validation partition;
- threshold grid 0.10, 0.15, ..., 0.90;
- direct CrossEncoder.predict() output;
- no additional sigmoid;
- complete many-match evaluation truth;
- Zhang FLR = 1 - correct / declared;
- Zhang MMR = 1 - correct / matchable test-A records.

Final-paper seeds: 42, 43, 44.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.model_selection import train_test_split

import analysis17_section5 as a17


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

PERSEED_PATH = RESULTS_DIR / "analysis17_pretrained_ce_perseed.csv"
PRIMARY3_PATH = RESULTS_DIR / "analysis17_pretrained_ce_primary3.csv"

PRIMARY_SEEDS = [42, 43, 44]
VALID_DATASETS = ["DBLP", "ECOM"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final-paper pretrained cross-encoder baseline."
    )
    parser.add_argument(
        "--datasets",
        default="DBLP,ECOM",
        help="Comma-separated subset of DBLP,ECOM.",
    )
    return parser.parse_args()


def zhang_metrics(
    raw: dict[str, float | int],
) -> dict[str, float | int]:

    declared = int(raw["declared"])
    correct = int(raw["correct"])
    false_links = int(raw["false_links"])
    missing_matches = int(raw["missing_matches"])
    n_true_records = int(raw["n_true_records"])

    if declared != correct + false_links:
        raise AssertionError(
            "Declared links must equal correct + false links."
        )

    flr = (
        1.0 - correct / declared
        if declared
        else math.nan
    )

    mmr = (
        1.0 - correct / n_true_records
        if n_true_records
        else math.nan
    )

    return {
        "FLR": flr,
        "MMR": mmr,
        "declared": declared,
        "correct": correct,
        "false_links": false_links,
        "missing_matches": missing_matches,
        "n_true_records": n_true_records,
    }


def run_seed(
    dataset: dict[str, Any],
    model: CrossEncoder,
    seed: int,
) -> dict[str, Any]:

    a17.seed_everything(seed)

    ids_train, ids_test = train_test_split(
        dataset["a_ids"],
        test_size=0.50,
        random_state=seed,
    )

    # Build exactly the same supervised pair set as Analysis 17.
    train_pairs = a17.build_train_pairs(
        ids_train,
        dataset["retained"],
        dataset["truth"],
        dataset["base_top50"],
    )

    labels = [
        label
        for _, _, label in train_pairs
    ]

    # Construct the same 80/20 pair-level partition.
    # The 80% side is deliberately NOT used to fit the pretrained CE.
    fit_indices, validation_indices = train_test_split(
        list(range(len(train_pairs))),
        test_size=a17.LR_VAL_FRAC,
        random_state=seed,
        stratify=labels,
    )

    validation_pairs = [
        train_pairs[index]
        for index in validation_indices
    ]

    fields = dataset["cfg"]["fields"]

    validation_text = [
        (
            a17.serialise(
                dataset["df_a"].loc[ida],
                fields,
            ),
            a17.serialise(
                dataset["df_b"].loc[idb],
                fields,
            ),
        )
        for ida, idb, _ in validation_pairs
    ]

    validation_labels = [
        label
        for _, _, label in validation_pairs
    ]

    # IMPORTANT:
    # direct CrossEncoder.predict(), with no additional sigmoid.
    validation_scores = a17.ce_predict(
        model,
        validation_text,
    )

    threshold, validation_f1 = a17.choose_threshold(
        validation_scores,
        validation_labels,
    )

    test_structure = a17.build_test_structure(
        ids_test,
        dataset["truth"],
        dataset["base_top50"],
    )

    def score_candidates(
        ida: str,
        candidates: list[str],
    ):
        text_a = a17.serialise(
            dataset["df_a"].loc[ida],
            fields,
        )

        candidate_text = [
            (
                text_a,
                a17.serialise(
                    dataset["df_b"].loc[idb],
                    fields,
                ),
            )
            for idb in candidates
        ]

        return a17.ce_predict(
            model,
            candidate_text,
        )

    raw_metrics = a17.evaluate(
        test_structure,
        score_candidates,
        lambda score: score > threshold,
    )

    metrics = zhang_metrics(raw_metrics)

    result = {
        "dataset": dataset["name"],
        "seed": seed,
        "model": a17.CE_MODEL_NAME,
        "model_parameters_updated": False,
        "score_path": "CrossEncoder.predict() direct",
        "n_train_a": len(ids_train),
        "n_test_a": len(ids_test),
        "n_pair_fit_partition_unused": len(fit_indices),
        "n_validation_pairs": len(validation_pairs),
        "threshold": threshold,
        "validation_f1": validation_f1,
        **metrics,
    }

    print(
        f"{dataset['name']} seed={seed}: "
        f"threshold={threshold:.2f}; "
        f"FLR={metrics['FLR']:.6f}; "
        f"MMR={metrics['MMR']:.6f}; "
        f"declared={metrics['declared']}; "
        f"correct={metrics['correct']}"
    )

    return result


def run_dataset(
    dataset_name: str,
    device: str,
) -> list[dict[str, Any]]:

    dataset = a17.load_dataset(dataset_name)

    print(
        f"\n{dataset_name}: constructing pretrained "
        "bi-encoder top-50 blocks"
    )

    base_model = SentenceTransformer(
        a17.BI_ENCODER_NAME,
        device=device,
    )

    a17.prepare_baseline(
        dataset,
        base_model,
    )

    # Only base_top50 is required from this point onward.
    del base_model

    dataset.pop("base_emb_a", None)
    dataset.pop("base_emb_b", None)
    dataset.pop("a_pos", None)
    dataset.pop("b_pos", None)
    dataset.pop("string_cache", None)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f"{dataset_name}: loading untouched pretrained "
        f"cross-encoder {a17.CE_MODEL_NAME}"
    )

    model = CrossEncoder(
        a17.CE_MODEL_NAME,
        num_labels=1,
        max_length=a17.CE_MAX_LEN,
        device=device,
    )

    # There is deliberately no model.fit() anywhere in this script.
    rows = [
        run_seed(
            dataset,
            model,
            seed,
        )
        for seed in PRIMARY_SEEDS
    ]

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rows


def save_outputs(
    rows: list[dict[str, Any]],
) -> None:

    frame = pd.DataFrame(rows).sort_values(
        ["dataset", "seed"]
    )

    frame.to_csv(
        PERSEED_PATH,
        index=False,
    )

    aggregate = (
        frame.groupby(
            "dataset",
            as_index=False,
        )
        .agg(
            n_splits=("seed", "count"),
            FLR=("FLR", "mean"),
            MMR=("MMR", "mean"),
            FLR_sd=("FLR", "std"),
            MMR_sd=("MMR", "std"),
        )
    )

    aggregate["FLR_se"] = (
        aggregate["FLR_sd"]
        / aggregate["n_splits"] ** 0.5
    )

    aggregate["MMR_se"] = (
        aggregate["MMR_sd"]
        / aggregate["n_splits"] ** 0.5
    )

    aggregate.to_csv(
        PRIMARY3_PATH,
        index=False,
    )

    print("\nPrimary-three pretrained CE results")
    print("=" * 78)
    print(
        aggregate.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(f"  {PERSEED_PATH}")
    print(f"  {PRIMARY3_PATH}")


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
            f"Unknown datasets: {sorted(invalid)}"
        )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 78)
    print(
        "Analysis 17 pretrained cross-encoder "
        "final-paper baseline"
    )
    print("=" * 78)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Datasets: {datasets}")
    print(f"Seeds: {PRIMARY_SEEDS}")
    print(f"Device: {device}")
    print("Cross-encoder fit(): NEVER")
    print("Additional sigmoid: NO")
    print(
        "Metrics: Zhang FLR/MMR from "
        "declared/correct/matchable counts"
    )

    rows: list[dict[str, Any]] = []

    for dataset_name in datasets:
        rows.extend(
            run_dataset(
                dataset_name,
                device,
            )
        )

    save_outputs(rows)


if __name__ == "__main__":
    main()
