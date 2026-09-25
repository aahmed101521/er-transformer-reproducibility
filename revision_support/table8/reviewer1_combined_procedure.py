from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

import recover_table8_mcse as r

ROOT = Path(__file__).resolve().parent
PRED_PATH = ROOT / "results" / "table8_predictions.csv.gz"
OUT_PERSEED = ROOT / "results" / "reviewer1_combined_perseed.csv"
OUT_SUMMARY = ROOT / "results" / "reviewer1_combined_summary.csv"

QUANTILES = [("Q25-train", 0.25), ("Q50-train", 0.50)]


def bool_series(s: pd.Series) -> pd.Series:
    return r.as_bool_series(s)


def mcse(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=float)
    if len(arr) <= 1:
        return float("nan")
    return float(np.std(arr, ddof=1) / math.sqrt(len(arr)))


def compute_gaps_for_ids(
    ids: list[str],
    top50: dict[str, list[str]],
    lr,
    emb_a: dict[str, torch.Tensor],
    emb_b: dict[str, torch.Tensor],
    ds: dict,
) -> np.ndarray:
    fields = ds["cfg"]["fields"]
    gaps = np.empty(len(ids), dtype=float)
    for idx, ida in enumerate(ids):
        candidates = top50[ida]
        features = r.cosine_feature_matrix(
            [(ida, idb) for idb in candidates],
            emb_a,
            emb_b,
            fields,
            ds["a_pos"],
            ds["b_pos"],
        )
        scores = lr.predict_proba(features)[:, 1]
        top2 = np.partition(scores, -2)[-2:]
        top2.sort()
        gaps[idx] = float(top2[1] - top2[0])
    return gaps


def chosen_metrics(test: pd.DataFrame, hard: np.ndarray) -> dict[str, float | int]:
    lr_decl = bool_series(test["ft_lr_declared"]).to_numpy()
    ce_decl = bool_series(test["ce_direct_declared"]).to_numpy()
    lr_cat = test["ft_lr_category"].astype(str).to_numpy()
    ce_cat = test["ce_direct_category"].astype(str).to_numpy()
    has_truth = bool_series(test["has_true_match"]).to_numpy()

    declared = np.where(hard, ce_decl, lr_decl)
    category = np.where(hard, ce_cat, lr_cat)

    d = int(declared.sum())
    c = int((category == "correct").sum())
    f = int((category == "false_link").sum())
    m = int(has_truth.sum())

    if d != c + f:
        raise AssertionError(f"declared={d} != correct+false={c+f}")

    flr = 1.0 - c / d if d else 0.0
    mmr = 1.0 - c / m if m else 0.0
    phi = float(hard.mean())

    return {
        "n_test": int(len(test)),
        "n_routed": int(hard.sum()),
        "phi": phi,
        "declared": d,
        "correct": c,
        "false_links": f,
        "n_true_records": m,
        "FLR": float(flr),
        "MMR": float(mmr),
        "ce_evaluations": int(hard.sum()) * r.K,
        "all_ce_evaluations": int(len(test)) * r.K,
        "ce_eval_reduction": 1.0 - phi,
    }


def main() -> None:
    if not PRED_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PRED_PATH}. Run this script inside the completed "
            "table8_mcse_recovery_bundle directory."
        )

    pred = pd.read_csv(PRED_PATH, dtype={"ida": str})
    required = {
        "dataset", "seed", "ida", "has_true_match", "ft_lr_gap",
        "ft_lr_declared", "ft_lr_category",
        "ce_direct_declared", "ce_direct_category",
    }
    missing = required - set(pred.columns)
    if missing:
        raise KeyError(f"Prediction file missing columns: {sorted(missing)}")

    rows: list[dict] = []

    print("=" * 100)
    print("REVIEWER 1 — COMBINED PROCEDURE WITH TRAINING-DERIVED ROUTING THRESHOLDS")
    print("=" * 100)
    print("Routing cutoffs are the 25th and 50th percentiles of OUTER-TRAINING-record")
    print("discrimination gaps. They are fixed before being applied to the test records.")
    print("Cross-encoder decisions are the historical fine-tuned CE decisions from Table 8.")
    print("Rest-case decisions use the fine-tuned bi-encoder + logistic regression.")
    print()

    for dataset in ["DBLP", "ECOM"]:
        ds = r.load_dataset(dataset)
        paper_name = ds["cfg"]["paper_name"]
        print("-" * 100)
        print(paper_name)
        print("-" * 100)

        for seed in r.PRIMARY_SEEDS:
            r.seed_everything(seed)
            ids_train_arr, ids_test_arr = train_test_split(
                ds["a_ids"], test_size=0.50, random_state=seed
            )
            ids_train = list(ids_train_arr)
            ids_test = list(ids_test_arr)

            # Load the already recovered fine-tuned encoder. No transformer training occurs here.
            model_dir = r.model_path(dataset, seed)
            if not r.marker_path(model_dir).exists():
                raise FileNotFoundError(
                    f"Recovered fine-tuned model not found: {model_dir}\n"
                    "Do not retrain from this script. Use the completed Table-8 recovery directory."
                )
            model = SentenceTransformer(str(model_dir), device="cuda" if torch.cuda.is_available() else "cpu")

            fields = ds["cfg"]["fields"]
            emb_a = r.encode_fields(model, ds["df_a"], fields, f"{dataset} seed {seed} A")
            emb_b = r.encode_fields(model, ds["df_b"], fields, f"{dataset} seed {seed} B")

            train_pairs = r.build_train_pairs(
                ids_train, ds["retained"], ds["truth"], ds["base_top50"]
            )
            x_train = r.cosine_feature_matrix(
                [(a, b) for a, b, _ in train_pairs],
                emb_a, emb_b, fields, ds["a_pos"], ds["b_pos"]
            )
            y_train = np.asarray([y for _, _, y in train_pairs], dtype=int)
            lr, link_threshold, val_f1 = r.calibrate_lr(x_train, y_train, seed)

            # Select routing cutoffs only from the outer training records.
            train_gaps = compute_gaps_for_ids(
                ids_train, ds["base_top50"], lr, emb_a, emb_b, ds
            )

            test = pred[(pred["dataset"] == dataset) & (pred["seed"] == seed)].copy()
            if set(test["ida"].astype(str)) != set(map(str, ids_test)):
                raise AssertionError(f"Test IDs do not align for {dataset} seed {seed}")

            # Optional fidelity check: recompute test gaps and require close agreement with Table-8 output.
            test_lookup = test.set_index("ida")
            calc_test_gaps = compute_gaps_for_ids(
                ids_test, ds["base_top50"], lr, emb_a, emb_b, ds
            )
            saved_test_gaps = np.asarray([float(test_lookup.loc[str(i), "ft_lr_gap"]) for i in ids_test])
            max_gap_diff = float(np.max(np.abs(calc_test_gaps - saved_test_gaps)))
            if max_gap_diff > 1e-8:
                raise RuntimeError(
                    f"LR gap reproduction failed for {dataset} seed {seed}: max diff {max_gap_diff:.3g}"
                )

            for rule, q in QUANTILES:
                cutoff = float(np.quantile(train_gaps, q))
                hard = test["ft_lr_gap"].to_numpy(dtype=float) <= cutoff
                met = chosen_metrics(test, hard)
                rows.append({
                    "dataset": dataset,
                    "paper_dataset": paper_name,
                    "seed": seed,
                    "routing_rule": rule,
                    "training_gap_quantile": q,
                    "routing_cutoff": cutoff,
                    "link_threshold": float(link_threshold),
                    "validation_f1": float(val_f1),
                    "max_test_gap_reproduction_diff": max_gap_diff,
                    **met,
                })
                print(
                    f"seed {seed} {rule:9s}: cutoff={cutoff:.6f}, "
                    f"phi={met['phi']:.3f}, FLR={met['FLR']:.6f}, MMR={met['MMR']:.6f}, "
                    f"CE evals={met['ce_evaluations']:,}/{met['all_ce_evaluations']:,}"
                )

            del model, emb_a, emb_b
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    perseed = pd.DataFrame(rows)
    perseed.to_csv(OUT_PERSEED, index=False)

    summary_rows: list[dict] = []
    for keys, g in perseed.groupby(["dataset", "paper_dataset", "routing_rule"], sort=True):
        dataset, paper_name, rule = keys
        summary_rows.append({
            "dataset": dataset,
            "paper_dataset": paper_name,
            "routing_rule": rule,
            "n_splits": len(g),
            "phi_mean": float(g["phi"].mean()),
            "phi_MCSE": mcse(g["phi"]),
            "FLR_mean": float(g["FLR"].mean()),
            "FLR_MCSE": mcse(g["FLR"]),
            "MMR_mean": float(g["MMR"].mean()),
            "MMR_MCSE": mcse(g["MMR"]),
            "ce_evaluations_mean": float(g["ce_evaluations"].mean()),
            "all_ce_evaluations_mean": float(g["all_ce_evaluations"].mean()),
            "ce_eval_reduction_mean": float(g["ce_eval_reduction"].mean()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_SUMMARY, index=False)

    print("\n" + "=" * 100)
    print("COMBINED PROCEDURE — mean (MCSE), seeds 42–44")
    print("=" * 100)
    for row in summary.itertuples(index=False):
        print(
            f"{row.paper_dataset:13s} {row.routing_rule:9s}  "
            f"phi {row.phi_mean:.3f} ({row.phi_MCSE:.4f})   "
            f"FLR {row.FLR_mean:.3f} ({row.FLR_MCSE:.4f})   "
            f"MMR {row.MMR_mean:.3f} ({row.MMR_MCSE:.4f})   "
            f"CE evals {row.ce_evaluations_mean:.0f}/{row.all_ce_evaluations_mean:.0f} "
            f"(reduction {100*row.ce_eval_reduction_mean:.1f}%)"
        )

    print("\nSaved:")
    print(f"  {OUT_PERSEED}")
    print(f"  {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
