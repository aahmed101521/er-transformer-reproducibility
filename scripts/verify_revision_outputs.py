"""Verify the September 2026 revision-specific saved outputs.

This is a lightweight audit of the recovered Table 7 pretrained cross-encoder,
final Table 8 Zhang-MMR/MCSE results, and the training-derived combined routing
analysis. It does not train models.
"""
from __future__ import annotations

from pathlib import Path
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def close(a: float, b: float, tol: float = 5e-7) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    failures: list[str] = []

    pre = pd.read_csv(require(RESULTS / "analysis17_pretrained_ce_primary3.csv"))
    expected_pre = {
        "DBLP-Scholar": (0.06098450096011158, 0.025968855664361468),
        "Abt-Buy": (0.26664528118381003, 0.2778804682686383),
    }
    for row in pre.itertuples(index=False):
        exp = expected_pre[str(row.paper_dataset)]
        if not close(row.FLR_mean, exp[0]) or not close(row.MMR_mean, exp[1]):
            failures.append(f"pretrained CE mismatch: {row.paper_dataset}")

    t8 = pd.read_csv(require(RESULTS / "analysis17_table7_ftlr_primary3.csv"))
    expected_t8 = {
        ("DBLP", "Q1", "Cross-encoder"): (0.033214198448536236, 0.02931501019242644),
        ("DBLP", "Q1", "Logistic regression"): (0.0374951481772882, 0.045209017325758695),
        ("DBLP", "Q2", "Cross-encoder"): (0.03425245591778273, 0.021752175217521746),
        ("DBLP", "Q2", "Logistic regression"): (0.03146381793009027, 0.06309202348806313),
        ("DBLP", "Q3", "Cross-encoder"): (0.008249682704511393, 0.0102944361630292),
        ("DBLP", "Q3", "Logistic regression"): (0.006414326664418842, 0.041180929811820755),
        ("DBLP", "Q4", "Cross-encoder"): (0.0, 0.0),
        ("DBLP", "Q4", "Logistic regression"): (0.0, 0.0),
        ("ECOM", "Q1", "Cross-encoder"): (0.12099483204134365, 0.3995098039215687),
        ("ECOM", "Q1", "Logistic regression"): (0.2301929804438837, 0.7058823529411765),
        ("ECOM", "Q2", "Cross-encoder"): (0.05114285714285716, 0.1308641975308642),
        ("ECOM", "Q2", "Logistic regression"): (0.10133232438482713, 0.25925925925925924),
        ("ECOM", "Q3", "Cross-encoder"): (0.015729812661498716, 0.07160493827160493),
        ("ECOM", "Q3", "Logistic regression"): (0.017283950617283977, 0.017283950617283977),
        ("ECOM", "Q4", "Cross-encoder"): (0.0, 0.0049382716049382784),
        ("ECOM", "Q4", "Logistic regression"): (0.0, 0.0),
    }
    seen = set()
    for row in t8.itertuples(index=False):
        key = (str(row.dataset), str(row.quartile), str(row.model))
        if key not in expected_t8:
            continue
        seen.add(key)
        exp = expected_t8[key]
        if not close(row.FLR_mean, exp[0]) or not close(row.MMR_mean, exp[1]):
            failures.append(f"Table 8 mismatch: {key}")
    if seen != set(expected_t8):
        failures.append("Table 8 rows incomplete")

    combo = pd.read_csv(require(RESULTS / "reviewer1_combined_summary.csv"))
    expected_combo = {
        ("DBLP", "Q25-train"): (0.2464322120285423, 0.01684993270238566, 0.03311647606772772),
        ("DBLP", "Q50-train"): (0.4847094801223242, 0.017495983678245675, 0.023718366735494583),
        ("ECOM", "Q25-train"): (0.33271719038817005, 0.0495125817985563, 0.14849044978434997),
        ("ECOM", "Q50-train"): (0.589648798521257, 0.04206819759242216, 0.1447935921133703),
    }
    for row in combo.itertuples(index=False):
        key = (str(row.dataset), str(row.routing_rule))
        exp = expected_combo[key]
        if not all(close(v, e) for v, e in zip((row.phi_mean, row.FLR_mean, row.MMR_mean), exp)):
            failures.append(f"combined-routing mismatch: {key}")

    if failures:
        print("REVISION OUTPUT VERIFICATION: FAIL")
        for item in failures:
            print(" -", item)
        raise SystemExit(1)

    print("REVISION OUTPUT VERIFICATION: PASS")
    print("Pretrained CE, final Table 8, and combined-routing saved outputs match the revision audit.")


if __name__ == "__main__":
    main()
