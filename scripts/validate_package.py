#!/usr/bin/env python3
"""Structural checks and SHA-256 verification for the archival package.

This runs with the Python standard library only. A PASS is NOT a model rerun.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md", "REPRODUCIBILITY.md", "DETERMINISM.md",
    "DATA_LICENSE.md",
    "environment/ENVIRONMENT_PROVENANCE.md",
    "requirements.txt", "requirements-recovery.txt",
    "requirements-historical-core.txt", "reproduce.py", "reproduce_revision.py",
    "paper/main.tex", "paper/references.bib",
    "data/dblp/DBLP1.csv", "data/dblp/Scholar.csv",
    "data/dblp/DBLP-Scholar_perfectMapping.csv",
    "data/abt_buy/Abt.csv", "data/abt_buy/Buy.csv",
    "data/abt_buy/abt_buy_perfectMapping.csv",

    # Frozen August top-50 candidate blocks used by the untouched
    # pretrained-cross-encoder reconstruction.
    "cache/DBLP_top50.json",
    "cache/ECOM_top50.json",

    "results/revision_20260922/table8_primary3_final_mcse.csv",
    "results/revision_20260922/reviewer1_combined_summary.csv",
    "results/revision_20260922/analysis17_pretrained_ce_primary3.csv",
    "revision_support/table8/recover_table8_mcse.py",
    "revision_support/table8/reviewer1_combined_procedure.py",
    "revision_support/table8/cache/DBLP_top50.json",
    "revision_support/table8/cache/ECOM_top50.json",
    "environment/final_recovery_pip_freeze.txt",
    "environment/historical_core_versions.txt",

    # Verified September 2026 full-pipeline environment.
    "environment/full_pipeline_core_versions.txt",
    "environment/full_pipeline_pip_freeze.txt",
    "environment/full_pipeline_pip_check.txt",
    "environment/full_pipeline_nvidia_smi.txt",
)


def main() -> None:
    missing = [p for p in REQUIRED if not (ROOT / p).is_file()]
    if missing:
        raise SystemExit("STRUCTURE FAIL: missing " + ", ".join(missing))
    import sys
    sys.path.insert(0, str(ROOT))
    from reproduce import STAGES
    stage_scripts = [ROOT / "scripts" / stage["command"][0] for stage in STAGES]
    missing_stages = [str(p) for p in stage_scripts if not p.is_file()]
    if missing_stages:
        raise SystemExit("STAGE FAIL: missing " + ", ".join(missing_stages))

    manifest = ROOT / "FILE_MANIFEST.sha256"
    if not manifest.is_file():
        raise SystemExit("HASH FAIL: FILE_MANIFEST.sha256 missing")
    entries = 0
    for line in manifest.read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        digest, relpath = line.split("  ", 1)
        p = ROOT / relpath
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise SystemExit(f"HASH FAIL: {relpath}")
        entries += 1
    print(f"STRUCTURE AND HASH PASS: {len(REQUIRED)} required files, "
          f"{len(stage_scripts)} main stages, {entries} manifest entries")
    print("Scope: archived files only; no training or inference was run.")


if __name__ == "__main__":
    main()
