#!/usr/bin/env python3
"""Explicit, separate revision workflow for Table 8 and Table 11.

This script does NOT run as part of the Tables 4-10 reproduce.py driver.
--verify-saved checks archived CSVs only (no training).
--all re-executes six fine-tuned bi-encoder cells, then routing; it reuses
historical pretrained CE decisions and does not retrain that model.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUPPORT = ROOT / "revision_support" / "table8"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-saved", action="store_true", help="Audit archived outputs; no models run")
    mode.add_argument("--all", action="store_true", help="Recover Table 8 then re-evaluate Table 11")
    mode.add_argument("--table8", action="store_true", help="Recover Table 8 only")
    mode.add_argument("--combined-only", action="store_true", help="Run Table 11 from previously generated checkpoints and predictions")
    mode.add_argument("--dry-run", action="store_true", help="Print the revision workflow; run nothing")
    p.add_argument("--gpu", default="0", help="Physical GPU to expose; default 0")
    return p.parse_args()


def run_file(path: Path, cwd: Path, env: dict[str, str], dry_run: bool) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing revision script: {path}")
    cmd = [sys.executable, str(path)]
    if path.name == "recover_table8_mcse.py":
        cmd += ["--device", "cuda"]
    print("STAGE:", path.name, flush=True)
    print("COMMAND:", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)


def main() -> None:
    a = parse_args()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    if a.verify_saved:
        run_file(ROOT / "scripts" / "verify_revision_outputs.py", ROOT, env, False)
        return
    if a.dry_run:
        for name in ("recover_table8_mcse.py", "reviewer1_combined_procedure.py"):
            run_file(SUPPORT / name, SUPPORT, env, True)
        print("DRY RUN ONLY. Table 8 retrains six bi-encoder cells; Table 11 reuses their checkpoints.")
        return
    if a.table8 or a.all:
        run_file(SUPPORT / "recover_table8_mcse.py", SUPPORT, env, False)
    if a.combined_only or a.all:
        pred = SUPPORT / "results" / "table8_predictions.csv.gz"
        if not pred.exists():
            raise FileNotFoundError(
                f"Missing {pred}. Run `python reproduce_revision.py --all` first. "
                "The archived summary CSVs do not substitute for record-level predictions."
            )
        run_file(SUPPORT / "reviewer1_combined_procedure.py", SUPPORT, env, False)
    print("Revision workflow complete. Inspect the generated results and compare with archived references.")


if __name__ == "__main__":
    main()
