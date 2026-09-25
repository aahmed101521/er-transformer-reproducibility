"""
reproduce.py
============

Single entry point for reproducing the results reported in:

    Text-based Entity Resolution Using Transformer Models
    Abrar Ahmed, Li-Chun Zhang, Ralf Münnich, and Martin Vogt

Run from the repository root:

    python reproduce.py

The workflow is intentionally resumable. The underlying scientific scripts
save completed cells/checkpoints and reuse them where their own protocol
permits.

The default pipeline reproduces the analyses required by the final manuscript.
Historical diagnostic scripts remain in scripts/ but are not mixed into the
paper-facing workflow.

A CUDA-capable GPU is required for the transformer-training stages.
By default this driver exposes GPU 0 only. Use --gpu N to select another GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
RESULTS = ROOT / "results"
REPRODUCED = ROOT / "reproduced"

RESULTS.mkdir(parents=True, exist_ok=True)
REPRODUCED.mkdir(parents=True, exist_ok=True)


STAGES: list[dict[str, object]] = [
    {
        "name": "blocking-floor",
        "description": (
            "Compute rank statistics and the irreducible blocking MMR floor "
            "from the complete candidate universe."
        ),
        "command": [
            "analysis15_rank_statistic.py",
        ],
    },
    {
        "name": "representation-scoring",
        "description": (
            "Representation x scoring comparison for the standard "
            "blocking settings."
        ),
        "command": [
            "analysis16_clean_2x2.py",
        ],
    },
    {
        "name": "transformer-primary",
        "description": (
            "Primary three-seed baseline, fine-tuned bi-encoder, and "
            "fine-tuned cross-encoder experiment."
        ),
        "command": [
            "analysis17_section5.py",
            "--phase",
            "all",
            "--primary-only",
        ],
    },
    {
        "name": "discrimination-gap",
        "description": (
            "Final record-difficulty analysis using fine-tuned "
            "bi-encoder LR gaps on fixed pretrained top-50 blocks, "
            "compared with the direct-score cross-encoder."
        ),
        "command": [
            "analysis17_table7_ftlr.py",
        ],
    },
    {
        "name": "pretrained-cross-encoder",
        "description": (
            "Untouched pretrained cross-encoder baseline using direct "
            "CrossEncoder.predict() scores and no model fitting."
        ),
        "command": [
            "analysis17_pretrained_ce.py",
        ],
    },
    {
        "name": "training-fractions",
        "description": (
            "Ten-seed cross-encoder training-fraction experiment. "
            "This produces the fitted checkpoints required by the "
            "final direct-score reconstruction."
        ),
        "command": [
            "analysis18_section7.py",
            "--phase",
            "all",
        ],
    },
    {
        "name": "training-fractions-direct",
        "description": (
            "Final direct-score reconstruction of the training-fraction "
            "experiment, including Zhang FLR/MMR."
        ),
        "command": [
            "analysis18_section7_direct.py",
        ],
    },
    {
        "name": "ce-blocksize",
        "description": (
            "Primary three-seed independent cross-encoder block-size "
            "sensitivity experiment."
        ),
        "command": [
            "analysis19_ce_blocksize.py",
            "--phase",
            "all",
            "--primary-only",
        ],
    },
    {
        "name": "discriminative-scorers",
        "description": (
            "Primary three-seed LR, SVM, XGBoost, and MLP "
            "discriminative-scorer comparison."
        ),
        "command": [
            "analysis20_discriminative_scorers.py",
            "--phase",
            "all",
            "--primary-only",
        ],
    },
    {
        "name": "nested-bias",
        "description": (
            "Ten-seed SRSWOR/bootstrap nested-subsample bias experiment. "
            "Verified Analysis 18 targets are reused."
        ),
        "command": [
            "analysis22_section332_bias.py",
            "--phase",
            "run",
            "--target-reuse",
            "require",
        ],
    },
    {
        "name": "nested-bias-aggregate",
        "description": (
            "Aggregate the original Analysis 22 cells required by the "
            "direct-score reconstruction."
        ),
        "command": [
            "analysis22_section332_bias.py",
            "--phase",
            "aggregate",
        ],
    },
    {
        "name": "nested-bias-direct",
        "description": (
            "Final direct-score reconstruction of the nested-bias "
            "experiment using Zhang FLR/MMR."
        ),
        "command": [
            "analysis22_section332_bias_direct.py",
            "--phase",
            "run",
        ],
    },
    {
        "name": "paper-tables",
        "description": (
            "Reconstruct the paper-facing tables from saved record counts "
            "using Zhang FLR/MMR and verify them against the LaTeX manuscript."
        ),
        "command": [
            "build_paper_tables.py",
        ],
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the ER Transformer paper."
    )

    parser.add_argument(
        "--gpu",
        default="0",
        help=(
            "Physical CUDA device to expose to the GPU stages. "
            "Default: 0."
        ),
    )

    parser.add_argument(
        "--from-stage",
        choices=[str(stage["name"]) for stage in STAGES],
        default=None,
        help=(
            "Resume from this stage. Earlier stages are skipped."
        ),
    )

    parser.add_argument(
        "--only",
        choices=[str(stage["name"]) for stage in STAGES],
        default=None,
        help="Run one stage only.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the complete execution plan without running "
            "any scientific script."
        ),
    )

    parser.add_argument(
        "--include-crosscheck",
        action="store_true",
        help=(
            "After the paper-facing pipeline, also run the independent "
            "Analysis 21 ten-seed fine-tuned-bi-encoder cross-check."
        ),
    )

    return parser.parse_args()


def stage_command(
    command: Sequence[str],
) -> list[str]:
    script = SCRIPTS / command[0]

    if not script.exists():
        raise FileNotFoundError(
            f"Required reproduction script is missing: {script}"
        )

    return [
        sys.executable,
        str(script),
        *command[1:],
    ]


def printable_command(
    command: Sequence[str],
) -> str:
    return " ".join(
        f'"{part}"' if " " in str(part) else str(part)
        for part in command
    )


def selected_stages(
    args: argparse.Namespace,
) -> list[dict[str, object]]:

    if args.only is not None:
        return [
            stage
            for stage in STAGES
            if stage["name"] == args.only
        ]

    if args.from_stage is None:
        return list(STAGES)

    names = [
        str(stage["name"])
        for stage in STAGES
    ]

    index = names.index(args.from_stage)
    return list(STAGES[index:])


def run_stage(
    stage: dict[str, object],
    env: dict[str, str],
    run_log: dict[str, object],
    dry_run: bool,
) -> None:

    name = str(stage["name"])
    description = str(stage["description"])
    raw_command = list(stage["command"])

    command = stage_command(raw_command)

    print("\n" + "=" * 88)
    print(f"STAGE: {name}")
    print("=" * 88)
    print(description)
    print()
    print(printable_command(command))

    if dry_run:
        return

    started = utc_now()

    record: dict[str, object] = {
        "name": name,
        "description": description,
        "command": command,
        "started_at": started,
        "status": "running",
    }

    run_log["stages"].append(record)
    write_run_log(run_log)

    try:
        subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        record["status"] = "failed"
        record["returncode"] = exc.returncode
        record["finished_at"] = utc_now()
        write_run_log(run_log)

        print(
            f"\nReproduction stopped because stage "
            f"{name!r} failed with exit code {exc.returncode}."
        )
        print(
            f"After fixing the problem, resume with:\n\n"
            f"    python reproduce.py --from-stage {name}"
        )
        raise

    record["status"] = "completed"
    record["returncode"] = 0
    record["finished_at"] = utc_now()

    write_run_log(run_log)


def write_run_log(
    payload: dict[str, object],
) -> None:
    path = REPRODUCED / "reproduction_run.json"

    temporary = path.with_suffix(".json.tmp")

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")

    temporary.replace(path)


def main() -> None:
    args = parse_args()

    env = os.environ.copy()

    # Explicitly freeze project-root resolution for all recovered scripts.
    env["ER_PROJECT_ROOT"] = str(ROOT)

    # The original GPU-only scripts require exactly one visible device.
    # Exposing one physical GPU here makes the single start command
    # deterministic on multi-GPU machines as well.
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    stages = selected_stages(args)

    if args.include_crosscheck:
        stages.append(
            {
                "name": "ft-biencoder-crosscheck",
                "description": (
                    "Optional independent ten-seed fine-tuned-bi-encoder "
                    "cross-check retained from Analysis 21."
                ),
                "command": [
                    "analysis21_table5_mc10.py",
                    "--phase",
                    "all",
                ],
            }
        )

    print("=" * 88)
    print("ER TRANSFORMER REPRODUCTION")
    print("=" * 88)
    print(f"Project root : {ROOT}")
    print(f"Python       : {sys.executable}")
    print(f"GPU          : {args.gpu}")
    print(f"Stages       : {len(stages)}")

    if args.dry_run:
        print("Mode         : DRY RUN - no experiment will execute")
    else:
        print("Mode         : EXECUTE")

    print("\nExecution plan:")

    for number, stage in enumerate(
        stages,
        start=1,
    ):
        command = stage_command(
            list(stage["command"])
        )

        print(
            f"  {number:02d}. "
            f"{stage['name']}"
        )
        print(
            f"      {printable_command(command)}"
        )

    run_log: dict[str, object] = {
        "paper": (
            "Text-based Entity Resolution Using Transformer Models"
        ),
        "authors": [
            "Abrar Ahmed",
            "Li-Chun Zhang",
            "Ralf Münnich",
            "Martin Vogt",
        ],
        "entry_point": "python reproduce.py",
        "project_root": str(ROOT),
        "python": sys.executable,
        "gpu": str(args.gpu),
        "started_at": utc_now(),
        "dry_run": bool(args.dry_run),
        "stages": [],
    }

    if not args.dry_run:
        write_run_log(run_log)

    for stage in stages:
        run_stage(
            stage=stage,
            env=env,
            run_log=run_log,
            dry_run=args.dry_run,
        )

    if args.dry_run:
        print(
            "\nDry run complete. No scientific script was executed."
        )
        return

    run_log["finished_at"] = utc_now()
    run_log["status"] = "completed"
    write_run_log(run_log)

    print("\n" + "=" * 88)
    print("REPRODUCTION COMPLETE")
    print("=" * 88)
    print(f"Results      : {RESULTS}")
    print(
        f"Run manifest : "
        f"{REPRODUCED / 'reproduction_run.json'}"
    )


if __name__ == "__main__":
    main()