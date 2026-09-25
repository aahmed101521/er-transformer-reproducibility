# Text-based Entity Resolution Using Transformer Models

**Authors:** Abrar Ahmed, Li-Chun Zhang, Ralf Münnich, Martin Vogt

This repository is the reproducibility package for the revised manuscript **Text-based Entity Resolution Using Transformer Models**.

The scientific reproduction work was verified in September 2026 on an NVIDIA L40. The main Tables 4–10 workflow is exposed through a single driver, `python reproduce.py`, containing 13 dependency-ordered stages. For the verification reported here, those stages were executed individually with `--only` in the same clean environment so that each stage could be audited and preserved separately. The final paper-table reconstruction checked 116 manuscript-facing values and reported zero failures.

The separate revision workflow, `python reproduce_revision.py --all`, freshly trains the six Table 8 bi-encoder cells and then evaluates the combined-routing procedure used for Table 11.

The package has not yet been assigned a permanent public repository URL or DOI. No placeholder URL should be treated as an active deposit.

## Which command does what?

| Command | Result | Evidence level |
| --- | --- | --- |
| `python scripts/validate_package.py` | Check required package structure and SHA-256 manifest | Structural audit; no model fitting |
| `python reproduce.py --dry-run` | Show the 13-stage Tables 4–10 execution plan | No scientific script executed |
| `python reproduce.py` | Run the complete Tables 4–10 driver | Main reproduction entry point |
| `python reproduce.py --only NAME` | Run one named stage from the same driver | Used for the September 2026 audited verification |
| `python reproduce_revision.py --dry-run` | Show the separate Table 8/Table 11 revision plan | No model fitting |
| `python reproduce_revision.py --all` | Re-train six bi-encoder cells, reconstruct Table 8 and re-run Table 11 routing | Fresh revision re-execution |
| `python reproduce_revision.py --verify-saved` | Verify saved revision outputs | Saved-output audit |
| `python scripts/verify_revision_outputs.py` | Check saved Table 7, Table 8 and Table 11 outputs | Saved-output audit |

Run all commands from the package root.

`python reproduce.py --from-stage NAME` can be used to resume the main driver. `python reproduce.py --only paper-tables` requires the necessary upstream generated results and should not be treated as a zero-compute stand-alone command on a fresh archive.

## Dataset, truth and statistics

The working package contains the Magellan DBLP-Scholar and Abt-Buy CSV data used by the experiments. Redistribution terms must be checked before the final public archive is released; if redistribution is not permitted, the public deposit should provide a documented acquisition/preparation step instead of bundling the source data.

The file-A record is the split and evaluation unit. One positive partner per matchable file-A record is used for supervised fitting; all known valid partners remain in the evaluation truth and are excluded from negatives. Only retrieved top-k candidates are scored at test time.

The final linkage-error definitions used by the manuscript are:

```text
FLR = 1 - correct / declared
MMR = 1 - correct / n_true_records
```

The historical quantity

```text
psi = missing_matches / n_true_records
```

remains in some intermediate outputs but is not the final manuscript MMR.

Primary-table splits use seeds 42, 43 and 44. Ten-replication studies use seeds 42 through 51.

The transformer models are:

- bi-encoder: `all-MiniLM-L6-v2`
- cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`

## Verified software environment

The verified September 2026 full-pipeline environment is specified by the root `requirements.txt`.

Core versions:

- Python 3.12.3
- NumPy 2.5.3
- pandas 3.0.6
- scikit-learn 1.9.1
- PyTorch 2.5.1+cu121
- sentence-transformers 6.1.0
- transformers 5.17.0
- datasets 5.0.1
- accelerate 1.15.0
- XGBoost 3.3.0
- jellyfish 1.2.1

The reproduction server used CUDA 12.1 on an NVIDIA L40.

The complete transitive environment and machine-level evidence are stored in:

- `environment/full_pipeline_pip_freeze.txt`
- `environment/full_pipeline_core_versions.txt`
- `environment/full_pipeline_pip_check.txt`
- `environment/full_pipeline_nvidia_smi.txt`

`pip check` reported no broken requirements.

Earlier environment records are retained separately for provenance:

- `environment/historical_core_versions.txt` and `requirements-historical-core.txt` contain a partial historical manifest.
- `environment/final_recovery_*` and `requirements-recovery.txt` describe an earlier September 2026 recovery environment.

These historical/recovery records should not be confused with the verified full-pipeline environment above.

## Creating a clean environment

With Python 3.12 available:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Then inspect the driver without running experiments:

```bash
python reproduce.py --dry-run
```

A CUDA-capable GPU is required for fitting stages. Verify the local driver and GPU separately with `nvidia-smi` and `torch.cuda.is_available()`.

## Main Tables 4–10 reproduction

The single main entry point is:

```bash
CUDA_VISIBLE_DEVICES=0 python reproduce.py 2>&1 | tee reproduce_full.log
```

The driver contains 13 stages:

1. blocking-floor
2. representation-scoring
3. transformer-primary
4. discrimination-gap
5. pretrained-cross-encoder
6. training-fractions
7. training-fractions-direct
8. ce-blocksize
9. discriminative-scorers
10. nested-bias
11. nested-bias-aggregate
12. nested-bias-direct
13. paper-tables

During the September 2026 audit, these same stages were executed one by one with `python reproduce.py --only <stage>` in the same verified environment. This allowed each stage to be compared independently with archived scientific outputs. The final paper-table stage reported 116/116 manuscript-facing checks passed.

Saved analysis outputs are under `results/`. Reconstructed paper-facing tables are under `reproduced/paper_tables/`. Driver status information is stored under `reproduced/`.

This command does not regenerate Table 11.

## Frozen candidate blocks for the pretrained cross-encoder

The Stage 5 untouched pretrained-cross-encoder reconstruction uses the frozen top-50 candidate blocks:

- `cache/DBLP_top50.json`
- `cache/ECOM_top50.json`

These files are part of the reproducibility input. They preserve the archived candidate sets used for the reported pretrained-cross-encoder experiment.

The Stage 5 script validates their dataset, bi-encoder, block-size and record-coverage metadata before scoring. They should not be silently rebuilt with a later retrieval implementation.

## Table 8 and Table 11 revision workflow

The separate revision workflow is:

```bash
python reproduce_revision.py --dry-run
CUDA_VISIBLE_DEVICES=0 python reproduce_revision.py --all \
  2>&1 | tee reproduce_revision.log
```

This workflow freshly trains six `all-MiniLM-L6-v2` bi-encoder cells for Table 8, reconstructs the revised Table 8 quantities and then evaluates the combined-routing procedure for Table 11.

The Table 8 recovery reuses historical cross-encoder decisions supplied under:

```text
revision_support/table8/historical_ce_predictions/
```

It does not retrain the cross-encoder in this revision-specific workflow.

The Table 11 routing experiment uses training-derived 25th- and 50th-percentile discrimination-gap cutoffs together with the freshly recovered Table 8 predictions and bi-encoder checkpoints.

For a fast verification of saved revision outputs:

```bash
python reproduce_revision.py --verify-saved
```

or

```bash
python scripts/verify_revision_outputs.py
```

The September 2026 fresh revision run completed successfully and the saved revision verifier passed.

## Verified September 2026 reproduction evidence

The reproduction audit established, among other checks:

- blocking-floor outputs reproduced byte-for-byte;
- representation/scoring summaries reproduced byte-for-byte;
- primary transformer scientific outputs reproduced exactly;
- the ten-seed training-fraction experiment reproduced all substantive values across 100 per-seed rows;
- the primary cross-encoder block-size experiment reproduced all scientific values across 18 rows;
- the discriminative-scorer experiment reproduced all scientific values across 144 primary-seed rows after excluding runtime/provenance metadata;
- the nested-bias MC10 summary reproduced byte-for-byte;
- direct-score summaries were independently recomputed from their per-seed outputs and agreed without differences;
- the revised Table 8 workflow freshly trained six bi-encoder cells and passed the revision verifier;
- the combined-routing/Table 11 results were regenerated; and
- the final paper-table reconstruction passed 116/116 manuscript-facing checks.

Detailed run logs, comparison reports and stage manifests are retained in `reproduction_evidence/`.

## Determinism and comparison criteria

See `DETERMINISM.md`.

The package does not claim that fine-tuned checkpoint files are bitwise deterministic. The reproduction claim concerns the scientific outputs used in the paper.

For scientific-field comparisons where a floating-point tolerance was needed, the audit used `rtol=1e-12` and `atol=1e-12` and found no changed scientific values in the reported checks.

For the recovered Table 8 bi-encoder cells, comparison to the archived audit is made at six-decimal precision.

The reported Monte Carlo standard errors quantify variability across data splits or study replications. They are not estimates of same-seed GPU nondeterminism and are not used as a reproduction tolerance.

## Package validation

After the final package contents have been frozen, regenerate `FILE_MANIFEST.sha256` and run:

```bash
python scripts/validate_package.py
```

Saved revision outputs can additionally be checked with:

```bash
python scripts/verify_revision_outputs.py
```

The SHA-256 manifest should be regenerated only after all final documentation, scripts and retained outputs have been fixed.

## Paper and public deposit status

The cumulative reviewer-highlighted LaTeX manuscript is stored under `paper/`.

At the time of this package preparation, the final public repository/archive and DOI have not yet been created and verified. The manuscript, README and response letter should be updated with the real repository URL and DOI only after the deposit exists.

The intended final archival workflow is:

1. freeze and validate the package;
2. create the public source-code repository;
3. create a permanent archive/release, for example through Zenodo;
4. verify the public URL/DOI; and
5. insert those identifiers into the manuscript and reviewer response.

See `environment/ENVIRONMENT_PROVENANCE.md` for environment provenance and `DETERMINISM.md` for the distinction between replication variability and same-seed GPU reproducibility.


## Licensing

Software code authored for this reproducibility package is released under the
MIT License; see `LICENSE`.

The DBLP-Scholar and Abt-Buy benchmark datasets are third-party materials and
are not relicensed under the MIT License. Their provenance and the licensing
statement provided by the original distributor are documented in
`DATA_LICENSE.md`.

Manuscript text included for reproducibility is not covered by the software
license and remains subject to the applicable author/publisher rights.
