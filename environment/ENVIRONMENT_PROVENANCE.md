# Software environment provenance

This document distinguishes the software environments evidenced by the project. They should not be merged into a single historical claim.

## Verified full-pipeline environment — September 2026

The final Tables 4–10 reproducibility verification and the separate Table 8/Table 11 revision workflow were tested on an NVIDIA L40 using Python 3.12.3.

The verified direct package versions are:

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

CUDA was available through PyTorch with CUDA runtime 12.1, and the verified GPU was an NVIDIA L40.

`pip check` reported no broken requirements.

The root `requirements.txt` records these tested direct dependency pins.

The complete transitive environment and machine evidence are captured in:

- `full_pipeline_core_versions.txt`
- `full_pipeline_pip_freeze.txt`
- `full_pipeline_pip_check.txt`
- `full_pipeline_nvidia_smi.txt`

These files describe the environment used for the September 2026 verified reproduction work.

### Scope of the verification

The main `reproduce.py` driver exposes 13 dependency-ordered stages for Tables 4–10.

For the September 2026 verification, all 13 stages were executed in the same clean environment. They were run individually through the driver's `--only` mechanism rather than as one uninterrupted shell process. This was done so that each stage could be audited independently and its evidence retained.

The final paper-table reconstruction checked 116 manuscript-facing values and reported zero failures.

The separate `reproduce_revision.py --all` workflow was also executed in this environment. It freshly trained the six Table 8 bi-encoder cells and then regenerated the combined-routing results used for Table 11. The post-run revision verifier passed.

This verified environment is therefore the recommended environment for reproducing the revised package.

## Historical Analysis 20 manifest — partial evidence

`historical_core_versions.txt` and `requirements-historical-core.txt` are derived from `results/analysis20_discriminative_scorers_manifest.json`.

That historical manifest records:

- Python 3.12.3
- NumPy 2.5.1
- pandas 3.0.3
- scikit-learn 1.9.0
- PyTorch 2.5.1+cu121
- XGBoost 3.3.0

It does not establish historical versions for:

- sentence-transformers
- transformers
- datasets
- accelerate
- jellyfish

No complete historical `pip freeze` survived.

Accordingly, this partial manifest should be treated as provenance evidence for that historical run, not as a complete installation recipe.

The recorded scikit-learn version differs from the final verified environment (1.9.0 historically versus 1.9.1 in the verified reproduction environment). The September 2026 discriminative-scorer audit nevertheless reproduced all scientific values in the checked primary-seed rows; the version field itself was one of the metadata differences recorded by that audit.

## September 22 Table 7/8 recovery environment

`final_recovery_core_versions.txt` and `final_recovery_pip_freeze.txt` record an earlier September 2026 recovery environment.

That environment used:

- Python 3.12.3
- NumPy 2.5.3
- pandas 3.0.6
- scikit-learn 1.9.1
- PyTorch 2.5.1+cu121
- sentence-transformers 6.1.0
- transformers 5.17.0
- datasets 5.0.1
- accelerate 1.15.0

The recovery environment was run on an NVIDIA L40.

XGBoost was not installed in that environment because it was not needed for the specific Table 7/Table 8 recovery work.

The corresponding freeze contains remnants of CUDA dependency packages from PyTorch wheel changes. It is retained as a provenance record and should not be treated as the preferred clean installation specification.

`requirements-recovery.txt` records the tested core recovery stack.

## Why the environments are kept separate

Three kinds of environment evidence therefore exist:

1. **Historical partial evidence** from retained experiment metadata.
2. **Recovery environment evidence** from the September Table 7/Table 8 reconstruction work.
3. **Verified full-pipeline environment evidence** from the final September 2026 reproducibility verification.

The third is the environment that should be used for reproducing the revised package. The earlier two are retained because they document how historical or recovery results were produced and explain small provenance differences, such as the scikit-learn patch-version change.

The existence of several environment records should not be interpreted as a claim that all historical experiments were originally produced under the final verified environment.

## Determinism and reproducibility interpretation

The environment evidence should be read together with `DETERMINISM.md`.

No claim is made that GPU-trained checkpoint files are bitwise identical across machines or CUDA configurations.

The September 2026 verification instead compared the scientific outputs used by the paper. Depending on the stage, these checks included exact comparison of scientific fields, byte-for-byte output comparison, recomputation from per-seed records and manuscript-facing table verification.

Where a floating-point comparison was required, the scientific-field audit used `rtol=1e-12` and `atol=1e-12`. The recovered Table 8 historical audit quantities were checked at six-decimal precision.

Monte Carlo standard errors quantify variation across study replications or data splits. They are not used as a tolerance for GPU nondeterminism.

## Rebuilding the verified environment

The root `requirements.txt` contains the verified direct dependency pins.

A new Python 3.12 environment can be created with:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

After installation, compare the core package versions with `full_pipeline_core_versions.txt` and inspect the driver with:

```bash
python reproduce.py --dry-run
```

GPU availability should be checked independently with `nvidia-smi` and `torch.cuda.is_available()`.

## Public archival status

These environment records are intended to accompany the final public reproducibility package.

A permanent repository/archive URL or DOI should be added to the package and manuscript only after the final archive has been created and independently verified. No provisional identifier should be presented as an active public deposit.
