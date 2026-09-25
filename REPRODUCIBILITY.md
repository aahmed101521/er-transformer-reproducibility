# Reproduction status and evidence

See `README.md` for commands and `environment/ENVIRONMENT_PROVENANCE.md`
for separate historical, recovery and verified full-pipeline environments.

## Distinguish three reproducibility levels

1. **Saved-output audit:** `python scripts/verify_revision_outputs.py`
   checks archived values; this does not rerun models.
2. **Revision re-execution:** `python reproduce_revision.py --all`
   reconstructs Table 8 from bundled raw data, top-50 caches and
   historical CE decisions, then re-evaluates the combined routing
   procedure for Table 11. It retrains six *bi-encoder* cells, not the CE.
3. **Full paper pipeline:** `python reproduce.py` exposes 13 dependency-
   ordered stages for Tables 4-10. In September 2026 all 13 stages were
   executed in the same clean, verified environment. For auditability, the
   verification ran the stages individually with `--only` rather than as one
   uninterrupted shell process. The final paper-table stage checked 116
   manuscript-facing values with zero failures. Table 11 remains part of the
   separate revision workflow.

The independent `--only paper-tables` stage needs upstream files such as
`results/analysis18_fraction_sweep_direct_mc10.csv`; these are not all
bundled. Run the upstream scientific stages first rather than treating
`--only paper-tables` as a standalone zero-compute command.

## Verified September 2026 evidence

- The 13-stage Tables 4-10 workflow was executed in the verified full-pipeline
  environment on the NVIDIA L40.
- The final paper-table reconstruction passed 116/116 manuscript-facing
  checks.
- The Stage 5 pretrained cross-encoder reproduced the archived decisions when
  evaluated on the bundled frozen top-50 candidate blocks.
- The ten-seed training-fraction experiment reproduced all substantive values
  across 100 per-seed rows.
- The primary three-seed block-size experiment reproduced all scientific
  values across 18 rows.
- The discriminative-scorer experiment reproduced all scientific values
  across 144 primary-seed rows.
- The nested-bias MC10 output reproduced byte-for-byte.
- The direct-score reconstruction summaries were independently recomputed
  from their per-seed outputs and agreed without differences.
- The separate Table 8 revision workflow freshly trained six bi-encoder cells,
  and the post-run revision verifier passed.
- The combined-routing/Table 11 results were regenerated from those Table 8
  checkpoints and passed the saved-output revision audit.
- Complete environment evidence, including `pip freeze`, `pip check`, core
  package versions and GPU information, is stored under `environment/`.

## Remaining before final reviewer attestation

The scientific reruns are complete. The remaining work is archival:

- freeze the final package contents;
- regenerate `FILE_MANIFEST.sha256`;
- run `python scripts/validate_package.py`;
- rerun the saved-output and dry-run validators; and
- deposit the final package at a real, verified public repository/archive and
  insert its URL/DOI in the manuscript and package documentation.

No placeholder URL or DOI should be presented as an active public deposit.

