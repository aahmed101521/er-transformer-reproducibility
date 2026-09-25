# Random seeds, determinism, and comparison criteria

The primary comparison tables use file-A-level split seeds 42, 43, 44.
The ten-replication studies use seeds 42 through 51. Scripts seed the
Python, NumPy and PyTorch RNGs where applicable.

We do not claim that fine-tuned checkpoint files are bitwise deterministic.
GPU kernels, device configuration, library versions and execution order can
affect trained weights even when the same random seed is supplied.

The reproduction claim concerns the scientific outputs used in the paper:
record-level decisions, declared/correct/missing counts, FLR/MMR values and
the corresponding table summaries.

The three-seed and ten-seed MCSEs measure variability across data splits or
study replications. They are not estimates of same-seed GPU nondeterminism
and are not used as a reproduction tolerance.

September 2026 verified evidence:

- All 13 stages exposed by `reproduce.py` were executed in the same clean,
  verified NVIDIA L40 environment. The stages were run individually with
  `--only` so that each stage could be audited and preserved separately.
- The final paper-table reconstruction checked 116 manuscript-facing values
  and reported zero failures.
- The ten-seed training-fraction experiment reproduced all substantive values
  in 100 per-seed rows exactly.
- The cross-encoder block-size experiment reproduced all scientific values in
  the 18 primary-seed rows exactly.
- The discriminative-scorer experiment reproduced all scientific values in
  the 144 primary-seed rows exactly; differences were limited to runtime,
  provenance path and the recorded scikit-learn patch version.
- The nested-bias MC10 summary reproduced byte-for-byte.
- The Table 8 revision workflow freshly trained six bi-encoder cells and the
  regenerated outputs passed the revision verifier. The archived historical
  quartile lambda/psi quantities agree to six decimal places.
- The untouched pretrained cross-encoder reproduced the archived decisions
  when evaluated on the frozen top-50 candidate blocks distributed with the
  package.

For the scientific-field comparisons where numerical tolerance was required,
the audit used `rtol=1e-12` and `atol=1e-12` and found no changed scientific
values. This is an empirical reproduction criterion for the tested
environment, not a claim that trained checkpoint bytes are identical.

For the recovered Table 8 bi-encoder cells, comparison to the archived audit
is made at six-decimal precision.

A rerun should report (a) seed-level declared/correct/matchable counts,
(b) FLR/MMR under the final definitions, (c) displayed aggregate means and
MCSEs, and (d) any deviations from the archived references. Thresholds or
archived references should not be silently changed merely to force a pass.
