# Reviewer 3 Comment 1 — Stage 1 correction log

Changes to the September 22 archive (no model calculations altered):

1. Updated the bundled paper to the cumulative version through Reviewer 2 Comment 5.
2. Separated **partial historical** version evidence from the **captured recovery**
   environment and the **proposed, unverified full-pipeline** environment.
3. Added `requirements-historical-core.txt` and `requirements-recovery.txt`; revised
   root `requirements.txt` is explicitly only the proposed combined test stack.
4. Rewrote the determinism attestation to distinguish split MCSE from *unmeasured*
   fixed-seed GPU repeat-run tolerance.
5. Added the dedicated `reproduce_revision.py` entry point for Table 8/11,
   which is separate from the 13-stage Tables 4-10 main pipeline.
6. Made the absence of bundled fine-tuned checkpoints/Table 8 prediction rows,
   the standalone paper-table stage's upstream dependency, full GPU rerun status,
   and the missing public archive/DOI explicit in the README.
7. Added a standard-library structural/hash verifier.

No claim of full from-scratch pipeline success, same-seed determinism,
verified public deposit, or a captured single common environment is made.

Next: fresh server installation/dry run/full pipeline and verified deposit.
