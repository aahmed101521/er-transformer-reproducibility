# Revision-specific Table 8 and combined-procedure recovery

This directory preserves the self-contained recovery workflow used during the September 2026 revision.

## Table 8

From this directory, with the pinned environment active:

```bash
CUDA_VISIBLE_DEVICES=0 python recover_table8_mcse.py --device cuda
```

The script reconstructs the fine-tuned bi-encoder/logistic discrimination-gap analysis for seeds 42-44, verifies the historical August-30 quartile outputs, and writes final Zhang-definition MMR and MCSE results.

## Combined routing procedure

After the Table 8 recovery has produced the recovered fine-tuned bi-encoder checkpoints and `results/table8_predictions.csv.gz`, run:

```bash
CUDA_VISIBLE_DEVICES=0 python reviewer1_combined_procedure.py
```

This fixes the routing cutoffs from the 25th and 50th percentiles of the *training-record* discrimination gaps, applies them unchanged to the test records, pools counts for FLR/MMR, and reports the realised routing proportion and cross-encoder pair counts.

Reference copies of the final CSV outputs used in the revised manuscript are included in this directory and under the top-level `results/revision_20260922/` folder.
