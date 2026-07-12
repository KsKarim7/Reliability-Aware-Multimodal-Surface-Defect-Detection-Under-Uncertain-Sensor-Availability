# Full Pipeline Audit — MISDD-MM

## Context

This repo implements a multimodal (image + depth, possibly point cloud) industrial
anomaly detection system, extending a CLIP-based baseline (MISDD-MM) with four
innovations: CorrelatedPromptMLP, DynamicPromptGenerator, GranularTextGuidance,
and SensorSentinel. It's evaluated on MVTec 3D-AD and Eyecandies under simulated
missing-modality conditions (missing rate η). Results already exist in
`ablation_results/`, `ablation_results_v2/`, `ablation_results_v3/`, and `RESULTS.md`.

## Your job

Do a genuine end-to-end audit of this codebase — not a check against a fixed list.
Read the code, trace real data through it, run things where you can, and form your
own independent view of where it's fragile, inconsistent, or doing something other
than what the documentation/thesis claims. The list below is a starting point and
must not be treated as the full scope — if you find something wrong that isn't on
this list, that's exactly what this audit is for. Prioritize anything that could
silently change a reported number over cosmetic issues.

## Known leads to chase down (confirm or dismiss each, don't skip)

1. **Point cloud modality may be disabled.** In `train_cls.py`, all `pc`-related
   lines (feature extraction, gallery building) appear commented out — training
   seems to run on image + depth only, despite the repo containing a full
   PointNet2/point_transformer stack for 3D point cloud processing. Determine:
   - Is this true throughout the actual training/eval path (`train_cls.py`,
     `train_seg.py`, `test_cls.py`, `test_seg.py`), or only in some files?
   - Does `MISDD_MM/model.py` (whichever variant is actually imported) even have
     a functioning `encode_pc` path, or is it half-removed there too?
   - Cross-check against the thesis/paper text and `README.md`/`RESULTS.md` claims
     about "multimodal" fusion — is 2-modality vs. 3-modality what's actually
     being claimed as the contribution?
   - Report this as a clear finding either way: "point cloud is intentionally
     unused because X" or "point cloud appears accidentally disabled, and
     re-enabling it would require Y."

2. **`params.py` imports a package (`PromptAD`) that doesn't exist in this repo**
   (the real package is `MISDD_MM`). Confirm whether this script is dead/unused,
   and whether any other file has similar stale imports left over from an earlier
   version of the project.

3. **Five near-duplicate model files** in `MISDD_MM/`: `model.py`, `model_full.py`,
   and four timestamped `model_backup_*.py` files. Confirm exactly which file
   `MISDD_MM/__init__.py` imports, diff it against the others, and flag if any
   ablation script in `ablation_models/` accidentally imports a stale backup
   instead of the current model.

4. **Three overlapping ablation-result directories** (`ablation_results/`,
   `_v2/`, `_v3/`) with inconsistent coverage (v3 only has `full_model` rows per
   seed, missing the single-innovation ablations). Determine which directory is
   actually canonical/current, and whether `RESULTS.md`'s numbers trace back
   cleanly to one consistent source or are stitched from multiple versions.

5. **Score-fusion formula in `utils/metrics.py`**:
   `img_scores = 1.0 / (1.0/max_map_scores + 1.0/img_scores)`. Confirm this
   matches the intended methodology (vs. e.g. a simple average or max), and
   check it can't silently divide by zero or produce NaN/Inf when either term
   is 0.

6. **`cal_pro_metric`'s hardcoded mask-binarization threshold (0.45)** — confirm
   this matches the actual ground-truth mask value convention for *both*
   MVTec 3D-AD and Eyecandies, since they may not encode masks identically.

7. **`save_check_point` only persists a fixed list of keys** (feature galleries +
   text features) — confirm this is intentional and that the actually-learned
   prompt/module weights are recoverable some other way, not silently dropped.

8. **The "silent skip" failure mode already found once** (LicoriceSandwich on
   Eyecandies, caught via zero-value audit). Re-run the zero-value audit
   (`awk -F',' 'NR>1 && ($2==0 || $3==0 || $4==0)'`) across every current CSV in
   all three ablation_results directories, not just the ones already patched.

## Areas to audit independently (use your own judgment on depth)

- **Data pipeline**: `datasets/dataset.py`, `datasets/mvtec3d.py`,
  `datasets/eyescandies.py` — trace one real sample end-to-end (raw file →
  transform → batch), verify train/test split has zero filename overlap, verify
  the missing-modality mask is applied at the claimed rate η and not some other
  value, verify no label leakage between normal/abnormal splits.
- **Model architecture**: for each of the 4 innovations, write a small isolated
  test with synthetic input tensors — check output shapes, no NaN/Inf, gradients
  actually flow back to the parameters. Verify the near-zero Xavier init
  (gain 0.01) hits only the intended new modules, not the pretrained CLIP
  backbone.
- **Training loop** (`train_cls.py`, `train_seg.py`): confirm optimizer
  (SGD, lr, momentum, weight_decay), scheduler (CosineAnnealingLR), and epoch
  count match what's documented/claimed. Check for any parameters that should
  be frozen but aren't, or vice versa.
- **Evaluation**: confirm per-image and per-pixel metrics are computed on
  identically-ordered data (a silent misalignment here inflates/deflates every
  number without ever raising an error).
- **Config/CLI consistency**: cross-check `params.py`, `create_ablations.py`,
  and every `run_*.sh` script for argument mismatches (e.g. the known
  `--gpu-id` default-1-on-single-GPU issue documented in `README.md` — check if
  similar footguns exist for other flags).
- **Reproducibility**: if you can actually execute training/eval, re-run one
  known config (e.g. seed 111, bagel, full_model) and diff the result against
  the recorded CSV value before trusting anything else you find.

## Deliverable

Write a findings report (`PIPELINE_AUDIT_FINDINGS.md`) with:
- Each finding categorized as **Confirmed bug / Confirmed-but-intentional /
  Needs a decision from Khalid / Cosmetic**.
- File + line references for every finding.
- For anything that could change a reported number, state clearly whether
  existing results in `RESULTS.md` are affected and how.
- A short "safe to proceed to statistical testing" verdict at the end — yes,
  no, or yes-with-caveats, and why.
