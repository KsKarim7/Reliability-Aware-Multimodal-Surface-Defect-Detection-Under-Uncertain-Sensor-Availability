# Pipeline Audit Findings — MISDD-MM

Audit date: 2026-07-12. Scope: full end-to-end audit per `01_PIPELINE_AUDIT_PROMPT.md`.
Method: static code audit of every file in the results pipeline, provenance recomputation
of every reported number from raw CSVs, log analysis of all five training logs, git-history
verification of which code produced which result version, and CPU-only dynamic tests
(executed against the real dataset and the real model classes; outputs quoted verbatim
below). The seed-111 reproduction run was **skipped by explicit decision** because the
missing-rate sweep was training live on the GPU throughout the audit — recorded as a
limitation. The live run was verified healthy before and after the audit
(`misdd-training.service` active, η=0.5 seed 222 progressing).

---

## Executive summary

The **bookkeeping is clean**: every cell of every table in `RESULTS.md` and `CLAUDE.md`
(V1, V2, V3, Eyecandies) recomputes exactly from a CSV in the repo, and the run scripts'
zero-value guards work. The **pipeline itself has two confirmed data bugs and one deep
architectural finding** that change what the results *mean*:

1. RGB↔depth pairs are shuffled for ~99% of MVTec3D `good` samples (loader never sorts).
2. Test images are fed to CLIP in BGR while the feature gallery is built in RGB.
3. The missing-aware prompt machinery — where innovations 1, 2, and 4 live — receives
   **zero gradient from the main loss** (encoder call is under `@torch.no_grad()`) and is
   **entirely unused in the I-AUROC test path**. In the `innov1_only`, `innov2_only`, and
   `innov4_only` ablations, the innovation modules are **frozen at random initialization
   for the whole run**.

Finding 3 is the one that most affects the planned statistical validation: per-innovation
ablation deltas are mechanistically indistinguishable from seed noise, and testing them
for significance would formalize noise. See the verdict at the end.

---

## A. Confirmed bugs

### A1. MVTec3D `good`-split RGB↔depth pairing is misaligned (HIGH impact)

`datasets/mvtec3d.py:25-26` — for the `good` defect type, `img_paths` and `pc_path`
globs are **never sorted** (defect types are sorted at lines 36-38). Glob returns
filesystem (ext4 hash) order, which differs between the `rgb/` and `xyz/` directories.

Dynamic test result (real dataset, all 10 categories, train + test):

```
bagel/train/good:   244/244 PAIRS MISALIGNED  (e.g. img[0]=158 pc[0]=164)
bagel/test/good:     21/22  PAIRS MISALIGNED
... (every category: 95-100% of good samples misaligned; ID sets identical, order differs)
```

Consequences:
- **Every MVTec3D training sample** (train split is all `good`) pairs an RGB image with
  the depth map of a *different object instance*. Depth is derived from `pc_paths`
  (`datasets/dataset.py:36-38`), so the depth modality itself is wrong per sample.
- SensorSentinel (innovation 4) computes quality weights over mismatched RGB/depth pairs.
- At test time, `good` samples fuse the RGB score of sample X with the depth score of
  sample Y (both normal, so labels are unaffected — but scores mix instances).
  Anomalous test samples are correctly paired (their globs are sorted).
- The misalignment is **deterministic per filesystem but not portable across machines**
  — a fresh clone on a different disk can produce different pairings and different numbers.

Eyecandies is **not affected**: `datasets/eyescandies.py` pairs by index string
(`_load_good_split`, lines 76-92).

Effect on reported numbers: affects **absolute** values of all MVTec3D results
identically across configs and seeds (same seed → same loader order → same pairing), so
*internal* comparisons (ablation deltas, seed variance, V1/V2/V3) remain like-for-like.
The comparison against the *published* baseline is discussed in C3.

Fix if desired: add `img_paths.sort(); pc_path.sort()` in the `good` branch — but note
this **changes all MVTec3D numbers** and would require re-running everything to keep
consistency.

### A2. Train/test color-space mismatch (BGR vs RGB) (MEDIUM-HIGH impact)

`train_cls.py:97,136` converts dataset images (cv2 BGR) with `cv2.COLOR_BGR2RGB` when
building the feature gallery and during training. The **test loop does not**:
`train_cls.py:276` does `Image.fromarray(f.numpy())` on the raw BGR array. So every test
image is encoded by CLIP with red and blue channels swapped, while the normal-feature
gallery and the training-time features are RGB. (`_convert_to_rgb` in the transform is a
PIL mode conversion, not a channel swap — it does not fix this.)

Same pattern in `train_seg.py`. Affects both datasets, all configs, all seeds, uniformly —
internal comparisons remain valid, absolute numbers are depressed by an unknown amount
(likely worse on color-heavy Eyecandies than on MVTec3D surfaces).

### A3. Main loss cannot train the missing-aware prompts — innovations 1/2/4 are frozen in their own ablations (CRITICAL for interpretation)

`MISDD_MM/model.py:585-592` — `encode_image_missing` is decorated `@torch.no_grad()`,
in **all 12 model variants** (verified by grep across `model.py`, `model_full.py`, all
backups, all ablation files). This is the only place the missing-aware prompts enter the
CLIP encoder. Therefore the main SCL/triplet loss (`train_cls.py:179-230`) backpropagates
**only into the text prompt learners** (`img_prompt_learner`, `depth_prompt_learner`).

Dynamic test (real `Missing_PromptLearner`, simulating the exact training data flow):

```
6a MAIN-LOSS path (encode under no_grad): mpl params receiving grad = 0/171
6b GRANULAR-style path: gradients reach dynamic_image_gen 6/6, correlated_* 30/36, ...
```

The **only** gradient path into `Missing_PromptLearner` (host of innovations 1 and 2 and
the prompt parameters innovation 4 blends) is the granular auxiliary loss (innovation 3),
weighted at most 0.1 with linear warmup. Consequences per config:

| Config | granular module | Gradient into Missing_PromptLearner |
|---|---|---|
| `innov1_only` | not instantiated (line 568 commented) | **zero — frozen at random init all 50 epochs** |
| `innov2_only` | not instantiated | **zero — frozen** |
| `innov4_only` | not instantiated (module is deterministic anyway) | **zero — frozen** |
| `innov3_only` | instantiated | trains (via aux loss only) |
| `full_model` | instantiated | trains (via aux loss only, weight ≤ 0.1) |

Note the gradient-diagnostic CSVs (`CLAUDE.md` "Gradient diagnostic") measured real
nonzero gradients for innov1/innov2 in full_model — those gradients exist, but they come
exclusively through the granular loss, not the main loss. The "gradient interference"
narrative is really about interference *within the auxiliary loss path*.

### A4. Missing-aware prompts are ignored in the I-AUROC test path (CRITICAL for interpretation)

`MISDD_MM/model.py:834-845` — the `'cls'` forward branch calls `self.encode_image(imgs)`
and `self.encode_image(depths)` **without prompts**; the `all_prompts_image`,
`all_prompts_depth`, and `missing_flag` arguments are received and discarded.
(`MISDD_MM/CLIPAD/model.py:217-219` confirms prompts are only injected when
`missing_type` is passed.) The headline I-AUROC therefore never touches the
missing-aware prompts at inference. The pixel maps from the cls run (P-AUROC/AUPRO in the
CSVs) are gallery-based and equally prompt-free. Only the `'seg'` branch
(`model.py:806-832`, used by `train_seg.py`, which produced none of the reported results)
uses the prompts at inference.

**Combined effect of A3+A4:** the only causal path from innovations 1/2/4 to any reported
number is: (frozen random or weakly-trained) prompts → train-time image features (computed
under `no_grad`, used as constants in the loss) → text-prompt-learner gradients → text
features → test scores. The per-innovation ablation deltas in `RESULTS.md` §1.1
(77.11–77.22, spread 0.11pp) are consistent with this being **seed-noise plus a fixed
random perturbation**, not learned innovation behavior. The four-innovation story as
currently framed ("each innovation contributes...") is not supported by the code's actual
gradient/inference paths.

This may be *inherited* from the upstream MISDD-MM design (the earliest backup,
`model_backup_20260522_1839.py:371`, already has the `no_grad`), which would explain why
the reproduced baseline matches the published number — but that makes it an upstream flaw
the thesis builds on, not a justification.

### A5. Silent zero-row failure mode — mechanism identified, one guard gap remains

The LicoriceSandwich-style silent skip is explained by two pieces of code:
- `utils/csv_utils.py:9-22` — `write_results` pre-creates **every** class row as `0.00`
  when a CSV doesn't exist; a crashed/skipped class simply keeps its zeros.
- `train_cls.py:305-310` — `except Exception: p_roc=0.0; pro_auc=0.0` silently zeroes
  pixel metrics if `metric_cal_pix` throws.

Zero-value audit re-run across **all four** results dirs plus `result/` (verbatim results
in §E): all *current, cited* CSVs are clean. Remaining gap: the missing-rate sweep's
validation (`run_missing_rate_sweep.sh:27,48`) checks only column 2 (`$2==0`) — a
pixel-metric zero from the A5 exception path would pass validation. The v2/v3 runners
check all three columns.

### A6. `_is_full_model` detection misclassifies `innov3_only` as full model

`train_cls.py:156-161` detects "full model" via `hasattr` checks on
`granular_text_guidance`, `dynamic_image_gen`, `correlated_prompt_image`. All ablation
variants **instantiate every module as an attribute** (only `forward` differs), and
`innov3_only` keeps granular — so `innov3_only` passes the check. Confirmed empirically:
the ten `NameError: _compute_grad_diagnostics` tracebacks in `ablation_v2_segment1.log`
all fired on `innov3_only` runs (the diagnostic only runs when `_is_full_model` is true).

Historical impact: **none on recorded results** — `git show 30125c9:train_cls.py`
confirms V2 ran with hardcoded `max_norm=1.0` for all configs, and V3 re-ran only
full_model. But with the **current** code, any innov3_only re-run silently gets the 2×
clip and writes grad-diag CSVs labeled as full-model diagnostics. Also note the current
sweep runs full_model with `--max_norm` default 1.0 → effective clip 2.0 via the ×2 rule,
which is the intended V3 configuration.

---

## B. Confirmed-but-intentional (verify framing in the paper)

### B1. Point cloud modality is disabled — the system is RGB-D, not 3-modality (Lead 1)

- All `pc` feature extraction/gallery lines are commented out in `train_cls.py`
  (91-92, 98, 101, 104, 109-110, 116-117, 121, ...), `train_seg.py`, `test_cls.py`,
  `test_seg.py`.
- **No `encode_pc` method exists in any of the 12 model variants** (only the dead
  `build_pc_feature_gallery`, `model.py:689`). Re-enabling would require writing the
  entire PC encoding path, not just uncommenting.
- Yet `model.py:522` still **instantiates** `CLIPAD.PointTransformer` (GPU memory + RNG
  consumption), `model.py:574` creates an unused `self.proj` parameter, checkpoints save
  zero-filled `pc_feature_gallery1/2`, and `--pc_lambda` (`train_cls.py:417`) is unused
  (it only appears in the wandb run name).
- "Depth" is itself derived from the point cloud's Z channel
  (`datasets/dataset.py:37`), so the point cloud data is used — as a 2D depth image
  through the frozen CLIP image encoder, not through the PointNet2/point_transformer stack.

**Verdict: intentionally unused** (consistently removed across every training/eval file;
the Pointnet2 stack is vestigial). Action needed: make sure the thesis/paper claims
"multimodal = RGB + depth", and consider whether the `Pointnet2_PyTorch` build steps in
`README.md` §6 are still needed at all (they are only needed for the dead import chain
`model.py:14` → `utils/pointnet2_utils.py`).

### B2. Reported metric is best-epoch-on-test-set

`train_cls.py:312-320` evaluates the test set every epoch and keeps the max I-AUROC
(P-AUROC/AUPRO ride along from whichever epoch won). All reported numbers are therefore
"best test AUROC over 50 epochs" — a form of test-set model selection. This is common in
few-shot AD literature but must be disclosed, and it mildly inflates all numbers equally.

### B3. Checkpoints do not persist learned weights; test_cls.py is dead code (Lead 7)

`save_check_point` (`train_cls.py:22-36`) saves only feature galleries plus a
`'text_features'` key that **does not exist** in the state dict (real keys are
`img_text_features`/`depth_text_features`) — so the learned text features are *not*
saved, and the pc galleries that are saved are zero tensors. The learned prompt/module
weights are dropped entirely. They are **not recoverable**: results exist only as CSVs
from the in-training eval; reproducing any number means retraining.

This is survivable *because* `test_cls.py` is already stale dead code: it unpacks a
6-tuple from a dataset that returns 8 (`test_cls.py:37`), unpacks 2 values from
`get_dataloader_from_args` which returns 3 (line 81), and calls `model(img, 'cls')`
against a signature requiring 7 arguments (line 53). It would crash on the first batch.
`test_seg.py` has the same staleness. Nothing in the results pipeline uses them.

### B4. Score fusion is harmonic, and numerically safe in practice (Lead 5)

`utils/metrics.py:24` — `1/(1/max_map + 1/img)` is half the harmonic mean (the same
parallel-combination form used in the seg branch, `model.py:817-820`), so it is a
deliberate design idiom, not a typo. Edge-case test results: fused score is `0.0` when
either input is `0`, `inf` for negative inputs, with only silent RuntimeWarnings. In
practice `img_scores` are softmax outputs in (0,1) and `max_map_scores` are
`(1-cos)/2 ≥ 0`, so neither pathological case can realistically occur. Dismissed as a
numeric risk; **must be described as harmonic fusion in the methodology** (not average/max).

### B5. `cal_pro_metric` 0.45 threshold is a no-op for both datasets (Lead 6)

Masks are binarized to {0,1} upstream (`train_cls.py:289` `m[m>0]=1`; Eyecandies masks
also pass `gt[gt>0]=255` at load), and `specify_resolution` resizes masks with
`INTER_NEAREST` (`utils/eval_utils.py:15`) — verified dynamically: mask values after
resize are exactly `{0.0, 1.0}`. The 0.45 threshold therefore behaves identically for
MVTec 3D-AD and Eyecandies. **Dismissed.**

### B6. Missing-modality semantics and rate are correct (verified empirically)

`missing_setting` with `missing_type='both'` produces η/2 image-missing + η/2
depth-missing, **never both missing on the same sample**, total missing fraction exactly
η, deterministic per seed (verified for η ∈ {0.3, 0.5, 0.7, 0.9}). Applied to train and
test alike. The paper should define "both" precisely as "either modality may be missing"
rather than "both missing simultaneously".

Other verified-fine items: train/test filename overlap is **zero** for all 10 MVTec3D
categories (tested); test dataloader is `batch_size=1, shuffle=False` so image/pixel
metrics are order-aligned (`datasets/__init__.py:51`, area 10 clean); CLIP backbone is
frozen (optimizer at `train_cls.py:124-126` covers only prompt learners + granular; SGD
lr=0.02/momentum=0.9/wd=5e-4 + CosineAnnealing T_max=50 match documentation); the
gain-0.01 Xavier init is constructor-scoped to the innovation modules only (verified:
`|w|max ≈ 0.0016` at init, CLIP untouched); granular prompt templates cover all 20
classes (`MISDD_MM/ad_prompts.py:30,97`).

---

## C. Needs a decision from Khalid

### C1. How to present the four innovations, given A3+A4

Options, roughly in order of honesty-preserving effort:
(a) Reframe: innovations as *training-time feature modulation* for text-prompt learning,
with the granular loss as the only learning signal — and drop per-innovation causal
claims for innov1/2/4; (b) fix the architecture (remove `no_grad` from
`encode_image_missing` during training, use prompts in the cls test path) and re-run
everything — a different paper, likely different numbers; (c) reposition the paper on the
robustness/missing-rate story (already the stated plan in `CLAUDE.md` "Paper Positioning")
and report the ablation as inconclusive. **Statistical testing of per-innovation deltas
before this decision would be testing noise.**

### C2. `ablation_results/innov1_3.csv` and `innov1_4.csv` are byte-identical

`diff` confirms identical files (old 2-column format, p_roc=0.00, both mtime Jul 1).
Two different configs cannot legitimately produce identical per-class values; one is a
mislabeled copy. Neither is cited in `RESULTS.md` tables (only `innov2_4` is, and that
file is distinct and clean). Delete both or re-run if pairwise ablations are needed.

### C3. Baseline provenance: the 73.83 comparison row was never produced by this repo

`RESULTS.md:16,29` uses the published MISDD-MM number for all three seeds. The claimed
reproduction run ("matches exactly", `CLAUDE.md`) left no CSV in the repo, and there is
**no baseline model variant** in `ablation_models/` to re-run it with. Given A1/A2 affect
absolute numbers, the +3.64pp headline vs the published baseline is only valid if the
upstream code shares the same data-loading behavior (plausible — the loader looks
inherited — but unverified). Decision: either recover/recreate the baseline reproduction
CSV and a `model_baseline.py`, or soften the headline claim to "vs published baseline".

### C4. Missing-rate sweep mixes training regimes with the η=0.7 points

The live sweep (η=0.3/0.5/0.9) runs full_model with the **V3 code** (granular warmup +
effective clip 2.0). At η=0.7 you have V1 (77.47, pre-fix code) and V3 (76.71). A rate
curve is only internally consistent if the η=0.7 point is **V3's 76.71**, not V1's 77.47.
Decide before plotting/testing the sweep.

### C5. Stale old-format CSVs still on disk

`result/mvtec3d/both/0.9/csv/*` (all three seeds), `result/mvtec3d/both/0.5/csv/Seed_333`,
and `ablation_results/eyescandies_full.csv` are July-1-era 2-column files with
`p_roc=0.00`. The sweep will overwrite the `result/` ones as it reaches those configs;
`eyescandies_full.csv` duplicates `eyescandies_full_model.csv`'s I-AUROC (74.15) minus
pixel metrics. Nothing currently cites them, but they are zero-audit tripwires — consider
deleting or archiving them.

### C6. `logit_scale` used without `.exp()`

`model.py:699,733` and `train_cls.py:194,215` use raw `logit_scale` (= ln(1/0.07) ≈ 2.66)
as the softmax temperature; canonical CLIP applies `.exp()` (≈ 100), as CLIPAD itself does
(`CLIPAD/model.py:260-262`). At test time both cls and seg scores are 2-class softmaxes,
so ranking (and hence all AUROC/AUPRO numbers) is unaffected; it only softens the training
loss. Likely inherited from upstream. Fixing it would invalidate comparability with
V1/V2/V3 — recommend documenting, not changing, unless a full re-run happens anyway.

---

## D. Cosmetic

- `params.py:3` imports the nonexistent `PromptAD` package — dead script, the only stale
  import repo-wide (Lead 2 confirmed; verified by grep over all `.py`/`.sh`).
- `create_ablations.py:4,116,119` hardcodes `/home/p3766/` (a previous machine's user) —
  cannot run as-is; it is a one-shot generator whose outputs are already committed.
- `README.md` §10 documents the `--gpu-id` default-1 footgun, but it was fixed in commit
  `b9c1928` (default is 0 at `train_cls.py:395`) — stale doc. README §5 also says conda
  env `misdd_mm` while everything else uses `ramsdd`.
- `datasets/dataset.py:43` builds good-sample masks as `np.zeros([img.shape[0], img.shape[0]])`
  (H×H instead of H×W) — harmless for these square datasets.
- `train_cls.py:259` — `if _is_full_model if '_is_full_model' in dir() else False:` works
  but should be `if locals().get('_is_full_model', False):`.
- `test_cls.py:93-95` prints "Pixel-AUROC" for an I-AUROC value (moot; dead code).
- Layer-0 clones of `correlated_prompt_image/depth` are never used (`model.py:443` loop
  starts at 1) — 6/36 of those parameters are dead weight (confirmed in test 6b).
- Training runs with `model.eval()` throughout (`eval_mode` at `train_cls.py:87`;
  `train_mode` never called) — intentional for frozen-CLIP prompt tuning, worth a comment.
- `batch-size` 400 ≥ every train set → exactly **one optimizer step per epoch** (50 total
  updates per class) — visible as `0/1` in the tqdm logs; a methodological fact worth
  stating in the paper.
- `CLAUDE.md` says seed111 missing at η=0.3 — now present (Jul 11); doc is stale.
- Model file inventory (Lead 3): `MISDD_MM/__init__.py:1` imports `.model`;
  `model.py` == `model_full.py` byte-identical right now (sweep runs full model — correct);
  the four `model_backup_*.py` are development snapshots (652→793 lines), none imported by
  anything; `ablation_models/*.py` are standalone copies used only via file-copy by the
  runner scripts, never imported as modules — no stale-import risk found.

---

## E. Zero-value audit (Lead 8) — verbatim summary

Scanned: `ablation_results/`, `ablation_results_v2/`, `ablation_results_v3/`,
`ablation_results_missing_rate/`, `result/**/csv/`. Findings:

| File(s) | Zeros | Interpretation |
|---|---|---|
| `ablation_results_v2/**`, `ablation_results_v3/**`, `ablation_results_missing_rate/**`, `ablation_results/seed*/**`, `eyescandies_seed*/**` | none | clean (LicoriceSandwich patch verified present: 92.96/91.07/73.98) |
| `result/mvtec3d/both/0.5/csv/Seed_222-results.csv` | 4 all-zero rows (peach, potato, rope, tire) | **live in-progress run** (peach was mid-epoch-28 during the audit) — expected |
| `result/mvtec3d/both/0.9/csv/*`, `0.5/Seed_333`, `ablation_results/{innov1_3,innov1_4,eyescandies_full}.csv` | `p_roc=0.00`, no `pro_auc` column | July-1-era 2-column format; superseded/uncited (see C5) |

Log audit: `ablation_v2_segment{2,3,4}.log`, `ablation_v3_fullmodel.log`,
`missing_rate_sweep.log` — zero Tracebacks/CUDA errors/FAILED lines. `segment1.log` has
10 Tracebacks + 1 FAILED, all the `_compute_grad_diagnostics` NameError on innov3_only
(see A6), fixed in commit `30125c9` and successfully re-run.

Results provenance recomputation: every I-AUROC mean in `RESULTS.md` §1.1/§2.1 and every
`CLAUDE.md` V1/V2/V3 table cell matches the CSV recomputation to the 0.01 digit, and V3
values trace to `result/mvtec3d/both/0.7/csv/` files with mtimes inside the V3 run window
(Jul 8 20:28 – Jul 9 05:33). **No stitching inconsistencies found** (Lead 4: `ablation_results`
= V1, `_v2` = V2, `_v3` = V3 full_model with individuals deliberately reused from V2 —
matches the documented intent).

---

## Verdict: safe to proceed to statistical testing?

**Yes-with-caveats for the robustness/missing-rate story; NO for per-innovation ablation
claims as currently framed.**

Safe to test now (numbers are internally consistent, provenance verified):
- Seed variance analyses, V1 vs V2 vs V3 comparisons, and the missing-rate curve —
  **provided** the η=0.7 point uses V3 (C4) and the sweep completes cleanly.
- Full-model vs *reproduced-on-this-pipeline* configurations (everything shares the same
  A1/A2 data path, so deltas are like-for-like).

Not safe to test yet:
- **Per-innovation ablation significance** (innov1/2/4 vs baseline vs full): A3+A4 mean
  these configs differ from each other only through frozen random modules perturbing
  train-time features and RNG divergence. A significance test cannot rescue a broken
  causal path; decide C1 first.
- **The +3.64pp headline vs the published baseline** rests on an unverifiable
  reproduction (C3) on a pipeline with two absolute-value bugs (A1, A2). Statistical
  testing against 73.83 should wait until the baseline provenance is settled.

Audit limitations: no training reproduction run was executed (GPU occupied by the live
sweep — by user decision); `train_seg.py`/Eyecandies data were audited statically and via
loader-level tests only; upstream MISDD-MM source was not available to confirm which bugs
are inherited.

---

# ADDENDUM (2026-07-12, same day): Path B executed — corrected pipeline (V4)

Khalid chose Path B: fix the flaws and re-run everything. The missing-rate sweep was
stopped (its remaining output would have been obsolete old-code results). All fixes below
are applied and verified; the V4 campaign produces `ablation_results_v4/`.

## Fixes applied

| Audit finding | Fix | Where |
|---|---|---|
| A1 pairing misaligned | sort `good` globs (+ sorted defect_types for portability) | `datasets/mvtec3d.py` |
| A2 BGR at test | same BGR→RGB conversion as training in the test loop | `train_cls.py` |
| A3 no-grad prompts | `@torch.no_grad()` removed from `encode_image_missing`; CLIP backbone explicitly frozen via `requires_grad_(False)` (incl. surgery-created attention); in-place ops in CLIPAD made autograd-safe (hook clones, `x[0]=x_ori[0]` → `torch.cat`, residual `+=` → out-of-place) | `MISDD_MM/model.py`, `MISDD_MM/CLIPAD/transformer.py` |
| A4 prompts unused in cls | cls branch now encodes with `encode_image_missing` + prompts under `torch.no_grad()` | `MISDD_MM/model.py` forward |
| **NEW: deep prompts disconnected** | discovered during Path B: deep compound prompts (layers 1–5, innovation 1's output) were injected only into the surgery path `x`, which attention never reads — architecturally a no-op. Now also injected into the attended path `x_ori` (MaPLe-style) | `MISDD_MM/CLIPAD/transformer.py` step-i injection |
| A5 silent pixel zeros | pixel-metric exceptions now print a loud stderr traceback | `train_cls.py` |
| A6 `_is_full_model` misdetection | heuristic removed; clipping is explicit `--max_norm`; grad diagnostics behind explicit `--grad_diag` flag | `train_cls.py` |
| B2 best-epoch on test set | evaluation runs **once at the final epoch** — no test-set model selection | `train_cls.py` fit() |
| B3 useless checkpoints | checkpoints now persist galleries + text features (correct keys) + all learned module weights | `train_cls.py` save_check_point |
| B1 dead pc machinery | `PointTransformer` instantiation, pc galleries, pc methods, dead imports removed | `MISDD_MM/model.py` |
| C2 duplicate pairwise CSVs | `create_ablations.py` regenerates all variants incl. `innov2_4`; stale duplicates superseded | `ablation_models/` |
| C3 no runnable baseline | `baseline` config added (all four innovations off) and included in the V4 campaign | `create_ablations.py`, `ablation_models/model_baseline.py` |

Training-loop consequence: gradients through the encoder no longer fit a full-dataset
batch — `--batch-size` default is now 32 (≈8–10 optimizer steps/epoch instead of 1).

## Hardening added after a near-miss

The first V4 launch crashed: regenerated variants had empty deep-prompt lists because
`create_ablations.py`'s `disable_correlated` commented out the `append` lines (the old
committed variants had been hand-patched after generation — regeneration dropped those
hand fixes). Worse, stopping the crashed segment mid-run left an ablated `model.py`,
and `create_ablations.py`'s copy of `model.py`→`model_full.py` then poisoned the source
of truth. Recovered from git + re-applied fixes. Permanent guards now in place:
- `create_ablations.py` treats `model_full.py` as **read-only source**; it never copies
  `model.py` into it, and `disable_correlated` now *replaces* the appends
  (`append(cross_image)`) instead of deleting them.
- `run_v4_ablation.sh` has a `trap ... EXIT` that restores `model.py` even when killed.
- `/tmp/variant_forward_test.py` (structural forward+grad test of every variant) must
  pass before any campaign launch.

## Verification evidence

- GPU gradient probe (real model, real CLIP): **158/171** `Missing_PromptLearner` params
  receive gradient from the main-loss path (was **0/171**); the 13 without gradient are
  the architecturally-dead layer-0 correlated clones + the unused
  `common_prompt_complete` fallback. `correlated_prompt_image[1]` grad norm 11.5
  (innovation 1 functional for the first time). CLIP: 0 trainable params, 0 grad tensors.
- 2-epoch bagel smoke test (full model): trains at ~1.25 s/step (batch 32, no OOM,
  2.8 GB), final eval I-AUROC 67.56 / P-AUROC 92.34 / AUPRO 68.09, checkpoint saved,
  no errors. Baseline variant smoke-tested end-to-end the same way.
- All 9 model variants pass the structural forward/gradient test.

## Post-launch discovery: text-anchor overfitting under the honest protocol

The first V4 launch produced chance-level I-AUROC (49–64) despite healthy pixel metrics.
Systematic decomposition (component AUROCs now printed at every eval — permanent
`[component-diag]` instrumentation in `train_cls.py`) isolated it in three steps:

1. **Optimization budget** — batch-32 stepping = ~450 optimizer steps vs the 50 full-batch
   steps the protocol was tuned for. Fixed with gradient accumulation: microbatches of 32
   accumulate into **one step per epoch** (exact V1–V3 budget, true gradients). Also moved
   the feature-gallery build to after training using prompted features (gallery now matches
   the test-time representation). This fully restored pixel metrics and the map component
   (87–92 AUROC) but not the fused score.
2. **Ruled out**: deep-prompt injection (disabling it: textual still chance), moving
   features (visual prompts frozen at lr=0: textual still chance).
3. **Root cause: the learned text anchors overfit with training length.** Epoch sweep on
   bagel baseline (textual-only AUROC): ep5 = 51, ep10 = 55, **ep25 = 72**, ep50 = 52 —
   discrimination develops, peaks ~ep25, then collapses; and at ep50 the collapsed scores
   corrupt the harmonic fusion (fused 64 vs map 88). V1–V3 never saw this because
   best-epoch-on-test selection cherry-picked the sweet spot; the honest final-epoch
   protocol exposes it. **Campaign schedule set to Epoch 25** (single-class pilot,
   disclosed), which also halves campaign runtime.

Verified final config on bagel seed 111 (final-epoch, corrected pipeline, Epoch 25):

| Config | I-AUROC | P-AUROC | AUPRO | vs old |
|---|---:|---:|---:|---|
| baseline (25 ep) | 87.40 | 93.35 | 77.77 | — (baseline never runnable before) |
| **full_model (25 ep)** | **92.56** | **93.59** | **78.84** | V1 84.25 / V3 85.07, both best-epoch-inflated |

The full-vs-baseline gap (+5.2pp) is now a real, gradient-connected innovation effect.
Two diagnostic knobs added during the investigation remain (defaults = intended behavior):
`--visual_prompt_lr` (separate lr for the missing-prompt learner, default = `--lr`) and
`MISDD_XORI_INJECT=0` (env gate disabling deep-prompt injection, for ablations).

## Consequences for reporting

V1/V2/V3 numbers and the partial missing-rate sweep were produced by the flawed pipeline
and are **superseded** — they remain on disk/git as history but must not be mixed with V4
numbers. The V1→V3 "gradient interference" narrative described dynamics of the auxiliary
loss only and does not carry over. The V4 campaign (baseline + 4 single innovations +
full model × seeds 111/222/333, η=0.7, Epoch 25, MVTec 3D-AD) establishes the new primary
table; statistical validation should run on V4 results only. Note: V3-era checkpoints for
seed-111 bagel–dowel were overwritten by verification runs (they were gallery-only and
their CSV results are preserved); foam–tire remain intact.
