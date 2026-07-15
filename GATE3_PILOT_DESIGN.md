# Gate 3 pilot design: synthetic-anomaly contrastive prompt objective

## Question this experiment answers

The ceiling result (measurement spine, 2026-07-15) says: *the current all-normal
objective cannot improve the map branch* — prompt-induced feature shifts are
defect-blind and rank-preserving. The one untested route is an objective that has
actually seen anomalies. This pilot tests whether **synthetic-anomaly contrastive
prompt training produces defect-sensitive feature shifts that survive gallery
cancellation** — i.e., whether prompts can beat the frozen-CLIP kNN baseline when
given anomaly supervision.

**A negative result is a publishable strengthening of the ceiling claim** (it then
covers anomaly-aware objectives too, on this architecture). A positive result
revives the prompt architecture as a contribution. Either way the paper gains.

## Synthetic anomaly generation (applied on-the-fly, training only)

Per microbatch, each sample gets a corrupted twin with probability 0.5:

1. **CutPaste** (texture defects): rectangular patch (2–15% of area) copied from a
   random location in the same image, pasted elsewhere with random rotation.
2. **Perlin blend** (blob defects): Perlin-noise mask thresholded to a blob,
   filled with noise-perturbed content from another training image, alpha-blended.

Type chosen 50/50 per twin. The same mask corrupts RGB and depth (geometric
consistency); depth values inside the mask are offset by a random scalar
(±5–20% of depth range) to mimic geometry defects. The binary corruption mask is
kept — it labels which PATCHES are anomalous. Seeded per (class, seed) for
reproducibility.

## Loss (added to the existing SCL objective, weight `--syn_weight`, default 0.2)

Patch-level textual contrast, targeting the map branch's own features:

- Corrupted-patch mid-features (mask-selected, both hook depths) → pushed toward
  the abnormal text anchor and away from the normal anchor (CE over the 2-anchor
  softmax, temperature = logit_scale).
- Clean patches of the same corrupted image → normal anchor (keeps the shift
  differential, not global).
- Global CLS term intentionally EXCLUDED: the spine showed global CLS carries no
  defect signal; supervising it would re-inject noise.

Why patch-level: the map score is per-patch kNN distance; only a patch-level
differential shift can beat gallery cancellation. This is the mechanism the
quadrant probe showed is missing — the loss now explicitly demands it.

## What is trained

Same parameter surface as V5 (prompt learners + LayerScale gammas; CLIP frozen).
No architecture changes. The synthetic twin batches flow through
`encode_image_missing` exactly like normal batches (missing-flags applied to twins
identically, so the missing-modality protocol is preserved).

## Pilot protocol (single seed, 4 classes, ~4 GPU-h)

- Classes: **peach, cookie** (map-weak, where headroom exists), **bagel, dowel**
  (map-strong, regression guards). Seed 111.
- Configs: `full_model` and `baseline` (baseline has prompts too — if synthetic
  supervision helps at all, it should help both; full_model tests whether the
  innovations add anything under a real objective).
- Protocol otherwise identical to the running v5k campaign: Epoch 25, batch 32,
  map scoring, 3-NN, final-epoch eval, `--root-dir ./result_diag` (campaign
  artifacts untouched).
- Reference: the v5k campaign's own seed-111 numbers (same protocol, no syn loss).
- Success criterion (pre-registered): mean over the 4 classes improves > +0.5pp
  over reference with no class regressing > 1pp. Anything less = ceiling extended
  to anomaly-aware objectives; report as such.
- Diagnostics per run: `[component-diag]`, `[gamma-diag]`, plus a new
  `[syn-diag]`: AUROC of corrupted-vs-clean patch separation on a held-out
  training slice (checks the loss actually learned the synthetic task — if this
  is high but test AUROC unchanged, synthetic↛real transfer is the story; if
  low, the objective failed to train and the pilot is inconclusive, not negative).

## Implementation plan (all behind flags, default = current behavior)

- `utils/syn_anomaly.py`: mask + corruption generators (numpy/torch, no new deps;
  Perlin via bilinear-upsampled random grids).
- `train_cls.py`: `--syn_anomaly` (bool, default False), `--syn_weight`,
  corrupted-twin construction in the microbatch loop, patch loss on mid-features
  (hooks already exist), `[syn-diag]` print.
- `run_gate3_pilot.sh`: 4 classes × 2 configs, sequential, after v5k completes.
- Structural test + 2-epoch smoke run before the pilot, as always.

## Risks / honesty notes

- Synthetic≠real: CutPaste/Perlin may not resemble MVTec3D defects (peach bruises
  are subtle color gradients). The [syn-diag] metric separates "didn't learn" from
  "didn't transfer".
- The corrupted twin shares the gallery with its clean source — training pushes
  corrupted patches away from BOTH anchors and gallery; no test-set contact.
- k=3 and all other protocol choices stay frozen; nothing is tuned on the pilot
  classes beyond the pre-registered criterion.
