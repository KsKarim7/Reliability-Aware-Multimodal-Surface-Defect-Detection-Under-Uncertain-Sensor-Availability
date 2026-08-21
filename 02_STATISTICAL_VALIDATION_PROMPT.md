# Statistical Validation — MISDD-MM (V5 final protocol)

> **This file replaces the pre-audit version.** The earlier version was written when the
> claim under test was "four innovations improve on the baseline," and it drew on the
> V1–V3 result directories. Both are now obsolete: the pipeline audit established that the
> V1–V3 improvements were measurement artifacts, and the empirical phase has since produced
> a two-dataset, three-seed evidence base under a corrected pipeline.

---

## What is being tested now

The central claim is no longer an improvement. It is a **bounded null**:

> Under an all-normal training objective, prompt-side innovations do not measurably change
> gallery-based anomaly detection performance. Across seven configurations, two datasets, and
> three seeds, every configuration lies within a narrow band of the baseline.

This inverts the statistical task. **A difference test that fails to reject does not
establish this claim** — absence of evidence is not evidence of absence, and a reviewer will
say so immediately. The primary analysis must be an **equivalence test**, producing a
positive statement of the form "the effect is bounded within ±δ at 95% confidence."

Difference tests still belong in the report, but as supporting analysis, not the headline.

---

## Canonical data sources

Use **only** these. Everything else in the repository is superseded.

| Study | Directory | Shape |
|---|---|---|
| MVTec 3D-AD main table | `ablation_results_v5_3nn/seed{111,222,333}/` | 7 configs × 3 seeds × 10 classes |
| Eyescandies main table | `ablation_results_v5_eyescandies/seed{111,222,333}/` | 7 configs × 3 seeds × 10 classes |
| Missing-rate sweep | `ablation_results_v5_missing_rate/eta{0.3,0.5,0.9}/seed*/` | baseline + full_model only |
| Gate-3 pilot | `result_diag_gate3_{baseline,full_model}/…/Seed_111-results.csv` | 4 real classes, seed 111 |
| k-selection record | `ablation_results_v5/seed111/` | 1-NN scoring, seed 111 only |

η = 0.7 for the missing-rate curve comes from `ablation_results_v5_3nn/`, not a separate run.

**Do not use** `ablation_results/`, `_v2/`, `_v3/`, `_v4/`, `ablation_results_missing_rate/`,
or `RESULTS.md`. `_v4/` may be cited only when explicitly discussing the pre-LayerScale
pipeline, and must be labelled as such.

Every table in the report carries the producing commit hash. Main results: `e4db663` (or
later). V4-era measurements: tag `v4-campaign-code` (`29251d7`), with the ≤2.4e-4 reproduction
tolerance stated inline.

---

## Known data characteristics

Established during pre-analysis; treat as given rather than re-deriving from scratch, but
spot-check at least one.

**Pairing collapses the seed variance.** Raw baseline seed-means swing considerably
(MVTec SD 0.91 pp; Eyescandies SD 5.77 pp), but paired per-seed differences are tight —
seed-level SD 0.02–0.06 pp on MVTec, 0.05–0.15 pp on Eyescandies. The seed effect moves all
configurations together and is removed by pairing. Per-class SD is roughly 10× the seed-level
figure and is where the residual noise lives.

**Classes are strongly non-exchangeable.** Per-class difficulty ranges from potato (~54) to
rope (~89). The same class across seeds is correlated. Treating 30 class-seed cells as
independent observations would illegitimately shrink confidence intervals.

**Seeds are genuine independent replicates.** `setup_seed()` seeds the RNG before the
missing-modality mask is drawn, so each seed differs in both initialisation and mask. They
cannot be decomposed — seed variance is combined init+mask variance. Note this as a
limitation. (`datasets/seeds_mvtec3d/` is vestigial and unreferenced; masks are generated on
the fly.)

**Gate-3 CSVs contain legitimate zero rows.** Each has 10 rows but only 4 real ones (bagel,
cookie, dowel, peach); the remaining 6 are `0.00` placeholders, not failures. Filter them
before any aggregation. This is the one place where the project's usual "zero row = failure"
rule does not apply.

**Efficiency figures.** Parameter overhead is an exact deterministic count
(7,293,790 / 211,592,193 = 3.447%) with no variance. Latency (~11.12 ms/img, ~90 img/s) is
the mean of 20 timed iterations from a single probe run; per-iteration dispersion was not
recorded. Report it as a single measurement, or re-run capturing standard deviation — do not
present it with implied precision it does not have.

---

## Step 1 — Pre-specify the equivalence margin, before running anything

Choose δ and justify it on grounds **independent of the observed differences**. Circular
justification (picking δ after seeing that the differences are small) is exactly what a
reviewer will look for. Defensible bases include:

- the magnitude of improvement claimed by this line of work (~3.5 pp in the pre-audit results,
  and comparable figures in related prompt-tuning papers);
- the spacing between published baseline values across missing rates (~1–4 pp), as a proxy for
  what the field treats as a meaningful difference;
- a practical-detectability argument grounded in run-to-run variation.

State δ, its justification, and the fact that it was fixed in advance, **at the top of the
report**. If different δ values are reported for MVTec and Eyescandies, justify each
separately — Eyescandies has larger paired variance and may not support as tight a bound.

Report both the pre-specified δ and the **achieved** bound (the narrowest δ the data would
have supported at 95%). The second is informative; only the first is the claim.

---

## Step 2 — Primary analysis: equivalence

For each configuration versus baseline, on each dataset independently:

1. **TOST** (two one-sided tests) on the paired per-seed differences against ±δ, n = 3.
   Report both one-sided p-values, the conclusion, and the 90% CI (the interval whose
   containment within ±δ is equivalent to TOST at α = 0.05).
2. **95% CI on the mean paired difference** — this is the number that carries the paper.
   With n = 3 the t-critical value is 4.303, so report the interval honestly rather than
   leaning on the point estimate.
3. **Bootstrap CI** over seeds and classes (≥ 10,000 resamples), resampling in a way that
   respects the blocking structure. Report alongside the parametric interval; if they
   disagree materially, the bootstrap is the headline and the disagreement is itself a
   finding.

Because n = 3 gives only 2 degrees of freedom, state explicitly what the design can and
cannot support. A tight equivalence bound is achievable given the observed paired SDs; a
powerful difference test is not. Say so rather than letting a reader infer power that is not
there.

---

## Step 3 — Secondary analysis: unit of analysis and structure

**Headline unit is the seed** (n = 3 paired), because a seed is one complete replicate and
exchangeability holds at that level.

**Also fit a mixed model** using all 30 class-seed observations, with class as a blocking
factor and seed as a random effect. This uses the full data while modelling class-difficulty
structure and seed clustering. Report the fixed-effect estimate for configuration with its CI,
and compare against the seed-level result. Where they agree, the conclusion is robust to the
analysis choice; where they diverge, flag it prominently — per the project's standing rule
that any conclusion which flips under a reasonable alternative analysis is presented with
maximum caution.

**Do not** present a naive n = 30 test treating class-seed cells as independent.

Report per-class differences descriptively (tables, not tests), and note:

- `innov1_only` is the only configuration with a systematic negative sign on both datasets
  (−0.05 MVTec, −0.22 Eyescandies) and the largest per-class SD — a faint residual of the
  pre-LayerScale damage, tamed but not exactly zero;
- peach/seed333 shows a small consistent positive cluster (+0.80 to +0.84 across innov4,
  innov2_3_4, full_model) — the one directionally repeatable innovation-bearing effect,
  though within noise. Worth a sentence; not worth a claim.

---

## Step 4 — Supporting difference tests

Run these, but frame them as secondary:

- paired t-test and Wilcoxon signed-rank per configuration versus baseline, both datasets;
- effect sizes (Cohen's d) with CIs alongside every p-value, never a p-value alone;
- Benjamini-Hochberg correction across the family of configuration-versus-baseline
  comparisons, noting which results survive and which do not.

State plainly that a non-significant difference test does not establish equivalence, and that
the equivalence analysis in Step 2 is what supports the claim.

---

## Step 5 — Cross-dataset consistency

- Formally compare seed-to-seed variance between MVTec and Eyescandies (Levene or an F-test on
  variances), replacing the descriptive claim with a tested one.
- Test whether the configuration-versus-baseline differences themselves differ between
  datasets — i.e. does the ceiling hold to the same degree on both, or is one dataset closer
  to a real effect?
- Report the two datasets separately throughout. Do not pool.

---

## Step 6 — The 3-NN scoring claim

This is the one genuinely positive result, and its framing needs care.

`k = 3` was selected on seed 111 and frozen before seeds 222/333 ran. However, **only seed 111
currently has scores under both k = 1 and k = 3** — `ablation_results_v5/` contains seed 111
only. Without k = 1 numbers for the remaining seeds, there is no paired k1-versus-k3
comparison on held-out data, and the claim "the +0.89 pp improvement is confirmed on held-out
seeds" is not supported by the saved artifacts.

**Before weakening the claim, check whether it can be repaired cheaply.** `k` is an
evaluation-time parameter and does not affect training. If the `v5_3nn` MVTec checkpoints for
seeds 222/333 still exist, rescoring them at k = 1 is a pure evaluation pass, not a retraining
campaign, and it restores the held-out validation completely. Check first; report what you
find.

- If the checkpoints exist: rescore at k = 1, run the paired k1-versus-k3 comparison across
  all three seeds, and the held-out framing stands as originally intended.
- If they do not: state the weaker true claim — k was frozen on seed 111 before the remaining
  seeds ran, and the final k = 3 results on 222/333 are consistent with seed 111 — and note
  explicitly that a paired held-out comparison would require retraining. Correct `README.md`,
  which currently states the stronger version.

Either way, report the per-class k1-versus-k3 table for whatever seeds are available, and note
that peach (+2.87) contributes disproportionately to the +0.89 pp mean. A mean carried
substantially by one class is a weaker claim than a uniform improvement, and hiding that
behind an average is precisely the kind of thing this project's audit exists to prevent.

---

## Step 7 — Missing-rate curve

Coverage is `baseline` and `full_model` only, at η ∈ {0.3, 0.5, 0.9} from
`ablation_results_v5_missing_rate/` plus η = 0.7 from the main table. No per-innovation
missing-rate data exists.

- Test whether the **degradation slope** across η differs between baseline and full_model.
- Report the full−baseline gap at each η with CIs.
- Report absolute degradation from η = 0.3 to η = 0.9 for both configurations.
- State the coverage limitation explicitly; do not imply per-innovation robustness results
  that were not run.

---

## Step 8 — Gate-3 pilot

The pre-registered criterion (> +0.5 pp mean over the seed-111 reference, no class regressing
more than 1 pp) is evaluable as a paired comparison over **4 classes, single seed**. The
matched reference is `ablation_results_v5_3nn/seed111/` restricted to those same four classes.

- Filter the 6 placeholder zero rows first.
- Report the paired per-class differences and whether the criterion was met.
- State the design limitation plainly: 4 classes, one seed, so the result is a pilot outcome
  against a pre-registered threshold, not a powered test.
- Note that pre-registration is what makes reporting this null credible, and cite
  `GATE3_PILOT_DESIGN.md`.

---

## Step 9 — Verification and adversarial self-check

- Re-derive at least one reported aggregate directly from the per-class CSVs and confirm it
  matches to the reported precision.
- Confirm the 60 canonical CSVs are 11-line with no unexplained zero rows (excepting the
  Gate-3 placeholders).
- Flag any conclusion that would change under a different reasonable analysis choice —
  parametric versus bootstrap, seed-level versus mixed model, different δ.
- Independently inspect the data for anything statistically suspicious not already documented:
  outliers, unexpected correlation structure, classes behaving unlike the rest, aggregates
  that do not recompute. Report whatever you find, including things not on this list.

---

## Scope statements that must appear in the report

- Results are **RGB + depth**, two modalities. The point-cloud path exists in the repository
  but is disabled throughout — no `encode_pc` method exists in any model variant, and depth is
  derived from the point cloud's Z-channel and processed as a 2D image. Do not describe this
  as three-modality.
- Evaluation is **final-epoch, single evaluation, no best-epoch test-set selection**.
- Training uses gradient accumulation to **one optimizer step per epoch** (50 steps per class),
  reproducing the original full-batch budget.
- `--missing_type both` means either modality may be missing on a given sample, never both
  simultaneously; total missing fraction is exactly η.
- Seed variance bundles initialisation and missing-mask variance and cannot be decomposed from
  this data.

---

## Deliverable

`STATISTICAL_VALIDATION_REPORT.md` containing:

1. The pre-specified equivalence margin, its justification, and confirmation it was fixed
   before analysis.
2. The equivalence result per configuration per dataset — TOST outcome, 95% CI on the mean
   paired difference, bootstrap CI — as the headline.
3. The mixed-model result alongside the seed-level result, with any divergence flagged.
4. Supporting difference tests with effect sizes and multiple-comparison correction.
5. Cross-dataset consistency results.
6. The 3-NN claim at whatever strength the artifacts support, with the checkpoint check
   reported either way.
7. Missing-rate slope comparison with its coverage limitation.
8. Gate-3 pilot outcome against the pre-registered criterion.
9. A plain-language verdict per configuration: *equivalent to baseline within ±δ*,
   *inconclusive*, or *differs from baseline*.
10. All scope statements above.
11. Any data-quality or statistical issue discovered that was not already documented.
12. Producing commit hash for every table.

Where a claim cannot be supported at the strength previously stated, say so and correct the
affected document (`README.md`, `CLAUDE.md`) rather than leaving the stronger version in place.
