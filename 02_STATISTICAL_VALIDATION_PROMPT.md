# Statistical Validation — MISDD-MM Results

## Context

This should run **after** `01_PIPELINE_AUDIT_PROMPT.md` has been completed and its
findings resolved (especially the point-cloud-disabled question, the
model-file-duplication question, and the three-overlapping-ablation-directories
question). If those aren't resolved yet, stop and flag that first — statistically
validating numbers from a pipeline with unresolved correctness questions just
produces a precise answer to the wrong question.

Existing results live in `ablation_results/`, `ablation_results_v2/`,
`ablation_results_v3/`, and are summarized in `RESULTS.md`. The core claim is:
four innovations (CorrelatedPromptMLP, DynamicPromptGenerator,
GranularTextGuidance, SensorSentinel) improve on a CLIP-based baseline for
missing-modality anomaly detection on MVTec 3D-AD and Eyecandies, tested across
3 seeds (111, 222, 333).

## Your job

Don't just run the tests listed below and stop. Independently inspect the actual
CSV data first — check for anything statistically suspicious that isn't already
called out here (outliers, non-independence between "seeds," suspiciously round
numbers, inconsistent row counts between configs, category-level results that
don't sum/average to the reported aggregate, etc.) before running any test.
Report what you find even if it's not in this checklist.

## Step 1 — Consolidate one canonical dataset before testing anything

- Pick one canonical source among `ablation_results/`, `_v2/`, `_v3/` (or merge
  deliberately with clear, written justification for every merge decision).
- Re-run the zero-value audit on the canonical set:
  `awk -F',' 'NR>1 && ($2==0 || $3==0 || $4==0) {print FILENAME, $0}'`
  across every file. Do not proceed past a file with an unresolved zero row.
- Confirm the **point-cloud-disabled finding** from the pipeline audit is
  factored in here: if point cloud was supposed to be a modality and wasn't
  used, note explicitly that these are 2-modality (not 3-modality) results in
  every output table and report, so the statistical claims aren't
  misrepresenting what was actually tested.
- Produce one clean, versioned CSV (or set of CSVs) that every test below reads
  from, and save it alongside the report.

## Step 2 — Define what's actually testable given the real sample sizes

- Note explicitly: n=3 seeds is a small sample. Any p-value should be reported
  alongside an effect size (Cohen's d or similar) and a confidence interval, not
  presented alone.
- The baseline value (73.83 on MVTec) is a **fixed single number**, not a
  distribution — it cannot be paired-t-tested against a 3-seed distribution.
  Handle this case differently from innovation-vs-innovation comparisons (see
  below).
- Check whether "seeds" are truly independent replicates (different random
  init, same data) or whether any confound (e.g. different hardware runs at
  different times) could bias comparisons — call this out if found.

## Step 3 — Run the following comparisons

1. **Full model vs. each single-innovation ablation** (MVTec and Eyecandies
   separately): paired comparison across the 3 matched seeds.
   - Use paired t-test AND Wilcoxon signed-rank (report both; n=3 makes exact
     tests more honest than relying on normality assumptions).
   - Report mean difference, 95% CI, and effect size for each pairing.

2. **Full model vs. fixed baseline** (MVTec, where baseline has no
   distribution): report the baseline's position relative to the full-model
   3-seed 95% CI. State plainly whether the baseline falls outside that
   interval — this is the honest way to support the claim without misapplying
   a paired test.

3. **Seed-variance comparison, MVTec vs. Eyecandies**: formally test whether
   the seed-to-seed standard deviation is significantly different between the
   two datasets (e.g. Levene's test or an F-test on variances) rather than
   relying on the descriptive claim already in `RESULTS.md` section 2.2.

4. **Ablation contribution via bootstrap**: for each innovation's marginal
   contribution (full model vs. full-minus-that-innovation, or single-innovation
   vs. baseline), bootstrap-resample across seeds and categories (≥1000
   resamples) to get a CI on the delta — this is more robust than a t-test with
   only 3 seeds and should be the headline number if it disagrees with the
   simple mean comparison.

5. **Per-category consistency (Eyecandies)**: using the per-category breakdown
   already in `RESULTS.md` section 2.3, check whether the innovations' benefit
   is consistent across categories or concentrated in a few — a large-magnitude
   average driven by 1-2 categories is a different (weaker) claim than a
   consistent per-category improvement.

6. **Multiple comparisons**: since several innovation combinations are being
   compared against baseline/each other, apply a correction (e.g.
   Benjamini-Hochberg) where multiple p-values are reported together, and note
   where an effect survives correction and where it doesn't.

## Step 4 — Sanity checks before finalizing

- Re-derive at least one reported aggregate number (e.g. the 3-seed mean in
  `RESULTS.md`) directly from the canonical CSVs to confirm no
  transcription error between raw data and the written report.
- Flag any comparison where the statistical conclusion would flip under a
  different reasonable analysis choice (e.g. different canonical-CSV pick from
  Step 1, or parametric vs. non-parametric test) — these are the results to
  present with the most caution.

## Deliverable

Write `STATISTICAL_VALIDATION_REPORT.md` with:
- The canonical dataset used and how it was chosen.
- Every test run, its result, effect size, and CI.
- A plain-language verdict per innovation: "robustly supported," "directionally
  supported but weak given n=3," or "not statistically supported."
- An explicit statement of the point-cloud/modality scope of these results.
- Any statistically suspicious data-quality issue found that wasn't already
  flagged, per the "Your job" note above.
