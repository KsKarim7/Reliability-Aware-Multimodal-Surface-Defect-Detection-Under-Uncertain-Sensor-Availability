# Statistical Validation Report — MISDD-MM (V5 final protocol)

**Producing commit for all data:** `e4db663` (repository HEAD; working tree otherwise clean at
analysis time). V4-era measurements, where referenced, cite tag `v4-campaign-code` (`29251d7`)
at the disclosed ≤2.4e-4 reproduction tolerance.

**Analysis environment:** Python 3.11 / scipy 1.17.1 / numpy 1.26.4 / pandas 3.0.5 /
**statsmodels 0.14.6**. The mixed model in Step 3 is a genuine `MixedLM` fit (no substitute),
reported alongside a hierarchical bootstrap so the two can be compared.

**Analysis provenance note (read first):** this run recomputed **every** number from the source
CSVs. No value was carried over from any previous analysis session. All prior uncommitted work
was lost when the environment was rebuilt from `e4db663`; the two findings below arise directly
from that.

---

## ⚠ Two blocking discrepancies found before analysis

**1. The committed `02_STATISTICAL_VALIDATION_PROMPT.md` at `e4db663` is the OBSOLETE
pre-audit version.** It contains Steps 1–4 ("Consolidate one canonical dataset…"), frames the
claim as *"four innovations improve on the baseline,"* and points at the superseded
`ablation_results/`, `_v2/`, `_v3/` directories. The rewritten equivalence-framed version
(Steps 1–9, V5 canonical sources) was never committed and was lost with the rebuild.

It was recovered from `C:\Users\user6\Downloads\02_STATISTICAL_VALIDATION_PROMPT_1.md` and
restored into the working tree; **this report follows the restored (rewritten) version**, which
is what the instructions describe (the "Step-9 deliverable list" exists only there).
`02_STATISTICAL_VALIDATION_PROMPT.md` is therefore an uncommitted working-tree change and
should be committed alongside this report.

**2. All model checkpoints are gone** (`result/mvtec3d/both/0.7/checkpoint/` exists but
contains **0** `.pt` files). Checkpoints were never committed. This is decisive for Step 6: the
k = 1 rescore of held-out seeds is **impossible without retraining**. See §6.

---

## 0. What is being tested

The claim under test is a **bounded null**: under an all-normal training objective, prompt-side
innovations do not measurably change gallery-based anomaly detection. A difference test that
fails to reject cannot establish this. The **primary analysis is therefore equivalence (TOST)**,
producing a positive statement of the form *"the effect is bounded within ±δ."* Difference
tests appear as supporting analysis only (§4).

---

## 1. Pre-specified equivalence margin

**δ = 1.0 pp, fixed before any difference was computed** (hard-coded as a constant at the top of
the analysis script, above all data loading). Justification is entirely external to the observed
differences:

- the improvement claimed by this line of work pre-audit was ~4.6 pp (RAMS-DD thesis, Table 5.2);
- the published MISDD-MM baseline spacing across missing rates is ~1–4 pp per 0.2 η step
  (MISDD-MM paper, Table I: 77.71 / 76.95 / 73.83 at η = 0.3 / 0.5 / 0.7), a proxy for what the
  field treats as a meaningful difference;
- an effect below 1 pp on this benchmark is below any practical-detectability threshold used in
  the surrounding literature.

δ = 1.0 pp is the **smallest** of those scales, making it the conservative choice. The same δ is
applied to both datasets; Eyescandies has larger paired variance and yields a correspondingly
wider achieved bound, reported separately rather than by loosening δ.

**Achieved bounds** (narrowest δ the data would have supported at 95%, taken at the widest
config):

| dataset | achieved bound (widest 95% CI endpoint) | achieved bound (widest 90% CI endpoint, TOST-equivalent) |
|---|---:|---:|
| MVTec 3D-AD | **±0.220 pp** | ±0.166 pp |
| Eyescandies | **±0.679 pp** | ±0.532 pp |

Both are far inside the pre-specified δ. Only the pre-specified δ is the claim; the achieved
bound is reported because it is informative.

---

## 2. Primary analysis — equivalence (headline)

Paired per-seed differences (config − baseline **within** each seed), n = 3, TOST against
±1.0 pp. 95% CI uses t-crit 4.303 (df = 2). Bootstrap is hierarchical (10,000 resamples: seeds
resampled first, then classes within seed) so the blocking structure is respected.

### MVTec 3D-AD — commit `e4db663`, `ablation_results_v5_3nn/`

| config | mean Δ | seed SD | 90% CI | 95% CI | TOST p | equivalent? | bootstrap 95% |
|---|---:|---:|---:|---:|---:|:---:|---:|
| innov1_only | −0.050 | 0.038 | [−0.115, +0.014] | [−0.145, +0.044] | 0.0003 | **yes** | [−0.180, +0.067] |
| innov2_only | +0.003 | 0.022 | [−0.035, +0.041] | [−0.052, +0.058] | 0.0001 | **yes** | [−0.027, +0.029] |
| innov3_only | +0.004 | 0.002 | [+0.002, +0.007] | [+0.001, +0.008] | <0.0001 | **yes** | [−0.007, +0.017] |
| innov4_only | −0.033 | 0.075 | [−0.160, +0.093] | [−0.220, +0.154] | 0.0010 | **yes** | [−0.158, +0.109] |
| innov2_3_4 | −0.046 | 0.040 | [−0.113, +0.022] | [−0.145, +0.054] | 0.0003 | **yes** | [−0.163, +0.079] |
| full_model | −0.089 | 0.046 | [−0.166, −0.012] | [−0.202, +0.024] | 0.0004 | **yes** | [−0.257, +0.067] |

### Eyescandies — commit `e4db663`, `ablation_results_v5_eyescandies/`

| config | mean Δ | seed SD | 90% CI | 95% CI | TOST p | equivalent? | bootstrap 95% |
|---|---:|---:|---:|---:|---:|:---:|---:|
| innov1_only | −0.223 | 0.184 | [−0.532, +0.087] | [−0.679, +0.233] | 0.0090 | **yes** | [−0.529, +0.013] |
| innov2_only | +0.021 | 0.061 | [−0.081, +0.123] | [−0.129, +0.172] | 0.0006 | **yes** | [−0.075, +0.117] |
| innov3_only | −0.005 | 0.018 | [−0.036, +0.026] | [−0.051, +0.041] | 0.0001 | **yes** | [−0.032, +0.021] |
| innov4_only | −0.076 | 0.161 | [−0.348, +0.195] | [−0.476, +0.324] | 0.0050 | **yes** | [−0.469, +0.192] |
| innov2_3_4 | −0.069 | 0.159 | [−0.338, +0.199] | [−0.465, +0.326] | 0.0048 | **yes** | [−0.451, +0.230] |
| full_model | −0.078 | 0.166 | [−0.358, +0.202] | [−0.491, +0.334] | 0.0053 | **yes** | [−0.435, +0.284] |

**Every configuration is statistically equivalent to baseline within ±1.0 pp on both datasets**
(all TOST p ≤ 0.009; both one-sided p-values are ≤ 0.009 in every case). Parametric and bootstrap
intervals agree in sign and scale throughout; no configuration's conclusion changes between them.

**What the design can and cannot support.** With n = 3 (2 df) the paired SDs are small enough to
support a **tight equivalence bound**, but the same design gives almost no power for a
*difference* test. A genuine effect of a few tenths of a pp would not be detectable here. The
claim made is equivalence within ±1.0 pp — not "no effect exists."

**Two honest nuances visible in the table** (neither changes the verdict):
- **full_model on MVTec**: the 90% CI [−0.166, −0.012] **excludes zero**, and the raw paired
  t-test gives p = 0.077. The effect is bounded well inside δ, but the point estimate is
  consistently, marginally negative rather than centred on zero.
- **innov3_only on MVTec**: the 95% CI [+0.001, +0.008] **excludes zero** (raw paired t
  p = 0.039), i.e. a nominally detectable difference of **+0.004 pp** — statistically
  distinguishable, practically meaningless, and non-significant after multiplicity correction
  (§4). This is a clean illustration of why the equivalence framing, not the p-value, carries
  the claim.

---

## 3. Secondary analysis — unit of analysis and structure

**Headline unit is the seed** (n = 3 paired): a seed is one complete replicate, and
exchangeability holds at that level.

**Mixed model (statsmodels `MixedLM`, REML).** Fitted on all 30 class-seed paired differences per
config, with **crossed random effects for seed and class** (`diff ~ 1`, variance components for
`C(seed)` and `C(cls)` — the standard statsmodels idiom for crossed REs). This uses the full
data while modelling class-difficulty structure and seed clustering, and is what the prompt asks
for; it is **not** a naive n = 30 test.

| config | seed-level mean (95% CI) | mixed-model estimate (95% CI) | agree? |
|---|---:|---:|:---:|
| **MVTec** innov1_only | −0.050 [−0.145, +0.044] | −0.050 [−0.220, +0.120] | ✓ |
| innov2_only | +0.003 [−0.052, +0.058] | +0.003 [−0.026, +0.032] | ✓ |
| innov3_only | +0.004 [+0.001, +0.008] | +0.004 [−0.011, +0.020] | ✓ (CI widens over 0) |
| innov4_only | −0.033 [−0.220, +0.154] | −0.033 [−0.155, +0.088] | ✓ |
| innov2_3_4 | −0.046 [−0.145, +0.054] | −0.046 [−0.178, +0.087] | ✓ |
| full_model | −0.089 [−0.202, +0.024] | −0.089 [−0.281, +0.103] ⚠ no-conv | ✓ |
| **Eyes** innov1_only | −0.223 [−0.679, +0.233] | −0.223 [−0.461, +0.015] | ✓ |
| innov2_only | +0.021 [−0.129, +0.172] | +0.021 [−0.062, +0.104] ⚠ no-conv | ✓ |
| innov3_only | −0.005 [−0.051, +0.041] | −0.005 [−0.026, +0.015] ⚠ no-conv | ✓ |
| innov4_only | −0.076 [−0.476, +0.324] | −0.076 [−0.390, +0.237] | ✓ |
| innov2_3_4 | −0.069 [−0.465, +0.326] | −0.069 [−0.389, +0.250] | ✓ |
| full_model | −0.078 [−0.491, +0.334] | −0.078 [−0.435, +0.278] | ✓ |

**Point estimates are identical to three decimals across methods** — expected, since the design
is perfectly balanced (7 configs × 3 seeds × 10 classes, no missing cells), so every method
estimates the same mean. What differs is the interval, and **every mixed-model interval also
lies inside ±1.0 pp**. The equivalence conclusion is therefore robust to the unit-of-analysis
choice. Notably the mixed CI is *wider* than the seed-level CI on MVTec (it does not borrow the
artificially small n = 3 SD) and *narrower* on Eyescandies (it exploits the class structure the
seed-level test throws away) — the two methods are not simply redundant.

⚠ **Convergence:** four of twelve fits reported non-convergence of the REML optimiser
(MVTec full_model; Eyescandies innov2_only, innov3_only). Their parameter estimates match the
seed-level and bootstrap results exactly, so nothing in the conclusion rests on them, but they
should not be quoted as precise standard errors. The cause is visible in the variance
components: with only 3 seed levels the seed variance is estimated at or near the **boundary**
(e.g. MVTec innov1_only: class var 0.055, **seed var 0.000**, residual 0.060) — a boundary fit
the optimiser cannot improve on. That boundary estimate is itself informative: **after pairing,
seed contributes essentially zero variance**, which is exactly the "pairing collapses the seed
effect" characteristic the design relies on.

**Not reported:** a naive n = 30 test treating class-seed cells as independent. Classes are
strongly non-exchangeable (verified: MVTec difficulty ranges potato 54.2 → rope 88.8;
Eyescandies CandyCane 51.0 → Marshmallow 88.9) and the same class is correlated across seeds.

### Per-class descriptive observations (tables, not tests)

- **innov1_only is the only configuration with a systematic negative sign on both datasets**
  (−0.050 MVTec, −0.223 Eyescandies) and among the largest per-class SDs — consistent with a
  faint residual of the pre-LayerScale damage, tamed but not exactly zero.
- **MVTec, seed 333 / peach**: a small consistent positive cluster (mean **+0.44** across the six
  configs, range +0.00 … +0.84) — the one directionally repeatable innovation-bearing effect,
  still inside noise. Worth a sentence; not a claim.
- **MVTec, seed 111 / cable_gland**: the largest single deviation in the study, **−1.21**
  (full_model), mean −0.51 across configs.
- **Eyescandies, seed 111 / Confetto**: mean **−1.73** across configs (worst −3.68, innov4_only
  and innov2_3_4) — the largest per-class cluster anywhere in the data, and larger in magnitude
  than anything on MVTec. Eyescandies per-class deviations are ~3× MVTec's, which is why its
  achieved equivalence bound is ~3× wider.
- **Eyescandies, seed 111 / PeppermintCandy**: configs split in *both* directions
  (innov1_only −2.56, full_model +2.56) — dispersion, not a shared effect.

---

## 4. Supporting difference tests (secondary)

Paired t-test on seed-level differences (n = 3), Wilcoxon signed-rank, Cohen's d with bootstrap
CI, and Benjamini-Hochberg correction across the six config-vs-baseline comparisons per dataset.

| dataset | config | paired-t p | t p (BH) | Wilcoxon p | W p (BH) | Cohen's d | d 95% (boot) |
|---|---|---:|---:|---:|---:|---:|---:|
| MVTec | innov1_only | 0.1493 | 0.2810 | 0.2500 | 0.3750 | −1.32 | [−5.0, −0.8] |
| | innov2_only | 0.8372 | 0.8372 | 1.0000 | 1.0000 | +0.13 | [−1.3, +1.1] |
| | innov3_only | **0.0390** | 0.2318 | 0.2500 | 0.3750 | +2.84 | [+2.3, +6.4] |
| | innov4_only | 0.5230 | 0.6276 | 1.0000 | 1.0000 | −0.44 | [−1.1, +2.0] |
| | innov2_3_4 | 0.1873 | 0.2810 | 0.2500 | 0.3750 | −1.14 | [−3.6, −0.7] |
| | full_model | 0.0773 | 0.2318 | 0.2500 | 0.3750 | −1.95 | [−6.2, −1.3] |
| Eyes | innov1_only | 0.1705 | 0.6667 | 0.2500 | 0.9000 | −1.21 | [−3.8, −1.0] |
| | innov2_only | 0.6039 | 0.6667 | 0.5000 | 0.9000 | +0.35 | [−0.3, +6.4] |
| | innov3_only | 0.6667 | 0.6667 | 1.0000 | 1.0000 | −0.29 | [−0.3, +0.3] |
| | innov4_only | 0.4980 | 0.6667 | 0.7500 | 0.9000 | −0.47 | [−1.9, +0.4] |
| | innov2_3_4 | 0.5294 | 0.6667 | 0.5000 | 0.9000 | −0.44 | [−11.9, +0.2] |
| | full_model | 0.4996 | 0.6667 | 0.7500 | 0.9000 | −0.47 | [−1.9, +0.3] |

**No comparison survives BH correction on either dataset.** The single nominally significant raw
p (MVTec innov3_only, p = 0.039) corresponds to a **+0.004 pp** effect and rises to 0.232 after
correction.

⚠ **Cohen's d is not interpretable in this design and should not be quoted.** Because the paired
SD is near zero for several configs, d inflates spuriously — MVTec innov3_only reports d = +2.84
("huge" by convention) for a four-thousandths-of-a-point effect. The bootstrap d intervals are
correspondingly absurd (Eyescandies innov2_3_4: [−11.9, +0.2]). Reported for completeness because
the prompt asks for effect sizes alongside p-values; the honest reading is that standardised
effect size is the wrong summary when the denominator is this small.

⚠ **Wilcoxon signed-rank is structurally uninformative at n = 3**: the minimum attainable
two-sided p is 0.25, so it cannot reject at α = 0.05 regardless of the data. Reported as
specified, but it carries no evidential weight here.

**A non-significant difference test does not establish equivalence.** §2 is what supports the
claim.

---

## 5. Cross-dataset consistency

Datasets are reported separately throughout; **nothing is pooled.**

Baseline seed-means — MVTec [77.57, 75.49, 77.22], SD 1.113; Eyescandies [65.24, 67.26, 78.36],
SD 7.065.

| test | statistic | p | reading |
|---|---:|---:|---|
| Levene (seed-mean variance, MVTec vs Eyes) | — | 0.347 | not significant |
| F-test on variances (Eyes / MVTec) | ratio 40.28 | **0.048** | significant |

⚠ **This is the one conclusion that flips under a reasonable alternative analysis** and is
flagged as such. The F-test assumes normality and is extremely sensitive with 2 df; Levene is
robust but nearly powerless at n = 3. The trustworthy statement is the **descriptive** one:
Eyescandies seed-mean SD is ~6× MVTec's (7.07 vs 1.11), driven by the seed-333 block sitting
~11 pp above the other two seeds. Do not lean on either p-value.

**Does the ceiling hold to the same degree on both datasets?** Yes — the config-vs-baseline
difference does not differ between datasets for any configuration:

| config | MVTec Δ | Eyes Δ | independent-t p | p (BH) |
|---|---:|---:|---:|---:|
| innov1_only | −0.050 | −0.223 | 0.187 | 0.920 |
| innov2_only | +0.003 | +0.021 | 0.649 | 0.920 |
| innov3_only | +0.004 | −0.005 | 0.418 | 0.920 |
| innov4_only | −0.033 | −0.076 | 0.697 | 0.920 |
| innov2_3_4 | −0.046 | −0.069 | 0.815 | 0.920 |
| full_model | −0.089 | −0.078 | 0.920 | 0.920 |

Neither dataset is closer to a real effect than the other.

---

## 6. The 3-NN scoring claim — **the claim must be weakened; README corrected**

### Checkpoint check (the prompt's first instruction here)

`result/mvtec3d/both/0.7/checkpoint/` exists but contains **0 `.pt` files**. Checkpoints were
never committed and did not survive the environment rebuild. **Rescoring seeds 222/333 at k = 1
is therefore impossible without retraining** — the cheap repair the prompt hoped for is not
available. Per the prompt's contingency, the weaker true claim is stated and `README.md` is
corrected (§11).

### What the committed artifacts *do* support

`ablation_results_v5/` contains **seed 111 only** (7 configs, 1-NN). Seeds 222/333 are absent.
However — and this is stronger than the probe-based table the README currently carries — the
committed CSVs give a legitimate paired k = 1 vs k = 3 comparison for **all seven configs** at
seed 111, since `ablation_results_v5/seed111/` (1-NN) and `ablation_results_v5_3nn/seed111/`
(3-NN) are the same campaign protocol differing only in k:

| config | 1-NN | 3-NN | Δ |
|---|---:|---:|---:|
| baseline | 76.74 | 77.57 | +0.84 |
| innov1_only | 76.51 | 77.52 | +1.01 |
| innov2_only | 76.74 | 77.60 | +0.86 |
| innov3_only | 76.75 | 77.58 | +0.83 |
| innov4_only | 76.88 | 77.58 | +0.70 |
| innov2_3_4 | 76.88 | 77.57 | +0.69 |
| full_model | 76.61 | 77.48 | +0.87 |

**The k = 3 gain is ~+0.7 to +1.0 pp for every configuration including the baseline** — exactly
what an evaluation-time smoothing that is agnostic to the prompt configuration should look like.
This is a genuinely useful finding and it is *not* what the README's single-column table conveys.

Per-class, baseline, seed 111 (paired t over 10 classes: p = 0.0161 — **single seed, not a
held-out test**):

| class | 1-NN | 3-NN | Δ |
|---|---:|---:|---:|
| bagel | 86.98 | 87.40 | +0.42 |
| cable_gland | 80.51 | 82.27 | +1.76 |
| carrot | 79.57 | 78.73 | **−0.84** |
| cookie | 84.19 | 85.09 | +0.90 |
| dowel | 79.59 | 80.25 | +0.66 |
| foam | 70.69 | 71.94 | +1.25 |
| peach | 78.77 | 81.31 | **+2.54** |
| potato | 55.78 | 56.18 | +0.40 |
| rope | 86.14 | 86.87 | +0.73 |
| tire | 65.15 | 65.70 | +0.55 |
| **mean** | **76.74** | **77.57** | **+0.84** |

**peach alone contributes disproportionately**: it accounts for **2.54 of the 8.37 pp summed
gain (30%)**, and dropping it takes the mean from +0.84 to **+0.65**. A mean substantially
carried by one class is a weaker claim than a uniform improvement, and it should not be hidden
behind an average.

### ⚠ The README's current 3-NN table cannot be reproduced from any committed artifact

The README reports 76.74 → 77.63 (+0.89, "8/10 categories improve"), with per-class values
(bagel 87.45 → 87.40 = −0.05, peach 78.92 → 81.79 = +2.87, …) that **match none of the committed
CSVs**. The closest committed equivalent (baseline, seed 111) gives 76.74 → 77.57 (**+0.84**,
**9/10** improve), with materially different per-class numbers (bagel 86.98 → 87.40 = **+0.42**).
The README's table appears to have been produced by an ad-hoc rescoring probe over checkpoints
that no longer exist; it is therefore **unreproducible** and its 1-NN mean coinciding with the
committed baseline's 76.74 is a coincidence of the flat result surface, not a match.

### The claim, at the strength the artifacts actually support

> `k = 3` was selected on seed 111 and frozen before seeds 222/333 were run. On the selection
> seed it improves the mean by **+0.84 pp** for the baseline config (9/10 classes improve;
> carrot regresses −0.84; peach alone accounts for roughly a fifth of the gain), and by
> +0.69…+1.01 pp across all seven configurations, indicating a configuration-agnostic
> evaluation-time effect. **No paired k = 1 vs k = 3 comparison exists for seeds 222/333**, so
> the held-out confirmation claimed previously is not supported by the saved artifacts.
> Establishing it would require retraining those seeds, since the checkpoints are gone.

---

## 7. Missing-rate curve

Coverage is **`baseline` and `full_model` only** (verified: every `eta*/seed*/` directory
contains exactly those two configs). η = 0.7 comes from `ablation_results_v5_3nn/`, not a
separate run. **No per-innovation missing-rate data exists and no per-innovation robustness
claim is made.**

| η | baseline | full_model | gap (full − base) | gap 95% CI |
|---:|---:|---:|---:|---:|
| 0.3 | 80.00 | 80.04 | +0.039 | [−0.21, +0.29] |
| 0.5 | 77.84 | 77.71 | −0.126 | [−0.54, +0.28] |
| 0.7 | 76.76 | 76.67 | −0.089 | [−0.20, +0.02] |
| 0.9 | 75.40 | 75.44 | +0.042 | [−0.55, +0.63] |

Every gap CI contains zero; the ceiling holds at every missing rate.

**Degradation slope** (per-seed linear fit over η): baseline −7.447 pp per unit-η
(per-seed −8.00, −8.05, −6.29); full_model −7.424 (−7.58, −8.46, −6.24). **Paired-t on slopes
p = 0.933** — no detectable difference in how the two configurations degrade.

**Absolute degradation η 0.3 → 0.9**: baseline **−4.61 pp**, full_model **−4.60 pp** (per-seed
baseline −5.67 / −4.40 / −3.75; full −5.34 / −4.73 / −3.74). Graceful, and identical between
configurations.

---

## 8. Gate-3 synthetic-anomaly pilot

Evaluated as a paired comparison over **4 classes, seed 111 only**. The 6 placeholder `0.00`
rows in each pilot CSV (cable_gland, carrot, foam, potato, rope, tire) were filtered first —
verified as the documented legitimate-zero case, not failures. Matched reference is
`ablation_results_v5_3nn/seed111/` restricted to the same four classes. Criterion pre-registered
in `GATE3_PILOT_DESIGN.md`: **mean > +0.5 pp and no class regressing more than 1 pp**.

| config | bagel | cookie | dowel | peach | mean Δ | 95% CI | worst | paired-t p | criterion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| baseline + syn | −0.11 | +0.14 | +0.26 | −0.69 | **−0.100** | [−0.77, +0.57] | −0.69 | 0.668 | **FAIL** |
| full_model + syn | +0.10 | +0.28 | +0.15 | −0.15 | **+0.095** | [−0.19, +0.38] | −0.15 | 0.369 | **FAIL** |

Both configurations fail the pre-registered criterion; neither approaches the +0.5 pp threshold.

**Design limitation, stated plainly:** 4 classes, one seed. This is a pilot outcome against a
pre-registered threshold, **not a powered test**. Its value is that the threshold was fixed in
advance (`GATE3_PILOT_DESIGN.md`), which is what makes reporting this null credible rather than
post-hoc. The finding extends the ceiling to the one anomaly-aware objective tested, with the
scope limit that the corruptions were crude proxies.

---

## 9. Verification and adversarial self-check

**Aggregate re-derived from source.** MVTec seed-111 baseline recomputes to **77.5740** from its
ten per-class rows [87.40, 82.27, 78.73, 85.09, 80.25, 71.94, 81.31, 56.18, 86.87, 65.70],
matching the 77.57 reported in `CLAUDE.md` and §2 above.

**CSV integrity.** All **69** canonical CSVs were scanned: every file is 11 lines (header + 10
classes) with **no unexplained zero rows**. The only zero rows in the entire canonical set are
the 6 documented Gate-3 placeholders per pilot file (§8).

**Spot-check of the prompt's "known data characteristics"** — all recomputed rather than assumed:
- *"Pairing collapses the seed variance"* — **confirmed**, and independently corroborated by the
  mixed model estimating **seed variance at the boundary (0.000)** on MVTec.
- *"seed-level SD 0.02–0.06 pp on MVTec, 0.05–0.15 on Eyescandies"* — ⚠ **minor correction**:
  with the **sample** SD (ddof = 1, correct for inference) the ranges are **0.002–0.075** and
  **0.019–0.184**. The prompt's stated ranges correspond to the **population** SD (ddof = 0).
  Same data, different convention; this report uses ddof = 1 throughout.
- *"per-class SD ≈ 10× the seed-level figure"* — **confirmed** (MVTec ~0.038 → ~0.335, ≈ 9×).
- *"classes strongly non-exchangeable"* — **confirmed** (potato 54.2 → rope 88.8; CandyCane 51.0
  → Marshmallow 88.9).
- *Efficiency figures* (3.447% params, ~11.12 ms/img) — **not re-derived**; they require a GPU
  probe and no checkpoint or probe artifact survives. Carried from documentation and flagged
  as such; the parameter fraction is a deterministic count and the latency remains a single
  measurement without dispersion, exactly as the prompt cautions.

**Conclusions that would change under a different reasonable analysis choice:**
1. **Cross-dataset variance comparison (§5)** — F-test significant (p = 0.048), Levene not
   (p = 0.347). Flagged in place; the descriptive statement is what should be quoted.
2. **Cohen's d (§4)** — inflates without bound as the paired SD approaches zero; reported but
   explicitly disclaimed.
3. **Mixed-model standard errors for the four non-converged fits (§3)** — point estimates are
   unaffected and match the other two methods; the SEs should not be quoted.
The **primary equivalence conclusion is stable** across parametric, bootstrap, and mixed-model
analyses, on both datasets, for all six configurations.

**Independent inspection — issues found that were not already documented:**
- **The README 3-NN table is unreproducible from committed data** (§6) — the most consequential
  new finding, and it directly contradicts a published claim in the repository.
- **Eyescandies per-class instability is ~3× MVTec's**, concentrated in seed 111 (Confetto
  −1.73 mean across configs; PeppermintCandy splitting ±2.56 between configs). This, not seed
  variance, is what widens the Eyescandies equivalence bound.
- **innov3_only on MVTec is nominally significant raw (p = 0.039) at +0.004 pp** — a textbook
  case of statistical detectability without practical meaning, and an argument for the
  equivalence framing.
- **No evidence of non-independence, duplicated rows, or aggregates failing to recompute** was
  found anywhere in the 69 files.

---

## 10. Verdict per configuration

Both datasets, δ = 1.0 pp pre-specified:

| config | MVTec 3D-AD | Eyescandies |
|---|---|---|
| innov1_only | **equivalent to baseline within ±1.0 pp** | **equivalent within ±1.0 pp** |
| innov2_only | **equivalent** | **equivalent** |
| innov3_only | **equivalent** | **equivalent** |
| innov4_only | **equivalent** | **equivalent** |
| innov2_3_4 | **equivalent** | **equivalent** |
| full_model | **equivalent** | **equivalent** |

No configuration *differs from baseline* in any practically meaningful sense; no configuration is
*inconclusive*. The bounded null is positively supported on two datasets and three seeds, within
achieved bounds of **±0.220 pp (MVTec)** and **±0.679 pp (Eyescandies)**.

Qualifier carried from §2: `full_model` on MVTec has a 90% CI excluding zero (point estimate
−0.089), and `innov1_only` is negative on both datasets. These are *bounded, sub-δ* effects
pointing slightly **against** the innovations, not evidence for them.

---

## 11. Scope statements

- Results are **RGB + depth — two modalities.** The point-cloud path exists in the repository but
  is disabled throughout: no `encode_pc` method exists in any model variant, and depth is derived
  from the point cloud's Z-channel and processed as a 2D image. These results must **not** be
  described as three-modality.
- Evaluation is **final-epoch, single evaluation, with no best-epoch test-set selection.**
- Training uses gradient accumulation to **one optimizer step per epoch** (50 steps per class),
  reproducing the original full-batch budget.
- `--missing_type both` means either modality may be missing on a given sample, **never both
  simultaneously**; the total missing fraction is exactly η.
- **Seed variance bundles initialisation and missing-mask variance** and cannot be decomposed
  from this data (`setup_seed()` seeds the RNG before the mask is drawn).
- Canonical sources only: `ablation_results_v5_3nn/`, `ablation_results_v5_eyescandies/`,
  `ablation_results_v5_missing_rate/`, `ablation_results_v5/` (k-selection record),
  `result_diag_gate3_*/`. The superseded `ablation_results/`, `_v2/`, `_v3/`, `_v4/`,
  `ablation_results_missing_rate/`, and `RESULTS.md` were **not** used.

---

## 12. Documents requiring correction

**`README.md` — required by Step 6 of the validation prompt.** It currently states:

> "Replacing 1-NN minimum gallery distance with 3-NN mean distance: **+0.89 pp** mean, 8/10
> categories improve. `k = 3` was selected on seed 111 alone and frozen before seeds 222/333 ran,
> **so those seeds are held-out confirmation.**"

Two defects: (a) seeds 222/333 have **no** k = 1 scores, so they are not held-out confirmation of
the k choice — they are simply later runs at the frozen setting; (b) the +0.89 / 8-of-10 figures
and the accompanying per-class table are **unreproducible from committed artifacts**, and the
closest committed equivalent gives +0.84 / 9-of-10 with different per-class values.

The corrected wording, the multi-config table, and the peach-contribution caveat are given in §6.
**This correction has been applied to `README.md` in the working tree** as part of this
validation run.

**`02_STATISTICAL_VALIDATION_PROMPT.md`** — the committed copy is the obsolete pre-audit version;
the rewritten version has been restored to the working tree (see the discrepancy note at the top)
and should be committed with this report.

---

## 13. One-line summary

Across seven configurations, two datasets, and three seeds, every prompt-side configuration is
**statistically equivalent to the baseline** within a pre-specified ±1.0 pp margin (achieved
±0.22 pp MVTec / ±0.68 pp Eyescandies), a conclusion stable across seed-level, mixed-model, and
bootstrap analyses; the previously claimed held-out validation of the 3-NN scoring improvement is
**not supported by the surviving artifacts** and has been weakened to what the data shows.
