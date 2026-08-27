# The reliability-corrected ceiling

## What we are bounding

A speech-emotion benchmark scores a model $f$ by accuracy against a label $Y$
that is the **majority vote of a panel of $R$ human raters**. The panel is a
sample. Run the labelling again with a different panel and $Y$ can change.

So $Y$ is a random variable, and the quantity a benchmark reports is

$$\text{acc}(f) \;=\; \mathbb{E}_{X,Y}\big[\mathbb{1}\{f(X)=Y\}\big].$$

$f$ is a deterministic function of the audio. The best it can do is the Bayes
rule against the label-generating process,

$$f^{*}(x) \;=\; \arg\max_{y} \; \Pr(Y=y \mid X=x),$$

which gives the **ceiling**

$$C(R) \;=\; \mathbb{E}_{X}\Big[\max_{y}\ \Pr\big(Y=y \mid X\big)\Big]. \qquad (1)$$

Nothing exceeds $C(R)$ except by fitting something other than $X$. That is the
whole claim, and it is just the Bayes rate written against a noisy label.

## Why not the textbook correction for attenuation

In psychometrics "reliability-corrected" normally means Spearman's disattenuation,

$$\rho_{xy}^{\text{true}} = \frac{\rho_{xy}}{\sqrt{\rho_{xx}\rho_{yy}}},$$

whose dual is the familiar ceiling: a perfectly valid predictor of a measure with
reliability $\rho_{yy}$ can correlate at most $\sqrt{\rho_{yy}}$ with it. That
identity is built on a linear, additive, continuous-error model
($Y = T + E$, $\mathrm{Cov}(T,E)=0$). Forced-choice emotion labels are
**unordered categorical**, and accuracy is not a correlation, so the square-root
law does not transfer — there is no $\sqrt{\alpha}$ shortcut here, and quoting one
would be wrong. Equation (1) is the categorical replacement: same idea (correct
the target for the unreliability of the measure), correct machinery
(Bayes rate under a measurement model instead of disattenuation under a linear one).

## Estimating $\Pr(Y \mid X=x)$

We never observe $\Pr(Y \mid X=x)$; we observe one panel's counts. Let clip $i$
have response counts $n_i \in \mathbb{N}^{K}$ over $K=6$ emotions from $R_i$ raters.

**Model.** Raters responding to clip $i$ draw independently from a clip-specific
categorical distribution $\pi_i$, and the $\pi_i$ are themselves drawn from a
Dirichlet prior centred on the marginal response profile of the clip's intended
emotion class:

$$\pi_i \sim \mathrm{Dirichlet}(a \, m_{c(i)}), \qquad n_i \mid \pi_i \sim \mathrm{Multinomial}(R_i, \pi_i).$$

$m_c$ is the observed response profile for intended class $c$ (so shrinkage goes
toward "what clips of this emotion usually pull", not toward uniform). The single
concentration $a$ is fitted by maximum likelihood on the Dirichlet-multinomial
marginal. Small $a$ means clips are heterogeneous; large $a$ means the counts are
close to multinomial.

**Posterior predictive consensus.** For a fresh panel of $R$ raters,

$$\Pr(Y=y \mid X=x_i) \;=\; \int \Pr\big(\text{majority}(N)=y\big)\ p(\pi_i \mid n_i)\, d\pi_i,
\qquad N \sim \mathrm{Multinomial}(R, \pi_i),$$

with $p(\pi_i \mid n_i) = \mathrm{Dirichlet}(n_i + a\,m_{c(i)})$ by conjugacy.
We evaluate it by Monte Carlo: draw $\pi_i$ from the posterior, draw a panel,
take the majority with uniform random tie-breaking, tabulate. Then

$$\widehat{C}(R) = \frac{1}{N}\sum_i \max_y \widehat{\Pr}(Y=y \mid X=x_i).$$

This ordering matters. Taking the max **after** integrating over the posterior is
what a real Bayes classifier does; taking the max of the raw empirical
proportions first (the naive plug-in) is the max of six noisy estimates and is
badly optimistic at $R_i \approx 7$ — it is the mistake that makes label noise
look smaller than it is.

**Uncertainty.** Clip-level cluster bootstrap: resample clips with replacement,
refit, recompute. We report the 2.5/97.5 percentile interval and $n$ everywhere.

## The assumption-light cross-check

The Dirichlet-multinomial can be misspecified. So we also compute a
**split-half held-out oracle** that assumes almost nothing:

1. Randomly split clip $i$'s raters into halves $A_i$, $B_i$.
2. The oracle predicts the modal response of $A_i$.
3. Score it against the majority vote of $B_i$.

This is a *measured* accuracy of a real predictor against a real held-out panel,
not a model-based extrapolation. It is **conservative** relative to $C(R)$ for two
reasons — the oracle sees only $R/2$ raters instead of the full posterior, and the
target panel is $R/2$ raters and so noisier than the real $R$-rater label. When
the two estimators agree, the parametric assumptions are not doing the work.

## $C(R)$ is a property of the protocol, not of emotion

As $R \to \infty$ the majority vote converges to $\arg\max_y \pi_{iy}$ and
$C(R) \to 1$ (excluding exact ties). **The ceiling is not a bound on how
knowable emotion is. It is a bound on how reproducible this labelling procedure
is at the panel size the benchmark actually used.** We therefore report
$\widehat{C}(R)$ as a curve in $R$, and the headline number at CREMA-D's actual
median panel size. A model that hits $C(R)$ has learned the modal percept of a
$R$-rater crowd — which is a real thing, but it is not "the emotion", and the
gap between those two is the point of the invariance half of this repo.

## Where the ceiling does *not* apply

If a paper evaluates against the **intended** (actor-directed) emotion rather
than crowd consensus, the label is deterministic given the clip and there is
**no label-noise ceiling at all** — only the ordinary Bayes error of the acoustic
signal, which these data cannot identify. Such a paper is free to exceed the
consensus ceiling without doing anything wrong.

The comparison against published SOTA is therefore only valid for papers that
scored against consensus/majority labels. We record the label target for every
SOTA number we cite and refuse to compare across it. Papers that do not state
their label target are reported as unspecified and excluded from the test.

## Assumptions, including the ones that weaken the claim

| # | Assumption | If it fails |
|---|---|---|
| A1 | Raters are exchangeable within a clip: $\pi_i$ is the same for every rater. | **The invariance half of this repo tests A1 directly and expects it to fail.** If raters differ systematically, $\pi_i$ is a mixture and $C(R)$ describes a *randomly composed* panel. That is still the object benchmarks use, so $C(R)$ remains the right bound for them — but it stops being a property of the clip. |
| A2 | The model has unlimited capacity and, in the limit, access to the clip's predictive distribution. | $C(R)$ is then a strict upper bound and no real model should approach it. This makes $C(R)$ *generous*; exceeding it is correspondingly more damning. |
| A3 | The benchmark label is the majority vote of $R$ raters with random tie-breaking. | Other protocols (drop ties, require a supermajority, agreement-filtered subsets) change $C(R)$, usually upward, because they discard the ambiguous clips. We report the tie rate and a drop-ties sensitivity. **A benchmark built only from high-agreement clips has a much higher ceiling and a much narrower claim.** |
| A4 | Dirichlet-multinomial captures the over-dispersion of the counts. | Shrinkage is mis-sized. The split-half estimator is the check; it shares none of this. |
| A5 | Test clips are exchangeable with the clips the ceiling was estimated on. | Speaker-dependent or easy-clip-biased splits sit at a different, higher ceiling. We flag the split protocol for every SOTA number. |
| A6 | Panel size $R$ used for the ceiling matches the panel behind the benchmark label. | The ceiling moves a lot in $R$ at small $R$. Hence the curve, not one number. |

Two more that are not assumptions but limits: this is CREMA-D, so **acted**
emotion with a known intended category — the *optimistic* case, easier than
spontaneous speech; and the ceiling is about the label, not about whether the
construct is worth measuring, which is the separate question the invariance
analysis asks.
