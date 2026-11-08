# Reviewer 1 Report

> **Reviewer Focus**: Methodological Rigor
> **Paper**: Test-Time Computing for Spatio-Temporal Weather Forecasting with Foundation Models
> **Target Venue**: Scopus-indexed applied machine learning / information sciences journal (master's-thesis paper)
> **Review Date**: 2026-11-07

## Summary

The paper proposes Spatial-Cluster Best-of-$N$ (referred to inconsistently as "SC-BoN" and "ST-TTC" — see Weaknesses), a training-free test-time computing framework that wraps a frozen time-series foundation model (primarily Chronos-2) for spatio-temporal weather forecasting. The method generates full-resolution "fine" candidates and coarse, cluster-aggregated "anchor" candidates from the same frozen backbone, prunes fine candidates with a physics-plausibility residual, and selects a final forecast using a composite verifier that scores trend (DTW), frequency (FFT $\ell_1$), and spatial-graph consistency against the best-matching anchor. The paper is explicitly a pre-results draft; I have reviewed the method and experimental design rather than the (placeholder) numbers.

## Overall Assessment

The high-level framing — frozen backbone, no gradient updates, generate/filter/verify/select — is clearly stated and formally distinguished from test-time adaptation (Eq. 1). The experimental design, on paper, is reasonably thorough (component ablation, compute-budget scaling, backbone transfer, clustering-strategy ablation, budget-split and weight-sensitivity analyses). However, once the placeholder numbers are set aside, the method description itself has several unresolved soundness and reproducibility gaps: core scoring terms ($E_{\text{trend}}$, $E_{\text{FFT}}$, $E_{\text{spatial}}$, the physics residual, the activity penalty) are given only in prose with no equations or units, there is no normalization scheme for combining variables of very different physical magnitude (temperature vs. wind), the physics-violation metric used for evaluation appears to be the same quantity used inside the filter (a circularity risk), and — most seriously — the paper's own Conclusion explicitly states the evaluation is "confined to a single reanalysis dataset (ERA5)," directly contradicting the two-dataset (ERA5 + WEATHER-5K) framing advertised in the Abstract, Problem Definition, and Experimental Settings. These are design- and rigor-level issues independent of whether the final numbers are filled in, and they need to be resolved before the paper is ready for review on its merits.

## Strengths

1. **Clear, formal frozen-backbone constraint**: Eq. 1 ($\theta^{(t)}=\theta$, $\nabla_\theta\mathcal{L}=\mathbf{0}$) gives a crisp, falsifiable definition of "training-free/gradient-free" that cleanly separates the contribution from test-time adaptation methods such as the directly competing ST-TTC of Chen and Liang and from Wang et al.'s TTA. This formalization is a genuine methodological strength and makes the scope of the claim auditable.
2. **Sensible two-scale design rationale**: The motivation for a coarse anchor branch — cheap, low-variance regional references against which expensive fine candidates can be verified without access to ground truth — is well argued in Section 3.1 and directly addresses the "verifier needs some reference signal" problem inherent to Best-of-$N$ in a spatial setting.
3. **Physically motivated pre-filter with a sensible failure-mode guard**: Using an advection/continuity PDE residual to prune unphysical candidates, combined with an explicit activity penalty to avoid rewarding trivially flat forecasts, targets a real, well-documented failure mode of generative time-series models (mode collapse to the mean). The authors are honest that this is a soft filter, since 2m temperature is not exactly conserved.
4. **Thorough planned ablation suite**: The design anticipates the right questions — per-component verifier ablation (Table 5), compute-budget scaling with latency reporting (Table 6, Fig. 2), cross-backbone generality (Table 7), clustering-strategy sensitivity (Table 8), and budget-split/weight-sensitivity analysis (Tables 9–10). This is a comprehensive battery of controls that, if executed well, would substantially support the causal claims the paper wants to make.
5. **Honest, substantive limitations section**: The Conclusion explicitly flags the soft-constraint nature of the physics term, the single-resolution/three-variable scope, the accuracy–compute trade-off, and the static (non-adaptive) clustering. This kind of self-critique is uncommon and commendable — though, as noted below, it also inadvertently exposes an inconsistency with earlier sections.

## Weaknesses

1. **The advertised two-dataset evaluation is not actually realized, and the paper contradicts itself about this** (Major): WEATHER-5K is introduced with a full dataset description (5,672 stations, 5 variables, 48h/24h setting) in Section 6.1 and is explicitly invoked in the Problem Definition ("our experiments cover both cases," Section 3.1) to justify the claim that the framework generalizes across gridded and irregular-station domains. Yet WEATHER-5K never appears again anywhere in the paper — not in the main results table, not in any of the five ablation/analysis subsections, not in the aggregate metrics table. The Conclusion then states plainly: "our evaluation is confined to a single reanalysis dataset (ERA5)... and to three variables." This is a direct self-contradiction between the Abstract/Problem Definition and the Conclusion, and it means the central "works on both continuous grids and irregular station networks" claim is currently untested in the results as presented.
   - *Suggestion*: Either include at least a parallel results/aggregate table for WEATHER-5K (even with placeholder numbers, consistent with the rest of the draft), or scope down the Abstract, contributions list, and Problem Definition to ERA5 only until station-network results exist, and remove the "two datasets" framing from the note/abstract until it is delivered.

2. **Core scoring functions are specified only in prose, with no equations, units, or default hyperparameter values** (Major): $E_{\text{trend}}$, $E_{\text{FFT}}$, $E_{\text{spatial}}$ (Section 5.4), the physics residual, and the activity penalty (Section 5.2) are all described narratively but never written as equations. The verifier weights $w_1,w_2,w_3$ are said to be "tuned on a validation split" but no values, search range, or tuning procedure are given anywhere, including in the ablation sections that sweep $w_3$. The number of clusters $M$ used in the *main* results is never stated (it only appears as $M=32$ in the later budget-allocation analysis). The kNN graph parameter $k$ and the Gaussian bandwidth $\sigma$ (Section 4) are never given numeric values. $K_A$ is never specified as a main-pipeline default. This is a reproducibility gap that has nothing to do with pending final numbers — these are method specifications that should be fixed and reportable regardless of whether the experiments have finished running.
   - *Suggestion*: Add a consolidated hyperparameter table (one row per symbol: $M$, $k$, $\sigma$, $N_A$, $N_C$, $K_A$, $w_1,w_2,w_3$, DTW window, FFT downsampling factor) with the default values used in the main experiments, and give explicit equations for all three verifier terms, the PDE residual, and the activity penalty.

3. **No normalization strategy for combining heterogeneous physical variables in the verifier or the physics filter** (Major): $E_{\text{trend}}$, $E_{\text{FFT}}$, and the PDE residual are computed jointly over temperature (t2m, ~250–320 K) and wind components (u10, v10, roughly $\pm$20 m/s). Without an explicit per-variable normalization or standardization step — which is never mentioned — a linear combination or joint distance metric over these variables will be dominated by whichever variable has the larger absolute scale (almost certainly temperature), biasing both the physics filter and the verifier toward optimizing temperature consistency while effectively ignoring wind consistency. This directly undermines the multivariate claim in the contributions list.
   - *Suggestion*: State explicitly whether each term is computed per-variable and then averaged/weighted, or on standardized (e.g., z-scored) fields, and report the normalization constants used.

4. **Possible metric–objective circularity in the physics-violation evaluation** (Major): The physics-plausibility residual $\mathcal{R}_{\text{phys}}$ used to prune candidates in Section 5.2 appears to be the same quantity reported as the "Physics Violation" evaluation metric in Table 6 (compute-budget scaling). If so, reporting that $\mathcal{R}_{\text{phys}}$ improves with more candidates is close to tautological — the method is explicitly selecting for low $\mathcal{R}_{\text{phys}}$, so of course $\mathcal{R}_{\text{phys}}$ improves; this does not by itself demonstrate improved physical realism in a way that is independent of the optimization target.
   - *Suggestion*: Either clarify that the evaluation metric is a genuinely independent physical diagnostic (e.g., computed from unused conservation laws, finer-scale residuals not used by the filter, or an external physics checker), or explicitly acknowledge the circularity and downweight the strength of claims built on $\mathcal{R}_{\text{phys}}$ improvements alone.

5. **The spatial-consistency term is a global roughness statistic, not a localized pattern-matching term, and can be satisfied by structurally wrong candidates** (Major): $E_{\text{spatial}}$ compares the *aggregate* Dirichlet energy (overall roughness) of a candidate's within-cluster residual to that of the history. Two candidates with very different, even spatially shifted or rotated, local anomaly patterns can have identical aggregate roughness and therefore identical $E_{\text{spatial}}$ scores. The term is therefore a necessary-but-not-sufficient check: it can rule out trivially flat or absurdly noisy candidates, but it cannot distinguish a spatially *coherent-looking but wrong* candidate from a spatially *coherent and right* one. This matters because spatial coherence (Moran's $I$) is the paper's headline claimed advantage.
   - *Suggestion*: Discuss this limitation explicitly, and consider a secondary, localized spatial-consistency diagnostic (e.g., a spatial cross-correlation or pattern-correlation term) in addition to the aggregate roughness functional, or justify in more detail why aggregate roughness is an adequate proxy.

6. **No statistical uncertainty quantification anywhere in the design** (Major): Nowhere in the paper is there a plan to report variance, confidence intervals, or significance tests across test windows or random seeds. Given that both branches draw *stochastic* samples from the backbone's predictive distribution, run-to-run variance at a fixed $N$ is a real concern, particularly at the smaller budgets ($N=1,2,4$) examined in the compute-scaling study. Percentage improvements reported as single numbers (e.g., "a further [X]\% reduction") are not interpretable without an estimate of their variability, and comparisons to the published baselines (PINT, ClimODE, ClimaX, PINFDiT; itself already caveated as not strictly controlled) compound this problem.
   - *Suggestion*: Report mean $\pm$ std (or bootstrap CIs) over multiple forecast origins/seeds for every reported number, and perform a paired significance test (e.g., paired bootstrap or Wilcoxon) at least for the headline single-pass-vs-ours and simple-scorer-vs-composite-verifier comparisons.

7. **Feasibility of the Direct CEM / TuRBO-1 search baselines is unclear at this problem's dimensionality** (Major): Section 6.3 states that Direct CEM and TuRBO-1 "run... directly in forecast (horizon) space." For ERA5 with $|\mathcal{V}|=2048$ locations, $C=3$ variables, and $L_F$ up to 128, the raw output space has on the order of $10^5$–$10^6$ dimensions. Cross-entropy-method Gaussian proposals and Gaussian-process surrogates (TuRBO) are well known to scale poorly beyond a few hundred dimensions without a learned low-dimensional embedding. The paper gives no indication of any dimensionality reduction for these baselines, which raises the concern that the comparison may be structurally unfair (the baselines may be practically inapplicable at this scale rather than genuinely inferior), or alternatively that there is an unstated reduction step that should be documented.
   - *Suggestion*: Explicitly state the search-space dimensionality used by each black-box optimizer baseline and, if a reduced representation (e.g., per-cluster or per-PCA-component parameterization) is used, describe it; otherwise discuss the scalability caveat directly.

8. **Dangling and undefined baseline ("defined above")** (Minor/Major — reproducibility): Section 6.2 refers to "the simple truth-free consensus scorer defined above," but no such definition appears anywhere earlier in the manuscript; the Baselines paragraph (Section 6.1) only names it ("consensus-based Best-of-$N$ without composite verifiers") without specifying how it scores or selects among candidates. As written, this baseline cannot be reproduced or even understood precisely.
   - *Suggestion*: Add an explicit definition (even a single sentence with an equation) of the consensus/simple scorer wherever it is first introduced, e.g., in Section 5 or 6.1.

9. **Naming collision between the proposed method and the directly contrasted prior work** (Minor, but confusing enough to flag under rigor of exposition): The paper's own method is formally named "Spatial-Cluster Best-of-$N$ (SC-BoN)" in the Introduction (Section 1) and in the Figure/Table 1 captions, but is called "ST-TTC" in the Abstract, Highlights, and essentially every subsequent section (Sections 6–8, Conclusion). "ST-TTC" is also the exact acronym used throughout the paper for the directly contrasted prior work by Chen and Liang (2025), which the authors are at pains to differentiate themselves from (gradient-based online calibration vs. frozen gradient-free search). Using the same acronym for both one's own method and the primary contrastive baseline is a serious internal-consistency problem that will confuse readers trying to track which numbers belong to which method.
   - *Suggestion*: Pick one name (I would suggest keeping "SC-BoN," since "ST-TTC" is already claimed by Chen and Liang in the literature) and apply it uniformly via find-and-replace across the whole manuscript.

10. **Short climatology window used for ACC** (Minor): ACC is measured "against the validation climatology" (Section 6.1), and the validation period is only two years (2015–2016 for ERA5). Standard practice in the weather-forecasting literature computes climatology from multi-decade averages; a two-year climatology is a noisy and potentially biased reference (it will reflect the particular weather of those two years rather than a long-run seasonal cycle), which could distort the ACC comparison against published baselines that likely use longer climatologies.
    - *Suggestion*: Either use a longer historical window for climatology (ERA5 is available back to 1940/1979) or justify the two-year choice and discuss its effect on ACC comparability with the published baselines.

11. **Missing bibliography entries for several cited works** (Minor): `references.bib` (in `paper/references.bib`) does not contain entries for several keys cited in the manuscript text, including `wang2025fredf`, `shuman2013gsp`, `scheuerer2015variogram`, `deboer2005cem`, `eriksson2019turbo`, and `han2024weather5k`. These citations would fail to resolve at compile time.
    - *Suggestion*: Add the missing `.bib` entries before the next LaTeX compilation pass; this is a simple fix but should not be left until final submission.

## Questions for Authors

1. Is the "Physics Violation ($\mathcal{R}_{\text{phys}}$)" metric reported in Table 6 computed identically to the residual used inside the physical-plausibility filter of Section 5.2, or is it an independent diagnostic? If identical, how do you plan to address the circularity concern (Weakness 4)?
2. What normalization (if any) is applied to temperature vs. wind-component fields before they are combined inside $E_{\text{trend}}$, $E_{\text{FFT}}$, $E_{\text{spatial}}$, and the physics residual? Please give the exact equations.
3. What are the default values of $M$, $k$ (kNN), $\sigma$ (Gaussian bandwidth), $K_A$, and $w_1,w_2,w_3$ used to produce the main results in Table 2/3, as opposed to the values used in the later sensitivity analyses (which use $M=32$, equal weights, etc.)?
4. Will WEATHER-5K results (a results table analogous to Table 2/3, plus at least the compute-budget and verifier ablations) appear in the final version, or should the two-dataset framing in the Abstract and Problem Definition be scoped down to ERA5-only to match the Conclusion's stated scope?
5. How is CRPS computed from the final pipeline output — is it computed over the full ensemble of $N_A$ (or $N_A+N_C$) candidates prior to selection (treating them as an ensemble), or over some other distributional representation? A single selected point forecast $\hat{x}^F$ does not by itself admit a CRPS without an accompanying predictive distribution.
6. In the Direct CEM / TuRBO-1 baselines, what is the actual dimensionality of the search space, and is any dimensionality reduction applied (e.g., searching per-cluster coefficients rather than the raw $|\mathcal{V}|\times C\times L_F$ tensor)? If not, how many optimizer iterations/budget $B$ are used, and is convergence verified?
7. Across how many independent forecast origins (rolling windows) in the test period are the reported metrics averaged, and will variance/confidence intervals be reported in the final version?

## Detailed Comments

### Abstract
- Uses "ST-TTC" for the proposed method's acronym, which collides with the name used throughout the paper for the directly contrasted prior work (Chen and Liang, 2025). See Weakness 9.
- The claim of "the clearest gains in spatial coherence" is reasonable framing given the verifier design, but note Weakness 5: the spatial term is an aggregate roughness statistic, so this claim should eventually be qualified by what Moran's $I$ actually captures (global autocorrelation, not localized pattern fidelity).

### Introduction
- The three stated research gaps (Section 1, enumerated list) are clearly motivated and map cleanly onto the three method components (filter, two-scale verification, early pruning), which is good structural discipline.
- The contribution bullets introduce "SC-BoN" as the method name, conflicting with "ST-TTC" used elsewhere (Weakness 9).
- The claim "multi-scale Dynamic Time Warping" (third bullet) is not matched by the Method section's description of $E_{\text{trend}}$, which uses a single, ordinary DTW between two trend series at matched real-time spans (Section 5.4) — there is no multi-scale DTW algorithm described (e.g., no hierarchical/pyramid DTW). Please reconcile the terminology or add the multi-scale mechanism.

### Related Work
- Thorough and mostly well-organized; the distinction between test-time search/editing and test-time parameter adaptation (Section 2.3) is a useful organizing frame.
- The contrast with Chen and Liang's ST-TTC (Section 2.3, repeated in Section 3.2) is articulated clearly on substance (gradient-free vs. gradient-based), which makes the shared acronym (Weakness 9) especially unfortunate — the substantive differentiation is undercut by the surface-level naming confusion.
- Consider situating the work also against large-scale, weather-specific foundation models (e.g., Pangu-Weather-style and FuXi/Aurora-style architectures), which, like GraphCast/GenCast, require end-to-end training but are a natural point of reference for "foundation models for weather," distinct from the general-purpose TS-FM category (Chronos/TimesFM/Moirai) currently discussed.

### Problem Definition
- Eq. 1's formalization of the frozen-backbone constraint is a genuine strength (Strength 1).
- The claim that "our experiments cover both cases [grid and station]" (Section 3.1) is not borne out by the Results section (Weakness 1) — this sentence should be revised or the evaluation completed.

### Method
- The four-stage pipeline overview (Section 4.1, Figure 1, Table 1) is clear at a conceptual level.
- Section 4.2 (clustering) and Section 5 (generation, verifier) lack several formal specifications noted in Weaknesses 2–5; as written, an independent implementer could not reproduce the exact scoring functions from the text alone.
- The min-min selection rule (Eq. 3) — minimizing over anchors $j$ and then over candidates $i$ — effectively evaluates up to $N_A \times N_C$ (candidate, anchor) pairs under a noisy, truth-free score. This is a larger implicit search than a standard single-reference Best-of-$N$, and plausibly increases exposure to the verifier-gaming/over-optimization risk that the paper itself cites from Huang et al. (2025) in Section 2.1. This interaction is not discussed and would be worth a sentence of analysis.

### Experimental Settings and Results
- The dataset split design (disjoint validation/test years for both ERA5 and WEATHER-5K) is appropriate.
- No data-contamination discussion: Chronos-2 and the other backbones are pretrained on large, heterogeneous corpora of unknown composition; the paper does not discuss whether ERA5 (a very widely used, public reanalysis dataset) could have been part of any backbone's pretraining data, which would compromise the "zero-shot" framing. This should at minimum be acknowledged as a caveat.
- As noted in Weakness 1, WEATHER-5K is described here but never appears again in any results, ablation, or analysis subsection.
- The baseline suite comparison against published methods (PINT, ClimODE, ClimaX, PINFDiT) is appropriately caveated as not strictly controlled (different setups), which I credit as good practice, but the juxtaposition in a single table with bolded "best" numbers still risks being read as a head-to-head comparison by non-careful readers.

### Ablation Study / Analysis
- The ablation structure (Section 7) is logically sound: leave-one-out on verifier terms, physics-filter toggle, and a random-selection control are the right set of comparisons to isolate component contributions.
- The compute-budget sweep (Section 7.2) and the budget-split / weight-sensitivity analyses (Section 8) do not state which dataset they use explicitly in all cases, though context suggests ERA5 throughout; please state this explicitly in each subsection header or caption.
- Defaults used across different analysis subsections are inconsistent in what is held fixed and what is reported (e.g., $M=32$ and equal weights for the budget-split analysis vs. unspecified $M$ and "tuned" weights for the main results) — see Weakness 2 and Question 3.

### Conclusion
- Commendably candid about limitations, but Limitation 2 ("our evaluation is confined to a single reanalysis dataset (ERA5)... and to three variables") directly contradicts the two-dataset framing established earlier in the paper (Weakness 1). This should be reconciled before the next revision, not left as an implicit admission in the final paragraph.

## Missing References

No fabricated or incorrect references were found; the related-work coverage is reasonably broad for this reviewer's area of focus. I would suggest the authors consider briefly situating their general-purpose TS-FM backbone choice against large-scale, weather-specific foundation model architectures (beyond GraphCast/GenCast, which are already cited), since these represent an important class of "foundation models for weather" that readers will expect to see acknowledged, even if excluded from direct comparison for requiring full training. I leave the detailed novelty-positioning assessment to Reviewer 2's report.

## Minor Issues
- Missing `.bib` entries for `wang2025fredf`, `shuman2013gsp`, `scheuerer2015variogram`, `deboer2005cem`, `eriksson2019turbo`, `han2024weather5k` (see Weakness 11).
- "SC-BoN" vs. "ST-TTC" naming inconsistency throughout (Weakness 9).
- Table 9 and Table 10 captions end with a stray " ." (e.g., "...on validation RMSE. ." and "...fixed trend/frequency ratio. .") — likely a LaTeX artifact from an inserted/removed sentence.
- Section 6.1 ("Forecast settings") never states the default $M$, $K_A$ used in the main pipeline (see Weakness 2).
- The dangling reference to a "simple truth-free consensus scorer defined above" (Section 6.2) with no actual prior definition (Weakness 8).

## Scores

| Criterion | Score (1-10) | Justification |
|-----------|:---:|---|
| Technical Soundness | 4 | Core scoring terms lack equations/normalization, and the physics-violation metric appears to reuse the filter's own optimization target. |
| Novelty | 6 | A reasonable, seemingly new combination of spectral-cluster anchoring, physics pre-filtering, and multi-term verification for frozen TS-FMs, though built largely from existing components (DTW, FFT distance, graph Laplacian smoothness, Best-of-$N$). |
| Significance | 5 | Plausible practical value if the two-dataset generality claim is actually delivered; currently unproven on the station-network case central to the paper's own framing. |
| Clarity | 5 | Generally well-written prose, but undermined by the SC-BoN/ST-TTC naming collision and a dangling baseline reference. |
| Completeness | 3 | WEATHER-5K is described but never evaluated anywhere in Results/Ablations/Analysis, and several default hyperparameters are never specified. |
| Reproducibility | 3 | Missing equations for all three verifier terms, the physics residual, and the activity penalty; missing default hyperparameter values; missing bib entries; no variance/seed reporting plan. |
| **Overall Score** | **4** | Promising design concept undercut by unresolved soundness/consistency gaps (dataset-scope contradiction, missing formal specification, possible metric circularity) that are independent of the pending final numbers. |

## Confidence

4/5. I have strong familiarity with test-time compute literature, graph/spectral methods, and physics-informed forecasting, and I traced every specific claim above directly to manuscript line references and cross-checked the bibliography file. My confidence is not a 5 only because I have not been able to inspect the authors' actual implementation to confirm whether some of the "missing" specifications (e.g., normalization) are in fact handled correctly in code but simply omitted from the text.

## Recommendation

**Weak Reject**

The underlying idea — a frozen, gradient-free, two-scale generate/filter/verify framework for spatio-temporal weather forecasting — is reasonable and clearly motivated, and the planned experimental design is appropriately comprehensive in scope. However, independent of the pending result numbers, the manuscript currently (1) does not deliver the two-dataset (grid + station network) evaluation it advertises and in fact contradicts itself about this in the Conclusion, (2) omits explicit equations and default hyperparameters for essentially all of its core scoring machinery, and (3) has at least one real risk of circularity in its physics-violation evaluation metric. I would upgrade to Borderline/Weak Accept once: the WEATHER-5K scope is either delivered or explicitly removed from the framing, the verifier/filter equations and a full hyperparameter table are added, the physics-metric circularity is resolved or acknowledged, and a plan for statistical uncertainty reporting is incorporated.
