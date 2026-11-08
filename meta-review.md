# Meta-Review: Area Editor Report

> **Paper**: Test-Time Computing for Spatio-Temporal Weather Forecasting with Foundation Models
> **Target Venue**: Scopus-indexed applied machine learning / information sciences journal (master's-thesis paper)
> **Date**: 2026-11-07
> **Reviewers**: 3 independent reviewers (R1 Methodological Rigor; R2 Novelty & Positioning; R3 Clarity & Completeness)

## Score Summary

| Criterion | R1 | R2 | R3 | Average |
|-----------|:--:|:--:|:--:|:-------:|
| Technical Soundness | 4 | 6 | 6 | 5.3 |
| Novelty | 6 | 5 | 6 | 5.7 |
| Significance | 5 | 5 | 6 | 5.3 |
| Clarity | 5 | 6 | 4 | 5.0 |
| Completeness | 3 | 4 | 4 | 3.7 |
| Reproducibility | 3 | 5 | 4 | 4.0 |
| **Overall** | **4** | **5** | **5** | **4.7** |

| Reviewer | Recommendation | Confidence |
|----------|---------------|:----------:|
| Reviewer 1 | Weak Reject | 4/5 |
| Reviewer 2 | Borderline | 4/5 |
| Reviewer 3 | Borderline | 4/5 |

## Paper Summary

The paper proposes a training-free, gradient-free test-time computing (TTC) framework that wraps a frozen time-series foundation model (primarily Chronos-2) for spatio-temporal weather forecasting. It draws full-resolution "fine" candidates and cheap, spectrally-clustered "coarse" regional anchors from the same frozen backbone, prunes the fine candidates with a soft advection/continuity PDE residual (plus an activity penalty against degenerate flat forecasts), then selects a final forecast with a three-term composite verifier that matches each survivor's trend (DTW), frequency spectrum (FFT $\ell_1$), and spatial-graph roughness (graph-Laplacian Dirichlet energy) against its best-matching anchor. The genuinely novel piece is the cross-scale decomposition that lets a cheap low-resolution forecast serve as a truth-free reference for verifying expensive high-resolution candidates in a Best-of-$N$ loop; the remaining ingredients are well-established components assembled into a coherent pipeline for an under-explored niche. The manuscript is a deliberate pre-results draft: all result tables carry placeholder values ([X], --), which, per instruction, I do not penalize.

## Consensus Strengths

1. **Crisp, falsifiable frozen-backbone formalization** (R1, R2, R3): Eq. 2 ($\theta^{(t)}=\theta$, $\nabla_\theta\mathcal{L}=\mathbf{0}$) turns the prose distinction between "test-time search" and "test-time adaptation" into an auditable, checkable constraint, cleanly separating the contribution from gradient-based calibration methods (Chen & Liang's work; Wang et al.'s TTA). R1 and R2 both call this a genuine methodological/positioning strength; R3 agrees it is clear and consistently referenced.

2. **Unusually thorough and well-reasoned ablation design** (R1, R2, R3): All three reviewers independently praise the sensitivity-analysis plan (per-component verifier leave-one-out, physics-filter toggle, random-selection control, compute-budget scaling with latency, cross-backbone transfer, clustering-strategy comparison, budget-split and weight sensitivity). R1 calls it "a comprehensive battery of controls"; R3 calls it evidence of "real methodological maturity." Each study clearly states what it isolates and why.

3. **Honest, non-overclaiming limitations discussion** (R1, R2, R3): The Conclusion candidly flags the soft (not hard) physics constraint, the single-resolution/three-variable scope, the compute-accuracy trade-off, and the static clustering. All three reviewers credit this self-critical framing as good practice, while noting the irony that it also exposes the dataset-scope contradiction (below).

4. **Substantive, specific related-work differentiation** (R1, R2, R3): Rather than vague "different domain" hand-waving, the paper gives precise mechanistic contrasts with the works it most overlaps (Chen & Liang's ST-TTC, PINFDiT, DEFT, SCALER). R2, the novelty specialist, independently verified these characterizations against public sources and found them accurate; R1 and R3 note the Related Work section is thorough and well-organized.

5. **The cross-scale (coarse-anchor) design rationale is sound** (R1, R2): R1 praises the motivation for a cheap, low-variance regional reference against which truth-free verification can be performed; R2 identifies the $Y_{\text{global}}/Y_{\text{local}}$ decomposition (trend/frequency at cluster scale, spatial term at residual scale) as "the most genuinely novel technical piece of the paper." This is the element I regard as the paper's real intellectual contribution.

6. **Physically-motivated pre-filter with an explicit failure-mode guard** (R1, R2): The advection/continuity residual plus the activity penalty against trivially flat forecasts targets a real, documented failure mode (collapse to the mean), and is honestly presented as a soft filter because surface temperature is not exactly conserved.

## Consensus Weaknesses

1. **The advertised two-dataset (ERA5 + WEATHER-5K) evaluation is structurally absent and the paper contradicts itself about it** (R1, R2, R3 — all three, flagged Major by each): WEATHER-5K is given a full dataset description (5,672 stations, 5 variables, 48h/24h setting) in Section 3.1 and Section 5.1, and is the sole justification for the paper's central "works on both continuous grids and irregular station networks" claim. Yet it appears in zero results tables, figures, or ablation subsections, and the Conclusion states plainly that "our evaluation is confined to a single reanalysis dataset (ERA5)." This is a direct self-contradiction between the Abstract/Problem Definition and the Conclusion. R3 additionally notes the ablations jump (Q1)-(Q3) to (Q5), with the missing (Q4) strongly suggesting a cross-dataset generality study was planned and dropped without reconciling the surrounding text. This is a missing *section/table structure*, not a pending-numbers issue, and so is in scope for review.

2. **Method name collides with the paper's own primary differentiator ("ST-TTC")** (R1, R2, R3 — all three, Major): The proposed method is called "SC-BoN" in the Introduction, contribution list, figure/table captions, and one row of the search-strategy table, but "ST-TTC" in the Abstract, Highlights, the entire Results/Ablation text, and the Conclusion. "ST-TTC" is also the exact acronym the paper assigns throughout to Chen & Liang's prior work, which it spends a full paragraph distinguishing itself from on the grounds of *not* being an ST-TTC-style gradient-based calibrator. A reader cannot tell from the acronym alone whether a given sentence describes the proposed method or the competing baseline. This actively undermines the paper's central positioning claim.

3. **Core scoring/filtering terms lack equations, and several main-pipeline default hyperparameters are never stated** (R1, R3 Major; R2 via Reproducibility): $E_{\text{trend}}$, $E_{\text{FFT}}$, $E_{\text{spatial}}$, the PDE residual $\mathcal{R}_{\text{phys}}$, and the activity penalty are all described only in prose; only the linear combination (Eq. 3) and the arg-min rule (Eq. 4) are written as equations. The main-results defaults for $M$, $k$ (kNN degree), $\sigma$ (Gaussian bandwidth), $K_A$, and the weight ratio $w_1{:}w_2{:}w_3$ are never given; the only numeric $M{=}32$, $K_A{=}5$ appear inside the later budget-allocation ablation and are never flagged as the main defaults. This is a reproducibility gap independent of pending numbers.

4. **Portability of the physics filter to WEATHER-5K's variable set is unaddressed** (R2, R3): The advection/continuity residual is formulated around temperature transported by orthogonal horizontal wind (u10, v10) — exactly ERA5's variables. WEATHER-5K provides wind *direction and speed* plus dewpoint and sea-level pressure, which the residual does not obviously cover. The paper never states how u/v are recovered from speed/direction for WEATHER-5K, nor how the filter treats variables it does not model. This is tightly coupled to Weakness 1: the "treats $\mathcal{V}$ uniformly" generality claim covers the spatial graph but has not been shown to extend to the physics component.

5. **Missing bibliography entries (`han2024weather5k`, `wang2025fredf`, `shuman2013gsp`, `scheuerer2015variogram`, `deboer2005cem`, `eriksson2019turbo`)** (R1, R2, R3 — all three, flagged Minor by each): *AE adjudication — discounted.* Per the editorial context, this is a build-configuration artifact: the reviewers read a stale local `paper/references.bib`, while the canonical Overleaf bibliography contains verified entries for all six keys. I treat this finding as a non-issue and do not carry it into the Must-Fix list. Authors should nonetheless ensure the submitted build uses the canonical `.bib` so the citations resolve.

## Points of Disagreement

### Technical soundness: is the method design sound, or are there rigor-level defects? (R1 = 4 vs R2 = 6, R3 = 6)

- **R1 argues**: Beyond the specification gaps, there are substantive rigor concerns — no normalization across heterogeneous variables (temperature ~250-320 K vs wind ~$\pm$20 m/s) so the verifier/filter will be dominated by temperature; a likely circularity in the physics-violation metric; the aggregate-roughness spatial term can be satisfied by structurally wrong candidates; no uncertainty quantification; and possibly infeasible CEM/TuRBO baselines at $10^5$-$10^6$ output dimensions. R1 therefore scores soundness 4.
- **R2 and R3 argue**: The design (cross-scale decomposition, PDE pre-filter, graph-Laplacian term) is coherent and well-motivated; they score soundness 6, flagging mainly that it "cannot be fully assessed" without equations/results.
- **AE Assessment**: R1 is the more penetrating read here, and its specific concerns are valid and were simply outside R2's and R3's declared focus. The normalization gap (R1 W3) and the circularity risk (R1 W4) are real and would materially affect results if left unhandled. However, these are *specification and analysis gaps*, not evidence that the core idea is unsound — none of them implies the method cannot work, only that it is under-specified and under-validated. I therefore place true technical soundness between the two camps: the concept is sound (as R2/R3 hold), but the execution has genuine rigor gaps that must be closed (as R1 holds). R1 is not an unjustified outlier; it is doing the job its focus demands.

### Reproducibility: are the hyperparameters specified? (R1 = 3, R3 = 4 vs R2 = 5)

- **R2 argues**: "Hyperparameters ($M$, $N_A$, $N_C$, $K_A$) and dataset splits are specified," scoring reproducibility 5.
- **R1 and R3 argue**: $M$ and $K_A$ are given *only* inside the Section 5.5 ablation ($M{=}32$, $K_A{=}5$) and never flagged as the main-results defaults; $k$, $\sigma$, and $w_1{:}w_2{:}w_3$ are never given at all; and no verifier/physics equations exist. They score 3-4.
- **AE Assessment**: R1 and R3 are factually correct here and R2 is too generous on this specific point. A careful trace of the manuscript confirms the main-pipeline defaults for $M$, $k$, $\sigma$, $K_A$, and the weight ratio are not stated anywhere. R2's own Reproducibility justification even concedes "exact verifier/physics-filter equations ... are not fully spelled out," which is slightly inconsistent with its score of 5. I side with R1/R3: reproducibility is currently weak (effective ~3-4) and the consolidated hyperparameter table is a Must-Fix.

### Novelty: how original is the contribution? (R2 = 5 vs R1 = 6, R3 = 6)

- **R2 argues** (as the novelty specialist): The niche (TTC for spatio-temporal forecasting) was opened by Chen & Liang, and TTC for frozen TSFMs by DEFT/SCALER; the paper's novelty is the *intersection* (frozen TSFM + continuum physics + cross-scale verification), which is real but narrower than the Abstract implies — hence 5.
- **R1 and R3 argue**: A reasonable, seemingly new combination for an under-explored setting — 6, while both explicitly note this is not their focus.
- **AE Assessment**: I weight R2 most heavily here by expertise. The honest characterization is "combination/positioning novelty at a well-defined intersection," not a fundamentally new algorithm. The cross-scale truth-free anchoring is the one piece with standalone originality. A ~5-6 novelty is appropriate and acceptable for the target Scopus-tier applied venue, provided the Abstract's blanket "has not been extended to spatio-temporal forecasting" claim is narrowed to the real, defensible gap (see Should-Fix).

Beyond these, the reviewers are in strong agreement; there are no genuine contradictions among them on the substantive findings, only differences in emphasis driven by their assigned focuses.

## Review Quality Assessment

All three reviews are thorough, well-evidenced, and traceable to specific manuscript locations; confidence 4/5 across the board is appropriate given the placeholder numbers. 

- **R1 (Methodological Rigor)** is the most incisive and adds the most unique value: the normalization gap, the physics-metric circularity, the aggregate-roughness limitation of $E_{\text{spatial}}$, the CRPS-from-a-point-forecast question, and the CEM/TuRBO dimensionality concern were caught by R1 alone and are all legitimate. Its Weak Reject is the harshest recommendation but is driven by substance, not pedantry. 
- **R2 (Novelty & Positioning)** did exemplary diligence, independently verifying the related-work characterizations against public sources, and uniquely caught the Abstract/Introduction self-contradiction, the un-delivered over-optimization claim, the over-broad Gap #1, and the missing Mean-of-$N$ baseline. Its one slip is the overly generous Reproducibility score (adjudicated above). 
- **R3 (Clarity & Completeness)** performed the most systematic structural audit — cross-checking every table/figure label, catching the missing (Q4) ablation, the "Step 2 without Step 1" table, the budget-split table/text mismatch, the undefined metrics (Moran's $I$, ACC, RMSE$_{\mathrm{lw}}$), and the "divergence vs Dirichlet energy" terminology slip.

The three reviews are highly complementary with minimal overlap beyond the three shared Major findings (dataset scope, naming, missing equations) and the discounted missing-citations artifact. This is a well-functioning review panel. The only correction I apply as AE is to discount the missing-citation finding as a build artifact (see Consensus Weakness 5).

## Prioritized Revision Plan

Authors should address the following issues **in order of priority**:

### Must Fix (Required for acceptance)

1. **Resolve the WEATHER-5K dataset scope** (R1, R2, R3): Choose one path and apply it everywhere. Either (a) add WEATHER-5K results as the missing "Q4" generality study — parallel main-results/aggregate and at least the compute-budget and verifier ablations, placeholder numbers are acceptable consistent with the rest of the draft — and rewrite the Conclusion's "confined to a single reanalysis dataset" caveat; or (b) demote WEATHER-5K to future work consistently across the Abstract, Problem Definition (Section 3.1's dual-domain framing), and Experimental Settings. The current state (promised twice, delivered nowhere, denied in the Conclusion) is not acceptable.

2. **Specify how the physics filter handles WEATHER-5K** (R2, R3) — only if path (a) above is chosen: State the speed/direction-to-(u,v) conversion and how the residual treats variables it does not model (dewpoint, sea-level pressure), so the "treats $\mathcal{V}$ uniformly" claim is honored by the physics component, not only the graph.

3. **Fix the SC-BoN / ST-TTC naming collision** (R1, R2, R3): Standardize on a single name for the proposed method — SC-BoN is the safer choice since "ST-TTC" is already claimed by Chen & Liang — and apply it by global replace across the Abstract, Highlights, every section, every caption, and the Conclusion. Reserve "ST-TTC" exclusively for the prior work.

4. **Add explicit equations and a consolidated default-hyperparameter table** (R1, R3; R2 via Reproducibility): Write out $E_{\text{trend}}$, $E_{\text{FFT}}$, $E_{\text{spatial}}$, $\mathcal{R}_{\text{phys}}$, and the activity penalty as equations, and add one table giving the main-results defaults for $M$, $k$, $\sigma$, $N_A$, $N_C$, $K_A$, $w_1{:}w_2{:}w_3$, the DTW window, and the FFT downsampling factor. These are method specifications fixable independently of final numbers.

5. **State the variable-normalization scheme** (R1): Specify explicitly whether $E_{\text{trend}}$, $E_{\text{FFT}}$, and $\mathcal{R}_{\text{phys}}$ are computed per-variable then combined, or on standardized (z-scored) fields, and report the normalization constants. Without this, temperature will dominate wind in every joint metric, silently invalidating the multivariate claim.

6. **Resolve the physics-violation metric circularity** (R1): Either demonstrate that the Table 6 "Physics Violation" metric is an independent diagnostic (e.g., a conservation law or finer-scale residual not used by the filter), or explicitly acknowledge that it reuses the filter's optimization target and downweight claims built on it.

### Should Fix (Strongly recommended)

1. **Define a statistical-uncertainty reporting plan** (R1): Report mean $\pm$ std or bootstrap CIs over multiple forecast origins/seeds, and paired significance tests (paired bootstrap or Wilcoxon) at least for single-pass-vs-ours and simple-scorer-vs-composite-verifier. Given both branches sample stochastically, single-number percentage gains are not interpretable without this.

2. **Reconcile the Abstract/Introduction novelty claim and soften overclaims** (R2, R3): Narrow the Abstract's "has not been extended to spatio-temporal forecasting" to the real gap (frozen, general-purpose, gradient-free FMs on continuous physical fields), narrow Gap #1 so it does not contradict the later PINFDiT discussion, and replace "strictly enforcing ... governing equations" with "softly enforcing / scoring consistency with" to match the honest soft-filter description.

3. **Define the reported metrics** (R3): Add one sentence (equation or precise citation plus interpretation) each for Moran's $I$, latitude-weighted RMSE, and ACC (spell out "anomaly correlation coefficient"), since the headline spatial-coherence claim rests on them.

4. **Define the "simple consensus scorer" baseline** (R1, R3): Give an explicit one-sentence/one-equation definition where it is first introduced; the current "defined above" reference has no antecedent. Also disambiguate it from the differently-defined "Best-of-$N$ (sample)" in the search-strategy table (e.g., "BoN-consensus" vs "BoN-verifier, single-scale").

5. **Add a Mean-of-$N$ / Median-of-$N$ aggregation baseline** (R2): This is the cheapest alternative use of the same sampling budget and the natural "does selection beat trivial aggregation" check.

6. **Document the CEM/TuRBO-1 search space** (R1): State the search-space dimensionality and any reduction (e.g., per-cluster parameterization); otherwise the comparison risks being structurally unfair at $10^5$-$10^6$ output dimensions.

7. **Fix table/text mismatches and terminology** (R1, R3): Add the promised spatial-coherence column to the budget-split table (or correct the text); either show "Step 1" or drop the "Step 2" framing of the $w_3$ sweep; state whether the main-results table is per-setting or pooled across the three history/horizon settings; reconcile "multi-scale DTW" and "graph-Laplacian divergence" with the Method's actual single-DTW / Dirichlet-energy descriptions.

### Nice to Have (Would strengthen the paper)

1. **Add a pseudocode/algorithm block** (R3) summarizing the four-stage pipeline (the template already imports `algorithm2e`).
2. **Acknowledge data-contamination risk** (R1): discuss whether public ERA5 could have entered any backbone's pretraining, which bears on the "zero-shot" framing.
3. **Address the over-optimization claim** (R2): either extend the $N$ sweep until degradation is visible, or soften Section 2.1 from "explicitly study" to "discuss," since only saturation (not degradation) is currently shown.
4. **Use a longer climatology window for ACC** (R1) or justify the two-year choice and its effect on comparability with published baselines.
5. **Situate against adjacent literature** (R1, R2): weather-specific foundation models (Pangu/FuXi/Aurora) and classical ensemble NWP post-processing as the closest within-domain "generate-score-select" precedent.
6. **Discuss two second-order rigor points** (R1): the min-min selection over $N_A \times N_C$ pairs increasing over-optimization exposure, and a localized spatial diagnostic to complement the aggregate-roughness $E_{\text{spatial}}$.

### Reviewer Questions Requiring Response

1. Is the Table 6 "Physics Violation ($\mathcal{R}_{\text{phys}}$)" metric identical to the residual used inside the filter, and if so how is the circularity addressed? (R1)
2. What normalization is applied across temperature and wind in the verifier and physics filter? Please give the exact equations. (R1)
3. What are the main-results defaults for $M$, $k$, $\sigma$, $K_A$, and $w_1{:}w_2{:}w_3$, as distinct from the $M{=}32$, $K_A{=}5$ used in the Section 5.5 sweep? (R1, R3)
4. Will WEATHER-5K results appear in the final version (and if so, how is the physics filter adapted to its speed/direction wind and to dewpoint/pressure), or should the two-dataset framing be scoped down to ERA5? (R1, R2, R3)
5. How is CRPS computed from the pipeline, given that a single selected point forecast $\hat{x}^F$ does not by itself admit a CRPS without a predictive distribution? (R1)
6. What is the actual search-space dimensionality for Direct CEM / TuRBO-1, and is any dimensionality reduction applied? (R1)
7. Across how many independent forecast origins are metrics averaged, and will variance/CIs be reported? (R1)
8. Why does the baseline suite omit a Mean-of-$N$ / Median-of-$N$ aggregation baseline? (R2)
9. Is there a removed "Q4" ablation (the subsections jump Q1-Q3 to Q5) — e.g., the intended cross-dataset generality study? (R3)

## Editorial Recommendation

**Decision: Major Revision**

### Justification

The paper addresses a genuinely under-explored and worthwhile niche — training-free, gradient-free test-time compute for *frozen, general-purpose* time-series foundation models on *continuous, physically-governed* spatial fields — and does so with a coherent design whose cross-scale truth-free anchoring is a real, if modest, contribution. All three reviewers recognize a sound core idea, a crisp frozen-backbone formalization, an unusually thorough ablation plan, and scholarly, specific related-work positioning. The average overall score (4.7/10) and the recommendation spread (one Weak Reject, two Borderline) reflect not a flawed concept but an execution that is currently under-specified and internally inconsistent. For the target Scopus-tier applied venue, the novelty (a well-scoped intersection rather than a new primitive) is acceptable provided the claims are correctly scoped.

The critical issues cluster into three Major findings on which all three reviewers independently agree — the orphaned-and-contradicted WEATHER-5K evaluation, the SC-BoN/ST-TTC naming collision with the paper's own key differentiator, and the absence of equations and default hyperparameters for the core scoring machinery — supplemented by R1's sharper rigor concerns (variable normalization, physics-metric circularity, no uncertainty quantification). Crucially, every one of these is addressable within a revision cycle: they require consistency passes, formal write-ups, a normalization statement, a circularity clarification, an uncertainty plan, and either delivery or honest de-scoping of the second dataset. None exposes a fundamental flaw in the method itself. All three reviewers explicitly state they would upgrade to Weak Accept / Borderline-Accept once these specific issues are resolved, which is the signature of a Major Revision rather than a Reject.

I decline Minor Revision because the issues are too numerous and substantive — a missing dataset evaluation, a full naming overhaul, the formalization of essentially all core scoring terms, and unresolved rigor questions go well beyond cosmetic fixes. I decline Reject because the contribution is sound, the positioning is honest, the review panel is unanimous that a clear path to acceptance exists, and the manuscript is explicitly a pre-results draft whose numbers are not yet in. Since the paper cannot be finally accepted until real results replace the placeholders in any case, Major Revision is the decisive and appropriate call: it requires the authors to close the structural and specification gaps and to deliver (or explicitly withdraw) the second-dataset claim before the completed-results version is judged.

### Conditions for Acceptance (if revision recommended)

1. **Dataset scope reconciled**: WEATHER-5K is either delivered as a parallel results/ablation set (with the physics-filter adaptation specified) or consistently demoted to future work across Abstract, Problem Definition, Experimental Settings, and Conclusion — with no remaining self-contradiction.
2. **Single, non-colliding method name** applied uniformly throughout (SC-BoN recommended), with "ST-TTC" reserved for Chen & Liang's prior work.
3. **All core terms given as equations** ($E_{\text{trend}}$, $E_{\text{FFT}}$, $E_{\text{spatial}}$, $\mathcal{R}_{\text{phys}}$, activity penalty) and a **consolidated default-hyperparameter table** ($M$, $k$, $\sigma$, $N_A$, $N_C$, $K_A$, $w_1{:}w_2{:}w_3$) added.
4. **Variable-normalization scheme stated** explicitly, with constants, so the multivariate verifier/filter is not dominated by temperature.
5. **Physics-violation circularity resolved** — either an independent diagnostic is used, or the reuse is acknowledged and dependent claims downweighted.
6. **A statistical-uncertainty reporting plan** (variance/CIs over origins/seeds, plus a paired significance test for the headline comparisons) incorporated into the completed-results version.
7. **Novelty/overclaim statements narrowed** so the Abstract and Gap #1 match the honest, defensible scope established in Related Work and the Method.
8. **Submitted LaTeX build uses the canonical Overleaf bibliography** so all citations resolve (the missing-key finding is a known build artifact, not a scholarship defect).
