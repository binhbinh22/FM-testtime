# Meta-Review: Area Editor Report

> **Paper**: Test-Time Computing for Spatio-Temporal Weather Forecasting with Foundation Models (Spatial-Cluster Best-of-$N$ / ST-TTC)
> **Target Venue**: `\journal{TBD}` (elsarticle preprint; applied-ML / time-series forecasting venue — venue not yet filled in)
> **Date**: 2026-10-11
> **Reviewers**: 3 independent reviewers (R1 methodological rigor, R2 novelty & positioning, R3 clarity & completeness)
> **Draft status**: Early draft; all numeric results are intentional placeholders (`--`, `[X]`, `[Y]`). Per editorial instruction, missing numbers are **not** penalized; factual/consistency errors and over-claims are weighted most heavily.

## Score Summary

Reviewers scored the six criteria below (there was no separate "Completeness" score; completeness was assessed qualitatively by R3 and is folded into Clarity/Reproducibility).

| Criterion | R1 | R2 | R3 | Average |
|-----------|:--:|:--:|:--:|:-------:|
| Soundness | 5 | 6 | 6 | 5.7 |
| Clarity | 6 | 7 | 5 | 6.0 |
| Novelty | 6 | 5 | 7 | 6.0 |
| Significance | 6 | 5 | 6 | 5.7 |
| Reproducibility | 3 | 5 | 5 | 4.3 |
| **Overall** | **5** | **5** | **5** | **5.0** |

| Reviewer | Recommendation | Confidence |
|----------|---------------|:----------:|
| Reviewer 1 (rigor) | Major Revision | not self-reported (AE-inferred: high — verified internal contradictions against source) |
| Reviewer 2 (novelty) | Major Revision | not self-reported (AE-inferred: high — independently checked DEFT/SCALER/PINFDiT against public sources) |
| Reviewer 3 (clarity) | Major Revision | not self-reported (AE-inferred: high — line-level cross-checks of figures/tables/citations) |

## Paper Summary

The paper proposes ST-TTC, a training-free test-time-computing wrapper around a frozen multivariate time-series foundation model (Chronos-2) for gridded ERA5 weather forecasting. The grid is spectrally clustered once; the frozen backbone then produces cheap coarse cluster-anchor forecasts and full-resolution fine candidates, a finite-difference physics residual pre-filters the fine candidates, and a three-term verifier (DTW trend, FFT periodicity, graph-Laplacian spatial consistency) selects the final forecast by scoring each survivor against its best-matching coarse anchor. The central and genuinely differentiated claim is improved spatial coherence (Moran's $I$) with accuracy at least maintained, obtained with no backbone retraining.

## Consensus Strengths

1. **Clean, falsifiable problem framing and TTC-vs-TTA scoping** (R1, R2): Section 3 gives precise notation (frozen $\theta$, i.i.d. draws, argmin selection) and the Related Work correctly separates test-time computing from test-time adaptation on a defensible criterion (whether $\theta$ is updated), preempting an obvious objection.
2. **Honest treatment of as-reported cross-paper comparisons** (R1, R2, R3): the Table 1 caption and Main Results prose explicitly flag that published-method numbers come from different setups and are "a reference rather than a strictly controlled comparison," and the limitations paragraph is specific and candid (soft physics filter, single dataset/resolution, static clustering).
3. **A thorough, claim-targeted ablation plan** (R1, R2, R3): Q1–Q5 plus the two sensitivity analyses are each aimed at a specific confound (verifier leave-one-out, compute scaling with a random-selection control, backbone generality, integration topology, clustering strategy). If executed as designed, this would isolate where the benefit comes from rather than merely tune hyperparameters.
4. **Well-integrated method narrative and figure** (R3): Section 4.1 maps cleanly onto Figure 1, whose internal verifier formula matches Eq. 3 verbatim; sentence-level prose is strong and free of generic-AI tells, uses "e.g." consistently, and largely avoids em-dash overuse.
5. **Correct "soft, approximate" framing of the physics term** (R1): the paper does not oversell a "physics-constrained" forecast; it states plainly that 2m temperature is not exactly conserved and the residual is only a soft filter.

## Consensus Weaknesses

1. **Related-work characterizations of the three closest works are factually imprecise** (R2, with R1/R3 noting the downstream Train?/positioning effects): DEFT's costly expert-oracle verifier is described as if it were a free autonomous scorer, and PINFDiT's scope is described as "single (non-spatial) series" despite its reported multivariate experiments. These are the highest-weight issues under the "factual-accuracy matters most" instruction, and ironically the DEFT correction *strengthens* the paper's novelty claim.
2. **Prose makes claims the tables do not contain** (R1, R3): the Main Results narrative reports Moran's $I$, aggregate normalized MSE, latitude-weighted RMSE and ACC, none of which appear in Table 1 or any companion table; CRPS and $\mathcal{R}_{\text{phys}}$ appear only as undefined column headers in Table 4; Moran's $I$ is never defined. A reader cannot check the paper's central claims against any table.
3. **The Q4 integration-topology table omits the method it is said to support** (R1, R3): the prose names "verification-only selection" as the winning topology, but Table 7 has no such row; "late" is also overloaded (the main method is called "late, post-hoc" while a different compared alternative is named "Late Anchoring").
4. **The Conclusion over-claims relative to the placeholder-aware body** (R1, R2): its first paragraph makes fully declarative success statements ("achieved best or comparable... delivered the most consistent improvement") with no placeholders, and Table 8's caption pre-commits to a positive result. This is the single largest over-claiming risk.
5. **Multiple internal inconsistencies between prose and populated (non-placeholder) cells** (R1, R2, R3): backbone roster (prose: Chronos-1/TimesFM/Moirai; Table 8: adds TimesFM-3, Moirai-2), context length (Table 3 gives Chronos-2 = Moirai = 512, contradicting the "long context"/"Moirai shorter" rationale and the 1024/96 setting named only in the Conclusion), physics-filter mechanism (hard top-$K_A$ cut in Section 4.3 vs. "down-weight" in Conclusion), and the $E(x^F)$ vs. $E(x^F;a_j)$ arity mismatch (Eq. 2 vs. Eq. 5).
6. **Reproducibility of the scoring machinery is under-specified beyond the missing numbers** (R1, with R2/R3 agreeing hyperparameters are otherwise well given): the physics residual, activity penalty, their combination rule, the FFT distance (bins/phase), and the graph-Laplacian functional are never written as equations, and $k$, $\sigma$, and the tuned $(w_1,w_2,w_3)$ appear nowhere, not even as `[TBD]`.

## Points of Disagreement

### Novelty: is the differentiation from prior work "clear" or "not yet trustworthy"?
- **R3 argues**: Novelty is a relative strength (7/10) — the Related Work differentiates ST-TTC from DEFT/SCALER/PINFDiT cleanly, separating what is shared from what differs.
- **R2 argues**: Novelty is the weakest axis (5/10) because the differentiation rests on at least two likely mischaracterizations (DEFT, PINFDiT) and one unacknowledged structural kinship (SCALER's coarse-anchor-then-refine), so the delta "cannot be trusted at face value" until corrected.
- **AE Assessment**: R2 is more persuasive. R3's clarity lens assessed whether the differentiation *reads* cleanly (it does) but took the underlying factual claims at face value; R2 actually verified them against the cited sources and found them wanting. The underlying contribution is genuinely novel (two-scale spatial Best-of-$N$ with a physics pre-filter and composite verifier on a frozen backbone), but the *argument* for it is currently built on inaccurate related-work statements. Resolution: treat novelty as real but provisional — it becomes solid only once the DEFT/PINFDiT/SCALER characterizations are corrected (MUST FIX 1, 2 and SHOULD FIX 6).

### Reproducibility: 3 vs. 5
- **R1 argues**: 3/10 — core formulas (physics residual, activity penalty, FFT distance, Laplacian functional) and $k,\sigma,(w_1,w_2,w_3)$ are absent even as placeholders, which would block faithful reimplementation.
- **R2/R3 argue**: 5/10 — the configuration hyperparameters ($M=32$, $N=20$ split 15/5, $K_A=5$) are specified concretely, which is commendable for a draft.
- **AE Assessment**: Both are correct about different things. The *configuration* is well specified; the *mathematical definitions* of the scoring terms are not, and those are what a reimplementer actually needs. For a method whose entire contribution is the verifier, the missing equations are the more serious gap, so R1's lower score is the better-calibrated signal. This is fully fixable (equations can go in an appendix) and should not, on its own, drive the decision.

### Compute accounting
- **R1 argues**: "# backbone calls" is not a fair compute unit because a coarse call (32 cluster-mean series, downsampled) and a fine call (2048 full-resolution series) differ greatly in cost; latency should be folded into the main table.
- Not raised by R2/R3.
- **AE Assessment**: R1 is correct and this is a valid point the other reviewers missed. It does not block the paper but should be addressed (SHOULD FIX 4) so that iso-compute comparisons mean what they claim.

Overall there is **no outlier reviewer and high agreement**: all three independently converge on Major Revision with an identical 5/10 overall, and the three highest-signal problem classes (prose-without-table, Q4 missing row, backbone/context-length inconsistencies, unhedged Conclusion) were triangulated by at least two reviewers each.

## Review Quality Assessment

All three reviews are thorough, fair, and appropriately refrained from penalizing missing numbers.
- **R2 (novelty)** added the most unique value by independently verifying DEFT, SCALER, and PINFDiT against public sources and confirming that `huang2025bestofn` is characterized correctly and that no citations are fabricated. The DEFT and PINFDiT findings are the most consequential in the whole review packet and would likely have been missed by a reviewer who did not check the sources. R2 also raised the only positioning-level gap (ensemble NWP post-processing as prior art).
- **R3 (clarity)** performed meticulous line-level cross-checking and surfaced exactly the factual/consistency class the editorial brief prioritizes (metrics-without-table, Figure 2 mismatch, Q4 missing row, the snell/huang citation slip).
- **R1 (rigor)** went deepest on method-level soundness and caught issues the others did not: the compute-unit incomparability, the min-over-anchors permissiveness, the absence of any variance/significance plan, and the context-length numeric contradiction.
- No review was unfair or based on misunderstanding. R1's reproducibility score of 3 is the only notable divergence and, as adjudicated above, is the better-calibrated of the two positions. All criticisms the authors must act on are well founded.

## Prioritized Revision Plan

Factual-accuracy risks are marked **[FACTUAL]**; internal-consistency risks are marked **[CONSISTENCY]**. Items raised by multiple reviewers are de-duplicated and attributed.

### Must Fix (required for acceptance)

1. **[FACTUAL] DEFT characterization** (Related Work 2.3 + Introduction para 2; R2): DEFT is described as a free, autonomous verifier that "edits its forecast," but it relies on a costly external expert evaluator under a tight query budget. Correct the description and reframe the contrast ("ST-TTC needs no costly oracle") as a *primary* novelty argument.
2. **[FACTUAL] PINFDiT scope claim** (Related Work 2.4; R2): "operates on a single (non-spatial) series" contradicts PINFDiT's reported multivariate experiments. Re-verify and replace with the precise claim actually supported (e.g., "physics energy computed per channel, with no explicit spatial adjacency"), citing the specific experiment/table. Also revisit the Train? column so PINFDiT's domain-pretrained backbone is not visually equated with Chronos-2's zero-shot status (R1, R2).
3. **[FACTUAL/CONSISTENCY] Table 7 (Q4) missing the "Verification-only (Ours)" row** (Section 6.4, `tab:ablation_topology`; R1, R3): add the row the prose names as the winner, and rename one of the two overlapping "late" concepts (e.g., "anchor-for-scoring" vs. "Late Anchoring" output-blending).
4. **[FACTUAL/CONSISTENCY] Metrics discussed but absent from any table** (Section 5.2, Table 1; R1, R3): add columns or a companion table for Moran's $I$, aggregate normalized MSE, latitude-weighted RMSE, and ACC; and define CRPS, $\mathcal{R}_{\text{phys}}$, and Moran's $I$ (with a standard citation for Moran's $I$) in the Metrics paragraph, since they are first used only as undefined table headers.
5. **[FACTUAL/CONSISTENCY] Figure 2 (`fig:scaling`) does not depict what the prose claims** (Section 6.2; R3): the text says error, physics-violation and latency are co-plotted on a common scale, but the figure shows a single RMSE curve. Either expand to a 2–3 panel / twin-axis figure (series normalized to their $N{=}1$ value) or rewrite the prose to describe only the RMSE curve and refer physics-violation/latency to Table 4.
6. **[FACTUAL] Simple-scorer baseline citation inconsistency** (Section 5.2 vs. Conclusion; R3): Section 5.2 cites `snell2024scaling`, the Conclusion cites `huang2025bestofn` for the same baseline. Use `snell2024scaling` consistently; reserve `huang2025bestofn` for the imperfect-verifier risk result.
7. **[FACTUAL/over-claim] Unhedged Conclusion and Table 8 caption** (Section 8, `tab:ablation_backbones`; R1, R2): the Conclusion's first paragraph asserts success with no placeholders and Table 8's caption pre-commits to a positive $\Delta$. Rewrite both with the bracketed-placeholder convention used in Sections 4–5, so no outcome language precedes the real numbers.
8. **[CONSISTENCY] Context-length contradiction** (Table 3 vs. Backbone-selection prose vs. Conclusion; R1, R3): Table 3 lists Chronos-2 = Moirai = Chronos = 512, which contradicts both the "Chronos-2 has a longer context / Moirai shorter" rationale and the 1024/96 history setting named only in the Conclusion. Reconcile the table, the rationale, and the evaluated history lengths.
9. **[FACTUAL/CONSISTENCY] Backbone roster mismatch** (Section 5.1/5.4 prose vs. Table 8; R1, R2, R3): prose promises Chronos-1/TimesFM/Moirai; Table 8 adds Chronos-2, TimesFM-3, and Moirai-2. Pin down version-exact checkpoint names and use them identically in Table 3, the Backbone paragraph, Section 5.4, and Table 8.
10. **[CONSISTENCY] History/horizon settings appear only in the Conclusion** (512/64, 512/128, 1024/96; Section 8 vs. Section 5.1; R3): state the evaluated $(L_H, L_F)$ pairs in Section 5.1 and clarify whether Table 1 reports one setting or an average across the three.
11. **[Soundness] Simple-scorer baseline not operationally defined** (Section 5.1 Baselines, Section 5.2; R1): the baseline is said to compute "MAE," but ground truth is unavailable at selection time by the paper's own problem definition. State explicitly what the simple scorer measures each candidate against (e.g., ensemble mean, persistence, climatology) and confirm it respects the same no-ground-truth constraint as the proposed verifier.

### Should Fix (strongly recommended)

1. **[CONSISTENCY] Physics-filter mechanism: discard vs. reweight** (Section 4.3 vs. Conclusion; R1): Section 4.3 implements a hard top-$K_A$ truncation, the Conclusion says the residual can only "down-weight." Use "hard pre-filter based on a soft (approximate) residual" consistently.
2. **[Reproducibility] Missing core equations and hyperparameter values** (Section 4.3–4.4; R1): add equations (appendix is fine) for the physics residual, the activity penalty, their combination rule, the FFT $\ell_1$ distance (bin range, phase vs. magnitude), and the graph-Laplacian smoothness functional; give numeric or `[TBD]` values for $k$, $\sigma$, and the tuned $(w_1,w_2,w_3)$.
3. **[Soundness] No statistical uncertainty is planned** (all result tables; R1): report mean ± std or bootstrap CI over sampling seeds and/or test windows, state the number of test windows (stride), and add a paired test (e.g., Wilcoxon) for the headline RMSE comparison against the single-pass floor — especially given the paper's own invocation of Best-of-$N$ degradation under an imperfect verifier.
4. **[Soundness] Compute metric comparability** (Table 1 "# calls"; R1): coarse and fine calls are not iso-cost. Report wall-clock/FLOPs alongside call count in the main table, or state the coarse:fine per-call cost ratio so "# calls" can be converted to comparable units.
5. **[Positioning] Missing ensemble-NWP prior art** (Related Work 2.4/2.5; R2): "generate multiple spatially coherent candidates and select/calibrate" is the premise of ensemble NWP post-processing (EMOS, rank histograms, best-member/analog selection) and generative ensembles (GenCast). Add a short paragraph acknowledging it and stating precisely how ST-TTC differs, to protect the "first TTC for spatio-temporal forecasting" framing.
6. **[Positioning] SCALER mechanistic contrast** (Related Work 2.3; R2): resolve the in-source TODO by reading SCALER in full and differentiating on mechanism (trained coarse model + active guidance vs. ST-TTC's frozen-backbone reuse + post-hoc verification), not only on domain (single-series vs. spatial).
7. **[CONSISTENCY] Verifier arity** (Eq. 2 vs. Eq. 5; R3): make the anchor-conditioning explicit from the first definition so $E(x^F)$ and $E(x^F;a_j)$ are visibly the same object.
8. **[CONSISTENCY] Advection vs. continuity terminology** (Section 4.3 "continuity (advection) residual" vs. Conclusion "advection and continuity residuals"; R3): choose one name and one symbol, and introduce $\mathcal{R}_{\text{phys}}$ as an equation rather than first in a table header.
9. **[Positioning] Train? column semantics** (Table 1; R1, R2): add a footnote distinguishing "no inference-time training" from "zero-shot w.r.t. the target domain," note that verifier weights/physics threshold are calibrated on a disjoint validation split, and state whether PINFDiT's backbone was pretrained on weather data.
10. **[Soundness] Justify the min-over-anchors matching rule** (Eq. 5; R1): the most permissive matching rule risks letting an outlier anchor validate a poor candidate. Either justify it or add an ablation against scoring vs. the ensemble mean/median of the anchors.

### Nice to Have (would strengthen the paper)

1. Justify DTW for $E_{\text{trend}}$ given that moving-average windows are matched across scales (the misalignment DTW tolerates may already be removed); consider a simpler L1/L2 (R1).
2. Verify that "i.i.d. sampling" is the correct characterization for all four backbones (quantile levels vs. stochastic draws), since this underpins the Best-of-$N$ diversity argument (R1).
3. Add a sensitivity sweep over the number of clusters $M$ (Q5 varies only the clustering *method* at fixed $M=32$); both R1 and R2 flag that the headline spatial-coherence claim may be sensitive to $M$.
4. Add a roadmap sentence enumerating Q1–Q5 at the start of Section 6, and either fold the two Analysis studies into the same numbering or justify the separation (R3).
5. Spell out acronyms at first use (FFT, PDE), cite WeatherBench (Rasp et al.) alongside GraphCast for the normalized-MSE convention, add a graph-Laplacian-as-regularizer reference for the scoring lineage, and cite a standard CRPS reference (R1, R2, R3).
6. Caption/typographic cleanups: stray double period in Table 5, em-dash in the `w3_sweep` caption, and premature bolding of placeholder cells in Tables 4 and 7 (R3).
7. Remove the Vietnamese TODO/`LƯU Ý` comments and fill `\journal{}` plus author/affiliation before submission, ensuring the flagged facts are resolved rather than merely deleted (R1, R2).
8. Add a code/data availability statement, hardware spec, software versions, and a random-seed policy (R1).
9. Align the Abstract headline metric with the body's (RMSE vs. aggregate normalized MSE) and replace `[Y]$\times$` with `$N\times$` once numbers are in (R3).
10. Map the three contribution bullets to the three motivating gaps, and revisit gap 3 (cost of full-horizon scoring) so the physics pre-filter's cost-reduction role is stated, not merely implied (R2, R3).

### Reviewer Questions Requiring Response

1. What does the simple-scorer Best-of-$N$ baseline's MAE measure against, given that ground truth is unavailable at selection time? (R1)
2. Is Chronos-2's multi-candidate output true i.i.d. sampling, or fixed quantile levels — and does this differ across Chronos-1/TimesFM/Moirai in a way that affects the "backbone-agnostic" claim? (R1)
3. How many test windows (start times, stride) make up the 2017–2018 evaluation, and is any repeated-seed or cross-window variance reported for the headline RMSE/Moran's $I$ improvements? (R1)
4. From PINFDiT's own experiments, is its physics-guided correction computed per channel, or does it already couple across series? This determines whether "does not model interactions between grid points" is accurate. (R2)
5. Does SCALER's coarse model require per-dataset training, and how does that sharpen the delta from ST-TTC's frozen-backbone coarse branch? (R2)
6. Which exact $(L_H,L_F)$ settings were run for Table 1, and is Table 1 a single setting or an average over 512/64, 512/128, 1024/96? (R3)
7. Is the "physics violation" in Table 4 the same quantity as the Section 4.3 filter residual, or a separate diagnostic? (R3)
8. In Table 7, what exactly defines "No Centroid (Isolated Stations)" — is it $w_1{=}w_2{=}w_3{=}0$, or a pipeline with no verifier at all? (R3)

## Editorial Recommendation

**Decision: Major Revision**

### Justification

The contribution is sound, timely, and sits in a genuinely underexplored niche: extending the generate-score-select test-time-computing paradigm to spatially coupled, physics-governed gridded forecasting on a frozen, general-purpose time-series foundation model, with spatial coherence as the primary differentiator. All three reviewers independently converged on this read (identical 5/10 overall, unanimous Major Revision), and the planned ablation suite is unusually thorough for a draft at this stage. None of the issues below requires abandoning the method or, in most cases, running new experiments.

The critical issues are concentrated in two categories that the editorial brief weights most heavily: factual-accuracy risks and internal consistency. Independent source verification by R2 established that the two most important related-work characterizations (DEFT's costly-oracle verifier, PINFDiT's scope) are likely inaccurate as written, and correcting them actually strengthens rather than weakens the novelty case. R1 and R3 jointly established a cluster of internal contradictions where the prose asserts things the tables and figures do not contain or support: metrics discussed but never tabulated, a Q4 table missing the very row its paragraph relies on, a figure that does not plot what the text says, a backbone roster and a context-length value that contradict the surrounding narrative, and a Conclusion that reads as a pre-written success story with a citation slip. Crucially, every one of these is fixable through re-verification and rewriting, and almost none depends on the still-pending numbers — which is precisely why they must be fixed before, not after, the numbers arrive.

I am therefore not recommending Minor Revision: the factual mischaracterizations and the prose/table/figure mismatches are too numerous and too load-bearing for the paper to be internally verifiable in its current state, and the Conclusion's unhedged narrative is an active over-claiming risk that must be corrected before any results are trusted to mean what the text says. Nor is rejection warranted: the core idea is coherent, the positioning is honest where it counts, and the defects are specification and consistency problems rather than flaws in the method itself. The appropriate outcome is a substantive revision pass that (a) corrects the related-work facts, (b) makes every table, figure, and citation match its surrounding prose, (c) adds the missing scoring equations and a variance/significance plan, and (d) re-hedges the Conclusion.

### Conditions for Acceptance (if revision recommended)

1. All eleven Must-Fix items are resolved, with the six factual/consistency items (DEFT, PINFDiT, Table 7 missing row, metrics-without-table, Figure 2, citation slip) corrected against the cited sources and the actual tables/figures.
2. The Conclusion and all table/figure captions use the same placeholder-aware, hedged style as Sections 4–5, with no outcome language preceding real numbers.
3. The scoring machinery is made reproducible: equations for the physics residual, activity penalty, FFT distance, and Laplacian functional are provided (appendix acceptable), and $k$, $\sigma$, and $(w_1,w_2,w_3)$ are given as values or explicit `[TBD]`.
4. A statistical-uncertainty protocol (seeds and/or test-window variance, number of test windows, and a paired significance test for the headline RMSE claim) is committed to and, once numbers exist, reported.
5. All in-source TODO/`LƯU Ý` comments and the unverified facts they flag (SCALER description, backbone versions, published-method variable coverage) are resolved, and the `\journal{}`/author fields are completed.
