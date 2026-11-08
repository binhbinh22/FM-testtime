# Reviewer 3 Report — Clarity & Completeness

> **Reviewer Focus**: Clarity & Completeness
> **Paper**: Test-Time Computing for Spatio-Temporal Weather Forecasting with Foundation Models
> **Target Venue**: TBD (applied-ML / time-series forecasting venue; `\journal{TBD}` in source)
> **Review Date**: 2026-10-11
> **Draft Status Acknowledged**: This is an early draft with intentional placeholder results (`--`, `[X]`). This review does **not** penalize missing numbers; it assesses narrative coherence, notational consistency, and structural completeness, which can and should be evaluated independently of final numeric results.

## Summary

The paper proposes Spatial-Cluster Best-of-$N$ (ST-TTC), a training-free test-time-computing framework that extends Best-of-$N$ search to spatio-temporal weather forecasting on a frozen Chronos-2 backbone. It generates full-resolution "fine" candidates and coarse cluster-level "anchors" from spectrally clustered regions, prunes fine candidates with a physics-plausibility residual filter, and selects the final forecast with a composite verifier combining trend (DTW), frequency (FFT), and spatial (graph-Laplacian) consistency terms against the best-matching anchor. The paper is organized as a conventional empirical ML paper (intro, related work, problem definition, method, experiments with a five-part ablation study plus analysis, conclusion) and is evaluated on ERA5 reanalysis data with results currently left as placeholders.

## Strengths

1. **Clear four-stage method narrative with a well-matched figure.** Section 4.1 walks through clustering, coarse branch, fine branch, and verifier/selection in one paragraph that maps cleanly onto Figure 1 (`fig:arch`), and the TikZ diagram's internal formula (`E = w_1 E_trend + w_2 E_FFT + w_3 E_spatial`) matches Equation 3 verbatim. This is exactly the kind of figure-prose-equation triangulation that makes a method section easy to follow, and it is one of the strongest parts of the paper.
2. **Honest, well-integrated limitations paragraph.** The second paragraph of the Conclusion (Section 8) explicitly lists three caveats (soft vs. hard physics constraint, single-dataset/resolution evaluation, accuracy-compute trade-off) and a fourth (static clustering). This is more candid and specific than the generic "future work" paragraphs common in draft papers, and each caveat is tied back to a specific section/table, which aids traceability.
3. **Ablation study is logically decomposed and each sub-study is motivated before its table.** Sections 6.1–6.5 each open with a one-paragraph statement of what question is being asked and why, before presenting the corresponding table, and each closes with an interpretive paragraph that references the table's specific columns. This "ask → show → interpret" pattern (e.g., Section 6.1 on verifier components, Section 6.5 on clustering strategies) is good practice and is followed consistently across all five ablations.
4. **Prose quality is genuinely good at the sentence level.** The writing avoids the generic AI-generated-text tells (no "delve into," "leverage," "showcase," "robust tapestry," etc.), uses "e.g." instead of "i.e." consistently, and largely avoids em-dash overuse (the one exception is noted below). Technical sentences are precise and hedge appropriately (e.g., "numbers for published methods... may use a different experimental setup, so they serve as a reference rather than a strictly controlled comparison," Table 1 caption).

## Weaknesses

1. **The main-results narrative cites metrics that never appear in any table (Major).** The Metrics paragraph (Section 5.1) promises five reportable quantities: per-variable MAE/RMSE, an aggregate normalized MSE, latitude-weighted RMSE, ACC, and Moran's $I$. Table 1 ("Main Results") only contains raw per-variable MAE/RMSE plus Train?/#calls columns. Yet the interpretive paragraph immediately following Table 1 (Section 5.2) makes its central claims using exactly the metrics absent from the table: "improving Moran's $I$ by [X]," "reduces aggregate normalized MSE by [X]%," "a modest gain of [X]% over the floor." A reader has no table to check these numbers against anywhere in the paper — Moran's $I$ only resurfaces, incidentally, as a column in the unrelated clustering ablation (Table 6), and aggregate normalized MSE/ACC/latitude-weighted RMSE appear in no table at all.
   - *Suggestion*: Either add columns for aggregate normalized MSE, latitude-weighted RMSE, ACC, and Moran's $I$ to Table 1 (perhaps as a compact second block), or add a dedicated secondary table of aggregate/spatial metrics referenced by the same paragraph that currently discusses them only in prose.

2. **Figure 2 does not depict what the surrounding prose says it depicts (Major).** Section 6.2 states: "Figure~\ref{fig:scaling} plots these quantities [error, physics-violation, latency] together... on a common scale." The actual TikZ plot has a single y-axis labeled "RMSE (lower is better)" and a single curve; CRPS/physics-violation and latency, both columns of the adjacent Table 4, are not plotted at all. This is a content-structure mismatch independent of whether the numbers are real or placeholder — even as a schematic, the figure should sketch three co-plotted (or triple-paneled) quantities if that is what the text claims.
   - *Suggestion*: Either expand the figure to a 2–3 panel layout (or twin/secondary axes) showing RMSE, physics violation, and latency together, with each series normalized to its own $N=1$ value so "common scale" is meaningful, or rewrite the prose to describe only the single RMSE-vs-$N$ curve the figure currently shows and move the physics-violation/latency discussion to refer to Table 4 directly instead of the figure.

3. **Table 7 (integration topology) omits the exact row the prose uses as the reference point (Major).** Section 6.4 concludes: "Verification-only selection, the design used in our main pipeline, achieves the best balance between accuracy and cost among the three topologies in Table~\ref{tab:ablation_topology}." But Table 7 lists only "No Centroid," two "Early Conditioning" variants, and two "Late Anchoring" variants — there is no row labeled "Verification-only (main pipeline / Ours)." The reader cannot verify the central claim of this ablation because its subject is absent from the table it is said to appear in.
   - *Suggestion*: Add an explicit "Verification-only (Ours, main pipeline)" row to Table 7 (likely the best values once populated), so the three-way comparison promised in the prose is actually present in the table.

4. **The Conclusion introduces concrete experimental settings that are never specified in Section 5 (Major).** The Conclusion states: "Across the history/horizon settings we evaluated (512/64, 512/128, 1024/96), ST-TTC achieved..." This is the only place in the entire manuscript these three $(L_H, L_F)$ pairs appear. Section 5.1 ("Dataset") never states which history/horizon lengths are used, and Table 1 shows no column or multiple rows distinguishing settings, despite the Main Results prose (Section 5.2) referring generically to "the evaluated history/horizon settings." A reader reaches the Conclusion with new, unexplained specifics that cannot be traced back to any experimental description or table breakdown.
   - *Suggestion*: State the evaluated $(L_H, L_F)$ pairs explicitly in Section 5.1 ("Dataset" or "Implementation" paragraph), and either show per-setting rows in Table 1 (or a supplementary table) or clarify in Section 5.2 that Table 1 reports results averaged/aggregated across these three settings.

5. **A citation for the "simple-scorer Best-of-$N$" baseline is inconsistent between sections (Major).** Section 5.2 attributes this baseline to Snell et al.: "a simple-scorer Best-of-$N$ \citep{snell2024scaling} that samples the same candidate budget..." The Conclusion instead attributes the same baseline to a different paper: "a simple-scorer Best-of-$N$ baseline \citep{huang2025bestofn} forming successively weaker floors beneath it." `huang2025bestofn` is introduced in Related Work as the source of a *theoretical risk result* about Best-of-$N$ degrading with an imperfect verifier, not as the originator of the baseline method itself — this looks like a copy/paste slip rather than an intentional second citation.
   - *Suggestion*: Use `\citep{snell2024scaling}` consistently for the baseline method in both places, and reserve `\citep{huang2025bestofn}` for discussions of the theoretical risk of naive $N$-scaling, as in Section 2.1.

6. **Several metrics used in tables are never defined in the Metrics paragraph (Major for completeness).** CRPS appears as a column header in Table 4 (`tab:ablation_compute`) and $\mathcal{R}_{\text{phys}}$ ("Physics Violation") appears in the same table, but neither is defined anywhere: the Metrics paragraph (Section 5.1) only defines MAE, RMSE, the aggregate normalized MSE, latitude-weighted RMSE/ACC, and backbone-call count. Moran's $I$ is used in three different tables/discussions but likewise has no formula or citation anywhere in the paper. For a methods-heavy readership this is a meaningful gap, since these three quantities carry real interpretive weight in the ablation narrative.
   - *Suggestion*: Add CRPS, $\mathcal{R}_{\text{phys}}$, and Moran's $I$ to the Metrics paragraph with at least a one-sentence definition or citation each (Moran's $I$ in particular should cite a standard reference, since it is central to the paper's "clearest advantage" claim).

7. **Notational mismatch between the formal problem statement and the realized verifier (Minor-Major).** Section 3 defines selection as $\hat{x}^F = \arg\min_i E(x_i^F)$ (Eq. 2), i.e., $E$ is a function of the candidate alone. Section 4.4 then defines the actual selection rule as $\hat{x}^F = \arg\min_i \min_j E(x_i^F; a_j)$ (Eq. 5), where $E$ additionally depends on a coarse anchor $a_j$ — a different arity for the same symbol. A careful reader comparing the two equations will notice the inconsistency.
   - *Suggestion*: In Section 3, write $E(x^F; x^H, \mathcal{G})$ or otherwise note that the verifier may depend on auxiliary quantities instantiated later, and keep the anchor-conditioning explicit from the first definition so Eq. 2 and Eq. 5 are visibly the same object.

8. **Backbone list in the ablation table does not match the backbone list announced in the text (Minor-Major).** Section 5.1 ("Backbone" paragraph) states the generality study uses "Chronos-1, TimesFM, and Moirai" alongside the Chronos-2 default — four backbones total. Table 6 (`tab:ablation_backbones`) instead lists five rows: Chronos-1, Chronos-2, TimesFM, **TimesFM-3**, and **Moirai-2**. "TimesFM-3" is never mentioned anywhere else in the paper, and "Moirai" becomes "Moirai-2" without explanation. (The authors' own inline comment flags this exact issue as unresolved — good self-awareness, but it should be fixed before the next full draft pass.)
   - *Suggestion*: Reconcile the backbone list across Section 5.1 and Table 6, pin down exact model/version names, and make sure the same four (or five, if TimesFM-3 is intentionally added) names are used consistently in both places.

9. **Table 6's context-length column undermines the "Backbone selection" rationale that cites it (Minor-Major).** Section 5.1 justifies choosing Chronos-2 partly because "Moirai... operates with a shorter maximum context than Chronos-2," citing Table 3 (`tab:fm_comparison`) as the source. Table 3 as currently filled in shows **Moirai's context length as 512, identical to Chronos-2's 512** — the table does not support, and in fact contradicts, the stated rationale. This is a logical inconsistency between prose and an already-populated (non-placeholder) table cell, not a missing-number issue.
   - *Suggestion*: Either correct the context-length values in Table 3 to reflect the actual (differing) context windows of each model, or rewrite the "Backbone selection" rationale to match whatever the verified table values turn out to be.

10. **Inconsistent terminology for the physics filter's residual (Minor).** Section 4.3 introduces "a continuity (advection) residual," treating "continuity" and "advection" as near-synonyms via the parenthetical. The Conclusion instead refers to "a physics pre-filter based on advection and continuity residuals" (plural), implying two distinct terms. Physically, the continuity equation (mass/energy conservation) and the advection equation (transport by a velocity field) are related but not identical; if the filter truly combines two separate residual terms, they should be named and (ideally) given separate symbols; if it is one term, the Conclusion's plural phrasing should be corrected.
   - *Suggestion*: Pick one consistent name (e.g., "advection residual") and symbol (e.g., $\mathcal{R}_{\text{phys}}$, introduced explicitly as an equation in Section 4.3 rather than appearing for the first time, undefined, as a table column header in Table 4) and use it uniformly in Sections 4.3, the ablation tables, and the Conclusion.

11. **No roadmap sentence orienting the reader to the five "Q1–Q5" ablation questions (Minor).** Section 6 jumps directly into "Ablation on Verifier Reward Components (Q1)" without a preceding sentence telling the reader that the section is organized around five questions, what they are, or why "Analysis" (Section 7: budget allocation, weight sensitivity) is kept structurally separate from the numbered Q1–Q5 ablations rather than continuing the same numbering (e.g., as Q6/Q7).
   - *Suggestion*: Add a 2–3 sentence roadmap at the start of Section 6 enumerating Q1–Q5, and either fold Section 7's two analyses into the same numbering scheme or state explicitly why they are treated as a separate "Analysis" rather than "Ablation" category.

12. **Abstract's headline metric does not match the body's headline metric (Minor).** The abstract claims a gain of "[X]% RMSE," while the Main Results paragraph's primary accuracy claim is framed in terms of "[X]% aggregate normalized MSE," a different (though related) quantity. The abstract also introduces an unexplained placeholder symbol "[Y]$\times$" for compute cost that is never used again in the body (the body consistently uses $N$, the number of backbone calls).
   - *Suggestion*: Once real numbers are available, make sure the abstract's headline metric is literally the same statistic reported as the headline number in Section 5.2, and replace "[Y]$\times$" with "$N\times$" (or state explicitly that $Y=N$) to match the body's notation.

## Questions for Authors

1. Which specific $(L_H, L_F)$ settings were actually run for the Main Results table (Table 1), and is Table 1 reporting a single setting or an average over the three settings named only in the Conclusion (512/64, 512/128, 1024/96)? This materially changes how the table should be read.
2. Is the "physics violation" reported in Table 4 the same quantity as the filter residual used in Section 4.3's physical-plausibility filter, or a separate, more detailed diagnostic computed only for this ablation? The shared symbol $\mathcal{R}_{\text{phys}}$ (first appearing in a table, never defined in the method) leaves this ambiguous.
3. In Table 7, what exactly distinguishes "No Centroid (Isolated Stations)" from the main pipeline's physics-filter-plus-verifier procedure when the coarse anchor's influence is removed — is it literally $w_3=w_1=w_2=0$, or a structurally different pipeline without any verifier at all? The text does not define this baseline's mechanics.
4. Can the aggregate normalized MSE, latitude-weighted RMSE, ACC, and Moran's $I$ promised in the Metrics paragraph be added as explicit table columns (in Table 1 or a companion table) in the next revision, so the Main Results narrative is checkable against a table rather than only against inline placeholder numbers?

## Detailed Comments

### Abstract
- Clear and appropriately scoped for a draft; correctly flags both placeholder numbers as `[X]`/`[Y]`. The metric-naming mismatch with the body (RMSE vs. aggregate normalized MSE) should be resolved once real numbers are in (Weakness 12).

### Introduction
- The "three concrete gaps" framing (paragraph 3) is a strong, specific motivation. However, the three contribution bullets that follow are not explicitly mapped back to the three gaps; a reader must infer which contribution addresses which gap (contribution 2 implicitly covers both gap 1 and part of gap 3 via the physics filter's pruning role, but this is never stated). A one-clause pointer per bullet (e.g., "(addressing Gap 1)") would tighten the logic.

### Related Work
- Well organized into five coherent subsections with clear differentiation statements at the end of each ("in contrast," "both, however," etc.) — this is good practice for positioning. The PINFDiT discussion (Section 2.4) is a good example: it clearly separates what is shared (inference-time physics residual, no retraining) from what differs (single-series vs. spatial, Langevin steering vs. finite-difference filter-then-verify).
- The forward reference to Table 3 (`tab:fm_comparison`) in Section 2.2 is resolved only much later in Section 5.1 — a large physical distance between citation and table. Consider either moving the table closer to Section 2.2 or placing a short inline mini-comparison there instead.

### Problem Definition
- Formally precise and appropriately concise. The arity mismatch between the $E(x^F)$ defined here and the $E(x^F; a_j)$ used in Section 4.4 (Weakness 7) is the main issue.

### Method
- Section 4.1 (Overview) and Figure 1 are the best-integrated part of the paper (Strength 1).
- Section 4.2 (Clustering) is self-contained and clearly motivated (haversine distance, pole convergence justification).
- Section 4.3 (Candidate Generation) never gives the physics residual an equation or a named symbol, which is then needed (and introduced without warning) in Table 4 as $\mathcal{R}_{\text{phys}}$ (Weakness 6, 10).
- Section 4.4 (Verifier) is clearly written term-by-term, but "FFT" (as in $E_{\text{FFT}}$) is never spelled out as "Fast Fourier Transform" in prose; a minor acronym-expansion gap.

### Experimental Settings and Results
- The Dataset/Backbone/Metrics/Implementation paragraphs are logically ordered and mostly complete, but leave several `[TBD]` items (checkpoint identifiers, context length, exact history/horizon settings) that go beyond "placeholder results" and into "placeholder methodology," which does affect reproducibility assessment (see Scores).
- Table 1 and its interpretive paragraph are the clearest illustration of Weakness 1 (metrics discussed but not tabulated).

### Ablation Study
- Internally the best-organized section of the paper; each Q1–Q5 subsection follows "motivate → table → interpret." The structural problems are confined to specific tables (Weaknesses 2, 3, 8) rather than the section's overall logic.

### Analysis
- Both sub-analyses (budget allocation, weight sensitivity) are clearly motivated and interpreted, but Table 5 (`tab:budget_split`)'s caption promises only RMSE while the preceding prose says it reports "error and spatial-coherence metrics" — another instance of the Weakness-1 pattern on a smaller scale. The caption also has a stray trailing period ("...validation RMSE. .").

### Conclusions
- Mostly a faithful, well-hedged summary, except for the unexplained introduction of the three $(L_H,L_F)$ settings (Weakness 4) and the citation inconsistency for the Best-of-$N$ baseline (Weakness 5).

## Missing References

No critical missing references identified for a clarity/completeness review (reference adequacy for novelty positioning is better assessed by the Reviewer 2 lens). One suggestion: if CRPS is retained as a reported metric, a standard citation for CRPS in probabilistic forecasting (e.g., Gneiting and Raftery) would help ground its definition, since none is currently given.

## Minor Issues

- Table 5 caption: stray double period, "...on validation RMSE. ." — remove extra period.
- Table `tab:w3_sweep` caption: "Step 2 — spatial weight..." uses an em-dash, inconsistent with the otherwise dash-light style used throughout the rest of the paper's captions and prose.
- "PDE" (Introduction, contributions bullet 2) is used without first spelling out "partial differential equation"; minor since the audience is technical, but inconsistent with the care taken elsewhere (e.g., DTW is expanded at first use).
- Table 3 (`tab:fm_comparison`): the "TimesFM" row has context length written as "[512]" in brackets (seemingly flagging uncertainty) while "Univariate/Multivariate" is "[TBD]" — these two placeholder conventions (bracketed value vs. TBD) are used inconsistently within the same table row.
- Table 7: only the "No Centroid" row has a bolded placeholder cell (`\textbf{--}`) under Inference Time, pre-emptively marking it as the presumed-best (fastest) value before any data exists; this bolding convention should wait until real numbers are available, or at least be applied consistently across all tables' placeholder cells.
- Section 6.1 and 6.3 each cite `\citep{snell2024scaling,cobbe2021verifiers}` and `\citep{snell2024scaling,liu2025videot1,ji2025survey}` respectively as general "test-time computing" support citations; while not wrong, some of these closing-sentence citation clusters feel slightly perfunctory and could be trimmed to the one or two most directly relevant works per sentence.

## Scores

| Criterion | Score (1-10) | Justification |
|-----------|:---:|---|
| Soundness | 6 | Conceptually coherent pipeline, but a real notational mismatch between the formal problem statement and the realized verifier, plus an undefined physics-residual symbol, make the formalism harder to fully verify as self-consistent. |
| Clarity | 5 | Sentence-level prose is strong, but multiple concrete figure-text, table-text, and citation mismatches (Weaknesses 1-5, 8-9) force the reader to reconcile contradictions rather than follow a single coherent narrative. |
| Novelty | 7 | Extending Best-of-$N$ test-time computing to a spatially coupled, physics-governed setting via a two-scale verifier is a reasonably novel combination, well differentiated from DEFT/SCALER/PINFDiT in Related Work. |
| Significance | 6 | A plausible, practically motivated niche (training-free improvement over frozen time-series FMs for spatio-temporal data) but significance cannot yet be judged without real numbers, and the scope is explicitly limited to one dataset/resolution. |
| Reproducibility | 5 | Many hyperparameters are specified exactly (M=32, N=20, N_A=15, N_C=5, K_A=5), which is good, but checkpoint identifiers, exact context length, and the history/horizon settings actually used are left unspecified (`[TBD]` or absent entirely), and there is no code/data availability statement. |
| **Overall** | **5** | **A promising, well-motivated draft whose core method section and ablation design are sound and clearly written, but which currently has enough internal figure/table/citation inconsistencies and undefined metrics that a careful reader cannot yet verify its own narrative — fixable issues, but requiring a real revision pass independent of filling in the placeholder numbers.** |

## Recommendation

**Major Revision**

The underlying method and ablation design are reasonable and the writing is largely free of superficial issues, but this review identified concrete, fixable structural problems (Weaknesses 1-5 in particular) that are independent of the missing numerical results: metrics discussed in prose with no supporting table, a figure that does not depict what the text says it depicts, a table missing the very row its surrounding paragraph refers to, an unexplained set of experimental settings appearing only in the Conclusion, and an inconsistent citation for a named baseline. None of these requires new experiments to fix, but together they currently prevent the paper from reading as a single, internally verifiable narrative. I would upgrade to Minor Revision or Accept once Weaknesses 1-5 are resolved and the remaining notational inconsistencies (7-10) are cleaned up, independent of whether the final numeric results are strong.
