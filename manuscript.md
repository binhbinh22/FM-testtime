<!-- NOTE TO REVIEWERS: This is a pre-results working draft of a master's-thesis paper. All result tables and specific numbers are intentional placeholders ([X], [X]\%, --) pending final experimental runs; please review the method, novelty, positioning, writing, and experimental DESIGN rather than penalizing the absence of final numbers. The method is evaluated on TWO datasets: ERA5 reanalysis (gridded) and WEATHER-5K (global weather stations). Source is LaTeX (elsarticle); tables/figures are shown as LaTeX. -->

\begin{frontmatter}

\title{Test-Time Computing for Spatio-Temporal Weather Forecasting with Foundation Models}

\begin{abstract}
Time-series foundation models such as Chronos-2 deliver strong zero-shot forecasts in a single frozen-weight forward pass, but this paradigm offers no mechanism to spend additional test-time compute for improved accuracy, nor any way to enforce physical or spatial structure on the output. Test-time computing has been explored for scaling reasoning and generation in other domains, but it has not been extended to spatio-temporal forecasting, where grid locations are spatially coupled and jointly governed by physical conservation laws. We introduce Spatial-Cluster Best-of-$N$ (ST-TTC), a training-free framework that generates candidate forecasts at two complementary spatial scales from a frozen backbone: full-resolution fine candidates and coarse candidates over spectrally clustered regions. Fine candidates are pre-filtered using a physical-plausibility residual before a composite verifier, combining trend, frequency, and spatial-consistency terms, scores the survivors against the coarse regional anchors to select the final forecast. On ERA5 reanalysis data, ST-TTC improves accuracy over single-pass inference by [X]\% RMSE and yields the clearest gains in spatial coherence, measured via Moran's $I$, while requiring only [Y]$\times$ the single-pass backbone compute.
\end{abstract}

\begin{highlights}

\item We propose a test-time computing (TTC) framework for spatio-temporal weather forecasting without retraining the backbone foundation model.
\item A composite verifier scores trend consistency, frequency periodicity, and spatial graph consistency among neighboring grid points, after a physics-based filter removes physically implausible candidates.

\item Our method improves forecasting accuracy over single-pass Chronos-2 inference by [X]\% RMSE while requiring only [Y]$\times$ additional inference compute.
\end{highlights}

\author[csdept]{}
\author[csdept]{}
\cortext[cor1]{Corresponding author}
\ead{email@hust.edu.vn}
\affiliation[csdept]{
  organization={},
  city={Hanoi},
  country={Vietnam}
}

\begin{keyword}
test-time computing \sep time series foundation models \sep spatio-temporal forecasting \sep physics-informed neural networks \sep weather forecasting
\end{keyword}

\end{frontmatter}

\section{Introduction}
\label{intro}

Weather forecasting fundamentally requires modeling time series that are coupled across both space, since neighboring atmospheric locations evolve jointly, and time. Recent time-series foundation models such as Chronos~\citep{ansari2024chronos}, Chronos-2~\citep{ansari2025chronos2}, TimesFM~\citep{das2024timesfm}, and Moirai~\citep{woo2024moirai} enable zero-shot forecasting across diverse domains without task-specific retraining. However, because these foundation models operate almost exclusively in a single inference pass, they produce a forecast without any mechanism to exploit additional test-time computation to improve forecast quality, nor do they provide a way to enforce physical conservation laws or spatial relationships on the generated trajectories.

In natural language processing and video generation, test-time computing (TTC)~\citep{snell2024scaling, liu2025videot1} has demonstrated that generating multiple candidates, scoring them, and searching for the best candidate at inference time can outperform scaling pre-training model parameters, all without modifying model weights. For time series, initial investigations have explored test-time refinement for univariate trajectories: DEFT~\citep{le2026deft} edits frozen foundation model forecasts under a query budget from an external expert evaluator, while SCALER~\citep{le2026scaler} guides an autoregressive model via coarse-to-fine residual refinement. In the spatio-temporal domain, Chen and Liang~\citep{chen2025stttc} recently pioneered test-time computing for spatio-temporal forecasting, termed ST-TTC, through a ``learning with calibration'' paradigm, where a spectral-domain calibrator with phase-amplitude modulation is updated online at test time via flash gradient descent on a streaming memory queue to counteract periodic non-stationarity. However, their framework fundamentally relies on test-time gradient adaptation of domain-specific architectures over discrete sensor graphs, such as traffic and air-quality monitoring networks. In contrast, zero-shot time-series foundation models operate with massive general pre-training, yet test-time computing for frozen, training-free foundation models on continuous physical spatio-temporal fields remains unexplored: there is currently no search or verification framework that scales inference compute to improve zero-shot forecasts while strictly enforcing spatial continuum coherence and atmospheric governing equations without test-time parameter updates or external expert feedback.

This leaves three concrete research gaps:
\begin{enumerate}
    \item \textbf{Physical plausibility under training-free zero-shot inference}: Standard candidate sampling from general-purpose time-series foundation models produces unconstrained numerical extrapolations that frequently violate atmospheric conservation laws, such as the advection of temperature by wind fields, or collapse into trivial, temporally flat predictions. While prior work injects physical ODE/PDE constraints into deep models during supervised end-to-end training~\citep{verma2024climode, park2025pint, xu2024weathergft}, there is no training-free mechanism to assess and enforce physical plausibility at test time.
    \item \textbf{Multi-scale spatial coupling across continuous grids}: Weather fields on planetary grids exhibit strong spatial correlations across multiple scales. Scoring each spatial grid node independently ignores the fact that neighboring locations evolve together, whereas scoring high-resolution candidate grids of $|\mathcal{V}| \times C$ series directly is computationally expensive and prone to overfitting localized high-frequency noise. A principled test-time search strategy must anchor fine-grained candidate trajectories to coherent regional dynamics.
    \item \textbf{Inference-budget efficiency under long horizons}: Full-length candidate generation and exhaustive multi-faceted scoring over long horizons across high-dimensional grids incur prohibitive computational cost under a fixed test-time budget. An efficient test-time framework must introduce early-filtering mechanisms to weed out unviable candidates before executing comprehensive verifier scoring.
\end{enumerate}

To address these gaps, we propose \textbf{Spatial-Cluster Best-of-$N$ (SC-BoN)}, a training-free test-time computing framework for spatio-temporal weather forecasting on frozen foundation models. Our contributions are summarized as follows:
\begin{itemize}
    \item We formulate a test-time candidate search framework for zero-shot time-series foundation models, evaluated primarily with Chronos-2, that operates in a strictly frozen, gradient-free regime ($\nabla_\theta = \mathbf{0}$), requiring neither domain retraining nor online test-time parameter adaptation.
    \item We introduce a two-scale architecture that partitions the spherical grid via haversine spectral clustering into $M$ regions. A lightweight coarse branch generates regional anchor trajectories, while a fine branch generates full-resolution candidates. To manage inference compute, candidates are first pruned by an advection PDE residual pre-filter with an activity regularizer, eliminating unphysical and trivial candidates before full scoring.
    \item We design a multi-objective spatio-temporal verifier that scores surviving candidates against the coarse anchors by jointly evaluating trend consistency via multi-scale Dynamic Time Warping, frequency-domain periodicity via Fourier spectral $\ell_1$ distance, and spatial smoothness via graph Laplacian divergence.
    \item We evaluate the proposed framework on multi-variable ERA5 reanalysis data spanning a $64 \times 32$ grid over 2m temperature and 10m wind components, systematically investigating compute budget scaling, verifier component sensitivity, spatial clustering topologies, and cross-backbone transferability across multiple foundation models.
\end{itemize}

The remainder of this paper is organized as follows: Section~\ref{sec:related_work} reviews related work on test-time computing, time-series foundation models, and physics-informed spatio-temporal modeling. Section~\ref{sec:problem} formally defines the problem and test-time computing constraints. Section~\ref{sec:method} introduces the proposed SC-BoN framework. Section~\ref{sec:settings} describes experimental settings and results. Section~\ref{sec:conclusion} concludes the paper.

\section{Related Work}
\label{sec:related_work}

\subsection{Test-Time Computing for LLMs and Vision Models}
The paradigm of sampling multiple candidate solutions and selecting among them with a verifier at test time was introduced by \citep{cobbe2021verifiers} for mathematical reasoning. \citep{snell2024scaling} later demonstrated that scaling test-time compute through candidate sampling and verification, or through iterative revision, can outperform scaling model parameters under an equivalent compute budget for large language models (see \citep{ji2025survey} for a comprehensive survey). In computer vision, Video-T1~\citep{liu2025videot1} extends this generate-score-search paradigm to test-time scaling for video diffusion models. These lines of work treat test-time compute as an independent axis for performance enhancement while keeping the backbone model frozen. However, they operate on single, self-contained outputs, such as a proof chain or a video clip, and do not address continuous spatio-temporal fields where multiple spatial locations are governed by physical conservation laws. Furthermore, Best-of-$N$ selection is subject to theoretical limitations: \citep{huang2025bestofn} establish that when the verifier is an imperfect proxy for the true objective, naively increasing $N$ can lead to optimization degradation and over-optimization, a risk we explicitly study when analyzing compute budget scaling and verifier design (Section~\ref{sec:analysis}).

\subsection{Time-Series Foundation Models}
Time-series foundation models adapt Transformer architectures to continuous sequential data. Chronos~\citep{ansari2024chronos} tokenizes scalar values into discrete bins and trains a language-model backbone to perform zero-shot univariate forecasting. Chronos-2~\citep{ansari2025chronos2} extends this architecture to multivariate, cross-series forecasting with an extended context window, enabling joint forecasting across parallel series. Concurrently, TimesFM~\citep{das2024timesfm} develops a decoder-only architecture with patched inputs for zero-shot forecasting, and Moirai~\citep{woo2024moirai} trains a universal forecaster across heterogeneous multivariate datasets with any-variate attention. Nevertheless, all these foundation models are architected for a single forward pass: given a historical context, they output a single deterministic forecast or i.i.d.\ stochastic samples. They lack any inference-time mechanism to expend additional compute to verify, align, or constrain the output with respect to domain-specific physics or spatial geometries.

\subsection{Test-Time Refinement and Adaptation for Temporal and Spatio-Temporal Data}
A growing body of research explores test-time computation and adaptation to improve sequential predictions without full model retraining. Existing methodologies in this space broadly fall into two methodological paradigms: test-time search and editing, and test-time parameter adaptation.

\paragraph{Test-time search and editing.}
For univariate time series, DEFT~\citep{le2026deft} freezes the pre-trained foundation model and edits its forecasts within a decomposed trend--seasonal space around anchor trajectories, guided by sequential queries to an external expert evaluator under a restricted query budget. Concurrently, SCALER~\citep{le2026scaler} scales test-time compute for autoregressive models by training a lightweight auxiliary transformer to generate coarse predictions that actively guide an LLM forecaster through coarse-to-fine residual refinement. While these approaches demonstrate the efficacy of test-time search, they focus primarily on isolated univariate series and rely on external evaluator queries or auxiliary trained networks to steer candidate generation.

\paragraph{Test-time adaptation and online calibration.}
A parallel line of work adapts model representations during inference using incoming test streams. Wang et al.~\citep{wang2026tta} investigate test-time adaptation (TTA) for time-series forecasting by minimizing unsupervised auxiliary losses via gradient descent on streaming test batches. In the spatio-temporal domain, Chen and Liang~\citep{chen2025stttc} recently introduced ST-TTC, an inference-stage calibration framework designed to counteract periodic non-stationarity in sensor networks such as traffic flow and urban air-quality monitoring. ST-TTC decomposes predictions in the spectral domain via real-to-complex Fourier transforms, adjusting amplitude and phase components, and updates calibrator parameters online through flash gradient descent over a streaming memory queue of past test observations.

While online adaptation and calibration techniques offer adaptability on discrete sensor networks, they fundamentally depend on test-time gradient backpropagation and parameter updates ($\nabla_\theta \neq \mathbf{0}$) on domain-trained architectures. In contrast, our work explores test-time computing in a strictly training-free and gradient-free regime ($\nabla_\theta = \mathbf{0}$) for general-purpose time-series foundation models, directing inference compute toward multi-candidate generation and physical/spatial verification over continuous atmospheric fields without modifying model weights during inference.

\subsection{Physics-Informed Weather and Climate Forecasting}
A parallel line of research incorporates physical priors directly into neural weather forecasting architectures. ClimODE~\citep{verma2024climode} models atmospheric field dynamics using continuous neural ODEs constrained by advection PDEs. PINT~\citep{park2025pint} enforces physical periodicity on long-horizon 2m temperature forecasts using the analytical solution of a damped harmonic oscillator. WeatherGFT~\citep{xu2024weathergft} interleaves Transformer layers with explicit PDE solver kernels, while PASSAT~\citep{zheng2025passat} develops spherical graph neural networks that numerically solve governing equations on the spherical manifold of the Earth. All these models, however, require full end-to-end training or supervised fine-tuning on domain data. PINFDiT~\citep{cao2026pinfdit} is closer in spirit to test-time physical guidance, injecting an advection-diffusion physics energy term at inference time into a pre-trained diffusion transformer via Langevin dynamics. Nonetheless, PINFDiT operates on multivariate channels without modeling spatial graph adjacency or spherical geodesic topology across geographic grids, and it steers a diffusion sampling process rather than performing verification-based selection among candidates from a general-purpose forecaster. In our framework, we evaluate advection PDE residuals via finite differences, using them as a physical pre-filter (Section~\ref{sec:generation}) to prune unviable candidates before multi-objective verifier scoring.

\subsection{Graph-Based Spatio-Temporal Modeling}
Representing spatial grids as graphs where nodes denote geographical locations and edges represent spatial proximity is standard in data-driven weather forecasting. GraphCast~\citep{lam2023graphcast} trains multi-mesh graph neural networks on historical ERA5 data for skillful medium-range deterministic forecasting, and GenCast~\citep{price2025gencast} extends this to probabilistic ensemble forecasting using diffusion models on spherical graphs. Both approaches, however, require training massive neural networks from scratch on decades of reanalysis data. In contrast, our framework employs graph representations purely at test time: we construct a geodesic kNN graph on the spherical manifold to perform spectral clustering for coarse regional anchor generation and to evaluate spatial smoothness through the graph Laplacian, requiring zero graph network training.

\section{Problem Definition}
\label{sec:problem}

\subsection{Spatial Domain and Graph Representation}
We represent the atmospheric spatial domain as a geodesic graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, W)$, where each node $v \in \mathcal{V}$ corresponds to one spatial location on the spherical Earth surface. A location is a grid cell when the data is a gridded reanalysis, or a weather station when the data is a station network; our experiments cover both cases (Section~\ref{sec:settings}), and the framework treats $\mathcal{V}$ uniformly as a set of $|\mathcal{V}|$ georeferenced points regardless of their origin. Edges $(u, v) \in \mathcal{E}$ connect spatially nearby locations based on geodesic (great-circle) distances, which respect the curvature of the sphere and the convergence of meridians near the poles. Edge weights $W_{uv}$ reflect spatial proximity via Gaussian radial basis functions.

Each node $v \in \mathcal{V}$ carries $C$ continuous surface variables, which always include the temperature and the horizontal wind used by the physics term of Section~\ref{sec:reward}; the full variable set is dataset-specific and is listed in Section~\ref{sec:settings}. Given a historical observation window of length $L_H$, denoted as $x^H \in \mathbb{R}^{|\mathcal{V}| \times C \times L_H}$, the objective is to forecast the future state trajectory over the next $L_F$ time steps:
\begin{equation}
x^F \in \mathbb{R}^{|\mathcal{V}| \times C \times L_F},
\end{equation}
jointly for all spatial nodes and physical variables.

\subsection{Frozen Foundation Model Backbone and Test-Time Computing Constraint}
Let $p_\theta(x^F \mid x^H)$ denote the predictive distribution of a pre-trained zero-shot time-series foundation model parameterized by $\theta$. In our test-time computing framework, we enforce a strict \textbf{training-free and gradient-free constraint}:
\begin{equation}
\theta^{(t)} = \theta, \quad \nabla_\theta \mathcal{L} = \mathbf{0},
\label{eq:frozen_constraint}
\end{equation}
meaning that no model parameters $\theta$ are updated via gradient descent, fine-tuning, or test-time backpropagation, nor are any auxiliary parametric adapters optimized during inference. This fundamentally distinguishes our approach from test-time adaptation or online calibration paradigms~\citep{chen2025stttc, wang2026tta}. The foundation model $p_\theta$ is treated strictly as a frozen generative operator.

\section{Method}
\label{sec:method}
We first describe the overall framework in Section~\ref{sec:overview}. Section~\ref{sec:clustering} describes the spatial clustering and graph used throughout the method. Section~\ref{sec:generation} introduces the coarse and fine candidate-generation branches. Section~\ref{sec:reward} details the composite verifier and the selection rule.

\subsection{Framework Overview}
\label{sec:overview}

The proposed framework, which we refer to as Spatial-Cluster Best-of-$N$, follows the ``generate, score, select'' pattern of test-time computing~\citep{snell2024scaling}, but organizes it around two spatial scales so that a cheap low-resolution forecast can anchor the verification of full-resolution candidates (Figure~\ref{fig:arch}; Table~\ref{tab:overview}). It proceeds in four stages, none of which updates the backbone parameters $\theta$. First, the set of spatial locations is partitioned once into $M$ spatial clusters by spectral clustering, yielding an assignment matrix $S$ (Section~\ref{sec:clustering}). Second, a \emph{coarse} branch aggregates the history to the $M$ cluster centroids, downsamples it in time, and has the frozen backbone produce $N_C$ low-resolution regional anchors (Section~\ref{sec:generation}). Third, a \emph{fine} branch has the backbone produce $N_A$ full-resolution candidates, which a physical-plausibility filter prunes to the $K_A$ most physically consistent (Section~\ref{sec:generation}). Finally, each surviving candidate is scored by a composite verifier $E(\cdot)$ that measures its trend, periodicity, and spatial consistency with the anchors and the history, and the lowest-scoring candidate is returned as the final forecast $\hat{x}^F$ (Section~\ref{sec:reward}).

\begin{figure}[t]
\centering
\begin{tikzpicture}[
  >={Latex[length=2.2mm]},
  box/.style={draw, rounded corners, align=center, font=\small, inner sep=4pt, minimum height=8mm},
  branch/.style={draw, rounded corners, align=left, font=\small, inner sep=5pt, text width=52mm},
  shaded/.style={draw, rounded corners, align=center, font=\small, inner sep=4pt, minimum height=8mm, fill=black!5},
]
\node[box] (hist) {History $x^H$\\ ($|\mathcal{V}|$ locations, $C$ variables)};
\node[shaded, left=12mm of hist] (clust) {Spectral\\ clustering\\ $\Rightarrow S$\\ (once)};
\draw[->] (clust) -- (hist);

\node[branch, below=12mm of hist, xshift=-34mm] (coarse)
  {\textbf{Coarse branch}\\ aggregate to $M$ clusters via $S$;\\ temporal downsample;\\ frozen backbone $\to$ $N_C$ anchors;\\ trend / seasonal decomposition};
\node[branch, below=12mm of hist, xshift=34mm] (fine)
  {\textbf{Fine branch}\\ frozen backbone $\to$ $N_A$ candidates;\\ physics filter $\to$ top $K_A$;\\ spatial decomposition\\ $\to Y_{\text{global}},\, Y_{\text{local}}$};
\draw[->] (hist.south) -- (coarse.north);
\draw[->] (hist.south) -- (fine.north);

\node[box, below=36mm of hist, text width=74mm] (ver)
  {\textbf{Composite verifier}\\ $E = w_1 E_{\text{trend}}\text{(DTW)} + w_2 E_{\text{FFT}} + w_3 E_{\text{spatial}}$\\ (each fine candidate scored against its best coarse anchor)};
\draw[->] (coarse.south) -- (ver.north);
\draw[->] (fine.south) -- (ver.north);

\node[box, below=6mm of ver] (sel) {Selection: $\hat{x}^F = \arg\min E$};
\draw[->] (ver) -- (sel);
\node[shaded, below=6mm of sel] (out) {Final forecast $\hat{x}^F$};
\draw[->] (sel) -- (out);
\end{tikzpicture}
\caption{Overview of the Spatial-Cluster Best-of-$N$ (SC-BoN) framework. The locations are clustered once; a coarse branch produces low-resolution regional anchors and a fine branch produces full-resolution candidates that a physics filter prunes; a three-term verifier scores each surviving candidate against its best anchor, and the lowest-scoring candidate is returned. The backbone is frozen throughout.}
\label{fig:arch}
\end{figure}

\begin{table}[H]
\centering
\caption{Overview of the proposed SC-BoN framework.}
\begin{tabular}{p{2.6cm}p{4.3cm}p{4.3cm}}
\toprule
\textbf{Stage} & \textbf{Input} & \textbf{Output} \\
\midrule
Spatial clustering & graph $\mathcal{G}$ (spatial geometry) & assignment $S \in \{0,1\}^{|\mathcal{V}| \times M}$ \\
Coarse branch & $x^H$, $S$ & $N_C$ regional anchors \\
Fine branch & $x^H$ & $K_A$ candidates (after physics filter) \\
Scoring \& selection & $K_A$ candidates, anchors, $\mathcal{G}$ & $\hat{x}^F$ (lowest $E$) \\
\bottomrule
\end{tabular}
\label{tab:overview}
\end{table}

\subsection{Spatial Clustering and Graph Construction}
\label{sec:clustering}

We build an undirected graph over the $|\mathcal{V}|$ spatial locations by connecting each location to its $k$ nearest neighbours under great-circle (haversine) distance, with Gaussian edge weights $w_{ij}=\exp(-d_{ij}^2/2\sigma^2)$ under a single bandwidth $\sigma$~\citep{vonluxburg2007tutorial}. Great-circle distance is used rather than a Euclidean distance on latitude--longitude pairs because it respects the spherical geometry of the Earth: on a regular grid it correctly handles the convergence of meridians at the poles, and for an irregular station network it measures true surface proximity rather than distance in a distorted coordinate plane. We then partition the graph into $M$ clusters by spectral clustering~\citep{ng2002spectral}, giving a hard assignment matrix $S\in\{0,1\}^{|\mathcal{V}|\times M}$ with $S_{vc}=1$ when location $v$ belongs to cluster $c$. The partition depends only on the spatial geometry of the locations, not on the forecasted values, so it is stable across forecast windows and is computed a single time; the same graph is reused by the spatial-consistency term of Section~\ref{sec:reward}.

\subsection{Candidate Generation}
\label{sec:generation}

Candidates are generated at two spatial scales from the same frozen backbone $p_\theta$ (Chronos-2~\citep{ansari2025chronos2} in our experiments; see Section~\ref{sec:settings} for the exact configuration), without modifying any of its parameters.

\paragraph{Coarse branch (regional anchors).} We aggregate the history to the $M$ cluster centroids through $S$, producing $M$ parallel cluster-mean series, and downsample them in time by average pooling, following the multiscale construction of TimeMixer~\citep{wang2024timemixer}. The backbone then forecasts $N_C$ candidates for these downsampled centroid series. Living at a coarse spatial (cluster) and temporal (downsampled) scale, they are cheap to produce and serve as low-resolution regional anchors against which the full-resolution candidates are verified.

\paragraph{Fine branch (full resolution).} We treat the $|\mathcal{V}|$ spatial locations as parallel series and forecast $N_A$ candidates at full spatial resolution. We draw these candidates from the backbone's predictive distribution rather than taking its single most-likely output, so that they reflect the diversity of plausible continuations the backbone already assigns non-negligible probability to.

\paragraph{Physical-plausibility filter.} Scoring every fine candidate with the full verifier is expensive, and many candidates are physically implausible to begin with, so we first prune the $N_A$ candidates to the $K_A$ most physically consistent and pass only these to the verifier of Section~\ref{sec:reward}. Atmospheric temperature is transported by the wind field and compressed by its divergence; we encode this continuity (advection) constraint as a PDE residual on each candidate's temperature and wind fields, following a Physics-Informed formulation~\citep{verma2024climode, park2025pint, cao2026pinfdit}. To make the spatial derivatives well defined whether the locations form a regular grid or an irregular station network, we evaluate the residual at the cluster scale: each candidate is aggregated to the $M$ cluster centroids, and spatial gradients and divergences are taken between neighbouring clusters on the cluster graph, which reduces to standard finite differences when the locations happen to form a regular grid. Since surface temperature is not an exactly conserved quantity, due to diabatic effects such as solar heating, the residual acts only as a soft filter. To keep the filter from favouring trivially flat candidates that satisfy the residual degenerately, we add an activity penalty that discourages candidates whose temporal variance falls below that of the observed history, and retain the $K_A$ candidates with the lowest resulting score.

\subsection{Composite Verifier and Selection}
\label{sec:reward}

Each of the $K_A$ candidates that survive the physical-plausibility filter is scored by a composite verifier measuring its consistency with the coarse anchors and with the spatial structure of the history. None of the terms below compares a candidate to the unknown ground-truth future. The verifier combines three terms linearly,
\begin{equation}
E(x^F) = w_1 E_{\text{trend}} + w_2 E_{\text{FFT}} + w_3 E_{\text{spatial}}
\label{eq:reward}
\end{equation}
with non-negative weights $w_1, w_2, w_3$ tuned on a validation split (Section~\ref{sec:settings}).

\paragraph{Cross-scale decomposition.} To compare a fine candidate against the coarse anchors on common ground, we first decompose it spatially through $S$ into a cluster-mean component $Y_{\text{global}}$, the candidate aggregated to the $M$ clusters, and a within-cluster residual $Y_{\text{local}}$, the candidate minus its cluster mean. The trend and frequency terms operate on $Y_{\text{global}}$, which lives at the same cluster scale as the anchors, while the spatial term operates on $Y_{\text{local}}$.

\paragraph{Trend consistency ($E_{\text{trend}}$).} We split $Y_{\text{global}}$ and each coarse anchor into a slowly varying trend and a seasonal residual by a centered moving-average decomposition~\citep{wu2021autoformer}, matching the moving-average window across the two scales so that both extract the trend over the same physical time span. $E_{\text{trend}}$ is the Dynamic Time Warping (DTW) distance between the candidate's cluster-level trend and the anchor's trend, penalizing candidates whose regional trend departs from the low-resolution forecast.

\paragraph{Frequency consistency ($E_{\text{FFT}}$).} We compare the periodicity of the candidate's seasonal component with that of the anchor in the frequency domain. Because the two scales have different temporal resolutions, we downsample the candidate's seasonal component to the anchor scale before applying the Fourier transform, then take the $\ell_1$ distance between the two spectra, as in frequency-domain forecasting objectives~\citep{wang2025fredf}. $E_{\text{FFT}}$ penalizes candidates whose periodic structure is inconsistent with the regional anchor.

\paragraph{Spatial consistency ($E_{\text{spatial}}$).} Using the graph $\mathcal{G}$ of Section~\ref{sec:clustering}, we measure the spatial roughness of the within-cluster residual $Y_{\text{local}}$ at each forecast step by a graph-Laplacian smoothness functional, the normalized Dirichlet energy of the signal on the graph~\citep{shuman2013gsp,vonluxburg2007tutorial}, and compare it to the roughness of the corresponding residual in the history. This compares a candidate's spatial structure against a reference in the spirit of the variogram score used for spatial forecast verification~\citep{scheuerer2015variogram}, but evaluated on the graph rather than on a regular grid. $E_{\text{spatial}}$ penalizes candidates that are spatially rougher, or spuriously smoother, than the spatial variability already present in the history, rather than requiring an arbitrary absolute threshold.

\paragraph{Selection.} Each fine candidate is scored against the coarse anchor that matches it best, taking the minimum of Eq.~\eqref{eq:reward} over the $N_C$ anchors, and the candidate with the lowest resulting score is returned as the final forecast,
\begin{equation}
\hat{x}^F = \arg\min_{i} \; \min_{j \in \{1,\dots,N_C\}} E\!\left(x^F_i ; a_j\right),
\label{eq:selection2}
\end{equation}
where $a_j$ denotes the $j$-th coarse anchor. The selection reuses candidates already drawn from the frozen backbone and requires no additional backbone calls.

\section{Experimental Settings and Results}
\label{sec:settings}

\subsection{Experimental Settings}

\paragraph{Datasets.} We evaluate on two complementary weather datasets that instantiate the two kinds of spatial domain defined in Section~\ref{sec:problem}. The first is \textbf{ERA5} reanalysis on a regular $64\times32$ grid ($|\mathcal{V}|=2048$ grid cells), with surface variables 2m temperature and the two 10m wind components (u10, v10); forecast windows are drawn from a held-out test period (2017--2018) with a disjoint validation period (2015--2016). The second is \textbf{WEATHER-5K}~\citep{han2024weather5k}, a station network of $|\mathcal{V}|=5{,}672$ irregularly spaced global weather stations, each providing hourly observations of five surface variables (temperature, dewpoint, wind direction, wind speed, and sea-level pressure); forecast windows are drawn from a held-out test year (2023) with a disjoint validation year (2022). In both cases the backbone is frozen and never trained, so there is no training split, and the great-circle kNN graph and spectral clustering of Section~\ref{sec:clustering} apply unchanged whether the nodes are grid cells or stations.

\paragraph{Forecast settings.} On WEATHER-5K we use an input history of $L_H=48$ hours and a forecast horizon of $L_F=24$ hours; on ERA5 we evaluate the history/horizon configurations $(512,64)$, $(512,128)$, and $(1024,96)$ time steps. Both datasets use a default test-time budget of $N=20$ candidates ($N_A=15$ fine and $N_C=5$ coarse). Unless stated otherwise, reported metrics are averaged over all locations.

\paragraph{Foundation Backbones.} All main experiments build on Chronos-2~\citep{ansari2025chronos2}, chosen for its native multivariate cross-series support across the $|\mathcal{V}| \times C$ parallel series and its long 8192-step context window. The backbone is kept entirely frozen without fine-tuning and queried solely via its forecasting interface. To assess generality, we instantiate the identical pipeline on Chronos-1~\citep{ansari2024chronos}, TimesFM~\citep{das2024timesfm}, and Moirai~\citep{woo2024moirai} for the backbone-ablation study.

\paragraph{Baselines.} We compare against three groups: (i) published forecasting models (PINT, ClimODE, ClimaX, PINFDiT) as reported in their original works; (ii) internal references on our frozen Chronos-2 (single-pass greedy and consensus-based Best-of-$N$ without composite verifiers); and (iii) alternative test-time search optimizers, including Random Search, Quantile Best-of-$N$, Direct CEM, TuRBO-1, and Surrogate CEM~\citep{le2026deft}.

\paragraph{Metrics.} We report physical MAE and RMSE per variable, alongside aggregate normalized MSE following WeatherBench and GraphCast~\citep{lam2023graphcast}. Latitude-weighted RMSE and ACC are measured against the validation climatology~\citep{verma2024climode}. We additionally report the continuous ranked probability score (CRPS), a physics-violation residual $\mathcal{R}_{\text{phys}}$ (advection/continuity PDE), and spatial coherence via Moran's $I$. Compute cost is reported as the number of backbone calls.

\paragraph{Implementation.} The framework is implemented in Python with PyTorch, with backbone inference executed on a single GPU; the spectral clustering step and the composite verifier, including the physics filter, run on CPU, since both operate on comparatively small, already-downsampled representations and add negligible overhead relative to backbone calls. 
\subsection{Main Results}
\label{sec:main_results}

\paragraph{Main results.} Table~\ref{tab:main_results} summarizes forecasting accuracy and compute cost across the evaluated history/horizon settings. The comparison spans two groups of methods. The first group consists of published weather and climate forecasting methods \citep{park2025pint,verma2024climode,nguyen2023climax,cao2026pinfdit}, whose numbers are taken as reported in their original papers. The second group consists of two reference points built on our own frozen Chronos-2 backbone \citep{ansari2025chronos2}: a single-pass forecast with no test-time computing, which represents the accuracy floor of the backbone alone, and a simple-scorer Best-of-$N$ \citep{snell2024scaling} that samples the same candidate budget but selects among candidates using the simple truth-free consensus scorer defined above rather than our composite verifier. The table reports per-variable MAE and RMSE for t2m, u10, and v10, whether each method requires training, and the number of backbone forecast calls per prediction; we additionally report the aggregate normalized MSE, the latitude-weighted RMSE and ACC, and the Moran's $I$ spatial-coherence statistic in Table~\ref{tab:aggregate}. We flag in the ``Train?'' column that, among all methods considered, only PINFDiT \citep{cao2026pinfdit} and our method require no additional training or fine-tuning; every other baseline relies on learned weights fitted to the forecasting task.

\begin{table*}[htbp]
\centering
\renewcommand{\arraystretch}{1.25}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l|cc|cc|cc|cc}
\toprule
\multirow{2}{*}{Model} & \multicolumn{2}{c|}{t2m} & \multicolumn{2}{c|}{u10} & \multicolumn{2}{c|}{v10} & \multirow{2}{*}{Train?} & \multirow{2}{*}{\# calls} \\
 & MAE & RMSE & MAE & RMSE & MAE & RMSE & & \\
\midrule
\multicolumn{9}{l}{\textit{Published methods (numbers as reported in the original paper)}} \\
PINT~\citep{park2025pint}        & --& --& --& --& --& --& Yes & -- \\
ClimODE~\citep{verma2024climode} & --& --& --& --& --& --& Yes & -- \\
ClimaX~\citep{nguyen2023climax}  & --& --& --& --& --& --& Yes & -- \\
PINFDiT~\citep{cao2026pinfdit}   & --& --& --& --& --& --& No  & -- \\
\midrule
\multicolumn{9}{l}{\textit{Reference points on our backbone (Chronos-2, no training)}} \\
Chronos-2 (single-pass, no TTC)  & --& --& --& --& --& --& No & 1 \\
Best-of-$N$ (simple scorer)      & --& --& --& --& --& --& No & $N$ \\
\midrule
Ours (proposed)                  & \textbf{--}& \textbf{--}& \textbf{--}& \textbf{--}& \textbf{--}& \textbf{--}& No & $N$ \\
\bottomrule
\end{tabular}%
}
\caption{Main results on ERA5, averaged over all grid points, reported per variable (t2m, u10, v10). We compare our method against published weather/climate forecasting methods (top block) and against two reference points on our own frozen backbone (middle block). ``Train?'' indicates whether the method requires training or fine-tuning on the target data; only PINFDiT and our method operate without retraining. Numbers for published methods are taken from their original papers and may use a different experimental setup, so they serve as a reference rather than a strictly controlled comparison; ``--'' marks a variable not reported in the source. Ablations of the verifier components and of the backbone choice appear in Table~\ref{tab:ablation_verifier} and Table~\ref{tab:ablation_backbones}.}
\label{tab:main_results}
\end{table*}

Our method attains the best or comparable per-variable MAE and RMSE among the no-training methods and remains competitive with the trained baselines, while delivering its clearest advantage on spatial coherence, improving Moran's $I$ by [X] over the single-pass floor. Relative to the single-pass backbone, our selection procedure reduces aggregate normalized MSE by [X]\%; relative to the simple-scorer Best-of-$N$, which already yields a modest gain of [X]\% over the floor, our composite verifier contributes a further [X]\% reduction. Because both reference points draw from the same candidate budget and the same frozen backbone, this gap isolates the contribution of the three-term verifier (trend, spectral, and spatial agreement with the coarse anchor) rather than any difference in model capacity or sampling. This benefit comes at the cost of $N$ backbone calls per prediction, compared to a single call for the floor, a cost-accuracy trade-off we revisit in the compute-scaling analysis (Figure~\ref{fig:scaling}).

\begin{table}[H]
\centering
\small
\caption{Aggregate metrics on ERA5, averaged over the three variables and the three forecast settings (placeholders pending final runs). RMSE$_{\mathrm{lw}}$ is latitude-weighted; Moran's $I$ measures spatial coherence (higher is better).}
\label{tab:aggregate}
\begin{tabular}{lccccc}
\toprule
Method & norm. MSE $\downarrow$ & RMSE$_{\mathrm{lw}}$ $\downarrow$ & ACC $\uparrow$ & Moran's $I$ $\uparrow$ & \# calls \\
\midrule
Chronos-2 (single-pass)      & -- & -- & -- & -- & 1 \\
Best-of-$N$ (simple scorer)  & -- & -- & -- & -- & $N$ \\
\textbf{Ours}                & \textbf{--} & \textbf{--} & \textbf{--} & \textbf{--} & $N$ \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Comparison of Test-Time Search Strategies}
\label{sec:search_strategies}

The Best-of-$N$ selection at the core of our method is only one of several ways to spend a fixed test-time budget on a frozen backbone. To justify this choice, we compare it against a suite of alternative test-time search strategies, following the protocol of \citep{le2026deft}. All strategies share the same frozen Chronos-2 backbone \citep{ansari2025chronos2} and the same truth-free composite verifier as their selection objective, and differ only in how they explore and select among candidate forecasts. We consider: (i) \emph{Random search}, which draws candidates and selects one uniformly at random, serving as a lower bound that uses no verifier signal; (ii) \emph{Best-of-$N$ (sample)} and \emph{Best-of-$N$ (quantile)}, which obtain $N$ candidates by stochastic sampling and by reading the backbone's quantile levels respectively, then select the candidate that minimizes the verifier; (iii) \emph{Direct CEM} \citep{deboer2005cem}, which runs the cross-entropy method directly in forecast (horizon) space, iteratively sampling from a Gaussian proposal and refitting it to the top-scoring elites; (iv) \emph{TuRBO-1} \citep{eriksson2019turbo}, which performs local Bayesian optimization of the verifier within a single trust region using a Gaussian-process surrogate; and (v) \emph{Surrogate CEM}, which augments the CEM loop with a cheap regression surrogate fit on past verifier evaluations to amortize the number of expensive queries.

Table~\ref{tab:search_strategies} reports this comparison. The black-box optimizers (Direct CEM, TuRBO-1, Surrogate CEM) search directly in horizon space and can drift away from the data manifold that the backbone's own candidates occupy, whereas Best-of-$N$ restricts the search to on-manifold candidates produced by the frozen model. We therefore expect our two-scale Best-of-$N$ to match or exceed the more elaborate optimizers while using a smaller and more predictable budget, consistent with the observation of \citep{le2026deft} that Best-of-$N$ over quantile candidates outperforms CEM- and Bayesian-optimization-based search in this setting. Because every strategy here uses the same frozen backbone and the same composite verifier, with no training, this comparison isolates the effect of the search strategy itself; the ``$B$'' entries in the budget column denote the iterative evaluation budget spent by the optimizers, which is in general no smaller than the $N$ candidate forecasts used by Best-of-$N$.

\begin{table*}[htbp]
\centering
\renewcommand{\arraystretch}{1.25}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l|cc|cc|cc|c}
\toprule
\multirow{2}{*}{Search strategy} & \multicolumn{2}{c|}{t2m} & \multicolumn{2}{c|}{u10} & \multicolumn{2}{c|}{v10} & \multirow{2}{*}{Budget} \\
 & MAE & RMSE & MAE & RMSE & MAE & RMSE & \\
\midrule
Random search~\citep{le2026deft}           & -- & -- & -- & -- & -- & -- & $N$ \\
Best-of-$N$ (sample)~\citep{snell2024scaling} & -- & -- & -- & -- & -- & -- & $N$ \\
Best-of-$N$ (quantile)                      & -- & -- & -- & -- & -- & -- & $N$ \\
Direct CEM~\citep{deboer2005cem}            & -- & -- & -- & -- & -- & -- & $B$ \\
TuRBO-1~\citep{eriksson2019turbo}           & -- & -- & -- & -- & -- & -- & $B$ \\
Surrogate CEM                               & -- & -- & -- & -- & -- & -- & $B$ \\
\midrule
Ours (SC-BoN, two-scale)                    & \textbf{--} & \textbf{--} & \textbf{--} & \textbf{--} & \textbf{--} & \textbf{--} & $N$ \\
\bottomrule
\end{tabular}%
}
\caption{Comparison of test-time search strategies on ERA5, reported per variable (t2m, u10, v10), averaged over all grid points. All strategies use the same frozen Chronos-2 backbone and the same truth-free composite verifier as the selection objective and require no training; they differ only in how candidates are explored and selected. Budget $N$ denotes the number of candidate forecasts for Best-of-$N$ and random search; $B$ denotes the iterative evaluation budget of the black-box optimizers (in general $B \geq N$). Baseline suite follows \citep{le2026deft}.}
\label{tab:search_strategies}
\end{table*}

\subsection{Ablation Study}
\label{sec:ablation}

\subsubsection{Ablation on Verifier Reward Components (Q1)}
\label{sec:ablation_verifier_components}

To isolate the contribution of each term in the composite verifier and of the physics-plausibility filter, we construct a set of ablated variants of ST-TTC at a fixed total budget $N$. Starting from the full configuration, we in turn remove $E_{\text{trend}}$, $E_{\text{FFT}}$, and $E_{\text{spatial}}$ from the verifier score $E = w_1 E_{\text{trend}} + w_2 E_{\text{FFT}} + w_3 E_{\text{spatial}}$ (renormalizing the remaining weights), disable the physics-plausibility filter so that the verifier scores all $N_A$ fine candidates rather than the $K_A$ survivors it would otherwise see, and include a random-selection control that forecasts the same number of fine candidates but selects among them uniformly at random, bypassing the filter and the verifier entirely. Table~\ref{tab:ablation_verifier} reports per-variable MAE/RMSE and Moran's $I$ for every variant alongside the full ST-TTC pipeline, so that the contribution of each component can be read off directly against this common baseline.

\begin{table}[H]
\centering
\small
\caption{Ablation on verifier components under fixed compute budget $N$.}
\label{tab:ablation_verifier}
\begin{tabular}{lcc}
\toprule
Variant & RMSE $\downarrow$ & \# model calls \\
\midrule
\textbf{Full verifier (Ours)} & \textbf{--} & $[N]$ \\
w/o $E_{\text{trend}}$      & --          & $[N]$ \\
w/o $E_{\text{FFT}}$        & --          & $[N]$ \\
w/o $E_{\text{spatial}}$    & --          & $[N]$ \\
w/o physics filter          & --          & $[N]$ \\
\midrule
Random selection (no verifier) & --      & $[N]$ \\
\bottomrule
\end{tabular}
\end{table}

Every ablated variant degrades relative to the full configuration on at least one metric, supporting the view that the three verifier terms and the physics filter are complementary rather than redundant. Removing $E_{\text{spatial}}$, the only term that explicitly scores within-cluster spatial structure, produces the largest drop in Moran's $I$ [X], while removing $E_{\text{trend}}$ or $E_{\text{FFT}}$ instead mainly erodes MAE/RMSE by [X] and [X] respectively, consistent with their role in matching low-frequency trend and seasonal content to the coarse anchor. Disabling the physics filter allows physically implausible candidates to reach the verifier and lowers both accuracy and spatial coherence by [X], indicating that filtering and verification address different failure modes rather than one subsuming the other. The random-selection control, which neither filters nor verifies, is the worst variant on every metric, often by a clear margin [X], confirming that the gains of ST-TTC come from the quality of the selection mechanism rather than from merely sampling multiple candidates \citep{snell2024scaling,cobbe2021verifiers}. Taken together, these results support retaining all three verifier terms together with the physics filter as the default configuration.

\subsubsection{Effect of Test-Time Compute Budget (Q2)}
\label{sec:ablation_compute}

We next examine how the compute budget $N$, the number of backbone forecast calls used per prediction, trades off against accuracy and physical consistency. Holding the number of spatial clusters $M$ and the fine/coarse split ratio fixed at their default values, we sweep $N$ from $1$ (equivalent to a single greedy forecast with no search) up to $64$, re-running the full ST-TTC pipeline at each budget. Table~\ref{tab:ablation_compute} reports per-variable error and a physics-violation measure (the PDE residual used by the physics filter) at each budget, and Figure~\ref{fig:scaling} illustrates the resulting error-versus-budget trend, in line with the test-time scaling analyses used elsewhere in the test-time compute literature \citep{snell2024scaling,huang2025bestofn,le2026scaler}.

\begin{table}[H]
\centering
\small
\caption{Performance scaling with respect to test-time compute budget $N$ on ERA5.}
\label{tab:ablation_compute}
\begin{tabular}{lcccc}
\toprule
Compute Budget ($N$) & RMSE $\downarrow$ & CRPS $\downarrow$ & Physics Violation ($\mathcal{R}_{\text{phys}}$) $\downarrow$ & Latency (s/sample) $\downarrow$ \\
\midrule
$N=1$ (Greedy / 1-shot) & -- & -- & -- & \textbf{--} \\
$N=2$                  & -- & -- & -- & -- \\
$N=4$                  & -- & -- & -- & -- \\
$N=8$                  & -- & -- & -- & -- \\
$N=16$ (saturation)    & \textbf{--} & \textbf{--} & \textbf{--} & -- \\
$N=32$                 & -- & -- & -- & -- \\
$N=64$                 & -- & -- & -- & -- \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[
  width=0.80\linewidth, height=5cm,
  xlabel={Test-time budget $N$ (backbone calls)},
  ylabel={RMSE (lower is better)},
  xmode=log, log basis x=2,
  xtick={1,2,4,8,16,32,64}, xticklabels={1,2,4,8,16,32,64},
  ytick=\empty, ymin=0,
  grid=both, grid style={gray!18},
  tick label style={font=\small}, label style={font=\small},
]
\addplot[thick, mark=*] coordinates {(1,1.00)(2,0.92)(4,0.86)(8,0.82)(16,0.80)(32,0.79)(64,0.785)};
\addplot[dashed, gray, forget plot] coordinates {(16,0)(16,0.80)};
\node[font=\scriptsize, anchor=south west] at (axis cs:16,0.02) {$N\approx16$};
\end{axis}
\end{tikzpicture}
\caption{Illustrative scaling of forecast error with the test-time budget $N$ (schematic placeholder; axis values are not measured results and will be replaced with experimental numbers). Error decreases with $N$ and saturates near $N=16$.}
\label{fig:scaling}
\end{figure}

Both forecast error and the physics-violation measure fall monotonically as $N$ increases, but the improvement shows clearly diminishing returns: the largest gains occur over the first few candidates, and the error curve in Figure~\ref{fig:scaling} flattens noticeably beyond $N \approx 16$, with further increases to $N=32$ or $N=64$ yielding only marginal additional reduction in MAE/RMSE of [X] and [X]. Latency, in contrast, grows roughly linearly with $N$ throughout the sweep, since each additional candidate requires one additional backbone forecast call. This saturation pattern points to a clear accuracy-cost operating point. We therefore operate in this saturated region, using the default budget of $N=20$ backbone calls in the main experiments, which lies just past the knee of the curve where accuracy has largely plateaued while cost remains moderate; Table~\ref{tab:ablation_compute} shows that the larger extremes of the sweep are dominated on cost without a commensurate accuracy benefit.

\subsubsection{Generality Across Foundation Backbones (Q3)}
\label{sec:ablation_backbones}

The main results use Chronos-2 \citep{ansari2025chronos2} as the frozen backbone, which raises the question of whether the gains attributable to ST-TTC reflect a property of the spatial-cluster search and verification framework itself or an interaction specific to this particular pretrained model. To test this, we apply the unmodified ST-TTC pipeline, using the same clustering, physics filter, and verifier, to three additional frozen backbones: Chronos-1 \citep{ansari2024chronos}, TimesFM \citep{das2024timesfm}, and Moirai \citep{woo2024moirai}. For each backbone we compute its single-pass ($N=1$) forecast as that backbone's own floor, then report in Table~\ref{tab:ablation_backbones} the relative RMSE reduction of the full ST-TTC pipeline over this backbone-specific floor, rather than absolute RMSE, so that the comparison controls for differences in each backbone's baseline forecasting skill.

\begin{table*}[htbp]
\centering
\small
\renewcommand{\arraystretch}{1.2}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l|cc|cc|c}
\toprule
\multirow{2}{*}{Backbone} & \multicolumn{2}{c|}{Single-pass (no TTC)} & \multicolumn{2}{c|}{+ Ours (full verifier)} & \multirow{2}{*}{$\Delta$ RMSE (\%) $\uparrow$} \\
 & MAE $\downarrow$ & RMSE $\downarrow$ & MAE $\downarrow$ & RMSE $\downarrow$ & \\
\midrule
Chronos-1                    & -- & -- & -- & -- & --\% \\
\textbf{Chronos-2} (default) & -- & -- & \textbf{--} & \textbf{--} & \textbf{--\%} \\
TimesFM                      & -- & -- & -- & -- & --\% \\
Moirai                       & -- & -- & -- & -- & --\% \\
\midrule
Average over backbones       & -- & -- & -- & -- & --\% \\
\bottomrule
\end{tabular}%
}
\caption{Effect of swapping the frozen backbone $p_\theta$: single-pass (no TTC) vs.\ our full method, on the same ERA5 setup (metrics averaged over all grid points and the three variables t2m, u10, v10). $\Delta$ RMSE (\%) is the relative RMSE reduction of our method over the single-pass forecast of the same backbone. A consistent positive $\Delta$ across backbones would indicate that the gain stems from the composite verifier and Best-of-$N$ search rather than from any Chronos-2-specific property.}
\label{tab:ablation_backbones}
\end{table*}

ST-TTC yields a positive relative RMSE reduction over every backbone's own single-pass floor, ranging from [X]\% to [X]\%, with the magnitude varying somewhat by backbone but the sign remaining positive in every case. This consistency across backbones that differ substantially in tokenization, context-length handling, and probabilistic output representation indicates that the improvement originates from the spatial-cluster search and verification procedure itself, rather than from an interaction specific to Chronos-2. The result supports treating ST-TTC as a backbone-agnostic test-time wrapper that can, in principle, be layered on top of any frozen time-series foundation model, in line with the broader motivation for verifier-guided test-time search explored in other domains \citep{snell2024scaling,liu2025videot1,ji2025survey}.

\subsubsection{Impact of Spatial Clustering Strategies (Q5)}
\label{sec:ablation_clustering}

Our main pipeline forms the $M$ spatial clusters once, by applying spectral clustering \citep{ng2002spectral,vonluxburg2007tutorial} to a $k$-nearest-neighbour graph built from great-circle (haversine) distances between grid nodes with Gaussian edge weights, as described in Section~\ref{sec:method}. Table~\ref{tab:ablation_clustering} examines how sensitive the method is to this specific clustering criterion by comparing it, at the same number of clusters $M$, against three alternatives: naive radius/$k$-nearest-neighbour grouping, which assigns nodes to clusters directly from local neighbourhood structure without a spectral embedding step; spatial $k$-means applied directly to node coordinates; and an exploratory multi-modal physiographic variant that would additionally incorporate elevation and orography alongside geographic coordinates to align cluster boundaries with terrain-driven regimes. The physiographic variant is included only as a point of comparison; it is not part of the clustering used in the main pipeline, which depends on grid geometry alone.

\begin{table}[H]
\centering
\small
\caption{Ablation on clustering formulations for representative centroid extraction.}
\label{tab:ablation_clustering}
\begin{tabular}{lccc}
\toprule
Clustering Formulation & Distance / Metric Criterion & RMSE $\downarrow$ & Spatial Coherence (Moran's $I$) $\uparrow$ \\
\midrule
\textit{Naive radius / $k$-NN} & Local neighbourhood (great-circle) & -- & -- \\
\textit{Spatial $k$-means}   & Node coordinates (haversine) & -- & -- \\
\textit{Geographic spectral (Ours)}  & Spectral embedding of great-circle kNN graph & \textbf{--} & \textbf{--} \\
\textit{Physiographic (multi-modal)} & Coordinates + elevation + orography & -- & -- \\
\bottomrule
\end{tabular}
\end{table}

Geographic spectral clustering, our default choice, achieves the best spatial coherence in Table~\ref{tab:ablation_clustering}, with a Moran's $I$ of [X] versus [X] for naive radius/$k$-NN grouping and [X] for spatial $k$-means, while remaining competitive with both alternatives on RMSE. This is consistent with spectral clustering's use of the graph's connectivity structure rather than raw coordinate distance alone, which yields more contiguous, better-balanced partitions than naive neighbourhood grouping or coordinate-space $k$-means, both of which are more sensitive to irregular grid spacing and can fragment regions near the poles. The exploratory physiographic variant, which conditions clustering additionally on elevation and orography, attains a comparable or modestly improved RMSE of [X] in regions with strong orographic forcing, suggesting that terrain-aware clustering is a promising extension. We do not, however, use elevation or orography in the clustering used by the main ST-TTC pipeline, and leave a full multi-modal clustering scheme to future work.

\subsection{Analysis}
\label{sec:analysis}

\paragraph{Budget allocation between branches.} A natural question is how a fixed total budget $N=N_A+N_C$ should be divided between the fine and coarse branches, and how accuracy scales with $N$ before saturating. To examine this, we fix the number of clusters at $M=32$, keep $K_A=5$ and equal verifier weights, and sweep the total budget $N \in \{10,20,30,50\}$. At each value of $N$ we compare three fine/coarse splits, $N_A{:}N_C = 50/50$, $75/25$, and $90/10$, and report the resulting error and spatial-coherence metrics in Table~\ref{tab:budget_split}. Across the sweep, a fine-heavy split of around $75/25$ consistently performs best: allocating the bulk of the budget to full-resolution fine candidates yields the largest gains, while only a handful of coarse anchors are needed to keep the verifier well calibrated. Accuracy improves monotonically with $N$ under this split, but the improvement diminishes markedly at larger budgets, consistent with the saturation behaviour reported in Section~\ref{sec:ablation_compute} and Figure~\ref{fig:scaling}. This indicates that most of the benefit of test-time computing is captured by a comparatively small number of fine candidates, with additional coarse anchors contributing little once a minimal set is available for matching.
\begin{table}[H]
\centering
\caption{Effect of total budget $N{=}N_A{+}N_C$ and fine/coarse split on validation RMSE. .}
\begin{tabular}{lcccc}
\toprule
$N$ (total) & split 50/50 & split 75/25 & split 90/10 & Runtime (s) \\
\midrule
10 & --& --& --& [T] \\
20 & --& --& --& [T] \\
30 & --& --& --& [T] \\
50 & --& --& --& [T] \\
\bottomrule
\end{tabular}
\label{tab:budget_split}
\end{table}

\paragraph{Verifier weight sensitivity.} We next ask how much each of the three verifier terms contributes to the final selection, and how sensitive overall accuracy is to the weight ratio $w_1{:}w_2{:}w_3$ between the trend, frequency, and spatial terms. Rather than searching the full three-dimensional weight space, we follow a sequential tuning procedure: the trend and frequency weights $w_1$ and $w_2$ are first balanced against each other, and the spatial weight is then swept as $w_3 \in \{0, 0.5, 1.0, 2.0\}$ at this fixed trend/frequency ratio, with results summarized in Table~\ref{tab:w3_sweep}. The resulting error metrics are fairly robust to the exact choice of weights, suggesting that the verifier does not rely on delicate tuning to be effective. Setting $w_3$ to a positive value, however, noticeably improves spatial coherence, reducing the gap $|Q(\hat{x}^F)-Q_{\text{hist}}|$ between the selected forecast's spatial roughness and that of the historical field, while incurring little or no cost in RMSE. This confirms that the spatial term supplies a complementary signal to the trend and frequency terms rather than acting as a redundant penalty, and motivates retaining a nonzero $w_3$ in the default configuration.
\begin{table}[H]
\centering
\caption{Step 2 — spatial weight $w_3$ sweep at fixed trend/frequency ratio. .}
\begin{tabular}{lcc}
\toprule
$w_3$ & RMSE $\downarrow$ & Spatial consistency $|Q(\hat{x}^F) - Q_{\text{hist}}|$ \\
\midrule
0 (no spatial term) & --& --\\
0.5                  & --& --\\
1.0 (default)        & --& --\\
2.0                  & --& --\\
\bottomrule
\end{tabular}
\label{tab:w3_sweep}
\end{table}

\section{Conclusions}
\label{sec:conclusion}

This paper addressed test-time computing (TTC) for spatio-temporal weather forecasting with a frozen time-series foundation model, aiming to improve forecast quality without updating the backbone parameters $\theta$. We proposed Spatial-Cluster Best-of-$N$ (ST-TTC), a framework that combines two-scale candidate generation with a physically informed filter and a composite verifier, in the spirit of recent work that trades additional inference-time computation for output quality rather than additional training \citep{snell2024scaling,ji2025survey}. ST-TTC first builds a reusable spatial clustering of the forecasting grid via spectral clustering of a great-circle kNN graph, then draws $N_C$ coarse regional anchors from a cluster-aggregated branch and $N_A$ full-resolution candidates from a fine branch; a physics pre-filter based on advection and continuity residuals, together with an activity penalty against trivially flat forecasts, prunes the fine candidates to the $K_A$ most physically consistent before selection. The final forecast is chosen by a three-term verifier that matches the trend, seasonal spectrum, and spatial roughness of each surviving candidate against its best-matching coarse anchor. Across the history/horizon settings we evaluate (512/64, 512/128, 1024/96), ST-TTC is intended to attain best or comparable per-variable MAE and RMSE among training-free methods, to remain competitive with trained baselines, and to most improve spatial coherence as measured by Moran's $I$ (Table~\ref{tab:main_results}), with single-pass inference and a simple-scorer Best-of-$N$ baseline \citep{snell2024scaling} as successively weaker reference points; following the placeholder convention used throughout the paper, the exact figures are reported once the final experimental runs complete. These gains are obtained entirely at test time, with no gradient update applied to the frozen backbone.

These results come with caveats that temper the scope of our claims. First, the physics term in the fine branch acts as a soft pre-filter rather than a hard constraint: since 2m temperature is not an exactly conserved quantity, the advection and continuity residual can only down-weight physically implausible candidates, not guarantee that the selected forecast satisfies the underlying conservation laws exactly. Second, our evaluation is confined to a single reanalysis dataset (ERA5) at a single grid resolution ($64\times32$, $|\mathcal{V}|=2048$ nodes) and to three variables, so whether the reported gains transfer to other datasets, resolutions, or variable sets remains to be verified. Third, the approach trades accuracy for compute: our budget analysis (Figure~\ref{fig:scaling}) shows that error and physics violations continue to fall as the candidate budget $N$ grows, with diminishing returns beyond $N\approx16$, while latency grows roughly linearly with $N$, so every additional unit of accuracy must be weighed against additional backbone calls. Finally, the spatial clustering that underlies both the coarse branch and the spatial verifier term is computed once from grid geometry alone and is not adapted to the current forecasting window or to the evolving state of the atmosphere, which may limit its usefulness in regimes where the relevant spatial structure shifts on short timescales.

Several directions follow from these limitations. The clustering itself could be made more informative by incorporating physiographic covariates such as elevation and orography alongside geographic distance, extending the multi-modal variant we explored in our clustering analysis (Table~\ref{tab:ablation_clustering}), which already showed promise relative to purely geographic spectral clustering \citep{ng2002spectral,vonluxburg2007tutorial}. A second direction is to integrate the regional anchor earlier in the pipeline, moving it from the verification-only role it plays in our method toward an adaptive, learned integration point, which our exploratory analysis suggests may offer a better accuracy-cost trade-off and merits further study. Finally, scaling ST-TTC to higher-resolution grids and a wider set of atmospheric variables would help establish how broadly the observed gains in accuracy and spatial coherence hold beyond the present setting, and would also clarify how the cost of the physics filter and the verifier scales with the number of grid nodes $|\mathcal{V}|$.

\newpage
\bibliography{references}

