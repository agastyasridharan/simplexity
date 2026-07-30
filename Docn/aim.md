# Project aim — discovering the MSP in transformer latents

*Authoritative theoretical frame for this project, written by Alec Boyd (2026-04-28). The handbook pages ([msp.md](handbook/msp.md), [transformers.md](handbook/transformers.md), [density_estimation.md](handbook/density_estimation.md), [papers.md](handbook/papers.md)) elaborate specific corners; this page states the aim itself.*

---

## 1. The mixed state / belief state of an HMM

The fundamental challenge of matching LLMs to computational mechanics is rooted in the HMM $\hat{T}^{(x)}$ and its mixed states

$$
| \rho(x_{0:L}) \rangle \;\equiv\; \frac{ \prod_{t=0}^{L-1} \hat{T}^{(x_t)}\, | \rho_{R_0} \rangle }{ \langle 1 |\, \prod_{t=0}^{L-1} \hat{T}^{(x_t)}\, | \rho_{R_0} \rangle } ,
$$

where

$$
| \rho_{R_0} \rangle \;\equiv\; \sum_{r} \Pr(R_0 = r)\, | r \rangle
$$

is the initial density vector over memory states and

$$
\langle 1 | \;\equiv\; \sum_{r}\, \langle r |
$$

is the transpose vector of all $1$'s, and $x_{0:L} \equiv x_0 x_1 \cdots x_{L-1}$.

> [transcription note: in the source the right-hand side of the $\langle 1|$ definition was $\sum_r \langle 1|$, which I read as a typo for $\sum_r \langle r|$ since the latter gives the row of ones. Confirm and I'll edit.]

This is an inherently predictive state in that it is a function of the past. It also has an explicit prediction of the next-token probability

$$
\Pr(X_L = x_L \mid X_{0:L} = x_{0:L}) \;=\; \langle 1 |\, \hat{T}^{(x_L)}\, | \rho(x_{0:L}) \rangle .
$$

## 2. The transformer latent

By comparison, a transformer (or neural network) LLM also has a latent state $S_L$ that is a function of the past,

$$
S_L \;=\; f(x_{0:L}),
$$

and it estimates the next-token probability via this latent state,

$$
\Pr\!\bigl(X_L^{\text{est}} = x_L \,\big|\, X_{0:L} = x_{0:L}\bigr) \;\equiv\; \Pr\!\bigl(X_L = x_L \,\big|\, S_L = f(x_{0:L})\bigr).
$$

The activations of the residual stream are often considered to be these latents. The context length $l$ tells us how far backward this function looks,

$$
f(x_{0:l}\, x_{l:L}) \;=\; f(x'_{0:l}\, x_{l:L}).
$$

## 3. The question — discovering the MSP

There is striking similarity between the $\varepsilon$-machine-adjacent MSP and the transformer LLM. They are both predictive models, meaning that they use functions of the past to capture the information shared between past and future. When

$$
\Pr\!\bigl(X_L = x_L \,\big|\, S_L = f(x_{0:L})\bigr) \;\approx\; \Pr\!\bigl(X_L = x_L \,\big|\, X_{0:L} = x_{0:L}\bigr),
$$

the transformer is a good model of the process. In this case we should expect the LLM to *at least* store copies of the causal states of the $\varepsilon$-machine, which can be mapped to MSP states (up to multiplicity).

In addition, the two fields are connected by **linearity**. The mixed states are embedded in a linear simplex, and the residual stream $S_t$ is in a continuous linear space. The hypothesis motivating a lot of recent work is that **the MSP is linearly embedded in the residual-stream latents of the transformer**. What this means is that, for an LLM trained to do next-token prediction on data taken from $\hat{T}^{(x)}$, the latent states $S_t$ for a particular data stream $x_{0:L}$ should map linearly to the belief states of $\hat{T}^{(x)}$. In other words, there exists a linear map $\hat{M}$ such that

$$
\hat{M}\, f(x_{0:L}) \;\approx\; | \rho(x_{0:L}) \rangle \qquad \text{for all } x_{0:L}.
$$

This is what is being tested when we linearly fit the latents to the MSP.

---

## 4. Model performance — the average log-loss rate

We want to evaluate how close our models come to the true process — quantitatively, in addition to the geometric question of §3. The natural quantity is the **average log-loss rate**

$$
\langle \ell \rangle \;\equiv\; \lim_{L \to \infty} \frac{ -\sum_{x_{0:L}} \Pr(X_{0:L} = x_{0:L})\, \ln \Pr(X^{\text{est}}_{0:L} = x_{0:L}) }{L} .
$$

This is closely aligned with the entropy rate, except it has an additional term that is the per-symbol relative entropy:

$$
\langle \ell \rangle \;=\; \lim_{L \to \infty} \frac{ -\sum_{x_{0:L}} \Pr(X_{0:L} = x_{0:L})\, \ln \frac{ \Pr(X^{\text{est}}_{0:L} = x_{0:L}) }{ \Pr(X_{0:L} = x_{0:L}) } }{L} \;+\; \lim_{L \to \infty} \frac{ -\sum_{x_{0:L}} \Pr(X_{0:L} = x_{0:L})\, \ln \Pr(X_{0:L} = x_{0:L}) }{L}
$$

$$
=\; \lim_{L \to \infty} \frac{ D_{\text{KL}}\!\bigl( X_{0:L} \,\big\|\, X^{\text{est}}_{0:L} \bigr) }{L} \;+\; \lim_{L \to \infty} \frac{ H[X_{0:L}] }{L} \;=\; d_\mu + h_\mu ,
$$

where $h_\mu \equiv \lim_{L \to \infty} H[X_{0:L}] / L$ is the **entropy rate** of $X$ (a process-only quantity, model-independent floor on $\langle \ell \rangle$), and $d_\mu \equiv \lim_{L \to \infty} D_{\text{KL}}\!\bigl( X_{0:L} \,\big\|\, X^{\text{est}}_{0:L} \bigr) / L$ is the **KL-divergence rate** between the true process and the estimated process.

$d_\mu$ is the most important measure of model fidelity: it captures the portion of the loss that depends on the model and is zero iff the model perfectly captures the process. In practice this is hard to estimate directly because the defining sum runs over exponentially many length-$L$ trajectories.

## 5. Estimating model performance via Barron's generalised SMB theorem

Barron's generalised Shannon–McMillan–Breiman theorem tells us that the KL-divergence rate can be estimated by evaluating a single long sequence's log-probability ratio:

$$
d_\mu \;=\; \lim_{L \to \infty} \frac{1}{L} \ln \frac{ \Pr(X_{0:L} = x_{0:L}) }{ \Pr(X^{\text{est}}_{0:L} = x_{0:L}) } .
$$

> [transcription note: the source had $\lim_{L \to 0} \ln\ln$; corrected to $\lim_{L \to \infty} \tfrac{1}{L} \ln$ here and below — the next-line decomposition makes the intended limit unambiguous.]

The true probability decomposes as

$$
\Pr(X_{0:L} = x_{0:L}) \;=\; \prod_{t=0}^{L-1} \Pr(X_t = x_t \mid X_{0:t} = x_{0:t}) \;=\; \prod_{t=0}^{L-1} \langle 1 |\, \hat{T}^{(x_t)}\, | \rho(x_{0:t}) \rangle ,
$$

i.e. each per-step factor is the next-token probability *conditioned on the mixed state at that step*. The estimated probability decomposes analogously through the latent state:

$$
\Pr(X^{\text{est}}_{0:L} = x_{0:L}) \;=\; \prod_{t=0}^{L-1} \Pr(X^{\text{est}}_t = x_t \mid S_t = f(x_{0:t})) .
$$

A new expression follows for the KL-divergence rate:

$$
d_\mu \;=\; \lim_{L \to \infty} \frac{1}{L} \ln \frac{ \prod_{t=0}^{L-1} \Pr(X_t = x_t \mid M_t = | \rho(x_{0:t}) \rangle) }{ \prod_{t=0}^{L-1} \Pr(X^{\text{est}}_t = x_t \mid S_t = f(x_{0:t})) } \;=\; \lim_{L \to \infty} \frac{1}{L} \sum_{t=0}^{L-1} \ln \frac{ \Pr(X_t = x_t \mid M_t = | \rho(x_{0:t}) \rangle) }{ \Pr(X^{\text{est}}_t = x_t \mid S_t = f(x_{0:t})) } .
$$

So we merely need to take the mixed state and the latent state of the neural network for every historical context and compare their next-token predictions, summing the log-ratios for a single very long sequence $x_{0:L}$ sampled from the true process generated by $\hat{T}^{(x)}$.

## 5b. A closed-form, strictly non-negative estimator of $d_\mu$

The single-sample Barron–SMB estimator of §5 is unbiased but has a per-position variance equal to $\operatorname{Var}_{x_t \sim P_t}[\ln P_t(x_t)/Q_t(x_t)]$, which can be substantial relative to the population value. Worse, when reported as $\hat d_\mu = \widehat{\langle \ell \rangle} - \hat h_\mu$, the Monte-Carlo noise in $\hat h_\mu$ adds further uncertainty, and the resulting estimate can become spuriously *negative* — even though $d_\mu$ is provably non-negative.

A cleaner estimator integrates out $x_t$ analytically at every position, replacing the single-sample log-ratio with the per-position KL divergence. Define

$$
\hat d^{\,\mathrm{KL}}_\mu \;\equiv\; \frac{1}{L} \sum_{t=0}^{L-1} D_\text{KL}\!\Bigl(\, \Pr(X_t \mid M_t = m_t) \,\Big\|\, \Pr(X^\text{est}_t \mid S_t = s_t) \,\Bigr) ,
$$

where the per-position KL is given in closed form by

$$
D_\text{KL}(P \,\|\, Q) \;=\; \sum_{x \in \mathcal{X}} P(x)\, \ln \frac{P(x)}{Q(x)} \;\geq\; 0 ,
$$

with equality iff $P = Q$. So $\hat d^{\,\mathrm{KL}}_\mu \geq 0$ pointwise — at every step $t$ and at every snapshot of training — without any clipping or subtraction.

**Same expectation, lower variance.** Taking the expectation of the Barron–SMB summand under $x_t \sim P_t$:

$$
\mathbb{E}_{x_t \sim P_t}\!\left[\, \ln \frac{P_t(x_t)}{Q_t(x_t)} \,\right] \;=\; D_\text{KL}(P_t \,\|\, Q_t) ,
$$

so $\hat d^{\,\mathrm{KL}}_\mu$ and the Barron–SMB $\hat d_\mu$ target the same population quantity. The KL form computes the inner expectation analytically per position, eliminating the sampling variance and removing the need for an MC estimate of $h_\mu$.

**Operational ingredients** (all already produced inside the test-evaluation loop):
- $P_t(x) = \langle 1 |\, T^{(x)}\, m_t = (m_t\, E)_x$, where $E_{ij} = p(x_j \mid s_i)$ is the emission matrix.
- $Q_t(x) = \operatorname{softmax}(\text{model}_\theta(x_{0:t}))_x$.

This is the preferred estimator for the prediction-fit goodness criterion in §7.

## 6. Training and testing data

We train on a sequence $x^{\text{train}}_{0:L^{\text{train}}}$ and evaluate on a separate test sequence $x_{0:L}$. For now, we use a **single training and a single test sequence**, but the framework extends straightforwardly to multiple sequences / batches.

## 7. Summary — two parallel evaluations

From an HMM $\hat{T}^{(x)}$ we have both an MSP and a neural network trained on data sampled from it. On the test sequence we evaluate the model in **two complementary ways**:

For each step $t$ of the test sequence the NN realises the latent $s_t = f(x_{0:t})$ and the MSP realises the mixed state $m_t = | \rho(x_{0:t}) \rangle$. (Mnemonic: $m$ for *mixed state*, $s$ for *transformer state*. The lowercase $s_t$ is local to this section; §2 uses uppercase $S_L$ for the same NN-latent quantity.) Both evaluations sum a per-step quantity over $t = 0, \ldots, L-1$:

1. **Prediction-fit goodness** — closed-form per-position KL (§5b), preferred for stability and non-negativity:

$$
\hat d^{\,\mathrm{KL}}_\mu \;=\; \frac{1}{L} \sum_{t=0}^{L-1} D_\text{KL}\!\bigl( \Pr(X_t \mid M_t = m_t) \,\|\, \Pr(X^\text{est}_t \mid S_t = s_t) \bigr) \;\geq\; 0 .
$$

Goes to $0$ as the model perfectly captures the process. The Barron-SMB single-sample form
$\hat d_\mu = \tfrac{1}{L} \sum_t \ln P_t(x_t)/Q_t(x_t)$ targets the same population quantity but adds variance from sampling $x_t$ and from estimating $h_\mu$; prefer $\hat d^{\,\mathrm{KL}}_\mu$ unless a single-sample form is specifically required.

2. **Geometric-fit goodness** — residual of the linear probe $\hat{M}$:

$$
\frac{1}{L} \sum_{t=0}^{L-1} \bigl\| \hat{M}\, s_t \,-\, m_t \bigr\|^2 .
$$

Goes to $0$ when the linear hypothesis of §3 holds exactly. Tells us whether the model has *internally* organised the latent space into the MSP geometry, regardless of whether its outputs are accurate.

The two evaluations are independent in principle — a model could fit well predictively but lack the geometric structure (e.g., uses an entirely different sufficient statistic), or have the geometry but be miscalibrated on outputs (e.g., poorly trained final-layer unembedding). For Mess3, where the emission matrix is invertible, predictive perfection implies geometric perfection up to the linear map $\hat{M}$; for cryptic processes (RRXOR), the two come apart.

---

## 8. Notation reconciliation with the handbook

The aim above and the existing handbook differ slightly in symbols. Both are correct; this table makes the mapping explicit so neither version is ambiguous.

| Aim (this page) | Handbook ([notation.md](handbook/notation.md)) | Meaning |
|---|---|---|
| $\hat{T}^{(x)}$ | $T^{(x)}$ | Substochastic transition tensor |
| $\lvert \rho(x_{0:L}) \rangle$ (column) | $\eta_t$ (row) | Belief / mixed state on $\Delta^{\lvert\mathcal{S}\rvert - 1}$ |
| $\lvert \rho_{R_0} \rangle$ | $\eta_\varnothing$ (typically $\pi$) | Initial belief over memory states |
| $\langle 1 \rvert$ | $\mathbf{1}^\top$ | Row of ones |
| $R_t$ (memory state) | $S_t$ (causal state) | Hidden state of the HMM |
| $r$ | $s_i$ | A memory-state value |
| $S_L$ (transformer latent) | $a^{(N)}_L \in \mathbb{R}^{d_\text{resid}}$ | Final-layer residual stream at position $L$ |
| $f(x_{0:L})$ | $a(\text{ctx})$ | Transformer's map from context to latent |
| $\hat{M}$ | $W$ (probe weight matrix) | Linear map: residual → belief |
| $l$ (context length) | $L$ (context length) | How far back $f$ depends on input |

**Style going forward.** In project-facing project documents (this page, future paper-style write-ups), prefer the aim notation: $\hat{T}^{(x)}$, $\lvert \rho \rangle$, $\hat{M}$, $R_t$. In the code-facing handbook and source ([src/](src/)), keep the row-vector / zero-based numerical conventions of `notation.md` because they map directly to NumPy / PyTorch.

There is **one collision worth flagging explicitly:** the symbol $S$ is used in opposite senses in the two notations.

- In the aim, $S_L$ = *transformer latent* (residual stream).
- In the handbook (and Crutchfield/Shalizi convention), $S$ = *causal state of the ε-machine*, $S_L$ = causal state at time $L$.

When ambiguous, write $a_L$ for the residual-stream activation (handbook convention) or be explicit ("the transformer latent $S_L$" vs "the causal state $S_L$ of the generator").

---

## 9. What this means for our code

Two parallel evaluations follow from §3 and §7:

### 9a. Geometric-fit evaluation (currently implemented)

The aim equation $\hat{M}\, f(x_{0:L}) \approx \lvert \rho(x_{0:L}) \rangle$ is exactly what `fit_linear_probe(...)` in [src/probe.py](src/probe.py) is solving:

$$
\hat{M},\ c \;=\; \arg\min_{M, c}\ \sum_{x_{0:L}}\, \bigl\lVert \lvert \rho(x_{0:L}) \rangle - M\, f(x_{0:L}) - c \bigr\rVert^2 \;+\; \lambda \lVert M \rVert_F^2 .
$$

(Bias term $c$ is included for numerical convenience; it can be absorbed into $\hat M$ by appending a constant feature to $f$.) Reported via $R^2$ on a held-out test sequence and by the visual match between the two panels of `simplex.png`. Our 5k-step AdamW run sits at $R^2 = 0.9905$ on Mess3.

### 9b. Prediction-fit evaluation: closed-form per-position KL (§5b)

At every position $t$ of a long test sequence, evaluate

- $P_t(x) \;=\; \langle 1 |\, \hat{T}^{(x)}\, m_t = (m_t\, E)_x$ — analytical next-token distribution from the HMM along the trajectory.
- $Q_t(x) \;=\; \operatorname{softmax}\bigl(\,\text{model}_\theta(x_{0:t})\,\bigr)_x$ — model's next-token distribution.

Then average the per-position KL:

$$
\hat d^{\,\mathrm{KL}}_\mu \;=\; \frac{1}{L} \sum_{t=0}^{L-1} D_\text{KL}(P_t \,\|\, Q_t) \;\geq\; 0 .
$$

Implemented in `main_simplexity.py` as `evaluate_kl(...)` and tracked alongside test cross-entropy at every test snapshot. Saved to `probe.npz` as `test_kls_steps` / `test_kls_values`; plotted on log y-axis in `relative_entropy.png`.

### 9c. Together

The two evaluations isolate complementary failure modes:

- High geometric $R^2$ + small $\hat d_\mu$ → model has the right structure *and* uses it correctly. ✓
- High geometric $R^2$ + large $\hat d_\mu$ → model encodes the belief state but its readout is miscalibrated (probably a final-layer training issue).
- Low geometric $R^2$ + small $\hat d_\mu$ → model is predictive but uses a non-MSP sufficient statistic. (Possible for invertible-emission processes like Mess3; requires probing a different basis.)
- Low geometric $R^2$ + large $\hat d_\mu$ → model is undertrained or undersized.

Mess3 currently shows $R^2 = 0.99$ and a cross-entropy at the floor ($\hat d_\mu$ implicitly small). Both axes will become more informative for RRXOR.

## 10. What we are *not* yet testing

The linearity hypothesis as stated is the *minimum* claim. Each of the following is a follow-up direction; none is strictly implied by the aim equation:

1. **Causality.** Do interventions on the MSP-direction in the residual stream actually change downstream predictions in the way the MSP would predict? (Currently we have *correlational decoding* only — Park et al. 2024's causal inner product gives the right framework.)
2. **Multi-layer / spread-out geometry.** Does the linear embedding live in the final layer, or distribute across layers? Shai et al. demonstrated the latter for RRXOR. Requires per-layer probing — `simplexity/analysis/layerwise_analysis.py` is the reference.
3. **Factored embedding.** When the generator factorises into independent components, do the factors embed in *orthogonal* subspaces of the residual stream? (Shai, Amdahl-Culleton et al. 2026.)
4. **Transducer extension.** When $\hat{T}^{(x)}$ is generalised to $\hat{T}^{(x \mid y)}$ — i.e., an ε-transducer (Barnett & Crutchfield 2015) — does the linearity claim still hold for the joint input–output belief state? **This is the core question motivating the parent folder "Transducer Decomposition."** *[Updated 2026-07 — this question is refined and operationalised in §12: the answer depends on whether the intermediate token is observed.]*
5. **Energetics.** Does the thermodynamic cost of the engine implementing $\hat{M}\, f$ obey the bounds in [Boyd, Crutchfield, Gu, Binder 2025](handbook/density_estimation.md#11-the-mle--thermodynamic-work-extraction-equivalence-boyds-framework)? This is the bridge from the MSP-recovery question to the user's energetics-of-predictive-intelligence agenda.

---

## 11. Where this aim sits in the broader project

| Page | Relationship to the aim |
|---|---|
| [wiki.md](wiki.md) | 103-entry annotated bibliography — the literature surrounding the aim |
| [reading_list.md](reading_list.md) | 10-paper reading spine if you only have time for the highest-impact references |
| [handbook/notation.md](handbook/notation.md) | Code-facing notation (row vectors, zero-indexed) reconciled in §8 above |
| [handbook/msp.md](handbook/msp.md) | Detailed construction of $\lvert \rho(x_{0:L}) \rangle$ |
| [handbook/processes.md](handbook/processes.md) | Explicit $\hat{T}^{(x)}$ tensors for Mess3, RRXOR, etc. |
| [handbook/transformers.md](handbook/transformers.md) | What the function $f$ looks like internally; QK/OV factorisation, residual stream as object |
| [handbook/density_estimation.md](handbook/density_estimation.md) | The fundamental theory of learning a sequence distribution; the entropy/KL decomposition of §4 here, the Boyd-2025 thermodynamic-overfitting framework as the bridge to energetics |
| [handbook/papers.md](handbook/papers.md) | Mathematical structures across the 10 anchor papers, organized thematically |
| [src/probe.py](src/probe.py) | The code that fits $\hat{M}$ and reports $R^2$ — the operational form of §9a; awaits the §9b $\hat d_\mu$ implementation |

---

## 12. Addendum (2026-07) — the observability refinement for transducers

*Added after the June-2026 meeting with Alec Boyd and the [`Coarse-Graining Roots`](Coarse-Graining%20Roots.pdf) note (§556). This section extends §10.4 only; §1–§11 above are unchanged. It states precisely what the "Transducer Decomposition" experiments test, and the one subtlety §10.4 did not yet capture: **the linearity answer depends on which symbols are observed.** The interactive companion is [`explorer.html`](explorer.html); the implementation is [`docs/transducer_decomposition_experiment.md`](../docs/transducer_decomposition_experiment.md).*

### 12.1 Setup and operators

An autonomous **input HMM** $\hat{T}^{(y)}$ on memory states $R$ emits an intermediate symbol $y$; a **transducer** $\hat{U}^{(x \mid y)}$ on memory states $S$ reads $y$ and emits an output $x$ (this $\hat{U}^{(x \mid y)}$ is exactly §10.4's $\hat{T}^{(x \mid y)}$). The joint hidden state is $(R, S)$, and the fully-observable joint operator is the tensor product

$$
\hat{V}^{(y,x)} \;=\; \hat{T}^{(y)} \otimes \hat{U}^{(x \mid y)},
\qquad
\Pr(y_{0:L}, x_{0:L}) \;=\; \langle 1 |\, \prod_{t=0}^{L-1} \hat{V}^{(y_t, x_t)}\, | \rho_0 \rangle .
$$

### 12.2 The observability axis (the refinement)

The linear-embedding claim of §3 has **three different predicted geometries**, depending on which symbols cross the visibility boundary:

| observed | operator | belief geometry | embedding dimension |
|---|---|---|---|
| both $y$ and $x$ | $\hat{V}^{(y,x)} = \hat{T}^{(y)} \otimes \hat{U}^{(x\mid y)}$ | **tensor product** $\lvert\rho_R\rangle \otimes \lvert\rho_S\rangle$ — *separable* | $(\lvert R\rvert - 1) + (\lvert S\rvert - 1)$ |
| only $x$ (hide intermediate $y$) | $\hat{W}^{(x)} = \sum_y \hat{T}^{(y)} \otimes \hat{U}^{(x\mid y)}$ | **entangled** — need not factor; fills the joint simplex | up to $\lvert R\rvert\lvert S\rvert - 1$ |
| only $y$ (hide output $x$) | $\sum_x \hat{V}^{(y,x)}$ | reduces to the **input HMM's MSP** | $\lvert R\rvert - 1$ |

The middle row is the crux: **coarse-graining the intermediate token entangles the two latent spaces.** Because $\hat{W}^{(x)}$ sums over the hidden $y$, the operators no longer separate as a tensor product, so the input and transducer beliefs cannot be read off independently — "complexification by coarse-graining" (Coarse-Graining Roots §556). Coarse-graining the *output* instead is a *simplification* (you recover the input HMM alone). Numerically confirmed: fully-observable beliefs are 100% rank-1 (separable); the coarse-grained beliefs are ~34% rank-1 for SNS→SNS and 0% for Fractal2→Fractal2.

### 12.3 Three probe targets and the erasure question

In every regime the belief lives on the same $(R,S)$ space, so we fit $\hat{M}$ (§9a) to three targets: the **joint** belief, the **input marginal** ($\to R$), and the **transducer marginal** ($\to S$). Under full observation all three are recovered; under coarse-graining the *joint* stays recoverable while the *marginals lag*. That lag is the operational form of the guiding question:

> **How much information does the transducer erase** about the input, once the intermediate token is hidden?

### 12.4 Reinterpretation of the earlier composite-Mess3 result

The prior hierarchical-HMM experiment hid its "driver" (the intermediate/selector variable), so it was already the coarse-grained $\hat{W}$ regime — *not* a clean input→output decomposition. Its high joint $R^2$ (0.98) alongside a lagging driver marginal ($R^2 \approx 0.94$; $0.56$ at layer 0) is therefore the **expected signature of entanglement**, not undertraining or a model failure. Reading that result through §12.2 is what resolves the confusion.

### 12.5 The two evaluations of §7, plus a separability diagnostic

Both §7 axes are computed in-notebook per regime: the **geometric fit** ($R^2$ of $\hat{M}$, §9a) and the **prediction fit** ($\hat{d}^{\mathrm{KL}}_\mu$, the closed-form per-position KL of §5b/§9b, together with the $\langle\ell\rangle - h_\mu$ entropy-rate gap). A third, purely analytical **separability fraction** (share of belief states that factor as a tensor product) labels the regime as separable vs entangled *before* any training. First confirmed instance — Stage A, IID $\to$ SNS, fully observed: $R^2 = 0.994$, $\hat{d}_\mu = 0.0006$ nats, loss at the entropy floor, all controls clean.

### 12.6 Feedback / partially observable (the §10.1 direction)

When the loop is closed — agent $\hat{T}$'s output feeds $\hat{U}$ and vice-versa — each agent carries a **modified-operator MSP** over its own $(S, X)$ (the fed-back output as an extra coordinate), and by Barnett–Crutchfield Corollary 1 the two beliefs evolve independently. This realises the partial-observability direction flagged in §10 and is implemented as a separate notebook ([`transducer_feedback_experiment.ipynb`](../notebooks/colab/transducer_feedback_experiment.ipynb)).

### 12.7 Where this lives in code

| Object | Location |
|---|---|
| Operators $\hat V$, $\hat W$, input-only + kernels (SNS, Fractal2, Mess3) | [`simplexity/generative_processes/transition_matrices.py`](../simplexity/generative_processes/transition_matrices.py) |
| Feed-forward + coarse-graining experiment (the `OBSERVE` switch) | [`notebooks/colab/transducer_decomposition_experiment.ipynb`](../notebooks/colab/transducer_decomposition_experiment.ipynb) |
| Feedback / partially-observable experiment | [`notebooks/colab/transducer_feedback_experiment.ipynb`](../notebooks/colab/transducer_feedback_experiment.ipynb) |
| Design writeup + metrics | [`docs/transducer_decomposition_experiment.md`](../docs/transducer_decomposition_experiment.md) |
