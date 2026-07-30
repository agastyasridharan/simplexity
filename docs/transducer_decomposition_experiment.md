# Transducer Decomposition Experiment

This experiment implements the **input→output transducer decomposition** described in
[`Docn/aim.md`](../Docn/aim.md) §10.4 and specified concretely by the feed-forward mode of
[`Docn/explorer.html`](../Docn/explorer.html). It is the genuine ε-transducer counterpart to the
hierarchical-HMM study in [`composite_mess3_experiment.md`](composite_mess3_experiment.md).

The question: when a transformer is trained on the **joint** input/output stream of a
transducer, does its residual stream linearly encode **both** the input-HMM belief state and the
output-transducer belief state?

## The process

- An autonomous **input HMM** `T^(y)` (states `R`) emits symbols `y`.
- A **transducer** reads `y` and emits `x`; its kernel is `T^(x|y)` (states `S`).
- The observable is the **composite token** `(y, x)`, encoded as `k = y * n_x + x`.
- The joint hidden state is `(R, S)` and the joint symbol-labeled kernel is

  ```
  T_joint^(y,x) = kron( T_input^(y), T_transducer^(x|y) )
  ```

  Summed over the composite token this is row-stochastic, so the joint is a valid HMM and the
  transformer can be trained on next-(composite-)token prediction exactly as in the flat case.

The three belief targets are marginals of the joint belief `eta_t` over `R x S`:

| target | definition | Stage A (IID input) |
|---|---|---|
| input belief | marginal over `R` | trivial (single point) |
| transducer belief | marginal over `S` | **countable SNS MSP on the 1-simplex** |
| joint belief | full `(R, S)` | equals transducer belief |

## Observation regimes (the coarse-graining axis — Coarse-Graining Roots §556)

The central control is **which tokens the transformer is trained on**, set by `OBSERVE`. All
three regimes share the same joint `(R, S)` state space but use a different symbol-labeled
operator, and hence predict a different belief geometry:

| `OBSERVE` | trained tokens | operator | predicted belief | separability |
|---|---|---|---|---|
| `joint` | `(y, x)` composite | `V^(y,x) = kron(T^(y), U^(x|y))` | tensor product (input ⊗ transducer) | ≈ 1 (separable) |
| `output_only` | `x` only (hide intermediate `y`) | `W^(x) = Σ_y kron(T^(y), U^(x|y))` | **entangled cascade** (fills `Δ^{RS−1}`) | < 1 |
| `input_only` | `y` only (hide output `x`) | `Σ_x kron(T^(y), U^(x|y))` | input HMM's MSP | 1 |

Coarse-graining the intermediate token **entangles** the input and transducer latent spaces:
their beliefs no longer factor, so the input/transducer marginals become only partially
recoverable. This is exactly the regime the old `composite_mess3` experiment was in (hidden
driver), which is why its driver R² lagged — the *expected signature of coarse-graining*, not a
model failure. Verified numerically: fully-observable beliefs are 100% rank-1 (separable);
coarse-grained beliefs are ~34% (sns→sns) or 0% (fractal2→fractal2).

**Process choices** (`INPUT_MODE` / `TRANSDUCER_MODE`): `iid`, `sns`, `fractal2`, `mess3`.
Stage A default is `OBSERVE='joint'`, `iid → sns` (clean separable baseline, countable SNS arc).
Flip `OBSERVE='output_only'` on the same process to measure **how much the transducer erases**.

**Stage C — feedback (partially observable):** implemented in
[`notebooks/colab/transducer_feedback_experiment.ipynb`](../notebooks/colab/transducer_feedback_experiment.ipynb).
Two agents T (states R) and U (states S) exchange symbols in a closed loop (T reads `x` emits `y`;
U reads `y` emits `x`). The transformer trains on the observed exchange token `(x, y)`; probes
target agent T's belief over `R`, agent U's **modified-operator belief** over `(S, X)` (the
partially-observable MSP), and their joint. Belief updates are a verbatim port of
`explorer.html` `sampleFeedback` and are deterministically recomputable from the observed stream
(verified: vectorized update matches the scalar reference exactly).

## Metrics reported

- **Geometric** — R²/MSE per target (joint, input marginal, transducer marginal) × per layer ×
  trained/untrained baseline.
- **Separability fraction** — % of analytical belief states that factor as a tensor product.
- **Predictive (`d_μ`)** — mean per-position KL `KL(P_t ‖ Q_t)` with `P_t` the analytical
  next-token distribution and `Q_t` the model softmax (aim §5b/§9b), plus the `CE − h_μ`
  entropy-rate gap. Both computed in-notebook.
- **Erase-gap** — compare marginal R² across a `joint` run and an `output_only` run on the same
  process.

## Library code

- [`simplexity/generative_processes/transition_matrices.py`](../simplexity/generative_processes/transition_matrices.py):
  - `driven_transducer(...)` — fully-observable joint kernel `V`.
  - `coarse_grained_transducer(...)` — intermediate-hidden operator `W^(x) = Σ_y kron(T^y, U^{x|y})`.
  - `input_only_operator(...)` — output-hidden operator (reduces to the input HMM).
  - `iid_sns_transducer(p_in, p_0, p_1)` — Stage A instance (registered in `HMM_MATRIX_FUNCTIONS`).
  - `_sns_kernel(p)`, `_fractal2_kernel(b, p, q)` — SNS / Fractal2 kernels (explorer parametrization).
- Tests: `test_iid_sns_transducer`, `test_iid_sns_transducer_marginal_input`,
  `test_driven_transducer_kron_structure`, `test_coarse_grained_transducer`, `test_input_only_operator`.

The notebook's numpy kernels are a verbatim port of `explorer.html`; they were cross-checked to
produce **identical** belief trajectories to the explorer's `sampleDrivenTransducer`, and the
library JAX kernel matches the numpy construction.

## What the notebook produces

[`notebooks/colab/transducer_decomposition_experiment.ipynb`](../notebooks/colab/transducer_decomposition_experiment.ipynb):

1. Build the three regime operators (`V`, `W`, input-only); compute the **regime-correct
   entropy-rate floor** in-notebook.
2. Plot the **three analytical belief manifolds** (product / entangled cascade / input-line) on
   the `(input marginal, transducer marginal)` plane, matching the explorer, with the
   separability fraction per regime.
3. Train a 4-layer transformer on the active-regime token stream (vocab depends on `OBSERVE`).
4. **Predictive fit**: `d_μ` (per-position KL) and the `CE − h_μ` gap.
5. Linear probes (per-layer + concatenated) for joint / input / transducer belief, each with an
   **untrained baseline**; under `output_only` the marginal lag quantifies the erasure.
6. Controls: sequence-level CV, shuffle, temporal split.
7. Ground-truth vs residual-stream readout of the **joint belief manifold** (product vs entangled)
   plus the transducer predicted-vs-true check.

## Running

GPU recommended (torch + numpy only; no JAX in the notebook). On **free-tier Colab (T4)** keep
`COLAB_FREE_T4 = True` in the config cell — it caps training to 100k steps and analysis to 20k
sequences (~40 min/run, avoids the ~12 GB OOM and idle-timeout); set it `False` on Colab Pro /
A100 for the full 1M-step run. Select a **T4 GPU** runtime, run cell 1 (installs
`transformer-lens`), then run the cells in order, **skipping the "Option B / load" cell** and
running the training cell instead.

For the erase-information study, run once with `OBSERVE='joint'` and once with
`OBSERVE='output_only'` on the same process, and compare the saved `config.json` marginal R² and
`separability`. Stage C (feedback) lives in `transducer_feedback_experiment.ipynb` and runs the
same way. To run the unit tests where a working JAX toolchain is available:

```bash
uv run --extra dev --extra pytorch pytest tests/generative_processes/test_transition_matrices.py -k "transducer or coarse or input_only"
```

> Note: on a Windows machine where `uv`'s interpreter has a broken `_ctypes` (missing libffi
> DLL), JAX cannot import and these tests cannot run locally; the belief math was instead
> verified in pure numpy against `explorer.html`. The notebook itself is torch/numpy only and
> does not require JAX.
