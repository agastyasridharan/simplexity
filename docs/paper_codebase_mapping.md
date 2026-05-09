# Paper-Codebase Mapping: "Transformers Represent Belief State Geometry in their Residual Stream"

Paper: Shai, Marzen, Teixeira, Oldenziel, Riechers (NeurIPS 2024). arXiv:2405.15943v3.

This document maps every experiment and figure in the paper to specific code in the Simplexity repository.

---

## Core Mathematical Correspondence

### HMM Definition (Section 2.1)

The paper defines edge-emitting HMMs with token-labeled transition matrices {T^(x)}, where T^(x)\_{i,j} = Pr(x, s\_j | s\_i).

- `simplexity/generative_processes/generalized_hidden_markov_model.py` — `GeneralizedHiddenMarkovModel`
  - `transition_matrices`: shape `(vocab_size, num_states, num_states)` — the paper's {T^(x)}
  - `__init__()` (lines 41-92): computes net transition matrix T = sum\_x T^(x), extracts principal right eigenvector for normalization, computes stationary distribution as left eigenvector of T
- `simplexity/generative_processes/hidden_markov_model.py` — `HiddenMarkovModel`
  - Stochastic special case: all T^(x) entries non-negative, rows of T sum to 1
  - Sets `normalizing_eigenvector = ones(num_states)` (line 63), reducing GHMM normalization to simple sum normalization

### Belief State Update (Eq. 1-2)

Paper Eq. 1: `eta' = eta * T^(x) / (eta * T^(x) * 1)`

HMM implementation (`hidden_markov_model.py:91-100`):

```python
state = state @ self.transition_matrices[obs]   # eta' = eta . T^(x)
return state / jnp.sum(state)                    # normalize by sum
```

GHMM implementation (`generalized_hidden_markov_model.py:137-145`):

```python
state = state @ self.transition_matrices[obs]
return state / (state @ self.normalizing_eigenvector)  # normalize by v
```

For HMMs the normalizing eigenvector is 1, so both implementations coincide.

### Linear Regression (Appendix A.5)

Paper: `min_{W,c} sum_i ||b_i - (W * a_i + c)||^2`

Implementation in `simplexity/analysis/linear_regression.py:56-95`:

```python
design = _design_matrix(x_arr, fit_intercept)       # [1|X] augmented design matrix
sqrt_w = jnp.sqrt(w_arr)[:, None]
weighted_design = design * sqrt_w                    # sqrt(w) * [1|X]
weighted_targets = y_arr * sqrt_w                    # sqrt(w) * y
beta, _, _, _ = jnp.linalg.lstsq(weighted_design, weighted_targets, rcond=None)
predictions = design @ beta                          # y_hat = [1|X] . beta
```

Where `beta[0]` = intercept c, `beta[1:]` = weight matrix W.

Metrics computed in `_regression_metrics()` (lines 32-53): MSE, RMSE, MAE, R-squared, weighted L2 distance.

---

## Transition Matrix Verification

All transition matrices are defined in `simplexity/generative_processes/transition_matrices.py`.

### Z1R Process (Figure 2)

Function: `zero_one_random(p)` (lines 345-365).

With p=0.5, produces:

```
T^(0) = [[0, 1, 0], [0, 0, 0], [0.5, 0, 0]]
T^(1) = [[0, 0, 0], [0, 0, 1], [0.5, 0, 0]]
```

**Status: Exact match with paper.**

### RRXOR Process (Figure 7, Appendix A.3)

Function: `rrxor(p1, p2)` (lines 271-288). State mapping: S=0, 0=1, 1=2, T=3, F=4.

With p1=0.5, p2=0.5, produces the 5x5 matrices from Appendix A.3:

```
T^(0) = [[0, 0.5, 0, 0, 0], [0, 0, 0, 0, 0.5], [0, 0, 0, 0.5, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0]]
T^(1) = [[0, 0, 0.5, 0, 0], [0, 0, 0, 0.5, 0], [0, 0, 0, 0, 0.5], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
```

Config: `tests/end_to_end/configs/generative_process/rrxor.yaml` uses p1=0.5, p2=0.5.

**Status: Exact match with paper.**

### Mess3 Process (Figure 5, Appendix A.3)

Function: `mess3(x, a)` (lines 124-152). Formula:

```
b = (1-a)/2,  y = 1-2x
T^(A) = [[ay, bx, bx], [ax, by, bx], [ax, bx, by]]
T^(B) = [[by, ax, bx], [bx, ay, bx], [bx, ax, by]]
T^(C) = [[by, bx, ax], [bx, by, ax], [bx, bx, ay]]
```

The paper (Appendix A.3) specifies T^(A)\_{0,0} = 0.765, T^(A)\_{0,1} = 0.00375, etc. Solving for parameters:

- ay = 0.765, bx = 0.00375, ax = 0.0425, by = 0.0675
- Solution: **a = 0.85, x = 0.05**

The codebase default config (`tests/end_to_end/configs/generative_process/mess3.yaml`) uses **x=0.15, a=0.6**, which produces T^(A)\_{0,0} = 0.42 — a less extreme fractal. The function structure is correct; calling `mess3(x=0.05, a=0.85)` reproduces the paper's exact matrices.

**Status: Function correct, default parameters differ from paper. Use x=0.05, a=0.85 to reproduce.**

Golden test data at codebase defaults: `tests/generative_processes/goldens/mixed_state_trees/mess3_x_0p15_a_0p6.npz`.

---

## Figure-by-Figure Code Mapping

### Figure 1 — Overview (belief geometry emerges during training)

High-level summary. The underlying computations involve:

- Theoretical prediction: `MixedStateTreeGenerator` in `simplexity/generative_processes/mixed_state_presentation.py:202-266`
- Residual stream analysis: `ActivationTracker.analyze()` in `simplexity/activations/activation_tracker.py:132-198`
- Training progression: analysis run at multiple training checkpoints

### Figure 2 — HMM illustration (Z1R process)

3-state HMM with transition matrices T^(0) and T^(1).

- Code: `zero_one_random(p=0.5)` at `transition_matrices.py:345-365`
- Verified exact match (see Transition Matrix Verification above)

### Figure 3 — Z1R Mixed-State Presentation

(A) Generative structure, (B) MSP predictive structure, (C) belief distributions in simplex, (D) belief geometry.

- MSP tree generation: `MixedStateTreeGenerator._generate_tree_data()` (lines 230-266) with `SearchAlgorithm.BREADTH_FIRST`
- Belief update per node: `get_child()` (lines 326-336):
  ```python
  unnormalized_belief_state = node.unnormalized_belief_state @ self.ghmm.transition_matrices[obs]
  belief_state = self.ghmm.normalize_belief_state(unnormalized_belief_state)
  ```
- Probability pruning: children with `probability < prob_threshold` are discarded (line 362)
- Golden data: `tests/generative_processes/goldens/mixed_state_trees/zero_one_random_p_0p5.npz`

### Figure 4 — Methodology (finding belief simplex in residual stream)

(A) Transformer architecture, (B) collect activations at all context positions, (C) find linear projection, (D) plot.

Pipeline code:

1. **Record activations (B)**: `prepare_activations()` in `activation_tracker.py:68-119`
   - Converts raw activations from PyTorch/JAX via `_to_jax_array()` (line 54)
2. **Deduplication (implicit)**: `build_deduplicated_dataset()` in `simplexity/utils/analysis_utils.py:272-314`
   - Groups inputs by prefix via `make_prefix_groups()` (line 9)
   - Keeps first occurrence of activations/beliefs per unique prefix (`dedup_tensor_first()`, line 25)
   - Sums probabilities across duplicate prefixes (`dedup_probs_sum()`, line 59)
3. **Linear regression (C)**: `linear_regression()` in `analysis/linear_regression.py:56-95`
4. **PCA for visualization (D)**: `compute_weighted_pca()` in `analysis/pca.py:16-83`

### Figure 5 — Main result: Mess3 fractal geometry in residual stream

(A) Mess3 process (3 states, vocab {A,B,C}), (B) ground-truth fractal belief geometry, (C) linear projection of final residual stream.

- Process definition: `mess3(x, a)` at `transition_matrices.py:124-152`
  - Paper uses x=0.05, a=0.85 (see Transition Matrix Verification)
- Ground truth geometry (B): computed via `MixedStateTreeGenerator` building the MSP tree
- Residual stream projection (C): `LinearRegressionAnalysis` applied to final layer activations, finding 2D subspace of 64-dimensional activations that best matches ground-truth belief distributions

### Figure 6 — Controls (nontriviality)

(A) Training emergence, (B) cross-validation, (C) shuffle control, (D) MSE comparison.

- **(A) Training emergence**: `ActivationTracker.analyze()` run at multiple training checkpoints
- **(B) Cross-validation**: 80/20 splits x 1000 iterations (per Appendix A.7). Train split fits regression, held-out data evaluated via `_compute_regression_metrics()` (linear_regression.py:98-120) with learned beta
- **(C) Shuffle control**: Randomly permute belief-state-to-activation correspondences, run regression. Collapses fractal to simplex center
- **(D) MSE bars**: `_regression_metrics()` (linear_regression.py:32-53) returns weighted MSE

### Figure 7 — RRXOR: belief geometry across layers

(A) RRXOR process (5 states, vocab {0,1}), (B) 4-simplex geometry (36 distinct belief states), (C) residual stream representation, (D) distance correlations, (E) per-layer and concatenated MSE.

- Process: `rrxor(p1=0.5, p2=0.5)` at `transition_matrices.py:271-288` (verified exact match)
- **(C) Concatenated layers**: `LinearRegressionAnalysis(concat_layers=True)` triggers concatenation in `activation_tracker.py:104-106`:
  ```python
  concatenated = jnp.concatenate(list(layer_acts.values()), axis=-1)
  layer_acts = {"concatenated": concatenated}
  ```
- **(D) Distance correlations**: Ground-truth Euclidean distance between belief state pairs vs. pairwise distances in regression `projected` output
- **(E) Per-layer MSE**: `LayerwiseAnalysis.analyze()` in `layerwise_analysis.py:175-200` iterates over all layers independently, computes MSE for each. Key formatting via `format_layer_spec()` in `analysis/metric_keys.py`

Key paper finding: RRXOR belief geometry is NOT in the final layer alone (unlike Mess3), but IS in the concatenation of all layers. Testable via `concat_layers=False` (per-layer) vs. `concat_layers=True` (concatenated).

### Figure S1 — Pre/post LayerNorm comparison

Same analysis pipeline applied to different activation hooks (before vs. after final LayerNorm). Both show similar belief geometry (MSE 0.0004 pre-LN vs 0.0003 post-LN). The hook names differ (e.g., `hook_resid_post` vs post-LN hook), formatted by `format_layer_spec()`.

### Figure S2 — Loss vs MSE relationship

Multiple training runs with varying validation loss plotted against regression MSE. Shows better next-token prediction correlates with more faithful belief state representation. Uses `linear_regression()` MSE at each loss level, with loss tracked by `simplexity/metrics/`.

---

## Full Analysis Pipeline

```
                      Training Data Generation
                      ========================
  HMM definition          Data + belief states          Training
  (transition_matrices.py) --> (generator.py) ---------> Transformer
        |                   generate_data_batch_          (external:
        |                   with_full_history()            TransformerLens)
        |                        |
        v                        v
  MSP Tree                  Activations + Ground Truth Beliefs
  (mixed_state_presentation.py)  |
        |                        v
        |                   Preprocessing
        |                   (analysis_utils.py)
        |                   build_deduplicated_dataset()
        |                        |
        |                   +----+----+
        |                   |         |
        v                   v         v
  Ground Truth         Linear Reg.   PCA
  Belief Geometry      (linear_      (pca.py)
  (for comparison)      regression.py)
                             |         |
                             v         v
                        MSE, R^2    Projections
                        (Fig 6D,7E)  (Fig 5C,7C)
```

### Key files in the pipeline

| Step | File | Key function |
|---|---|---|
| Define HMM | `generative_processes/transition_matrices.py` | `mess3()`, `rrxor()`, `zero_one_random()` |
| Build HMM | `generative_processes/builder.py` | `build_hidden_markov_model()` |
| Generate data + beliefs | `generative_processes/generator.py` | `generate_data_batch_with_full_history()` |
| Compute MSP tree | `generative_processes/mixed_state_presentation.py` | `MixedStateTreeGenerator.generate()` |
| Preprocess activations | `activations/activation_tracker.py` | `prepare_activations()` |
| Deduplicate by prefix | `utils/analysis_utils.py` | `build_deduplicated_dataset()` |
| Linear regression | `analysis/linear_regression.py` | `linear_regression()`, `layer_linear_regression()` |
| PCA | `analysis/pca.py` | `compute_weighted_pca()` |
| Layer-wise orchestration | `analysis/layerwise_analysis.py` | `LayerwiseAnalysis.analyze()` |
| Multi-analysis orchestration | `activations/activation_tracker.py` | `ActivationTracker.analyze()` |
| Analysis wrappers | `activations/activation_analyses.py` | `LinearRegressionAnalysis`, `PcaAnalysis` |

---

## What is Present vs. Missing

| Component | Present | Notes |
|---|---|---|
| Mess3 transition matrices | Yes | Function correct; use x=0.05, a=0.85 to match paper |
| RRXOR transition matrices | Yes | Exact match with p1=0.5, p2=0.5 |
| Z1R transition matrices | Yes | Exact match with p=0.5 |
| MSP tree generation | Yes | BFS/DFS with probability pruning |
| Belief state updates | Yes | Both HMM and GHMM variants |
| Linear regression pipeline | Yes | Standard + SVD variants, factored support |
| PCA visualization | Yes | Weighted PCA with variance thresholds |
| Layerwise analysis | Yes | Per-layer + concatenated |
| Activation tracker | Yes | Orchestrates preprocessing + analysis |
| Prefix deduplication | Yes | Groups by prefix, sums probs, keeps first activation |
| Figure notebooks | No | Removed in git history (commit 8097fa2) |
| Legacy training scripts | No | Removed in git history (commit 346be49) |
| Saved model weights | No | /examples/models/ not present |
| TransformerLens integration | No | Paper used TransformerLens; repo supports PyTorch/Equinox/Penzai frameworks |

---

## Transformer Architecture (Appendix A.6)

The paper trained transformers with:

- Context window: 10
- Activation: ReLU
- Head dimension: 8, model dimension: 64
- 4 layers, 1 attention head per layer
- MLP dimension: 256
- Causal attention masking, layer normalization
- SGD optimizer, batch size 64, learning rate 0.01, 1,000,000 epochs
- 64 sequences per batch from stationary distribution
- TransformerLens library for model instantiation and activation hooks

The Simplexity codebase does not currently contain TransformerLens-specific code, but supports multiple model frameworks via `simplexity/predictive_models/types.py` (Equinox, Penzai, PyTorch).
