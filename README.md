# Simplexity

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Astera-org/simplexity)

A library for exploring sequence prediction models from a Computational Mechanics perspective.

## Belief State Geometry in Hierarchical Sequential Data

An extension of [Shai et al. (NeurIPS 2024)](https://arxiv.org/abs/2405.15943) from flat HMMs to hierarchical HMMs with multi-timescale latent structure. A slow-switching driver (dwell ~20 steps) selects between two Mess3 transducer variants, creating a 6-state process where the transformer must simultaneously infer a fast variable (transducer state, observable in ~3 tokens) and a slow variable (driver identity, requiring ~15-30 tokens of statistical evidence).

A 4-layer transformer trained on next-token prediction linearly encodes the full 6D Bayesian posterior in its residual stream (R²=0.982, MSE=0.004), with the network allocating deeper computation to the slower inference problem and encoding driver and transducer in significantly separated subspaces.

See [`docs/composite_mess3_experiment.md`](docs/composite_mess3_experiment.md) for the full writeup with worked examples, belief evolution traces, and precise numerical results.

## Transducer Belief Decomposition (input HMM → output transducer)

A genuine input→output ε-transducer experiment (the feed-forward construction from `Docn/explorer.html`, per `Docn/aim.md` §10.4). An autonomous **input HMM** emits symbols `y` that drive a **transducer** emitting `x`; the observable is the composite token `(y, x)` and the joint kernel is `Kron(T_input^(y), T_transducer^(x|y))`. A transformer is trained on the joint stream and the residual stream is linearly probed for the **input belief**, the **transducer belief**, and the **joint belief** separately.

Staged via a single `INPUT_MODE` flag: Stage A (`'iid'`) isolates the transducer with an IID input (its SNS belief manifold is the countably-infinite mixed-state arc on the 1-simplex — an unambiguous reconstruction target); Stage B (`'sns'`/`'mess3'`) uses a structured input so both marginals are nontrivial. Adds an in-notebook entropy-rate floor and per-target untrained baselines.

- Library: `driven_transducer`, `iid_sns_transducer`, `_sns_kernel` in [`simplexity/generative_processes/transition_matrices.py`](simplexity/generative_processes/transition_matrices.py) (+ tests)
- Notebooks (torch+numpy only): feed-forward + coarse-graining regimes in [`notebooks/colab/transducer_decomposition_experiment.ipynb`](notebooks/colab/transducer_decomposition_experiment.ipynb); feedback / partially-observable (modified-operator MSP) in [`notebooks/colab/transducer_feedback_experiment.ipynb`](notebooks/colab/transducer_feedback_experiment.ipynb)
- Writeup: [`docs/transducer_decomposition_experiment.md`](docs/transducer_decomposition_experiment.md)

### What was added

**Library code:**
- [`simplexity/generative_processes/transition_matrices.py`](simplexity/generative_processes/transition_matrices.py) — added `composite_mess3()` function that constructs the (3, 6, 6) block transition matrix, registered in `HMM_MATRIX_FUNCTIONS`
- [`tests/generative_processes/test_transition_matrices.py`](tests/generative_processes/test_transition_matrices.py) — three tests: stochasticity/stationarity, block-diagonal recovery at epsilon=0, and block structure verification

**Experiment notebook:**
- [`notebooks/colab/composite_mess3_experiment.ipynb`](notebooks/colab/composite_mess3_experiment.ipynb) — self-contained Colab notebook (only requires `pip install transformer-lens`). Covers process definition, transformer training (1M steps), four linear probes (full belief, driver/transducer decompositions, timescale separation), five controls (untrained baseline, sequence-level CV, point-level CV, temporal split, shuffle), subspace analysis with belief-projection null distribution, and PCA visualizations

**Trained checkpoint:**
- [`notebooks/checkpoints/composite_mess3/`](notebooks/checkpoints/composite_mess3/) — trained model weights (`model_weights.pt`, 608 KB), all numerical results (`config.json`, `controls.json`, `subspace.json`, `probe_results.npz`), training history (`training_history.npz`, 1M loss values), and all figures (`figures/`)

**Documentation:**
- [`docs/composite_mess3_experiment.md`](docs/composite_mess3_experiment.md) — detailed writeup explaining the process, architecture, training, probing methodology, all results, and controls

### Reproduce on Colab

Upload `notebooks/checkpoints/composite_mess3/` as a zip and run the notebook using the load-checkpoint cell (~20 min, GPU required). To retrain from scratch, run the training cell instead (~1 hour on T4).

## Codebase Structure

```
simplexity/
  ├── data/                      # generated data
  ├── notebooks/                 # workflow and analysis examples
  ├── simplexity/                # core library code
  │   ├── configs/               # hydra configuration files
  │   ├── data_structures/       # data structures
  │   ├── evaluation/            # functions for model eval/validation
  │   ├── generative_processes/  # generative process classes
  │   ├── logging/               # parameter and metric logging
  │   ├── persistence/           # model loading and checkpointing
  │   ├── predictive_models/     # predictive model classes
  │   ├── training/              # model training functions
  │   ├── utils/                 # utils
  │   ├── train_model.py         # entrypoint for training models
  └── tests/                     # unit tests
```

## Usage

### Installation

[Install UV](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install the dependencies:

```bash
uv sync
```

to include the dev dependencies, run:

```bash
uv sync --extra dev
```

This should create a new python environment at `.venv` and install the dependencies.

### Training

Model training is configured using [Hydra](https://hydra.cc/) with config files specified in the `simplexity/configs` directory. To train a model using the configuration specified in `simplexity/configs/train_model.yaml`, run:

```bash
uv run python simplexity/train_model.py
```

An experiment comprised of several runs can be executed to perform a hyperparameter search or otherwise run multiple trials with different parameters using [Optuna](https://optuna.org/). To run an experiment using the configuration specified in `simplexity/configs/experiment.yaml`, run:

```bash
uv run python simplexity/run_experiment.py --multirun
```

#### Advanced Training Metrics (PyTorch)

The `TrainingMetricTracker` under `simplexity.metrics.metric_tracker` keeps
instantaneous (current loss, learning rate, parameter/gradient norms) and
cumulative metrics (tokens processed, running average loss, distance from
initialization, cumulative parameter-update norms) in sync with a PyTorch
optimizer. The tracker is wired into
`simplexity.training.train_pytorch_model.train`, but it can also be used directly
when writing custom training loops:

```python
import torch
from simplexity.metrics.metric_tracker import MetricTracker

model = build_model()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10_000)
tracker = MetricTracker(
    model,
    optimizer,
    metric_kwargs={
        "lr_schedule": lr_schedule,
    },
)

for step, batch in enumerate(train_loader, start=1):
    logits = model(batch["inputs"])
    loss = criterion(logits, batch["labels"])

    loss.backward()
    optimizer.step()
    lr_schedule.step()
    optimizer.zero_grad(set_to_none=True)

    tracker.step(
        step=step,
        loss=loss,
        tokens_in_batch=int(batch["labels"].numel()),
    )
    metrics = tracker.get_metrics()
    logger.log_metrics(step, metrics)
```

Each call returns a flat dictionary with keys such as `step/tokens`, `cum/tokens`,
`loss/step`, `loss/ma`, `lr/step`, `grads/l2_norm/step`, `params/update_l2_norm/cum`,
and `params/distance_from_init`, ready to be sent to any `Logger` implementation.
### Model Checkpointing

The `ModelPersister` class is responsible for saving and loading model checkpoints. The `LocalPersister` class saves checkpoints to the local file system, while the `S3Persister` class saves checkpoints to an S3 bucket.

#### Using S3 Storage

The `S3Persister`, can be configured using an `.ini` file, which should have the following structure:

```ini
[aws]
profile_name = default

[s3]
bucket = your_s3_bucket_name
prefix = your_s3_prefix
```

[AWS configuration and credential files](https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-files.html) can be used for authentication and settings. Authentication credentials should be specified in `~/.aws/credentials`. Settings like `region`, `output`, `endpoint_url` should be specified in `~/.aws/config`. Multiple different profiles can be defined and the specific profile to use can be specified in the `aws` section of the `.ini` file.
