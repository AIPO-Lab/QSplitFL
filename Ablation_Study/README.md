# QSplitFL Ablation Study

This folder contains the complete ablation study for the QSplitFL framework.
Each experiment isolates one design choice to measure its individual contribution
to the overall system performance.

All configurations are evaluated on CIFAR-10 using ResNet50 with 10 clients
and 100 training rounds, using a Dirichlet(alpha=0.5) non-IID data partition.

---

## What is tested

Eight configurations are defined, each varying one component while keeping
all others fixed at the full-model defaults:

| Configuration | What is varied |
|---|---|
| Full QSplitFL (M=3, lambda=0.05, 6D state) | Baseline — nothing varied |
| Single-head DQN (M=1) | Committee voting disabled |
| Committee M=5 | Larger committee size |
| No decay (lambda=0) | Reward decay factor removed |
| High decay (lambda=0.1) | Aggressive reward decay |
| Equal capability weights (wi=0.25) | Uniform vs. tuned metric weights |
| CPU-only state | Rich 6D state reduced to CPU metric only |
| Random split (no RL) | RL policy replaced by fixed mid-layer split |

---

## File structure

```
Ablation_Study/
├── ablation_configs.py   -- 8 AblationConfig dataclasses
├── resnet50_model.py     -- ResNet50 with split learning support (L=50, splits 25-49)
├── environment.py        -- FL environment with configurable state and reward
├── committee_dqn.py      -- Committee DQN supporting M=1, 3, or 5
├── voting_mechanism.py   -- Majority voting with Q-value tie-breaking
├── sfl_trainer.py        -- SFL round execution, FedAvg, and model evaluation
├── run_ablation.py       -- Main runner
├── plot_ablation.py      -- Plot and heatmap generation
├── requirements.txt
└── results/              -- Created automatically on first run
    ├── <config>_per_round.csv   (one per configuration)
    ├── ablation_summary.csv
    ├── accuracy_bar_chart.png
    ├── convergence_curves.png
    ├── split_layer_curves.png
    ├── results_heatmap.png
    └── reward_curves.png
```

---

## Running the experiments

Install dependencies:

```
pip install -r requirements.txt
```

Run all 8 configurations (GPU recommended):

```
python run_ablation.py
```

Quick smoke test (10 rounds, 5 batches per client):

```
python run_ablation.py --quick
```

Run specific configurations by index (0-indexed):

```
python run_ablation.py --configs 0 1 7
```

Re-generate plots from previously saved CSVs without retraining:

```
python plot_ablation.py
```

---

## Command-line options

| Flag | Default | Description |
|---|---|---|
| --quick | off | 10 rounds, 5 batches/client for fast testing |
| --configs N ... | all 8 | Indices of configurations to run |
| --rounds N | 100 | Number of training rounds per configuration |
| --clients N | 10 | Number of federated clients |
| --batches N | all | Max batches per client per round |
| --seed N | 42 | Global random seed |
| --outdir PATH | results/ | Directory for output files |

---

## Output files

After a full run, the results/ directory contains:

- One CSV per configuration with per-round accuracy, loss, split layer, and reward
- ablation_summary.csv with final accuracy, average split layer, and rounds to 80% accuracy
- Five plots covering accuracy comparison, convergence, split-layer selection, summary heatmap, and reward signal

---

## Design notes

**State representation.**
The full state is a 6-dimensional vector: average CPU, Memory, Battery, Network,
Overall capability, and capability standard deviation across the client cluster.
For the CPU-only variant the non-CPU dimensions are set to zero so the DQN
architecture remains identical across all configurations.

**Reward function.**
The decayed loss-drop reward is: r_t = -(L_t - L_{t-1}) * exp(-lambda*(t-1)).
Setting lambda=0 produces a flat reward where all rounds contribute equally.

**SFL execution.**
The split federated learning round (client forward, smashed data transmission,
server forward, backward, gradient return to client, FedAvg) is identical
across all 8 configurations. Only the RL policy that selects the split layer
differs between variants.

**Committee DQN.**
All committee members share a common encoder but maintain independent decision
heads. M=1 bypasses the voting step entirely. M must be odd to avoid ties;
ties are resolved by mean Q-value across all heads.
