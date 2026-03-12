# QSplitFL

QSplitFL is a research framework that applies Reinforcement Learning to automate split point selection in Federated Split Learning. Instead of manually choosing where to divide a neural network between clients and a server, a capability-aware Deep Q-Network (DQN) agent learns the optimal split point dynamically based on real-time client hardware conditions.

This project implements the algorithms described in the paper: **"Reinforcement Learning Design for Split Learning in a Federated Learning Client Cluster"**.

---

## Background

In Split Learning, a neural network is partitioned at a chosen layer. Clients compute the forward pass up to that layer and send intermediate activations (smashed data) to the server, which completes the forward pass, computes the loss, and sends gradients back. The choice of split point determines how much computation falls on the client versus the server and how much data is transmitted over the network.

Choosing this split point manually is suboptimal in heterogeneous environments where clients differ in CPU speed, memory, battery level, and network bandwidth. QSplitFL replaces the manual choice with a learned RL policy that adapts to client capabilities each round.

---

## Project Structure

```
QSplitFL/
├── complete_rl_implementation/     Full system with Committee DQN across multiple datasets
│   ├── core/                       Core RL source modules
│   ├── runners/                    Experiment entry points and run scripts
│   ├── plotting/                   Figure generation and analysis scripts
│   ├── utils/                      Maintenance and verification utilities
│   ├── tests/                      Test suite
│   ├── logs/                       Experiment log files
│   ├── results/                    Summary CSVs and result images
│   ├── assets/                     PDF and image assets
│   ├── MNIST/                      MNIST experiment results
│   ├── CIFAR10/                    CIFAR-10 experiment results
│   ├── CIFAR-100/                  CIFAR-100 experiment results
│   ├── FMNIST/                     Fashion-MNIST experiment results
│   ├── Paper_Figures/              Figures for publication
│   ├── Paper_Figures_New/          Updated publication figures
│   ├── Paper_Figures_Small/        Cropped and resized figures
│   └── Organized_Analysis_Results/ Aggregated analysis outputs
│
├── Capability_Aware_DQN_Implementation/    Standalone capability-aware DQN on MNIST
├── MNIST_SplitFL_Complete/                 Baseline split FL implementation on MNIST
└── Ablation_Study/                         8-configuration ablation study on CIFAR-10
```

---

## Modules

### complete_rl_implementation

The primary implementation. Supports MNIST, CIFAR-10, CIFAR-100, and Fashion-MNIST with ResNet50, MobileNetV4, and ConvNeXt backbones. Integrates all algorithms including Committee DQN, Reward Committee, and the complete multi-dataset training pipeline.

**core/** -- Python source modules

| File | Description |
|---|---|
| `client.py` | Client-side forward and backward pass logic |
| `server.py` | Server-side computation and FedAvg aggregation |
| `environment.py` | FL environment, capability state computation, reward function |
| `dqn_agent.py` | Capability-aware DQN with experience replay and target network |
| `committee_dqn.py` | Ensemble of M DQN heads with shared encoder and majority voting |
| `reward_committee.py` | Anti-reward-hacking ensemble of reward predictors |
| `voting_mechanism.py` | Majority vote logic for committee action selection |
| `models.py` | Shared neural network definitions |
| `resnet50_model.py` | ResNet50 model definition |
| `mobilenetv4_model.py` | MobileNetV4 model definition |
| `convnext_model.py` | ConvNeXt model definition |

**runners/** -- experiment entry points

| File | Description |
|---|---|
| `main_complete_rl.py` | Main training script for the full RL system |
| `run_all_auto.py` | Automated runner for all dataset/model combinations |
| `run_sequential_experiments.py` | Sequential multi-configuration experiment runner |
| `deploy_models.py` | Model deployment and evaluation script |

**plotting/** -- figure generation

| File | Description |
|---|---|
| `generate_all_plots.py` | Generate all result plots from saved data |
| `generate_comparison_plots_fmnist_cifar100.py` | Cross-dataset comparison figures |
| `generate_detailed_client_comparisons.py` | Per-client performance breakdown figures |
| `generate_new_figures.py` | Updated figure generation for revised paper |
| `generate_small_figures.py` | Cropped compact versions of figures |
| `aggregate_results_to_table.py` | Aggregate per-round results into summary tables |
| `organize_plots.py` | Sort and organize generated figures into folders |
| `regenerate_comparison_plots.py` | Regenerate comparison plots with updated formatting |
| `crop_figures.py` | Crop whitespace from exported figures |
| `update_all_plots_with_model_names.py` | Annotate plots with backbone model names |

**utils/** -- maintenance scripts

| File | Description |
|---|---|
| `enhance_logging.py` | Add structured logging to training scripts |
| `remove_refs.py` | Strip internal references from exported files |
| `update_server_robust.py` | Patch server.py with robustness improvements |
| `update_servers.py` | Bulk server configuration updates |
| `verify_connections.py` | Validate client-server tensor flow |
| `verify_split_ranges.py` | Check split point index validity across all models |

**tests/** -- test suite

| File | Description |
|---|---|
| `test_integration.py` | End-to-end integration tests |
| `test_main.py` | Unit tests for main training loop |
| `test_quick_run.py` | Fast smoke test for a single training round |
| `test_quick_log.py` | Verify logging output format |
| `test_resnet_mnist.py` | ResNet50 on MNIST correctness test |
| `automated_test.py` | Automated multi-config regression tests |

---

### Capability_Aware_DQN_Implementation

A self-contained version of the capability-aware DQN on MNIST. Designed for understanding and quick experimentation. Each module includes built-in tests.

| File | Description |
|---|---|
| `environment_capability_aware.py` | 6D state vector construction and decayed reward computation |
| `dqn_split_agent.py` | Q-network, experience replay buffer, epsilon-greedy exploration |
| `split_learning_utils.py` | Dynamic model splitting and FedAvg aggregation |
| `main_capability_aware_dqn.py` | Training loop with 6-panel result visualization |

---

### Ablation_Study

Systematically evaluates 8 design configurations on CIFAR-10 with ResNet50 across 10 clients and 100 rounds. Each configuration isolates one design choice to measure its individual contribution.

| File | Description |
|---|---|
| `ablation_configs.py` | Definitions of all 8 experimental configurations |
| `run_ablation.py` | CLI runner supporting quick, selective, and full runs |
| `sfl_trainer.py` | Core TrainRoundSFL logic shared across all configurations |
| `plot_ablation.py` | Comparative result visualization across configurations |
| `requirements.txt` | Python dependency list |

---

## Algorithms

**Algorithm -- Capability-Aware DQN with Experience Replay**

The agent observes a 6-dimensional cluster state vector representing the average CPU, memory, battery, and network capacity of selected clients, their weighted overall capability score, and capability variance across clients. It selects a split layer using epsilon-greedy exploration, receives a reward based on the change in global loss, and trains via temporal-difference updates with a target network.

**Algorithm -- TrainRoundSFL**

Each training round selects K clients, runs a split forward and backward pass with the RL-chosen split point, and applies FedAvg to aggregate updated client weights into the global model.

**Algorithm -- Committee DQN with Majority Voting**

An ensemble of M DQN agents shares a shallow encoder but maintains separate deep heads. The final action is determined by majority vote across all heads, reducing susceptibility to reward hacking.

---

## Key Equations

**Reward function (decayed loss-drop):**

```
r_t = -delta_L_t * exp(-lambda * (t - 1))
```

where `delta_L_t` is the change in global loss at round `t` and `lambda` controls early-round reward weighting.

**State vector:**

```
s_t = [avg_CPU, avg_Memory, avg_Battery, avg_Network, avg_Overall, capability_std]
```

This 6D representation achieves O(|K|) complexity versus O(d^3) for PCA-based approaches, while remaining interpretable.

---

## Ablation Configurations

| Config | Committee Size | RL | Reward Decay | State |
|---|---|---|---|---|
| Full QSplitFL | M=3 | Yes | lambda=0.05 | 6D full |
| Single-head DQN | M=1 | Yes | lambda=0.05 | 6D full |
| Large committee | M=5 | Yes | lambda=0.05 | 6D full |
| No reward decay | M=3 | Yes | lambda=0 | 6D full |
| CPU-only state | M=3 | Yes | lambda=0.05 | 1D CPU |
| High decay | M=3 | Yes | lambda=0.2 | 6D full |
| No RL (random split) | N/A | No | N/A | N/A |
| No RL (fixed split) | N/A | No | N/A | N/A |

---

## Installation

```bash
pip install torch torchvision numpy matplotlib
```

Requires Python 3.8 or later and PyTorch 1.13 or later.

---

## Usage

**Run the full RL system:**

```bash
cd complete_rl_implementation/runners
python main_complete_rl.py
```

**Run the standalone capability-aware DQN:**

```bash
cd Capability_Aware_DQN_Implementation
python main_capability_aware_dqn.py
```

**Run ablation studies:**

```bash
cd Ablation_Study

# Full run (GPU recommended)
python run_ablation.py

# Quick smoke test
python run_ablation.py --quick

# Run specific configurations by index
python run_ablation.py --configs 0 1 7
```

**Run tests:**

```bash
cd complete_rl_implementation/tests
python test_integration.py
python test_quick_run.py
```

---

## Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `num_rounds` | 100 | Training rounds per experiment |
| `num_clients` | 10 | Total clients in the federation |
| `k` | 6 | Clients selected per round |
| `committee_size` | 3 | Number of DQN heads in the committee |
| `learning_rate` | 0.001 | DQN optimizer learning rate |
| `gamma` | 0.2 | Discount factor (low, favoring immediate rewards) |
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_end` | 0.01 | Minimum exploration rate |
| `lambda_decay` | 0.05 | Reward decay constant |
| `memory_size` | 10000 | Experience replay buffer capacity |
| `batch_size` | 32 | Mini-batch size for DQN updates |

---

## Datasets and Models

**Datasets:** MNIST, CIFAR-10, CIFAR-100, Fashion-MNIST

**Models:** CNN, ResNet50, MobileNetV4, ConvNeXt

Data is partitioned across clients using a Dirichlet distribution (alpha=0.5) to simulate non-IID conditions, the standard heterogeneous federated learning benchmark.

---

## Results

Training produces per-round accuracy and loss logs, summary CSV files, and visualization plots. Results are saved to dataset-specific subdirectories under `complete_rl_implementation/` (e.g., `MNIST/`, `CIFAR10/`). Processed figures for publication are in `Paper_Figures/` and `Paper_Figures_New/`. Aggregated analysis is in `Organized_Analysis_Results/`.

Expected accuracy of the full QSplitFL system on MNIST after 100 rounds: above 95%.
