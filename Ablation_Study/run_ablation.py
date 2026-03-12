"""
Ablation Study Runner

Runs all 8 ablation configurations on CIFAR-10 / ResNet50 / 10 clients /
100 rounds, then saves per-round accuracy CSVs and a summary CSV.

Usage
-----
# Full run (GPU recommended):
    python run_ablation.py

# Quick smoke test (few batches per client, few rounds):
    python run_ablation.py --quick

# Run only specific configurations (0-indexed):
    python run_ablation.py --configs 0 1 7

Command-line options
--------------------
--quick          Use max_batches=5 and num_rounds=10 for a fast test.
--configs N ...  Indices of ablation configs to run (default: all 8).
--rounds N       Override number of training rounds (default: 100).
--clients N      Override number of clients (default: 10).
--batches N      Max batches per client per round (default: None = all data).
--seed N         Random seed (default: 42).
--outdir PATH    Directory for results (default: ./results).
"""

import argparse
import csv
import os
import sys
import time
import random
import copy

import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

# ── local modules ─────────────────────────────────────────────────────────────
from ablation_configs import ABLATION_CONFIGS, CONFIG_LABELS
from resnet50_model   import ResNet50
from environment      import FL_Environment
from committee_dqn    import CommitteeDQN
from sfl_trainer      import make_dirichlet_split, run_sfl_round, evaluate


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="QSplitFL Ablation Study")
    p.add_argument("--quick",   action="store_true",
                   help="Fast smoke test: 10 rounds, 5 batches/client")
    p.add_argument("--configs", type=int, nargs="+",
                   help="Which config indices to run (0–7); default: all")
    p.add_argument("--rounds",  type=int, default=100)
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--batches", type=int, default=None,
                   help="Max batches per client per round (None = all)")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--outdir",  type=str, default="results")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_cifar10(data_dir: str = "./data"):
    """Download (if needed) and return CIFAR-10 train + test datasets."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=True,  download=True, transform=transform_train
    )
    test_ds  = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    return train_ds, test_ds


# ─────────────────────────────────────────────────────────────────────────────
# Single-configuration training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_config(config, client_loaders, test_loader, num_rounds: int,
                 max_batches, device, seed: int):
    """
    Run the full training loop for one ablation configuration.

    Returns
    -------
    dict with keys:
        accuracies    : List[float] — per-round test accuracy
        losses        : List[float] — per-round cluster loss
        split_layers  : List[int]   — per-round chosen split layer
        rewards       : List[float] — per-round reward
        final_accuracy: float
        avg_split     : float
        rounds_to_80  : int  (num_rounds if never reached)
    """
    # ── reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ── ResNet50 constants ────────────────────────────────────────────────────
    TOTAL_LAYERS = ResNet50.TOTAL_LAYERS   # 50
    MIN_SPLIT    = ResNet50.MIN_SPLIT      # 25
    MAX_SPLIT    = ResNet50.MAX_SPLIT      # 49
    ACTION_SIZE  = ResNet50.ACTION_SIZE    # 25
    MID_SPLIT    = MIN_SPLIT              # fixed baseline = 25

    # ── global model ──────────────────────────────────────────────────────────
    global_model = ResNet50(num_classes=10, input_channels=3).to(device)

    # ── RL environment ────────────────────────────────────────────────────────
    env = FL_Environment(
        num_clients        = len(client_loaders),
        state_mode         = config.state_mode,
        capability_weights = config.capability_weights,
        lambda_decay       = config.lambda_decay,
        seed               = seed,
    )

    # ── RL agent (or None for random-split baseline) ──────────────────────────
    if config.use_rl:
        agent = CommitteeDQN(
            committee_size   = config.committee_size,
            state_size       = 6,
            action_size      = ACTION_SIZE,
            total_layers     = TOTAL_LAYERS,
            learning_rate    = 1e-3,
            gamma            = 0.2,
            epsilon_start    = 1.0,
            epsilon_end      = 0.01,
            kappa            = 0.05,
            buffer_capacity  = 10_000,
            batch_size       = 32,
            target_update_freq = 5,
        )
    else:
        agent = None

    # ── bookkeeping ───────────────────────────────────────────────────────────
    accuracies   = []
    losses       = []
    split_layers = []
    rewards      = []
    prev_loss    = float("inf")

    all_indices = list(range(len(client_loaders)))

    print(f"\n{'─'*72}")
    print(f"  Config : {config.name}")
    print(f"  Desc   : {config.description}")
    print(f"  M={config.committee_size}  λ={config.lambda_decay}  "
          f"state={config.state_mode}  RL={config.use_rl}")
    print(f"{'─'*72}")

    t0_config = time.time()

    for round_t in range(1, num_rounds + 1):

        t0 = time.time()

        # ── (1) Construct state ─────────────────────────────────────────────
        state = env.get_cluster_state(all_indices)

        # ── (2) Select split layer ──────────────────────────────────────────
        if config.use_rl:
            split_layer, info = agent.select_action(state)
        else:
            split_layer = MID_SPLIT    # fixed mid-layer (no RL)

        # ── (3) Execute SFL round (Algorithm 2) ─────────────────────────────
        cluster_loss = run_sfl_round(
            global_model  = global_model,
            client_loaders = client_loaders,
            split_layer   = split_layer,
            device        = device,
            local_lr      = 0.01,
            max_batches   = max_batches,
        )

        # ── (4) Evaluate global model ───────────────────────────────────────
        accuracy, test_loss = evaluate(global_model, test_loader, device)

        # ── (5) Compute reward ──────────────────────────────────────────────
        reward = env.compute_reward(cluster_loss, prev_loss, round_t)

        # ── (6) Get next state ──────────────────────────────────────────────
        env.update_capabilities()
        next_state = env.get_cluster_state(all_indices)
        done       = (round_t == num_rounds)

        # ── (7) RL update ───────────────────────────────────────────────────
        if config.use_rl and agent is not None:
            agent.store_transition(state, split_layer, reward, next_state, done)
            td_loss = agent.train_step()
            if agent.should_update_target(round_t):
                agent.update_target_networks()
            agent.decay_epsilon(round_t)

        # ── bookkeeping ─────────────────────────────────────────────────────
        accuracies.append(accuracy)
        losses.append(cluster_loss)
        split_layers.append(split_layer)
        rewards.append(reward)
        prev_loss = cluster_loss

        elapsed = time.time() - t0
        eps_str = f"ε={agent.epsilon:.3f}" if config.use_rl else "no-RL"
        print(
            f"  Round {round_t:3d}/{num_rounds}  "
            f"acc={accuracy:.4f}  loss={cluster_loss:.4f}  "
            f"split={split_layer}  reward={reward:+.4f}  "
            f"{eps_str}  [{elapsed:.1f}s]"
        )

    total_time = time.time() - t0_config
    print(f"\n  Finished in {total_time/60:.1f} min")

    # ── summary stats ─────────────────────────────────────────────────────────
    final_acc   = accuracies[-1]
    avg_split   = float(np.mean(split_layers))
    rounds_to_80 = num_rounds   # default: never reached 80 %
    for i, acc in enumerate(accuracies):
        if acc >= 0.80:
            rounds_to_80 = i + 1
            break

    print(f"  Final accuracy  : {final_acc*100:.2f}%")
    print(f"  Avg split layer : {avg_split:.1f}")
    print(f"  Rounds to 80%   : {rounds_to_80}")

    return {
        "accuracies":     accuracies,
        "losses":         losses,
        "split_layers":   split_layers,
        "rewards":        rewards,
        "final_accuracy": final_acc,
        "avg_split":      avg_split,
        "rounds_to_80":   rounds_to_80,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Result saving helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_per_round_csv(config_name: str, result: dict, outdir: str):
    """Save per-round metrics for one configuration."""
    safe_name = config_name.replace(" ", "_").replace("(", "").replace(")", "") \
                            .replace("=", "").replace("/", "").replace("λ", "lambda")
    filepath  = os.path.join(outdir, f"{safe_name}_per_round.csv")
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "accuracy", "cluster_loss", "split_layer", "reward"])
        for i, (acc, loss, sp, rw) in enumerate(zip(
            result["accuracies"], result["losses"],
            result["split_layers"], result["rewards"]
        ), start=1):
            w.writerow([i, f"{acc:.6f}", f"{loss:.6f}", sp, f"{rw:.6f}"])
    return filepath


def save_summary_csv(all_results: dict, outdir: str):
    """Save Table-5-style summary CSV across all configurations."""
    filepath = os.path.join(outdir, "ablation_summary.csv")
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Configuration", "Final Accuracy (%)",
                    "Avg Split Layer", "Rounds to 80% Acc"])
        for cfg_name, result in all_results.items():
            w.writerow([
                cfg_name,
                f"{result['final_accuracy']*100:.2f}",
                f"{result['avg_split']:.1f}",
                result["rounds_to_80"],
            ])
    return filepath


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── apply --quick overrides ───────────────────────────────────────────────
    num_rounds  = 10  if args.quick else args.rounds
    max_batches = 5   if args.quick else args.batches

    # ── device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*72}")
    print(f"  QSplitFL Ablation Study")
    print(f"{'='*72}")
    print(f"  Dataset  : CIFAR-10")
    print(f"  Model    : ResNet50 (L=50, splits 25–49)")
    print(f"  Clients  : {args.clients}")
    print(f"  Rounds   : {num_rounds}")
    print(f"  Batches  : {'all' if max_batches is None else max_batches} per client")
    print(f"  Device   : {device}")
    print(f"  Seed     : {args.seed}")
    print(f"{'='*72}\n")

    # ── CIFAR-10 ──────────────────────────────────────────────────────────────
    print("Loading CIFAR-10…")
    train_ds, test_ds = load_cifar10()

    # Non-IID partition via Dirichlet(α=0.5)
    client_subsets = make_dirichlet_split(
        train_ds, num_clients=args.clients, alpha=0.5, seed=args.seed
    )
    client_loaders = [
        data.DataLoader(subset, batch_size=128, shuffle=True,
                        num_workers=2, pin_memory=(device.type == "cuda"))
        for subset in client_subsets
    ]
    test_loader = data.DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda")
    )
    print(f"  Train samples per client: "
          f"{[len(s) for s in client_subsets]}")

    # ── select configs ────────────────────────────────────────────────────────
    selected_indices = args.configs if args.configs else list(range(len(ABLATION_CONFIGS)))
    selected_configs = [ABLATION_CONFIGS[i] for i in selected_indices]
    print(f"\nRunning {len(selected_configs)} configuration(s): "
          f"{[c.name for c in selected_configs]}\n")

    # ── output directory ──────────────────────────────────────────────────────
    os.makedirs(args.outdir, exist_ok=True)

    # ── main loop ─────────────────────────────────────────────────────────────
    all_results = {}
    t_total = time.time()

    for config in selected_configs:
        result = train_config(
            config         = config,
            client_loaders = client_loaders,
            test_loader    = test_loader,
            num_rounds     = num_rounds,
            max_batches    = max_batches,
            device         = device,
            seed           = args.seed,
        )
        all_results[config.name] = result

        # Save per-round CSV immediately (fault tolerant)
        per_round_path = save_per_round_csv(config.name, result, args.outdir)
        print(f"  Saved per-round CSV → {per_round_path}")

    # ── summary ───────────────────────────────────────────────────────────────
    summary_path = save_summary_csv(all_results, args.outdir)
    print(f"\nSaved summary → {summary_path}")

    # ── print results summary to console ──────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  Ablation Study Results")
    print(f"  (CIFAR-10, ResNet50, {args.clients} clients, {num_rounds} rounds)")
    print(f"{'─'*72}")
    print(f"  {'Configuration':<35} {'Acc %':>8} {'Avg Split':>10} {'→80%':>8}")
    print(f"{'─'*72}")
    for cfg_name, res in all_results.items():
        print(
            f"  {cfg_name:<35} "
            f"{res['final_accuracy']*100:>7.2f}%  "
            f"{res['avg_split']:>9.1f}  "
            f"{res['rounds_to_80']:>7}"
        )
    print(f"{'='*72}")
    print(f"\nTotal time: {(time.time()-t_total)/60:.1f} min")
    print(f"All results saved in: {os.path.abspath(args.outdir)}/")

    # Call the plot script automatically
    try:
        import plot_ablation
        plot_ablation.generate_all_plots(all_results, args.outdir, num_rounds)
        print("Plots generated.")
    except Exception as e:
        print(f"[Warning] Could not generate plots: {e}")
        print("Run  python plot_ablation.py  separately.")


if __name__ == "__main__":
    main()
