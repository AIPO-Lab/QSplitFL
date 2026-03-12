"""
Split Federated Learning Training Utilities

Implements Algorithm 2 (TrainRoundSFL) from the QSplitFL paper.

The SFL execution is identical for all 8 ablation configurations.
What differs across configs is only the RL policy that selects split_layer.

In-memory simulation:
  - No network sockets; all tensors live on the same device.
  - Client forward → smashed data (detached) → server forward → loss →
    server backward → gradient back to client → client backward → FedAvg.
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def make_dirichlet_split(dataset, num_clients: int, alpha: float = 0.5,
                          seed: int = 42):
    """
    Partition a torchvision dataset into `num_clients` non-IID subsets
    using a Dirichlet(alpha) distribution over class labels.

    Returns:
        List[Subset] — one per client.
    """
    from torch.utils.data import Subset

    rng     = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    n_cls   = int(targets.max()) + 1

    # Indices per class
    cls_indices = [np.where(targets == c)[0] for c in range(n_cls)]
    for idx in cls_indices:
        rng.shuffle(idx)

    # Dirichlet allocation
    proportions = rng.dirichlet(np.repeat(alpha, num_clients), size=n_cls)
    client_indices = [[] for _ in range(num_clients)]

    for c in range(n_cls):
        splits = (np.cumsum(proportions[c]) * len(cls_indices[c])).astype(int)
        splits = np.clip(splits, 0, len(cls_indices[c]))
        chunks = np.split(cls_indices[c], splits[:-1])
        for client_id, chunk in enumerate(chunks):
            client_indices[client_id].extend(chunk.tolist())

    subsets = [Subset(dataset, idxs) for idxs in client_indices]
    return subsets


# ---------------------------------------------------------------------------
# One SFL round  (Algorithm 2)
# ---------------------------------------------------------------------------

def run_sfl_round(
    global_model,
    client_loaders,
    split_layer: int,
    device,
    local_lr: float = 0.01,
    max_batches: int = None,
):
    """
    Execute one full SFL training round.

    Steps (Algorithm 2):
      1. Broadcast W_{t-1} and ℓ to all clients.
      2. Client forward  → smashed data A_k.
      3. Server forward  → ŷ_k.
      4. Loss + server backward → ∇A_k.
      5. Client backward using ∇A_k → ΔW_k^client.
      6. FedAvg: average client-side layer weights;
                 keep server-side layers from server model.

    Args:
        global_model:   Full ResNet50 (full mode) — template weights.
        client_loaders: List of DataLoaders, one per client.
        split_layer:    ℓ chosen by RL agent.
        device:         torch.device.
        local_lr:       SGD learning rate.
        max_batches:    Limit batches per client (None = use all data).

    Returns:
        cluster_loss (float): Mean cross-entropy loss over all clients.
    """
    criterion    = nn.CrossEntropyLoss()
    n_clients    = len(client_loaders)
    total_loss   = 0.0

    # Collect client-side state dicts (layers 1..ℓ trained per client)
    client_side_states  = []
    # Server-side state dict (trained on all clients' smashed data collectively)
    server_side_state   = None

    for loader in client_loaders:
        # --- create fresh client and server models from global weights ----------
        client_model = copy.deepcopy(global_model)
        client_model.configure_split('client', split_layer)
        client_model.to(device).train()

        server_model = copy.deepcopy(global_model)
        server_model.configure_split('server', split_layer)
        server_model.to(device).train()

        c_opt = optim.SGD(client_model.parameters(), lr=local_lr,
                          momentum=0.9, weight_decay=5e-4)
        s_opt = optim.SGD(server_model.parameters(), lr=local_lr,
                          momentum=0.9, weight_decay=5e-4)

        batch_losses, n_batches = [], 0

        for batch_idx, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device)
            labels = labels.to(device)

            c_opt.zero_grad()
            s_opt.zero_grad()

            # Step 2: Client forward  → smashed data
            smashed = client_model(images)                   # A_k

            # Detach before sending to server (simulates network boundary)
            smashed_d = smashed.detach().requires_grad_(True)

            # Step 3: Server forward  → prediction
            output = server_model(smashed_d)
            loss   = criterion(output, labels)

            # Step 4: Server backward  → gradient on smashed_d (=∇A_k)
            loss.backward()
            smashed_grad = smashed_d.grad.clone()

            # Server update (layers ℓ+1..L)
            s_opt.step()

            # Step 5: Client backward using received gradient
            smashed.backward(smashed_grad)
            c_opt.step()

            batch_losses.append(loss.item())
            n_batches += 1

        avg_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        total_loss += avg_loss

        client_side_states.append(client_model.state_dict())
        server_side_state = server_model.state_dict()   # overwritten each client

    # -------------------------------------------------------------------------
    # Step 6: FedAvg
    #   Client-side layers  → average across all clients
    #   Server-side layers  → from server model (trained on all smashed data)
    # -------------------------------------------------------------------------
    new_state = {}
    for key in global_model.state_dict().keys():
        if global_model.is_client_layer(key, split_layer):
            stacked    = torch.stack([cs[key].float()
                                      for cs in client_side_states])
            new_state[key] = stacked.mean(0).to(
                global_model.state_dict()[key].dtype
            )
        else:
            new_state[key] = server_side_state[key]

    global_model.load_state_dict(new_state)

    return total_loss / n_clients


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loader, device) -> tuple:
    """
    Evaluate the global model (full mode).

    Returns:
        (accuracy: float, avg_loss: float)
    """
    model.configure_split('full', None)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total, correct, running_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = running_loss / total if total > 0 else 0.0
    return accuracy, avg_loss
