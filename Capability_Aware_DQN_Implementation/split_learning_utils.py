"""
Split Learning Utilities
=========================

This module implements the core split learning mechanics:
1. Dynamic model splitting at arbitrary layers
2. Client-side forward pass (produces smashed data)
3. Server-side forward pass (completes the model)
4. Gradient backpropagation across the split
5. FedAvg aggregation of client-side models

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import logging
import copy

logger = logging.getLogger(__name__)


class SplitModel:
    """
    Wrapper for splitting a neural network at arbitrary layers.
    
    This class handles:
    - Splitting a model into client-side and server-side components
    - Forward passes through each component
    - Gradient flow across the split point
    """
    
    def __init__(self, full_model: nn.Module, split_layer: int, device: str = "cpu"):
        """
        Initialize split model.
        
        Args:
            full_model: Complete neural network model
            split_layer: Layer index where to split (client computes layers 0:split_layer)
            device: Device to run on ("cpu" or "cuda")
        """
        self.full_model = full_model
        self.split_layer = split_layer
        self.device = device
        
        # Extract layers
        self.layers = self._extract_layers(full_model)
        self.total_layers = len(self.layers)
        
        # Validate split point
        if split_layer < 1 or split_layer >= self.total_layers:
            raise ValueError(
                f"Invalid split_layer {split_layer}. "
                f"Must be in range [1, {self.total_layers-1}]"
            )
        
        # Create client-side and server-side models
        self.client_model = self._create_client_model()
        self.server_model = self._create_server_model()
        
        logger.info(f"Split model at layer {split_layer}/{self.total_layers}")
        logger.debug(f"Client layers: 0-{split_layer}")
        logger.debug(f"Server layers: {split_layer}-{self.total_layers}")
    
    def _extract_layers(self, model: nn.Module) -> List[nn.Module]:
        """Extract layers from model as a list."""
        layers = []
        
        # Handle different model architectures
        if hasattr(model, 'features') and hasattr(model, 'classifier'):
            # VGG-style: features + classifier
            for layer in model.features:
                layers.append(layer)
            for layer in model.classifier:
                layers.append(layer)
        elif hasattr(model, 'children'):
            # Sequential or custom model
            for child in model.children():
                if isinstance(child, nn.Sequential):
                    for layer in child:
                        layers.append(layer)
                else:
                    layers.append(child)
        else:
            # Treat as single layer
            layers = [model]
        
        return layers
    
    def _create_client_model(self) -> nn.Module:
        """Create client-side model (layers 0 to split_layer)."""
        client_layers = self.layers[:self.split_layer]
        return nn.Sequential(*client_layers).to(self.device)
    
    def _create_server_model(self) -> nn.Module:
        """Create server-side model (layers split_layer to end)."""
        server_layers = self.layers[self.split_layer:]
        return nn.Sequential(*server_layers).to(self.device)
    
    def forward_client(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through client-side model.
        
        Args:
            x: Input tensor
        
        Returns:
            smashed_data: Activations at the cut layer
        """
        x = x.to(self.device)
        with torch.set_grad_enabled(True):
            smashed_data = self.client_model(x)
        return smashed_data
    
    def forward_server(self, smashed_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through server-side model.
        
        Args:
            smashed_data: Activations from client-side model
        
        Returns:
            output: Final model predictions
        """
        smashed_data = smashed_data.to(self.device)
        with torch.set_grad_enabled(True):
            output = self.server_model(smashed_data)
        return output
    
    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        """
        Complete forward pass (for evaluation).
        
        Args:
            x: Input tensor
        
        Returns:
            output: Final model predictions
        """
        smashed_data = self.forward_client(x)
        output = self.forward_server(smashed_data)
        return output
    
    def get_client_model(self) -> nn.Module:
        """Get client-side model."""
        return self.client_model
    
    def get_server_model(self) -> nn.Module:
        """Get server-side model."""
        return self.server_model
    
    def update_split_layer(self, new_split_layer: int):
        """Update split layer dynamically."""
        self.split_layer = new_split_layer
        self.client_model = self._create_client_model()
        self.server_model = self._create_server_model()
        logger.info(f"Updated split layer to {new_split_layer}")


def train_client_split(
    client_model: nn.Module,
    server_model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = "cpu",
    epochs: int = 1
) -> Dict[str, float]:
    """
    Train client using split learning.
    
    Workflow:
    1. Client forward pass → smashed data
    2. Server forward pass → predictions
    3. Compute loss
    4. Server backward pass → gradients
    5. Client backward pass → update weights
    
    Args:
        client_model: Client-side model
        server_model: Server-side model
        train_loader: Training data loader
        optimizer: Optimizer for client model
        criterion: Loss function
        device: Device to use
        epochs: Number of local training epochs
    
    Returns:
        metrics: Dictionary with training metrics
    """
    client_model.train()
    server_model.eval()  # Server model is not trained here
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Client forward pass
            smashed_data = client_model(data)
            smashed_data.requires_grad = True
            
            # Server forward pass
            with torch.no_grad():
                output = server_model(smashed_data)
            
            # Compute loss
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Update client model
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            epoch_correct += pred.eq(target.view_as(pred)).sum().item()
            epoch_samples += data.size(0)
        
        total_loss += epoch_loss
        total_correct += epoch_correct
        total_samples += epoch_samples
    
    # Compute averages
    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples
    
    return {
        "loss": avg_loss,
        "accuracy": avg_accuracy,
        "num_samples": total_samples
    }


def fedavg_aggregate(client_models: List[nn.Module]) -> nn.Module:
    """
    FedAvg aggregation of client models.
    
    Computes weighted average of client model parameters.
    
    Args:
        client_models: List of client models to aggregate
    
    Returns:
        aggregated_model: Averaged model
    """
    if not client_models:
        raise ValueError("No client models to aggregate")
    
    # Create a copy of the first model for aggregation
    aggregated_model = copy.deepcopy(client_models[0])
    aggregated_dict = aggregated_model.state_dict()
    
    # Initialize aggregated parameters to zero
    for key in aggregated_dict.keys():
        aggregated_dict[key] = torch.zeros_like(aggregated_dict[key])
    
    # Sum all client parameters
    for client_model in client_models:
        client_dict = client_model.state_dict()
        for key in aggregated_dict.keys():
            aggregated_dict[key] += client_dict[key]
    
    # Average
    num_clients = len(client_models)
    for key in aggregated_dict.keys():
        aggregated_dict[key] = aggregated_dict[key] / num_clients
    
    # Load aggregated parameters
    aggregated_model.load_state_dict(aggregated_dict)
    
    logger.debug(f"Aggregated {num_clients} client models using FedAvg")
    
    return aggregated_model


def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str = "cpu"
) -> Tuple[float, float]:
    """
    Evaluate model on test set.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        criterion: Loss function
        device: Device to use
    
    Returns:
        accuracy: Test accuracy
        loss: Average test loss
    """
    model.eval()
    
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
    
    accuracy = correct / total
    avg_loss = test_loss / total
    
    return accuracy, avg_loss


# Example usage and testing
if __name__ == "__main__":
    print("Testing Split Learning Utilities...")
    
    # Create simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.fc1 = nn.Linear(32 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = x.view(-1, 32 * 7 * 7)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # Test split model
    print("\n--- Test 1: Model Splitting ---")
    model = SimpleModel()
    
    # Try different split points
    for split_layer in [2, 4, 6]:
        split_model = SplitModel(model, split_layer)
        
        # Test forward pass
        x = torch.randn(4, 1, 28, 28)
        smashed_data = split_model.forward_client(x)
        output = split_model.forward_server(smashed_data)
        
        print(f"Split at layer {split_layer}:")
        print(f"  Input shape: {x.shape}")
        print(f"  Smashed data shape: {smashed_data.shape}")
        print(f"  Output shape: {output.shape}")
    
    print("\n All tests passed!")


