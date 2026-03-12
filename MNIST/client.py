import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from models import CNNModel, ResNetFed, SimpleMNISTCNN, SimpleNN


class Client:
    """Federated Learning Client with Split Learning Support."""
    def __init__(self, client_id, dataset, num_classes=10):
        self.client_id = client_id
        self.local_data = data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0, persistent_workers=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleMNISTCNN(num_classes=num_classes).to(self.device)
        #self.model = SimpleNN()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        
        # Split learning attributes
        self.client_side_model = None
        self.split_layer = None

    def set_split_model(self, client_side_model, split_layer):
        """Set the client-side model for split learning."""
        self.client_side_model = client_side_model.to(self.device)
        self.split_layer = split_layer
        # Update optimizer to only train client-side parameters
        self.optimizer = optim.SGD(
            self.client_side_model.parameters(),
            lr=0.1, momentum=0.9, weight_decay=5e-4
        )

    def forward_split(self, images):
        """
        Forward pass through client-side model up to split layer.
        
        Args:
            images: Input images
            
        Returns:
            activations: Smashed data (activations at split layer)
        """
        if self.client_side_model is None:
            raise ValueError("Client-side model must be set before forward pass")
            
        self.client_side_model.train()
        images = images.to(self.device)
        
        # Forward pass
        activations = self.client_side_model(images)
        
        # Detach from graph to send to server (we'll attach gradients later)
        # But keep require_grad=True for the tensor we return so we can backward into it
        return activations.clone().detach().requires_grad_(True)

    def backward_split(self, activations, gradients):
        """
        Backward pass through client-side model using gradients from server.
        
        Args:
            activations: The activations that were sent to server (with graph attached)
            gradients: Gradients for the activations received from server
        """
        if self.client_side_model is None:
            raise ValueError("Client-side model must be set before backward pass")
            
        self.optimizer.zero_grad()
        
        # We need to re-forward or use hooks, but since we detached, we can't just .backward()
        # The standard way in PyTorch split learning is:
        # 1. Client forwards -> gets output (O)
        # 2. Sends O.detach() to server
        # 3. Server forwards, calculates loss, backwards -> gets dL/dO
        # 4. Server sends dL/dO to client
        # 5. Client computes O.backward(dL/dO)
        
        # However, we need the graph. 
        # In forward_split, we returned a detached tensor.
        # But we need to keep the graph for the backward pass.
        # So we should store the output with graph in the client state.
        
        # Let's fix forward_split to store the graph-connected output
        pass # This method is implemented via the state stored in forward_split logic below

    def train_step_split(self, images, server_grads=None):
        """
        Execute one step of split training.
        
        Phase 1 (Forward): If server_grads is None
            - Forward images -> activations
            - Store activations with graph
            - Return detached activations for server
            
        Phase 2 (Backward): If server_grads is provided
            - Use stored activations
            - Backward with server_grads
            - Optimizer step
        """
        if self.client_side_model is None:
            raise ValueError("Client-side model must be set")
            
        if server_grads is None:
            # Phase 1: Forward
            self.client_side_model.train()
            images = images.to(self.device)
            self.optimizer.zero_grad()
            
            self.current_activations = self.client_side_model(images)
            return self.current_activations.clone().detach().requires_grad_(True)
        else:
            # Phase 2: Backward
            if not hasattr(self, 'current_activations'):
                raise ValueError("No activations stored. Run forward phase first.")
                
            # Backpropagate gradients from server
            self.current_activations.backward(server_grads.to(self.device))
            self.optimizer.step()
            
            # Cleanup
            del self.current_activations
            return None

    def train(self, epochs=5):
        """Train the client's model locally (legacy method)."""
        self.model.train()
        epoch_losses = []
        epoch_accuracies = []
        
        for epoch in range(epochs):
            runningLoss = 0.0
            total, correct = 0, 0
            for images, labels in self.local_data:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                runningLoss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            
            epoch_loss = runningLoss / len(self.local_data)
            epoch_accuracy = correct / total
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)
            
            # Print detailed training info for each client to show local metrics
            print(f"Client ID: {self.client_id}, Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
            self.scheduler.step()

        # Return model state dict along with training metrics
        return {
            'model_state': self.model.state_dict(),
            'losses': epoch_losses,
            'accuracies': epoch_accuracies,
            'final_loss': epoch_losses[-1] if epoch_losses else 0.0,
            'final_accuracy': epoch_accuracies[-1] if epoch_accuracies else 0.0
        }

    def get_class_distribution(self):
        """Get class distribution in client dataset."""
        # Extract labels from the dataset
        labels = []
        for _, label in self.local_data.dataset:
            labels.append(label)
        
        # Convert to tensor and compute class counts
        labels_tensor = torch.tensor(labels)
        class_counts = torch.bincount(labels_tensor, minlength=10)  # Ensure 10 classes
        
        # Return normalized distribution
        return class_counts.float() / class_counts.sum()
