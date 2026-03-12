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
        self.local_data = data.DataLoader(dataset, batch_size=128, shuffle=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResNetFed().to(self.device)
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

    def train_split(self, epochs=5):
        """
        Train the client-side model for split learning.
        
        Returns:
            dict: Training metrics and model state
        """
        if self.client_side_model is None:
            raise ValueError("Client-side model must be set before split training")
            
        self.client_side_model.train()
        epoch_losses = []
        epoch_accuracies = []
        
        for epoch in range(epochs):
            running_loss = 0.0
            total, correct = 0, 0
            
            for images, labels in self.local_data:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                
                # Forward pass through client-side model
                client_activations = self.client_side_model(images)
                
                # For training, we need server-side gradients
                # In practice, this would be received from the server
                # For simulation, we'll use a simple proxy loss
                outputs = self._simulate_server_side(client_activations)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            
            epoch_loss = running_loss / len(self.local_data)
            epoch_accuracy = correct / total
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)
            
            print(f"Client ID: {self.client_id}, Split Training Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
            self.scheduler.step()
        
        return {
            'model_state': self.client_side_model.state_dict(),
            'losses': epoch_losses,
            'accuracies': epoch_accuracies,
            'final_loss': epoch_losses[-1] if epoch_losses else 0.0,
            'final_accuracy': epoch_accuracies[-1] if epoch_accuracies else 0.0
        }
    
    def _simulate_server_side(self, client_activations):
        """
        Simulate server-side forward pass for training.
        In practice, this would be done on the server.
        """
        # Simple proxy for server-side computation
        # In real implementation, this would be replaced with actual server-side model
        return torch.mean(client_activations.view(client_activations.size(0), -1), dim=1, keepdim=True).expand(-1, 10)
    
    def get_client_activations(self):
        """
        Get activations from client-side model for server processing.
        
        Returns:
            activations: Client-side model activations
            labels: Corresponding labels
        """
        if self.client_side_model is None:
            raise ValueError("Client-side model must be set before getting activations")
            
        self.client_side_model.eval()
        activations_list = []
        labels_list = []
        
        with torch.no_grad():
            for images, labels in self.local_data:
                images = images.to(self.device)
                activations = self.client_side_model(images)
                activations_list.append(activations.cpu())
                labels_list.append(labels)
        
        return torch.cat(activations_list, dim=0), torch.cat(labels_list, dim=0)

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
