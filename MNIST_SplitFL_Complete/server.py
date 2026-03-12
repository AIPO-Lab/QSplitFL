import torch
import torch.nn as nn
import torch.utils.data as data
from models import CNNModel, ResNetFed, SimpleMNISTCNN, SimpleNN


class Server:
    """Federated Learning Server with Split Learning Support."""
    def __init__(self, test_dataset, num_classes=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = ResNetFed().to(self.device)
        #self.global_model = SimpleNN().to(self.device)
        self.test_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=False)
        self.criterion = nn.CrossEntropyLoss()
        
        # Split learning parameters
        self.split_layer = None
        self.client_models = None
        self.server_side_model = None

    def set_split_layer(self, split_layer):
        """Set the split layer for split learning."""
        self.split_layer = split_layer
        
    def create_split_models(self, base_model):
        """
        Create client-side and server-side models based on split layer.
        
        Args:
            split_layer: Layer index where to split the model
        """
        if self.split_layer is None:
            raise ValueError("Split layer must be set before creating split models")
            
        # Get model architecture
        model_dict = base_model.state_dict()
        
        # Create client-side model (up to split layer)
        client_model = ResNetFed().to(self.device)
        client_dict = client_model.state_dict()
        
        # Create server-side model (from split layer onwards)
        server_model = ResNetFed().to(self.device)
        server_dict = server_model.state_dict()
        
        # Split the model parameters
        for key in model_dict.keys():
            if self._is_client_layer(key):
                client_dict[key] = model_dict[key]
            else:
                server_dict[key] = model_dict[key]
        
        client_model.load_state_dict(client_dict)
        server_model.load_state_dict(server_dict)
        
        self.client_models = client_model
        self.server_side_model = server_model
        
        return client_model, server_model
    
    def _is_client_layer(self, layer_name):
        """Determine if a layer belongs to the client side based on split layer."""
        # This is a simplified implementation
        # In practice, you'd need to map layer indices to actual layer names
        # For ResNet, we can approximate based on layer naming patterns
        client_patterns = ['conv1', 'layer1', 'layer2']
        
        if self.split_layer <= 2:
            return any(pattern in layer_name for pattern in client_patterns[:1])
        elif self.split_layer <= 4:
            return any(pattern in layer_name for pattern in client_patterns[:2])
        else:
            return any(pattern in layer_name for pattern in client_patterns[:3])
    
    def aggregate_split_models(self, client_models):
        """
        Aggregate client-side models using FedAvg for split learning.
        
        Args:
            client_models: List of client-side models
        """
        if self.client_models is None:
            raise ValueError("Split models must be created before aggregation")
            
        # Aggregate client-side models
        global_dict = self.client_models.state_dict()
        for key in global_dict.keys():
            # Stack client model parameters and compute mean
            client_params = []
            for model in client_models:
                client_params.append(model.state_dict()[key].float())
            global_dict[key] = torch.stack(client_params, 0).mean(0)
        
        self.client_models.load_state_dict(global_dict)
        
        # Update the global model with aggregated client-side parameters
        global_full_dict = self.global_model.state_dict()
        for key in global_dict.keys():
            global_full_dict[key] = global_dict[key]
        
        self.global_model.load_state_dict(global_full_dict)
    
    def forward_server_side(self, client_activations):
        """
        Forward pass through server-side model with client activations.
        
        Args:
            client_activations: Activations from client-side model
            
        Returns:
            outputs: Model predictions
        """
        if self.server_side_model is None:
            raise ValueError("Server-side model must be created before forward pass")
            
        return self.server_side_model(client_activations)

    def aggregate_models(self, client_models):
        """Aggregate client models using FedAvg (legacy method)."""
        global_dict = self.global_model.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.stack([client_models[i][key].float() for i in range(len(client_models))], 0).mean(0)
        self.global_model.load_state_dict(global_dict)

    def evaluate(self, test_dataset=None):
        """Evaluate the global model on the test dataset."""
        if test_dataset is not None:
            self.test_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=False)
            
        self.global_model.eval()
        correct, total = 0, 0
        total_loss = 0.0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.global_model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total
        avg_loss = total_loss / len(self.test_loader)
        return accuracy, avg_loss
