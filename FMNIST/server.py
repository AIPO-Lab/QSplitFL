import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from models import CNNModel, ResNetFed, SimpleMNISTCNN, SimpleNN


class Server:
    """Federated Learning Server with Split Learning Support."""
    def __init__(self, num_classes, model_instance, test_dataset=None):
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = model_instance.to(self.device)
        #self.global_model = SimpleNN().to(self.device)
        self.test_loader = data.DataLoader(test_dataset,  batch_size=128, shuffle=False, num_workers=0, persistent_workers=False)
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
        Both models are full instances but configured to execute partial forward passes.
        """
        if self.split_layer is None:
            raise ValueError("Split layer must be set before creating split models")
            
        # Create full instances for client and server models
        # Use the exact same class as global_model
        model_class = type(self.global_model)
        
        # Determine instantiation args
        # Check if model takes input_channels (new models) or not (legacy)
        try:
            if hasattr(self.global_model, 'conv1') and hasattr(self.global_model.conv1, 'in_channels'):
                 input_channels = self.global_model.conv1.in_channels
            elif hasattr(self.global_model, 'features') and hasattr(self.global_model.features[0], 'in_channels'):
                 input_channels = self.global_model.features[0].in_channels # MobileNet
            elif hasattr(self.global_model, 'downsample_layers'):
                 # ConvNeXt input channels hard to get from simple attribute check easily, assume standard
                 # Actually we can try-except
                 input_channels = 3 if self.global_model.__class__.__name__ != 'SimpleMNISTCNN' else 1
            else:
                 input_channels = 1 if 'MNIST' in self.global_model.__class__.__name__ else 3

            client_model = model_class(num_classes=self.num_classes, input_channels=input_channels).to(self.device)
            server_model = model_class(num_classes=self.num_classes, input_channels=input_channels).to(self.device)
        except TypeError:
            # Fallback for models not accepting input_channels
            client_model = model_class(num_classes=self.num_classes).to(self.device)
            server_model = model_class(num_classes=self.num_classes).to(self.device)

        # Load FULL state dict into both
        # This works because both are full models
        full_state = base_model.state_dict()
        client_model.load_state_dict(full_state)
        server_model.load_state_dict(full_state)
        
        # Configure split modes
        if hasattr(client_model, 'configure_split'):
            client_model.configure_split('client', self.split_layer)
            server_model.configure_split('server', self.split_layer)
        
        self.client_models = client_model
        self.server_side_model = server_model
        
        return client_model, server_model
    
    def _is_client_layer(self, layer_name):
        """Determine if a layer belongs to the client side based on split layer."""
        # Use model's own logic if available (ResNet50, MobileNetV4, ConvNeXt)
        if hasattr(self.global_model, 'is_client_layer'):
            return self.global_model.is_client_layer(layer_name, self.split_layer)
            
        # Fallback to legacy logic for SimpleMNISTCNN/ResNetFed
        return self._is_client_layer_legacy(layer_name)

    def _is_client_layer_legacy(self, layer_name):
        """Determine if a layer belongs to the client side based on split layer."""
        # Map split layers to SimpleMNISTCNN components (10-layer model)
        # Split 1: conv1, bn1
        # Split 2: conv1, bn1
        # Split 3: conv1, bn1, conv2, bn2
        # Split 4: conv1, bn1, conv2, bn2
        # Split 5: conv1, bn1, conv2, bn2, conv3, bn3
        # Split 6: all conv + fc1
        # Split 7: all conv + fc1, fc2
        # Split 8: all conv + fc1, fc2, fc3
        # Split 9: all conv + fc1, fc2, fc3, fc4
        
        if self.split_layer <= 2:
            client_patterns = ['conv1', 'bn1']
        elif self.split_layer <= 4:
            client_patterns = ['conv1', 'bn1', 'conv2', 'bn2']
        elif self.split_layer == 5:
            client_patterns = ['conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3']
        elif self.split_layer == 6:
            client_patterns = ['conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3', 'fc1']
        elif self.split_layer == 7:
            client_patterns = ['conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3', 'fc1', 'fc2']
        elif self.split_layer == 8:
            client_patterns = ['conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3', 'fc1', 'fc2', 'fc3']
        elif self.split_layer == 9:
            client_patterns = ['conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3', 'fc1', 'fc2', 'fc3', 'fc4']
        else:
            # Default or full
            client_patterns = ['conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3', 'fc1', 'fc2', 'fc3', 'fc4', 'fc5']
        
        return any(pattern in layer_name for pattern in client_patterns)
    
    def aggregate_split_models(self, client_models):
        """
        Aggregate client-side models using FedAvg for split learning.
        Only aggregates parameters belonging to the client side.
        """
        if self.client_models is None:
            raise ValueError("Split models must be created before aggregation")
            
        # Update global model with aggregated client parameters
        global_dict = self.global_model.state_dict()
        server_dict = self.server_side_model.state_dict()
        
        # Collect client states
        client_states = [model.state_dict() for model in client_models]
        
        for key in global_dict.keys():
            # Check if this parameter belongs to client side
            if self._is_client_layer(key):
                # Aggregate from clients
                params = [state[key].float() for state in client_states]
                global_dict[key] = torch.stack(params, 0).mean(0)
            else:
                # Update from server-side model (which was trained on server)
                global_dict[key] = server_dict[key]

        self.global_model.load_state_dict(global_dict)
        
        # Sync the reference models for next round
        self.client_models.load_state_dict(global_dict)
        self.server_side_model.load_state_dict(global_dict)
    
    def train_split_round(self, client_models, selected_clients, epochs=2):
        """
        Execute a complete split learning round with real gradient exchange.
        
        Args:
            client_models: List of client-side models
            selected_clients: List of selected client objects
            epochs: Number of training epochs per client
            
        Returns:
            dict: Round metrics including client losses and accuracies
        """
        if self.client_models is None or self.server_side_model is None:
            raise ValueError("Split models must be created before training round")
        
        # Optimizer for server-side model
        server_optimizer = torch.optim.SGD(
            self.server_side_model.parameters(), 
            lr=0.1, momentum=0.9, weight_decay=5e-4
        )
        
        # Track metrics
        all_client_losses = []
        all_client_accuracies = []
        server_losses = []
        
        self.server_side_model.train()
        
        # Train each client sequentially (or parallel in real world, but sequential here for simulation)
        for i, client in enumerate(selected_clients):
            print(f"Training Client {client.client_id}...")
            client_epoch_losses = []
            client_epoch_accs = []
            
            for epoch in range(epochs):
                running_loss = 0.0
                correct = 0
                total = 0
                
                # Iterate through client's data
                for images, labels in client.local_data:
                    labels = labels.to(self.device)
                    
                    # --- Step 1: Client Forward ---
                    # Client computes activations up to split layer
                    client_activations = client.train_step_split(images, server_grads=None)
                    client_activations = client_activations.to(self.device)
                    
                    # --- Step 2: Server Forward ---
                    # Server continues forward pass
                    server_optimizer.zero_grad()
                    outputs = self.server_side_model(client_activations)
                    loss = self.criterion(outputs, labels)
                    
                    # --- Step 3: Server Backward ---
                    # Backpropagate on server to get gradients for activations
                    loss.backward()
                    
                    # Get gradients w.r.t. client activations
                    activation_grads = client_activations.grad.clone()
                    
                    # Update server weights
                    server_optimizer.step()
                    
                    # --- Step 4: Client Backward ---
                    # Send gradients back to client for their update
                    client.train_step_split(images, server_grads=activation_grads)
                    
                    # Metrics
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                
                # Epoch metrics
                epoch_loss = running_loss / len(client.local_data)
                epoch_acc = correct / total
                client_epoch_losses.append(epoch_loss)
                client_epoch_accs.append(epoch_acc)
                print(f"    [Client {client.client_id}] Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc*100:.2f}%")
            
            # Store final metrics for this client
            all_client_losses.append(client_epoch_losses[-1])
            all_client_accuracies.append(client_epoch_accs[-1])
            server_losses.append(client_epoch_losses[-1]) # Server loss is the same
            
            # Clean up client state
            if hasattr(client, 'current_activations'):
                del client.current_activations
        
        # Step 5: Aggregate client-side models
        self.aggregate_split_models(client_models)
        
        return {
            'client_losses': all_client_losses,
            'client_accuracies': all_client_accuracies,
            'server_losses': server_losses,
            'avg_client_loss': np.mean(all_client_losses),
            'avg_client_accuracy': np.mean(all_client_accuracies),
            'avg_server_loss': np.mean(server_losses)
        }
    
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
            self.test_loader = data.DataLoader(test_dataset,  batch_size=128, shuffle=False, num_workers=0, persistent_workers=False)
            
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
