"""
Update server.py to use generic splitting logic suitable for all models.
Supports ResNet50, MobileNetV4, ConvNeXt (via is_client_layer)
and legacy CNNs (via fallback).
"""

import os
import re

datasets = ['MNIST', 'FMNIST', 'CIFAR10', 'CIFAR-100']

print("=" * 80)
print("UPDATING SERVER.PY FOR GENERIC MODEL SPLITTING")
print("=" * 80)

for dataset in datasets:
    print(f"\n[{dataset}] Updating server.py...")
    
    server_file = f"./{dataset}/server.py"
    
    if not os.path.exists(server_file):
        print(f"  ✗ File not found")
        continue
    
    with open(server_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Rename _is_client_layer to _is_client_layer_legacy
    if 'def _is_client_layer(self, layer_name):' in content:
        content = content.replace(
            'def _is_client_layer(self, layer_name):',
            'def _is_client_layer_legacy(self, layer_name):'
        )
        print("  ✓ Renamed _is_client_layer to _is_client_layer_legacy")
    
    # 2. Add new _is_client_layer method that dispatches
    new_method = '''    def _is_client_layer(self, layer_name):
        """Determine if a layer belongs to the client side based on split layer."""
        # Use model's own logic if available (ResNet50, MobileNetV4, ConvNeXt)
        if hasattr(self.global_model, 'is_client_layer'):
            return self.global_model.is_client_layer(layer_name, self.split_layer)
            
        # Fallback to legacy logic for SimpleMNISTCNN/ResNetFed
        return self._is_client_layer_legacy(layer_name)
'''
    
    # Insert new method before legacy method
    if 'def _is_client_layer_legacy' in content and 'def _is_client_layer(' not in content:
        content = content.replace(
            'def _is_client_layer_legacy',
            new_method + '\n    def _is_client_layer_legacy'
        )
        print("  ✓ Added generic _is_client_layer dispatcher")
        
    # 3. Fix create_split_models to use dynamic types and full state dict loading
    # We remove the manual filtering loop and load full dicts, 
    # BUT wait - load_state_dict will fail if we load full dict into partial model?
    # No, we agreed to load FULL model into client_model and server_model.
    
    # We need to replace the entire create_split_models method
    # It's safer to use regex to find the method body
    
    start_marker = 'def create_split_models(self, base_model):'
    end_marker = 'def _is_client_layer'
    
    if start_marker in content:
        # Construct new create_split_models
        new_create_split = '''    def create_split_models(self, base_model):
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
    
'''
        # Replace the method using regex matching everything from start to next def
        pattern = r'    def create_split_models\(self, base_model\):.*?    def '
        
        # This regex is tricky because of indentation and newlines. 
        # Better approach: Find start index and end index
        start_idx = content.find(start_marker)
        if start_idx != -1:
            # Find the NEXT method start
            next_method_idx = content.find('    def ', start_idx + len(start_marker))
            
            if next_method_idx != -1:
                # Replace content
                content = content[:start_idx] + new_create_split + content[next_method_idx:]
                print("  ✓ Replaced create_split_models method")
            else:
                print("  ⚠ Could not find end of create_split_models")
    
    
    # 4. Fix aggregate_split_models to use _is_client_layer properly
    # The current aggregation logic averages client models.
    # Since client_models are now FULL models (but only trained on client part),
    # we must ensure we ONLY copy the client-part weights to the global model.
    # The current code iterates all keys. WE MUST FILTER KEYS.
    
    start_agg = 'def aggregate_split_models(self, client_models):'
    if start_agg in content:
        new_agg = '''    def aggregate_split_models(self, client_models):
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
    
'''
        # Find start and end
        start_idx = content.find(start_agg)
        if start_idx != -1:
            next_method_idx = content.find('    def ', start_idx + len(start_agg))
            if next_method_idx != -1:
                content = content[:start_idx] + new_agg + content[next_method_idx:]
                print("  ✓ Replaced aggregate_split_models method")
                
    # Write back
    with open(server_file, 'w', encoding='utf-8') as f:
        f.write(content)

print("\n" + "=" * 80)
print("✓ SERVER.PY UPDATES COMPLETE")
print("=" * 80)
