"""
Download Fashion MNIST using TensorFlow and convert to PyTorch format
"""

import tensorflow as tf
import numpy as np
import os
import struct

print("Downloading Fashion MNIST using TensorFlow...")

# Load the Fashion MNIST dataset using TensorFlow
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

print(f"Training images shape: {train_images.shape}")
print(f"Testing images shape: {test_images.shape}")

# Create directory structure
data_dir = "./data/FashionMNIST/raw"
os.makedirs(data_dir, exist_ok=True)

print(f"\nSaving to PyTorch format in: {data_dir}")

def save_idx_file(filename, data, is_labels=False):
    """Save data in IDX format (MNIST/Fashion-MNIST format)"""
    filepath = os.path.join(data_dir, filename)
    
    with open(filepath, 'wb') as f:
        if is_labels:
            # Labels: magic number (2049), number of items
            magic = 2049
            f.write(struct.pack('>I', magic))
            f.write(struct.pack('>I', len(data)))
            f.write(data.astype(np.uint8).tobytes())
        else:
            # Images: magic number (2051), number of images, rows, cols
            magic = 2051
            f.write(struct.pack('>I', magic))
            f.write(struct.pack('>I', data.shape[0]))
            f.write(struct.pack('>I', data.shape[1]))
            f.write(struct.pack('>I', data.shape[2]))
            f.write(data.astype(np.uint8).tobytes())
    
    print(f"✓ Saved {filename}")

# Save training data
save_idx_file('train-images-idx3-ubyte', train_images, is_labels=False)
save_idx_file('train-labels-idx1-ubyte', train_labels, is_labels=True)

# Save test data
save_idx_file('t10k-images-idx3-ubyte', test_images, is_labels=False)
save_idx_file('t10k-labels-idx1-ubyte', test_labels, is_labels=True)

print("\n" + "="*60)
print("✅ Fashion MNIST dataset successfully downloaded and saved!")
print("="*60)
print("\nYou can now run: python main_complete_rl.py")
