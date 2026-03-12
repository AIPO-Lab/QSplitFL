import os
import shutil
import glob

# Configuration
SOURCE_DIRS = [
    "CIFAR-100_Results",
    "CIFAR10_Results",
    "FMNIST_Results",
    "MNIST_Results"
]
DEST_DIR = "Organized_Analysis_Results"

# Mapping keywords to folder names
CATEGORY_MAP = {
    "scalability": "Scalability_Analysis",
    "convergence": "Convergence_Analysis",
    "model_architecture": "Model_Architecture_Comparison",
    "comparison_by_clients": "Client_Impact_Analysis",
    "comparison_by_rounds": "Round_Impact_Analysis",
    
    # Additional patterns found in Uncategorized
    "_plot_clients": "Client_Impact_Analysis",  # Catches patterns like 'mnist_cnn_plot_clients100_rounds10.png'
    "accuracy_over_time": "Convergence_Analysis",
    "loss_over_time": "Convergence_Analysis",
    "reward_over_time": "Convergence_Analysis",
    "split_points": "Split_Layer_Analysis",
    "split_layer_analysis": "Split_Layer_Analysis",
    "comprehensive_analysis": "Comprehensive_Analysis",
    "exploration_rate": "Convergence_Analysis" # Epsilon decay is related to convergence/training progress
}

def organize_files():
    # Create destination directory
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        print(f"Created root directory: {DEST_DIR}")

    # Create subdirectories
    for subfolder in CATEGORY_MAP.values():
        path = os.path.join(DEST_DIR, subfolder)
        if not os.path.exists(path):
            os.makedirs(path)

    count = 0
    # Iterate through source directories
    for source_dir in SOURCE_DIRS:
        if not os.path.exists(source_dir):
            print(f"Warning: Source directory '{source_dir}' not found. Skipping.")
            continue

        # Find all PNG files
        png_files = glob.glob(os.path.join(source_dir, "*.png"))
        
        for file_path in png_files:
            filename = os.path.basename(file_path)
            
            # Determine category
            target_subfolder = "Uncategorized"
            for keyword, folder in CATEGORY_MAP.items():
                if keyword in filename:
                    target_subfolder = folder
                    break
            
            # Create target path
            dest_folder = os.path.join(DEST_DIR, target_subfolder)
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
                
            dest_path = os.path.join(dest_folder, filename)
            
            # Copy file
            try:
                shutil.copy2(file_path, dest_path)
                print(f"Copied: {filename} -> {target_subfolder}/")
                count += 1
            except Exception as e:
                print(f"Error copying {filename}: {e}")

    print(f"\nSuccess! Organized {count} files into '{DEST_DIR}'.")

if __name__ == "__main__":
    organize_files()
