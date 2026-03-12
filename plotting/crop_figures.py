"""
Script to crop client impact analysis figures.
Keeps only Accuracy Convergence (top-left) and Split Point Selection (bottom-right).
Removes Loss Minimization and RL Reward Signal plots.
"""

import os
from PIL import Image
import glob

# Source and destination directories
source_dir = r"c:\Users\nshadin\OneDrive - Kennesaw State University\QSplitFL\complete_rl_implementation\Paper_Figures"
dest_dir = r"c:\Users\nshadin\OneDrive - Kennesaw State University\QSplitFL\complete_rl_implementation\Paper_Figures_Cropped"

# Create destination directory
os.makedirs(dest_dir, exist_ok=True)

# Get all client comparison figures (not model architecture comparison)
patterns = [
    "*_CNN_comparison_by_clients_*.png",
    "*_ResNet50_comparison_by_clients_*.png",
    "*_MobileNetV4_comparison_by_clients_*.png",
    "*_ConvNeXt_comparison_by_clients_*.png"
]

files_to_process = []
for pattern in patterns:
    files_to_process.extend(glob.glob(os.path.join(source_dir, pattern)))

print(f"Found {len(files_to_process)} files to process")

for filepath in files_to_process:
    filename = os.path.basename(filepath)
    print(f"Processing: {filename}")
    
    # Open the image
    img = Image.open(filepath)
    width, height = img.size
    
    # The image has 4 subplots in 2x2 grid:
    # Top-left (Accuracy): roughly left half, top half
    # Top-right (Loss): roughly right half, top half  
    # Bottom-left (Reward): roughly left half, bottom half
    # Bottom-right (Split Point): roughly right half, bottom half
    
    # We want: Top-left (Accuracy) and Bottom-right (Split Point)
    # Calculate crop regions
    mid_x = width // 2
    mid_y = height // 2
    
    # Add some padding at top for title
    title_height = int(height * 0.06)  # Approximate title area
    
    # Crop Accuracy (top-left) - include title
    accuracy_box = (0, 0, mid_x, mid_y + int(height * 0.02))
    
    # Crop Split Point (bottom-right)
    split_point_box = (mid_x, mid_y - int(height * 0.02), width, height)
    
    # Extract subplots
    accuracy_img = img.crop(accuracy_box)
    split_point_img = img.crop(split_point_box)
    
    # Create new combined image - side by side
    new_width = accuracy_img.width + split_point_img.width
    new_height = max(accuracy_img.height, split_point_img.height)
    
    combined_img = Image.new('RGB', (new_width, new_height), 'white')
    combined_img.paste(accuracy_img, (0, 0))
    combined_img.paste(split_point_img, (accuracy_img.width, 0))
    
    # Save cropped image
    output_path = os.path.join(dest_dir, filename)
    combined_img.save(output_path, 'PNG', dpi=(300, 300))
    print(f"  Saved: {output_path}")

print(f"\nDone! Processed {len(files_to_process)} files.")
print(f"Cropped figures saved to: {dest_dir}")
