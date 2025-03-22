import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Basic configuration
image_folder = "extracted_letters"
output_csv = "output.csv"
grid_size = 3  # Default grid size
visualize = False  # Default visualization setting

# Get user input for parameters
print("=" * 50)
print("Letter Image Grid Analysis Tool")
print("=" * 50)

if not os.path.exists(image_folder):
    print(f"Error: Folder '{image_folder}' does not exist!")
    exit()

try:
    grid_size = int(input("Enter grid size (e.g., 3 for 3x3, 4 for 4x4): ").strip() or "3")
    if grid_size <= 0:
        raise ValueError("Grid size must be positive")
except ValueError:
    print("Invalid grid size. Using default (3x3).")
    grid_size = 3

visualize = input("Visualize grid division? (y/n): ").strip().lower() == 'y'

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)

# Create column headers
column_names = ["Image"]
    
for row in range(grid_size):
    for col in range(grid_size):
        column_names.append(f"Black_{row}_{col}")
        column_names.append(f"White_{row}_{col}")
        column_names.append(f"Black_Density_{row}_{col}")
        column_names.append(f"White_Density_{row}_{col}")

# Add global statistics columns
column_names.extend([
    "Total_Black_Pixels", 
    "Total_White_Pixels",
    "Total_Black_Density", 
    "Total_White_Density",
    "Letter_Height",
    "Letter_Width",
    "Aspect_Ratio"
])

# Results list
results = []

# Get list of image files
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

# Process each image with a progress bar
for image_name in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(image_folder, image_name)
    
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not load {image_name}. Skipping...")
        continue

    # Initialize data for this image
    row_data = [image_name]
    
    # Convert to binary using Otsu's method for adaptive thresholding
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Get image dimensions
    h, w = binary_image.shape
    cell_h, cell_w = h // grid_size, w // grid_size

    total_black_pixels = 0
    total_white_pixels = 0

    # Create visualization if enabled
    if visualize:
        vis_img = cv2.cvtColor(binary_image.copy(), cv2.COLOR_GRAY2BGR)

    # Process each cell in the grid
    for row in range(grid_size):
        for col in range(grid_size):
            # Define cell boundaries
            top = row * cell_h
            bottom = (row + 1) * cell_h if row < grid_size - 1 else h
            left = col * cell_w
            right = (col + 1) * cell_w if col < grid_size - 1 else w
            
            # Extract the cell
            cell = binary_image[top:bottom, left:right]
            
            # Draw grid lines for visualization
            if visualize:
                cv2.rectangle(vis_img, (left, top), (right, bottom), (0, 255, 0), 1)
            
            # Count pixels (in binary_inv, white pixels are the foreground/black parts of the letter)
            black_pixels = np.count_nonzero(cell == 255)  
            white_pixels = np.count_nonzero(cell == 0)
            total_pixels = cell.size
            
            # Calculate densities
            black_density = black_pixels / total_pixels
            white_density = white_pixels / total_pixels
            
            # Add to row data
            row_data.append(black_pixels)
            row_data.append(white_pixels)
            row_data.append(black_density)
            row_data.append(white_density)
            
            # Add to totals
            total_black_pixels += black_pixels
            total_white_pixels += white_pixels

    # Calculate global statistics
    total_pixels = h * w
    total_black_density = total_black_pixels / total_pixels
    total_white_density = total_white_pixels / total_pixels
    aspect_ratio = w / h if h > 0 else 0

    # Add global statistics to row data
    row_data.extend([
        total_black_pixels,
        total_white_pixels,
        total_black_density,
        total_white_density,
        h,
        w,
        aspect_ratio
    ])
    
    # Add row to results
    results.append(row_data)
    
    # Show visualization if enabled
    if visualize:
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Grid Analysis of {image_name}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# Convert to DataFrame and save to CSV
df = pd.DataFrame(results, columns=column_names)
df.to_csv(output_csv, index=False, float_format="%.4f")

print(f"Analysis complete! Results saved to {output_csv}")

# Show a sample of the results
print("\nSample of analysis results:")
print(df)