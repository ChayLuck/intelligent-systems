import os
import cv2
import numpy as np
import pandas as pd

# Input folder and output CSV file
image_folder = "extracted_letters"
output_csv = "output.csv"

# User input for grid size
grid_size = int(input("Enter grid size (e.g., 3 for 3x3): "))

# Column headers
column_names = ["Image"]
for row in range(grid_size):
    for col in range(grid_size):
        column_names.append(f"White_{row}_{col}")
        column_names.append(f"Black_{row}_{col}")
        column_names.append(f"White_Density_{row}_{col}")
        column_names.append(f"Black_Density_{row}_{col}")
column_names.append("Total_White_Pixels")
column_names.append("Total_Black_Pixels")
column_names.append("Total_White_Density")
column_names.append("Total_Black_Density")

# Results list
results = []

# Process each image in the folder
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: {image_name} is not loaded!")
        continue

    # Convert to binary
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    h, w = binary_image.shape
    cell_h, cell_w = h // grid_size, w // grid_size

    row_data = [image_name]
    total_white_pixels = 0
    total_black_pixels = 0

    # Process each cell
    for row in range(grid_size):
        for col in range(grid_size):
            cell = binary_image[row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w]
            white_pixels = np.count_nonzero(cell == 255)
            black_pixels = np.count_nonzero(cell == 0)
            total_pixels = cell_h * cell_w
            white_density = white_pixels / total_pixels
            black_density = black_pixels / total_pixels
            
            row_data.append(white_pixels)
            row_data.append(black_pixels)
            row_data.append(white_density)
            row_data.append(black_density)
            
            total_white_pixels += white_pixels
            total_black_pixels += black_pixels

    total_pixels = h * w
    total_white_density = total_white_pixels / total_pixels
    total_black_density = total_black_pixels / total_pixels

    row_data.append(total_white_pixels)
    row_data.append(total_black_pixels)
    row_data.append(total_white_density)
    row_data.append(total_black_density)
    results.append(row_data)

# Save to CSV
df = pd.DataFrame(results, columns=column_names)
df.to_csv(output_csv, index=False, float_format="%.2f")

print(f"Progress complete! The results are saved in {output_csv} file.")
