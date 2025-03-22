import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pytesseract

def analyze_letters(image_folder, output_csv, grid_size=3, visualize=False, 
                   recognize_letters=True, valid_letters="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    """
    Analyze letter images by dividing them into a grid and calculating pixel statistics.
    Optionally recognize the letter using OCR.
    
    Parameters:
    -----------
    image_folder : str
        Path to the folder containing letter images
    output_csv : str
        Path for the output CSV file
    grid_size : int
        Size of the grid (grid_size x grid_size)
    visualize : bool
        Whether to visualize the grid division for each image
    recognize_letters : bool
        Whether to use OCR to identify letters
    valid_letters : str
        String of valid characters to recognize
    """
    # Check if letter recognition is requested but not available
    if recognize_letters and not TESSERACT_AVAILABLE:
        recognize_letters = False
        print("Letter recognition disabled due to missing pytesseract.")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    
    # Create column headers
    column_names = ["Image"]
    if recognize_letters:
        column_names.append("Letter")
        
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
        
        # Recognize letter if enabled
        if recognize_letters:
            try:
                # Configure pytesseract for single character recognition with whitelist
                config = f"--psm 10 -c tessedit_char_whitelist={valid_letters}"
                letter = pytesseract.image_to_string(image, config=config).strip()
                
                # Validate the letter
                if len(letter) == 1 and letter in valid_letters:
                    row_data.append(letter)
                else:
                    print(f"Warning: {image_name} - Recognized '{letter}' not in valid letters.")
                    row_data.append(None)
            except Exception as e:
                print(f"OCR error for {image_name}: {e}")
                row_data.append(None)
        
        # Preprocessing: Apply Gaussian blur to reduce noise (optional)
        # image = cv2.GaussianBlur(image, (5, 5), 0)
        
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
            # Add recognized letter to visualization if available
            if recognize_letters and row_data[-1]:
                cv2.putText(vis_img, f"Letter: {row_data[-1]}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
    return df

def main():
    """
    Main function to run the letter analysis tool
    """
    print("=" * 50)
    print("Letter Image Grid Analysis Tool")
    print("=" * 50)
    
    # Get user input for parameters
    image_folder = "extracted_letters"
    if not os.path.exists(image_folder):
        print(f"Error: Folder '{image_folder}' does not exist!")
        return
    
    output_csv = "output.csv"
    
    try:
        grid_size = int(input("Enter grid size (e.g., 3 for 3x3, 4 for 4x4): ").strip() or "3")
        if grid_size <= 0:
            raise ValueError("Grid size must be positive")
    except ValueError:
        print("Invalid grid size. Using default (3x3).")
        grid_size = 3
    
    visualize = input("Visualize grid division? (y/n): ").strip().lower() == 'y'
    
    # OCR options
    recognize_letters = False
    if TESSERACT_AVAILABLE:
        recognize_letters = input("Use OCR to recognize letters? (y/n): ").strip().lower() == 'y'
        if recognize_letters:
            valid_letters = input("Enter valid letters to recognize (default: a-zA-Z): ").strip()
            if not valid_letters:
                valid_letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        else:
            valid_letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    else:
        valid_letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # Check if tesseract path needs to be set
    if recognize_letters:
            pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract.exe'
    
    # Run the analysis
    result_df = analyze_letters(
        image_folder, 
        output_csv, 
        grid_size, 
        visualize, 
        recognize_letters, 
        valid_letters
    )
    
    # Show a sample of the results
    print("\nSample of analysis results:")
    print(result_df.head())
    
    # Optional: Generate a simple visualization of the feature space
    if input("Generate feature visualization? (y/n): ").strip().lower() == 'y':
        try:
            plt.figure(figsize=(12, 8))
            
            # Create a heatmap of the black density values for the first few images
            subset = result_df.iloc[:min(10, len(result_df)), :].copy()
            
            # Extract only the density columns
            density_cols = [col for col in result_df.columns if 'Black_Density' in col and '_' in col][:grid_size*grid_size]
            
            # Create a pivot table for visualization
            heatmap_data = []
            for idx, row in subset.iterrows():
                image_name = row['Image']
                letter = row.get('Letter', 'Unknown')
                title = f"{image_name} ({letter})" if letter else image_name
                
                for col in density_cols:
                    _, r, c = col.split('_')
                    heatmap_data.append({
                        'Image': title,
                        'Row': int(r),
                        'Col': int(c),
                        'Density': row[col]
                    })
            
            heatmap_df = pd.DataFrame(heatmap_data)
            pivot_table = heatmap_df.pivot_table(index=['Image', 'Row'], columns='Col', values='Density')
            
            plt.figure(figsize=(15, 8))
            for i, img in enumerate(pivot_table.index.get_level_values(0).unique()):
                if i >= 10:  # Limit to 10 images
                    break
                plt.subplot(2, 5, i+1)
                img_data = pivot_table.loc[img]
                plt.imshow(img_data, cmap='Blues', interpolation='nearest')
                plt.title(img, fontsize=8)
                plt.colorbar(label='Black Density')
                plt.xticks(range(grid_size))
                plt.yticks(range(grid_size))
            
            plt.tight_layout()
            vis_filename = os.path.splitext(output_csv)[0] + "_visualization.png"
            plt.savefig(vis_filename)
            plt.show()
            print(f"Feature visualization saved as '{vis_filename}'")
        except Exception as e:
            print(f"Could not generate visualization: {e}")
    
if __name__ == "__main__":
    main()