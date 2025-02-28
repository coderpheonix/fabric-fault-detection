"""
Script to Rename Images in Dataset Folders

Renames images inside each defect category folder as:
Defect_Free_1, Defect_Free_2, ..., Holes_1, Holes_2, ..., etc.

Author: Shahidul Islam (Sawon)
"""

import os

# Define dataset folders
dataset_path = r"C:\study\Research\Image for research"
categories = ["Holes", "Stains", "Vertical_Defects", "Defect_Free"]

# Iterate through each category folder
for category in categories:
    folder = os.path.join(dataset_path, category)

    if not os.path.exists(folder):
        print(f"Warning: Folder '{folder}' does not exist. Skipping...")
        continue

    # Get list of image files
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()  # Sort to maintain order

    # Rename images with simple indexing format: Category_1, Category_2, ...
    for index, filename in enumerate(image_files, start=1):
        ext = os.path.splitext(filename)[1]  # Get file extension
        new_name = f"{category}_{index}{ext}"  # Example: Holes_1.png
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_name)

        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")

print("Image renaming completed successfully!")
