import os
import cv2

# Define dataset folders
dataset_path = r"C:\study\Research\Image for research"
categories = ["Holes", "Stains", "Vertical_Defects", "Defect_Free"]
output_size = (256, 256)  # Resize images to 256x256

for category in categories:
    folder = os.path.join(dataset_path, category)

    # Check if folder exists
    if not os.path.exists(folder):
        print(f"Warning: Folder '{folder}' does not exist. Skipping...")
        continue

    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)

        # Check if the file is an image
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
            if img is None:
                print(f"Skipping non-image file: {filename}")
                continue

            img = cv2.resize(img, output_size)  # Resize
            cv2.imwrite(img_path, img)  # Overwrite original image
            print(f"Processed: {category}/{filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("Data preprocessing completed successfully!")
