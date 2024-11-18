import numpy as np
import cv2
import os
import random


minValue = 70

# Balancing all the directories
def balance_directory(mainDir):
    main_directory = mainDir

    # Find the minimum number of images across all subdirectories
    min_image_count = None
    subdirs = [d for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))]
    for subdir in subdirs:
        subdir_path = os.path.join(main_directory, subdir)
        image_files = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
        count = len(image_files)
        
        if min_image_count is None or count < min_image_count:
            min_image_count = count

    print(f"Minimum number of images to keep in each subdirectory: {min_image_count}")

    # Delete extra images from each subdirectory
    for subdir in subdirs:
        subdir_path = os.path.join(main_directory, subdir)
        image_files = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
        
        # Shuffle and keep only the required number of images
        if len(image_files) > min_image_count:
            random.shuffle(image_files)
            files_to_delete = image_files[min_image_count:]  # Images exceeding the minimum count
            
            # Delete the extra images
            for file_name in files_to_delete:
                file_path = os.path.join(subdir_path, file_name)
                os.remove(file_path)
                # print(f"Deleted: {file_path}")

    print("Dataset balanced successfully.")


def func(path):    
    frame = cv2.imread(path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return res


# Main
input_path = "Dataset"               # Define the input path
balance_directory(input_path)

# Creating output directories
output_dir = "Processed_Dataset"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(f"{output_dir}/train"):
    os.makedirs(f"{output_dir}/train")
if not os.path.exists(f"{output_dir}/test"):
    os.makedirs(f"{output_dir}/test")



# Initialize variables
label = 0
var = 0
c1 = 0
c2 = 0

# Iterate over each subdirectory in "Dataset"
for subdir in os.listdir(input_path):
    subdir_path = os.path.join(input_path, subdir)
    if os.path.isdir(subdir_path):  # Check if it's a directory
        # Create corresponding train and test folders
        train_dir = os.path.join(output_dir, "train", subdir)
        test_dir = os.path.join(output_dir, "test", subdir)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Get list of all files in the subdirectory
        files = os.listdir(subdir_path)
        num = int(0.75 * len(files))                      # 75% for training, 25% for testing
        for i, file in enumerate(files):
            var += 1
            actual_path = os.path.join(subdir_path, file)
            train_path = os.path.join(train_dir, file)
            test_path = os.path.join(test_dir, file)
            
            bw_image = func(actual_path)
            
            # Save to train or test based on index
            if i < num:
                c1 += 1
                cv2.imwrite(train_path, bw_image)
            else:
                c2 += 1
                cv2.imwrite(test_path, bw_image)
        
        label += 1

print("Total images processed:", var)
print("Training images:", c1)
print("Testing images:", c2)
