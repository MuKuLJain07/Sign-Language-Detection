import os

main_dir = "Dataset"

for root, dirs, files in os.walk(main_dir):
    if root != main_dir:
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)  
        print(f"Cleared files in {root}")

print("All files deleted, directory structure intact.")
