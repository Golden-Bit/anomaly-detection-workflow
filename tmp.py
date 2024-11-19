import os
import cv2
from pathlib import Path
import shutil

root_dir = "outputs"
output_dir = "outputs_2"
os.makedirs(output_dir, exist_ok=True)

for subdir in os.listdir(root_dir):
    print(subdir)

    for file_name in os.listdir(f"{root_dir}/{subdir}"):
        print(file_name)
        if file_name == "input_frame.png":
            os.makedirs(f"{output_dir}/{subdir}", exist_ok=True)
            shutil.copy(f"{root_dir}/{subdir}/{file_name}",f"{output_dir}/{subdir}/{file_name}")

