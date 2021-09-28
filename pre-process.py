import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm

open_root_dir = open("path_of_train_dir.txt", "r")
root_dir = open_root_dir.read()
open_root_dir.close()
classify_train = os.path.join(root_dir, 'classify train')

def resize(folder):
    total = sum([len(files) for r, d, files in os.walk(folder)])
    with tqdm(total=total) as pbar:
        for folder_name_1 in os.listdir(folder):
            folder_1 = os.path.join(folder, folder_name_1)
            for folder_name_2 in os.listdir(folder_1):
                folder_2 = os.path.join(folder_1, folder_name_2)
                for classes_name in os.listdir(folder_2):
                    file = os.path.join(folder_2, classes_name)
                    
                    img = cv2.imread(file)
                    res = cv2.resize(img, (128, 128))
                    
                    cv2.imwrite(file, res)
                    
                    pbar.set_description("Resizing images to 128x128 pixels")
                    pbar.update()


resize(classify_train)
