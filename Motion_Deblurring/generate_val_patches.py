## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

##### Data preparation file for training Restormer on the GoPro Dataset ########

import cv2
import numpy as np
from glob import glob
from natsort import natsorted
import os
from tqdm import tqdm
from pdb import set_trace as stx
from joblib import Parallel, delayed
import multiprocessing

def val_files(file_):
    lr_file, hr_file = file_
    filename = os.path.splitext(os.path.split(lr_file)[-1])[0]
    lr_img = cv2.imread(lr_file)
    hr_img = cv2.imread(hr_file)

    lr_savename = os.path.join(lr_tar, filename + '.png')
    hr_savename = os.path.join(hr_tar, filename + '.png')

    w, h = lr_img.shape[:2]

    i = (w-val_patch_size)//2
    j = (h-val_patch_size)//2
                
    lr_patch = lr_img[i:i+val_patch_size, j:j+val_patch_size,:]
    hr_patch = hr_img[i:i+val_patch_size, j:j+val_patch_size,:]

    cv2.imwrite(lr_savename, lr_patch)
    cv2.imwrite(hr_savename, hr_patch)

############ Prepare Training data ####################
num_cores = 10
patch_size = 512
overlap = 256
p_max = 0

############ Prepare validation data ####################
val_patch_size = 256
src = '/home/mxt/106-48t/personal_data/mxt/Datasets/Deblur/RealBlur/RealBlur_J/test'
tar = '/home/mxt/106-48t/personal_data/mxt/Datasets/Deblur/RealBlur/RealBlur_J/val'

lr_tar = os.path.join(tar, 'input_crops')
hr_tar = os.path.join(tar, 'target_crops')

os.makedirs(lr_tar, exist_ok=True)
os.makedirs(hr_tar, exist_ok=True)

lr_files = natsorted(glob(os.path.join(src, 'blur', '*.png')) + glob(os.path.join(src, 'blur', '*.jpg')))
hr_files = natsorted(glob(os.path.join(src, 'sharp', '*.png')) + glob(os.path.join(src, 'sharp', '*.jpg')))

files = [(i, j) for i, j in zip(lr_files, hr_files)]

Parallel(n_jobs=num_cores)(delayed(val_files)(file_) for file_ in tqdm(files))
