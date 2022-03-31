import os
import argparse
import torch.nn as nn
import torch
from torch.utils.data import DataLoader

from data_RGB import get_test_data
from DeepRFT_MIMO import DeepRFT as mynet

from get_parameter_number import get_parameter_number
from tqdm import tqdm
from layers import *
import time


parser = argparse.ArgumentParser(description='Image Deblurring')
parser.add_argument('--input_dir', default='./Datasets/GoPro/test/blur', type=str, help='Directory of validation images')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

# model_restoration = mynet(inference=True)
model_restoration = mynet()
# print number of model
get_parameter_number(model_restoration)

# utils.load_checkpoint_compress_doconv(model_restoration, args.weights)
# print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# dataset = args.dataset
rgb_dir_test = args.input_dir
test_dataset = get_test_data(rgb_dir_test, img_options={})
test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

win = 256
all_time = 0.
with torch.no_grad():
    psnr_list = []
    ssim_list = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):

        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        input_    = data_test[0].cuda()
        filenames = data_test[1]
        _, _, Hx, Wx = input_.shape
        filenames = data_test[1]

        torch.cuda.synchronize()
        start = time.time()
        input_re, batch_list = window_partitionx(input_, win)
        restored = model_restoration(input_re)
        # print(restored[0].shape)
        restored = window_reversex(restored[0], win, Hx, Wx, batch_list)
        restored = torch.clamp(restored, 0, 1)
        # print(restored.shape)
        torch.cuda.synchronize()
        end = time.time()
        all_time += end - start
print('average_time: ', all_time / len(test_dataset))
