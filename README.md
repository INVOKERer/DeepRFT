[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deep-residual-fourier-transformation-for/deblurring-on-gopro)](https://paperswithcode.com/sota/deblurring-on-gopro?p=deep-residual-fourier-transformation-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deep-residual-fourier-transformation-for/deblurring-on-hide-trained-on-gopro)](https://paperswithcode.com/sota/deblurring-on-hide-trained-on-gopro?p=deep-residual-fourier-transformation-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deep-residual-fourier-transformation-for/deblurring-on-realblur-j-1)](https://paperswithcode.com/sota/deblurring-on-realblur-j-1?p=deep-residual-fourier-transformation-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deep-residual-fourier-transformation-for/deblurring-on-realblur-j-trained-on-gopro)](https://paperswithcode.com/sota/deblurring-on-realblur-j-trained-on-gopro?p=deep-residual-fourier-transformation-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deep-residual-fourier-transformation-for/deblurring-on-realblur-r)](https://paperswithcode.com/sota/deblurring-on-realblur-r?p=deep-residual-fourier-transformation-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deep-residual-fourier-transformation-for/deblurring-on-realblur-r-trained-on-gopro)](https://paperswithcode.com/sota/deblurring-on-realblur-r-trained-on-gopro?p=deep-residual-fourier-transformation-for)


# Intriguing Findings of Frequency Selection for Image Deblurring (AAAI 2023)
Xintian Mao, Yiming Liu, Fengze Liu, Qingli Li, Wei Shen and Yan Wang

**Paper**: https://arxiv.org/abs/2111.11745

**code**: https://github.com/DeepMed-Lab-ECNU/DeepRFT-AAAI2023

# Deep Residual Fourier Transformation for Single Image Deblurring
Xintian Mao, Yiming Liu, Wei Shen, Qingli Li and Yan Wang


**Paper**: https://arxiv.org/abs/2111.11745


## Network Architecture
<table>
  <tr>
    <td> <img src = "https://github.com/INVOKERer/DeepRFT/blob/main/images/framework.png" width="1200"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Overall Framework of DeepRFT</b></p></td>
  </tr>
</table>

## Installation
The model is built in PyTorch 1.8.0 and tested on Ubuntu 18.04 environment (Python3.8, CUDA11.1).

For installing, follow these intructions
```
conda create -n pytorch python=3.8
conda activate pytorch
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm kornia tensorboard ptflops
```

Install warmup scheduler

```
cd pytorch-gradual-warmup-lr; python setup.py install; cd ..
```

## Quick Run

To test the pre-trained models of Deblur and Defocus [Google Drive](https://drive.google.com/file/d/1FoQZrbcYPGzU9xzOPI1Q1NybNUGR-ZUg/view?usp=sharing) or [百度网盘](https://pan.baidu.com/s/10DuQZiXC-Dc6jtLc9YJGbg)(提取码:phws) on your own images, run 
```
python test.py --weights ckpt_path_here --input_dir path_to_images --result_dir save_images_here --win_size 256 --num_res 8 [4:small, 20:plus]# deblur
python test.py --weights ckpt_path_here --input_dir path_to_images --result_dir save_images_here --win_size 512 --num_res 8 # defocus
```
Here is an example to train:
```
python train.py
```


## Results
Experiment for image deblurring.
<table>
  <tr>
    <td> <img src = "https://github.com/INVOKERer/DeepRFT/blob/main/images/psnr_flops.png" width="1200"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Deblurring on GoPro Datasets.</b></p></td>
  </tr>
</table>

## Reference Code:
- https://github.com/yangyanli/DO-Conv
- https://github.com/swz30/MPRNet
- https://github.com/chosj95/MIMO-UNet
- https://github.com/codeslake/IFAN

## Citation
If you use DeepRFT, please consider citing:
```
@inproceedings{xint2023freqsel, 
    title = {Intriguing Findings of Frequency Selection for Image Deblurring},
    author = {Xintian Mao, Yiming Liu, Fengze Liu, Qingli Li, Wei Shen and Yan Wang}, 
    booktitle = {Proceedings of the 37th AAAI Conference on Artificial Intelligence}, 
    year = {2023}
    }
```
or
```
@inproceedings{,
    title={Deep Residual Fourier Transformation for Single Image Deblurring},
    author={Xintian Mao, Yiming Liu, Wei Shen, Qingli Li, Yan Wang},
    booktitle={arXiv:2111.11745},
    year={2021}
}
```
## Contact
If you have any question, please contact mxt_invoker1997@163.com
