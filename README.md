# Deep Residual Fourier Transformation for Single Image Deblurring
Xintian Mao, Yiming Liu, Wei Shen, Qingli Li and Yan Wang

code will be released soon

**Paper**: 


> **Abstract:** *It has been a common practice to adopt the ResBlock, which learns the difference between blurry and sharp image pairs, in end-to-end image deblurring architectures. Reconstructing a sharp image from its blurry counterpart requires changes regarding both low- and high-frequency information. Although conventional ResBlock may have good abilities in capturing the high-frequency components of images, it tends to overlook the low-frequency information. Moreover, ResBlock usually fails to felicitously model the long-distance information which is non-trivial in reconstructing a sharp image from its blurry counterpart. In this paper, we present a Residual Fast Fourier Transform with Convolution Block (Res FFT-Conv Block), capable of capturing both long-term and short-term interactions, while integrating both low- and high-frequency residual information. Res FFT-Conv Block is a conceptually simple yet computationally efficient, and plug-and-play block, leading to remarkable performance gains in different architectures. With Res FFT-Conv Block, we further propose a Deep Residual Fourier Transformation (DeepRFT) framework, based upon MIMO-UNet, achieving state-of-the-art image deblurring performance on GoPro, HIDE, RealBlur and DPDD datasets. Experiments show our DeepRFT can boost image deblurring performance significantly (e.g., with 1.09 dB improvement in PSNR on GoPro dataset compared with MIMO-UNet), and DeepRFT+ even reaches 33.23 dB in PSNR on GoPro dataset.* 

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

To test the pre-trained models of [Deblurring]() on your own images, run 
```
python test.py --weights ckpt_path_here --input_dir path_to_images --result_dir save_images_here
```
Here is an example to perform Deblurring:
```
python train.py
```


## Results
Experiment for image deblurring.

### Image Deblurring

<table>
  <tr>
    <td> <img src = "https://github.com/INVOKERer/DeepRFT/blob/main/images/psnr_params_flops.png" width="1200"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Deblurring on GoPro Datasets.</b></p></td>
  </tr>
</table>

Reference Code:
- https://github.com/yangyanli/DO-Conv
- https://github.com/swz30/MPRNet
- https://github.com/chosj95/MIMO-UNet

## Citation
If you use DeepRFT, please consider citing:

    @inproceedings{,
        title={Deep Residual Fourier Transformation for Single Image Deblurring},
        author={Xintian Mao, Yiming Liu, Wei Shen, Qingli Li, Yan Wang},
        booktitle={},
        year={}
    }

## Contact
If you have any question, please contact mxt_invoker1997@163.com or 51191214020@stu.ecnu.edu.cn
