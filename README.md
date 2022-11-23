

# Intriguing Findings of Frequency Selection for Image Deblurring (AAAI 2023)
Xintian Mao, Yiming Liu, Fengze Liu, Qingli Li, Wei Shen and Yan Wang


**Paper**: xxx

## Network Architecture
<table>
  <tr>
    <td> <img src = "https://github.com/INVOKERer/DeepRFT/blob/main/images/xxx.png" width="1200"> </td>
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


## Quick Run

To test the pre-trained models of Deblur [Google Drive](https://drive.google.com/file/d/1FoQZrbcYPGzU9xzOPI1Q1NybNUGR-ZUg/view?usp=sharing) or [百度网盘](https://pan.baidu.com/s/10DuQZiXC-Dc6jtLc9YJGbg)(提取码:phws) on your own images, run 
```
python test.py 
```
Here is an example to train:
```
python
```


## Results
Experiment for image deblurring.
<table>
  <tr>
    <td> <img src = "https://github.com/INVOKERer/DeepRFT/xxx.png" width="1200"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Deblurring on GoPro Datasets.</b></p></td>
  </tr>
</table>

## Reference Code:
- https://github.com/swz30/MPRNet
- https://github.com/chosj95/MIMO-UNet
- https://github.com/megvii-research/NAFNet
- https://github.com/megvii-research/TLC
- https://github.com/swz30/Restormer

## Citation
If you use , please consider citing:
```
@inproceedings{xint2023freqsel, 
    title = {Intriguing Findings of Frequency Selection for Image Deblurring},
    author = {Xintian Mao, Yiming Liu, Fengze Liu, Qingli Li, Wei Shen and Yan Wang}, 
    booktitle = {Proceedings of the 37th AAAI Conference on Artificial Intelligence}, 
    year = {2023}
    }
```
## Contact
If you have any question, please contact mxt_invoker1997@163.com
