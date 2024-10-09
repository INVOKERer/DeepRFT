## Training

1. To download GoPro training and testing data, run
```
python download_data.py --data train-test
```

2. Generate image patches from full-resolution training images of GoPro dataset
```
python generate_patches_gopro.py 
```

3. To train, run
```
./train_4gpu.sh Motion_Deblurring/Options/FNAFNet-width32-freqloss-4gpu.yml
```

**Note:** The above training script uses 4 GPUs by default. 

## Evaluation

Download the pre-trained model and place it in `./pretrained_models/`

#### Testing on GoPro dataset

- Download GoPro testset, run
```
python download_data.py --data test --dataset GoPro
```

- Testing
```
python val.py
```



#### To reproduce PSNR/SSIM scores of the paper on GoPro and HIDE datasets, run this MATLAB script

```
evaluate_gopro_hide.m 
```

#### To reproduce PSNR/SSIM scores of the paper on RealBlur dataset, run

```
evaluate_realblur.py 
```
