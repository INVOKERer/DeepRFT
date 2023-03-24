# Installation

This repository is built in PyTorch 1.10.0 and tested on Ubuntu 18.04 environment (Python3.8, CUDA11.3).
Follow these intructions

1. Clone our repository

2. Make conda environment
```
conda create -n pytorch110 python=3.8
conda activate pytorch110
```

3. Install dependencies
```
conda install pytorch=1.10.0 torchvision cudatoolkit=11.3 -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm ptflops
pip install seaborn einops kornia timm gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips
```

4. Install basicsr
```
python setup.py develop --no_cuda_ext
```

### Download datasets from Google Drive

To be able to download datasets automatically you would need `go` and `gdrive` installed. 

1. You can install `go` with the following
```
curl -O https://storage.googleapis.com/golang/go1.11.1.linux-amd64.tar.gz
mkdir -p ~/installed
tar -C ~/installed -xzf go1.11.1.linux-amd64.tar.gz
mkdir -p ~/go
echo "export GOPATH=$HOME/go" >> ~/.bashrc
echo "export PATH=$PATH:$HOME/go/bin:$HOME/installed/go/bin" >> ~/.bashrc
```

2. Install `gdrive` using
```
go get github.com/prasmussen/gdrive
```

3. Close current terminal and open a new terminal. 
