# How to setup DINO

```bash
$ git clone https://github.com/facebookresearch/dino

```

# WSL

```bash
$ conda ceate -y -n dino python==3.10 pip
$ pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
$ pip install openmim
$ mim install mmengine mmcv==2.1 mmdet
$ pip install "numpy<2"
```