# Model Training

In this section we will explain how you can pretrain a model with DINO using this repository.

## Prepare dataset

This repository ueses PyTorch's ImageFolder dataset 

```bash
$ python main_dino.py --arch vit_small --data_path data/MOT17/train --output_dir work_dirs
```

MMDetection training:

```bash
$ python main_dino.py --arch mmdet --data_path data/MOT17/test --image_size 256 --embed_dim 131072 --batch_size_per_gpu 1 --local_crops_number 2 --output_dir work_dirs
```