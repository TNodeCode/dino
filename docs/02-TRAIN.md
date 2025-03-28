# Model Training

In this section we will explain how you can pretrain a model with DINO using this repository.

## Prepare dataset

This repository ueses PyTorch's ImageFolder dataset 

```bash
$ python main_dino.py --arch vit_small --data_path data/MOT17/train --output_dir work_dirs
```

MMDetection training:

```bash
$ python main_dino.py --arch mmdet --data_path data/MOT17/train --image_size 256 --embed_dim 131072 --batch_size_per_gpu 1 --local_crops_number 2 --output_dir work_dirs
```

Deformable DETR Training:

```bash
python main_dino.py --arch mmdet:deformable-detr --data_path data/MOT17/train --image_size 512 --embed_dim 76800 --batch_size_per_gpu 1 --local_crops_number 1 --output_dir work_dirs

Deformable DETR Refine Training:

```bash
python main_dino.py --arch mmdet:deformable-detr-refine --data_path data/MOT17/train --image_size 512 --embed_dim 76800 --batch_size_per_gpu 1 --local_crops_number 1 --output_dir work_dirs
```

Faster RCNN Backbone Training:

```bash
python main_dino.py --arch mmdet:faster-rcnn --data_path ../mmdetection/data/MOT17_all/train/ --image_size 512 --embed_dim 256 --batch_size_per_gpu 8 --local_crops_number 8 --local_crops_scale 0.5 0.7 --global_crops_scale 0.7 1.0 --output_dir work_dirs/faster_rcnn_50--epochs 150 --saveckp_freq 1
```