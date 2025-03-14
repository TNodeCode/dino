# Model Training

In this section we will explain how you can pretrain a model with DINO using this repository.

## Prepare dataset

This repository ueses PyTorch's ImageFolder dataset 

```bash
$ python main_dino.py --arch vit_small --data_path data/MOT17 --output_dir work_dirs
```