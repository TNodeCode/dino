from mmdet.apis import init_detector, inference_detector
from mmdet.structures.det_data_sample import DetDataSample
from mmdet.models.layers import SinePositionalEncoding
from mmengine import Config
from mmdet.registry import MODELS as MODELS_DET

import torch
import torch.nn as nn
import os

class FasterRCNNBackbone(torch.nn.Module):
    def __init__(self, backbone_layer=-1):
        super().__init__()
        self.backbone_layer = backbone_layer
        config_file = 'mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
        checkpoint_file = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = init_detector(config_file, checkpoint_file, device=device)
        self.model = model.backbone
        self.reduce = nn.Conv2d(in_channels=2048,out_channels=1,kernel_size=1,stride=1,padding=0)
        del model
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("TOTAL PARAMS", total_params)
        print("TRAINABLE PARAMS", trainable_params)
        print(self)

    def forward(self, x):
        x = self.model(x)
        # we are only interested in the last layer
        x = x[-1]
        x = self.reduce(x)
        return torch.flatten(x, start_dim=1)


class DeformableDETRBackbone(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Load a base configuration file
        cfg = Config.fromfile('mmdetection/mmdet/configs/deformable_detr/deformable_detr_r50_16xb2_50e_coco.py')
        #cfg.model.backbone.frozen_stages=4
        # Build model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = init_detector(cfg, None, device=device)
        self.model = model.backbone
        self.reduce = nn.Conv2d(in_channels=2048,out_channels=1,kernel_size=1,stride=1,padding=0)
        del model
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("TOTAL PARAMS", total_params)
        print("TRAINABLE PARAMS", trainable_params)
        
    def forward(self, x):
        # out has shape ([layers=6, batch_size, n_queries=300, channels=256])
        #out = self.model(x, [DetDataSample(batch_input_shape=x.shape[2:], img_shape=x.shape[2:])])
        x = self.model(x)
        # we are only interested in the last layer
        x= x[-1]
        x = self.reduce(x)
        return torch.flatten(x, start_dim=1)


def get_mmdet_model(args):
    # We only want to train the backbone of Faster-RCNN
    if args.arch == "mmdet:faster-rcnn":
        model = FasterRCNNBackbone()
    elif args.arch == "mmdet:deformable-detr":
        model = DeformableDETRBackbone()
    else:
        raise Exception(f"{args.arch} is not supported")

    # return model and output dimension
    return model