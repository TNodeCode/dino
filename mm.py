from mmdet.apis import init_detector, inference_detector
from mmdet.structures.det_data_sample import DetDataSample
from mmdet.models.layers import SinePositionalEncoding
from mmengine import Config
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
        self.model = model
        self.fc = None
        self.head = None

    def forward(self, x):
        out = self.model.backbone(x)
        return torch.flatten(out[self.backbone_layer], start_dim=1)


class BBoxHead(nn.Module):
    def forward(self, hidden_states, references):
        return hidden_states

class DeformableDETRBackbone(torch.nn.Module):
            
    def __init__(self, *args, **kwargs):
        super().__init__()
        cfg = Config.fromfile('mmdetection/mmdet/configs/deformable_detr/deformable_detr_r50_16xb2_50e_coco.py')
        self.model = init_detector(config=cfg, checkpoint=None, device='cpu')
        self.model.bbox_head = BBoxHead()
        
    def forward(self, x):
        # out has shape ([layers=6, batch_size, n_queries=300, channels=256])
        out = self.model(x, [DetDataSample(batch_input_shape=x.shape[2:], img_shape=x.shape[2:])])
        # we are only interested in the last layer
        out= out[-1]
        out = torch.flatten(out, start_dim=1)
        return out


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