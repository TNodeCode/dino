from mmdet.apis import init_detector
from mmengine import Config
from mmengine.runner import save_checkpoint

import torch
import torch.nn as nn


def load_detector(config: Config) -> nn.Module:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = init_detector(config, None, device=device)
    return model


class FasterRCNNBackbone(torch.nn.Module):
    def __init__(self, config_file: str):
        super().__init__()
        cfg = Config.fromfile(config_file)
        cfg.model.backbone.frozen_stages=2
        model = load_detector(cfg)
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
    if args.arch == "mmdet:faster-rcnn-50":
        model = FasterRCNNBackbone(config_file='mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py')
    elif args.arch == "mmdet:faster-rcnn-101":
        model = FasterRCNNBackbone(config_file='mmdetection/configs/faster_rcnn/faster-rcnn_r101_fpn_1x_coco.py')
    elif args.arch == "mmdet:faster-rcnn-x101-32":
        model = FasterRCNNBackbone(config_file='mmdetection/configs/faster_rcnn/faster-rcnn_x101-32x4d_fpn_1x_coco.py')
    elif args.arch == "mmdet:faster-rcnn-x101-64":
        model = FasterRCNNBackbone(config_file='mmdetection/configs/faster_rcnn/faster-rcnn_x101-64x4d_fpn_1x_coco.py')
    elif args.arch == "mmdet:deformable-detr":
        model = DeformableDETRBackbone()
    else:
        raise Exception(f"{args.arch} is not supported")

    # return model and output dimension
    return model


def save_faster_rcnn_pretrained(detector_config: str, weights_file: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load the full state_dict from the pretrained model
    pretrained_state_dict = torch.load(weights_file, map_location=device)
    student = pretrained_state_dict['student']
    teacher = pretrained_state_dict['teacher']

    # Filter out only backbone keys (ignore 'reduce')
    backbone_student_state_dict = {
        k.replace('module.backbone.model.', ''): v for k, v in student.items()
        if k.startswith('module.backbone.model.') and 'reduce' not in k
    }

    def build_detector(config_file: str, backbone_dict, output_file: str):
        # Load into the Faster R-CNN model's backbone
        detector = load_detector(config_file=detector_config).train()
        missing, unexpected = detector.backbone.load_state_dict(backbone_student_state_dict, strict=False)
        if missing:
            raise ValueError("Missing keys", missing)
        if unexpected:
            raise ValueError("Unexpected keys:", unexpected)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

        torch.save(detector.state_dict(), output_file)

        # If you’re using MMDetection’s custom training setup and want to save in its format:
        #save_checkpoint(detector, output_file)

    build_detector(config_file=detector_config, backbone_dict=student, output_file="faster_rcnn_student.pth")
    build_detector(config_file=detector_config, backbone_dict=teacher, output_file="faster_rcnn_teacher.pth")
    