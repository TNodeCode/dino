from mmdet.apis import init_detector, inference_detector
import torch
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
        #print("FASTER RCNN IN", x.shape)
        out = self.model.backbone(x)
        #print("FASTER RCNN OUT", [f.shape for f in out])
        return torch.flatten(out[self.backbone_layer], start_dim=1)



def get_mmdet_model():
    # We only want to train the backbone of Faster-RCNN
    model = FasterRCNNBackbone()

    # return model and output dimension
    return model