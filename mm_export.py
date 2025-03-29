from mm import save_faster_rcnn_pretrained

save_faster_rcnn_pretrained(
    detector_config="mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py",
    weights_file="work_dirs/faster_rcnn_50/checkpoint0149.pth"
)