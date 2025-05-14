
import argparse
import os
from pathlib import Path
import torch
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device
from utils.plots import save_one_box
import cv2

def detect(weights='yolov7.pt', source='inference/images', img_size=640, conf_thres=0.25, save_img=True):
    # Initialize
    device = select_device('')
    model = torch.load(weights, map_location=device)['model'].float().eval()

    dataset = LoadImages(source, img_size=img_size)
    save_dir = Path('runs/detect/exp')
    save_dir.mkdir(parents=True, exist_ok=True)

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres, 0.45)[0]

        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0s.shape).round()

            for *xyxy, conf, cls in pred:
                label = f'{conf:.2f}'
                save_one_box(xyxy, im0s, file=save_dir / Path(path).name, BGR=True)
                cv2.putText(im0s, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(im0s, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)

        if save_img:
            cv2.imwrite(str(save_dir / Path(path).name), im0s)

if __name__ == '__main__':
    detect()
