from ultralytics import YOLO
import torch
from app.core.models import ImageElement
from app.core.utils import int_box_area

import os

MODELS_PATH = os.getenv("MODELS_PATH", "./../models")
YOLO_MODEL = YOLO(f"{MODELS_PATH}/icon_detect/model.pt")


def detect_objects(image, box_threshold, iou_threshold):
    w, h = image.size

    results = YOLO_MODEL.predict(source=image, conf=box_threshold, iou=iou_threshold)
    boxes = results[0].boxes.xyxy
    conf = results[0].boxes.conf

    xyxy = boxes / torch.Tensor([w, h, w, h]).to(boxes.device)
    bbox = xyxy.tolist()

    elements = [ImageElement('object', box, True, None, ['od'], i) for i, box in enumerate(bbox) if
                int_box_area(box, w, h) > 0]

    return elements
