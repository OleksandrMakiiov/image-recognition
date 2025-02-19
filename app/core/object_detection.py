from ultralytics import YOLO
import torch
from app.core.models import ImageElement
from app.core.utils import int_box_area


def detect_objects(image, model_name_path, box_threshold, iou_threshold):
    w, h = image.size

    model = YOLO(model_name_path)
    results = model.predict(source=image, conf=box_threshold, iou=iou_threshold)
    boxes = results[0].boxes.xyxy
    conf = results[0].boxes.conf

    xyxy = boxes / torch.Tensor([w, h, w, h]).to(boxes.device)
    bbox = xyxy.tolist()

    elements = [ImageElement('object', box, True, None, ['od'], i) for i, box in enumerate(bbox) if int_box_area(box, w, h) > 0]

    return elements
