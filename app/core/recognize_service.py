import numpy as np
import torch
from fastapi import File
from torchvision.ops import box_convert

from app.core.capture_service import capture
from app.core.models import RecognitionSettings
from app.core.models import RecognizedObject
from app.core.object_detection import detect_objects
from app.core.ocr_service import recognize_text
from app.core.utils import center_calc
from app.core.utils import denormalize_bbox
from app.core.utils import load_image
from app.core.utils import remove_overlap

import os

MODELS_PATH = os.getenv("MODELS_PATH", "./../models")


async def get_all_objects(image_file: File, settings: RecognitionSettings):
    image = await load_image(image_file)
    w, h = image.size

    # 1 use OCR
    ocr_elements = recognize_text(image)

    # 2 use YOLO detection
    obj_elements = detect_objects(image, f"{MODELS_PATH}/icon_detect/model.pt", settings.box_threshold,
                                  settings.iou_threshold)

    # 3 match and remove boxes in boxes
    elements = remove_overlap(obj_elements, ocr_elements, settings.iou_threshold)

    # 4 capture elements without text
    if settings.use_object_capturing:
        image_np = np.array(image)
        capture(elements, image_np, batch_size=settings.capture_batch_size)

    recognized_objects = []
    for el in elements:
        tensor = torch.tensor(el.bbox)
        box_coors = box_convert(boxes=tensor, in_fmt="xyxy", out_fmt="cxcywh").numpy()
        coors = denormalize_bbox(box_coors, w, h)
        center_x, center_y = center_calc(coors)

        recognized_objects.append(RecognizedObject(type=el.type, bbox=coors, center_x=center_x, center_y=center_y,
                                                   interactivity=el.interactivity, label=el.content or "undefined"))

    return recognized_objects
