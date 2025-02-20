from io import BytesIO
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from fastapi import File

from app.core.models import ImageElement


def int_box_area(box, w, h):
    x1, y1, x2, y2 = box
    int_box = [int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)]
    area = (int_box[2] - int_box[0]) * (int_box[3] - int_box[1])
    return area


def calc_IoU(box1, box2):
    # box1_tensor = torch.tensor([box1], dtype=torch.float)
    # box2_tensor = torch.tensor([box2], dtype=torch.float)
    #
    # iou = ops.box_iou(box1_tensor, box2_tensor).item()
    # return iou

    intersection = intersection_area(box1, box2)
    union = box_area(box1) + box_area(box2) - intersection + 1e-6
    if box_area(box1) > 0 and box_area(box2) > 0:
        ratio1 = intersection / box_area(box1)
        ratio2 = intersection / box_area(box2)
    else:
        ratio1, ratio2 = 0, 0
    return max(intersection / union, ratio1, ratio2)


def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def intersection_area(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def is_inside(potentially_inside_box, outside_box):
    intersection = intersection_area(potentially_inside_box, outside_box)
    ratio = intersection / box_area(potentially_inside_box)
    return ratio > 0.80


def remove_overlap(obj_boxes, ocr_boxes, iou_threshold):
    boxes = []
    boxes.extend(ocr_boxes)

    def gather_ocr_content(index, obj_box):
        ocr_labels = []
        for ocr_el in ocr_boxes:
            ocr_box = ocr_el.bbox
            if is_inside(ocr_box, obj_box) and ocr_el.content:
                ocr_labels.append(ocr_el.content)
                try:
                    boxes.remove(ocr_el)
                except:
                    print(f"Tried to delete {ocr_el.content}")
                    continue

        return " ".join(ocr_labels)

    for i, box1_elem in enumerate(obj_boxes):
        box1 = box1_elem.bbox
        is_valid_box = True
        for j, box2_elem in enumerate(obj_boxes):
            box2 = box2_elem.bbox
            if i != j and calc_IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2):
                is_valid_box = False
                break

        if is_valid_box:
            ocr_labels = gather_ocr_content(box1_elem.index, box1)

            if ocr_labels:
                source = box1_elem.source
                source.append('ocr')
                boxes.append(ImageElement('object', box1, True, ocr_labels, source, box1_elem.index))
            else:
                boxes.append(box1_elem)

        else:
            boxes.append(box1_elem)
            # print(f"Overlay box {box1_elem}")

    return boxes


async def load_image(image_file: File):
    image_bytes = await image_file.read()

    image_source = Image.open(BytesIO(image_bytes))
    if image_source.mode == 'RGBA':
        # Convert RGBA to RGB to avoid alpha channel issues
        image_source = image_source.convert('RGB')

    return image_source

async def load_gray_image(image_file: File):
    image_bytes = await image_file.read()
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)


def denormalize_bbox(coors: np.ndarray, img_width: int, img_height: int):
    cx, cy, w, h = coors

    x_min = (cx - w / 2) * img_width
    y_min = (cy - h / 2) * img_height
    x_max = (cx + w / 2) * img_width
    y_max = (cy + h / 2) * img_height

    return int(x_min), int(y_min), int(x_max), int(y_max)


def center_calc(coors: Tuple[float, float, float, float]):
    center_x = (coors[0] + coors[2]) // 2
    center_y = (coors[1] + coors[3]) // 2

    return center_x, center_y
