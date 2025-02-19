from typing import Tuple, List, Union
import torchvision.ops as ops
from torchvision.ops import box_convert
import torch
import supervision as sv
import numpy as np
from box_annotator import BoxAnnotator
import time

from PIL import Image
import numpy as np
import torch
from torchvision.ops import box_convert
import io
from app.core.ocr_service import recognize_text
from app.core.object_detection import detect_objects
from app.core.models import ImageElement
from app.core.utils import remove_overlap
from app.core.capture_service import capture


def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str], text_scale: float,
             text_padding=5, text_thickness=2, thickness=3) -> np.ndarray:
    """
    This function annotates an image with bounding boxes and labels.

    Parameters:
    image_source (np.ndarray): The source image to be annotated.
    boxes (torch.Tensor): A tensor containing bounding box coordinates. in cxcywh format, pixel scale
    logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
    phrases (List[str]): A list of labels for each bounding box.
    text_scale (float): The scale of the text to be displayed. 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web

    Returns:
    np.ndarray: The annotated image.
    """
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    xywh = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xywh").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [f"{phrase}" for phrase in range(boxes.shape[0])]

    box_annotator = BoxAnnotator(text_scale=text_scale, text_padding=text_padding,text_thickness=text_thickness,thickness=thickness) # 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web
    annotated_frame = image_source.copy()
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels, image_size=(w,h))

    label_coordinates = {f"{phrase}": v for phrase, v in zip(phrases, xywh)}
    return annotated_frame, label_coordinates




image_source = '/Volumes/Data/Documents/ui-parser/Screenshot 2025-02-19 at 09.16.07.png'
image_source = Image.open(image_source)
if image_source.mode == 'RGBA':
    # Convert RGBA to RGB to avoid alpha channel issues
    image_source = image_source.convert('RGB')

image_np = np.array(image_source)

#settings
text_threshold = 0.8
BOX_TRESHOLD=0.05
iou_threshold=0.5
box_overlay_ratio = max(image_source.size) / 3200
text_scale = 0.8 * box_overlay_ratio
text_thickness = max(int(2 * box_overlay_ratio), 1)
text_padding = max(int(3 * box_overlay_ratio), 1)
thickness = max(int(3 * box_overlay_ratio), 1)




ocr_elements = recognize_text(image_source, text_threshold, ['en'])
obj_elements = detect_objects(image_source, "/Volumes/Data/projects/UIParser/weights/icon_detect/model.pt", BOX_TRESHOLD, iou_threshold)
elements = remove_overlap(obj_elements, ocr_elements, iou_threshold)


capture(elements, image_np)


#make Tensors
filtered_boxes = torch.tensor([el.bbox for el in elements])
filtered_boxes = box_convert(boxes=filtered_boxes, in_fmt="xyxy", out_fmt="cxcywh")
phrases = [i for i in range(len(filtered_boxes))]

# draw boxes
annotated_frame, label_coordinates = annotate(image_source=image_np, boxes=filtered_boxes, logits=None,
                                                  phrases=phrases, text_scale=text_scale, text_padding=text_padding, text_thickness=text_thickness, thickness=thickness)


pil_img = Image.fromarray(annotated_frame)
buffered = io.BytesIO()
pil_img.save(buffered, format="PNG")
pil_img.show()


for i, box in enumerate(elements):
    print(f"ID: {i}, {box}")