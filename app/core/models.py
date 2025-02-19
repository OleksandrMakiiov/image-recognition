from pydantic import BaseModel
from typing import List, Any, Tuple


class ImageElement:
    def __init__(self, type, bbox, interactivity, content, source, index):
        self.type = type
        self.bbox = bbox
        self.interactivity = interactivity
        self.content = content
        self.source = source
        self.index = index

    def __str__(self):
        return f"content: {self.content}, interactivity: {self.interactivity}, type: {self.type}, source: {self.source}, index: {self.index}"


class RecognitionSettings(BaseModel):
    box_threshold: float = 0.05
    iou_threshold: float = 0.5


class RecognizedObject(BaseModel):
    type: str
    bbox: Tuple[float, float, float, float]
    center_x: int
    center_y: int
    interactivity: bool
    label: str

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    center_x: int
    center_y: int

class Word:
    def __init__(self, text, bounding_box: BoundingBox):
        self.text = text
        self.bounding_box = bounding_box

    def __str__(self):
        return f"Word(text: {self.text}, bounding_box: {self.bounding_box})"
