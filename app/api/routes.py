from fastapi import APIRouter, UploadFile, File, Body, Query
from typing import List, Optional

from app.core.recognize_service import get_all_objects
from app.core.models import RecognizedObject
from app.core.models import RecognitionSettings
from app.core.models import BoundingBox
from app.core.ocr_service import find_text_in_image
from app.core.match_img_service import find_image_in_image

router = APIRouter(prefix="/recognize", tags=["API"])


@router.post("/text", response_model=List[BoundingBox])
async def process_text(file: UploadFile = File(...), search_text: str = Query(...)):
    result = await find_text_in_image(file, search_text)
    return result


@router.post("/img", response_model=List[BoundingBox])
async def process_text(original_file: UploadFile = File(...), template_file: UploadFile = File(...)):
    result = await find_image_in_image(original_file, template_file)
    return result


@router.post("/all-objects", response_model=List[RecognizedObject])
async def process_text(image: UploadFile = File(...),
                       box_threshold: float = Query(0.05),
                       iou_threshold: float = Query(0.5),
                       capture_batch_size: int = Query(128),
                       use_object_capturing: bool = Query(True)):
    settings = RecognitionSettings(box_threshold=box_threshold, iou_threshold=iou_threshold, use_object_capturing=use_object_capturing, capture_batch_size=capture_batch_size)
    result = await get_all_objects(image, settings)
    return result
