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
                       settings: Optional[RecognitionSettings] = Body(default_factory=RecognitionSettings)):
    result = await get_all_objects(image, settings)
    return result