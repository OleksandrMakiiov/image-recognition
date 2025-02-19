from typing import List

import cv2
import numpy as np
from fastapi import File

from app.core.models import BoundingBox
from app.core.utils import center_calc
from app.core.utils import load_gray_image


async def find_image_in_image(original_file: File, template_file: File) -> List[BoundingBox]:
    # load files
    original_image = await load_gray_image(original_file)
    template_image = await load_gray_image(template_file)

    # Find
    result = cv2.matchTemplate(original_image, template_image, cv2.TM_CCOEFF_NORMED)

    # can be adjusted for better result
    threshold = 0.8
    loc = np.where(result >= threshold)

    # Get coordinated based on template size
    boxes = []
    if loc is not None and loc[0].size > 0:
        for pt in zip(*loc[::-1]):
            x1, y1 = pt
            x2 = x1 + template_image.shape[1]
            y2 = y1 + template_image.shape[0]
            center_x, center_y = center_calc((x1, y1, x2, y2))
            boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, center_x=center_x, center_y=center_y))

    return boxes
