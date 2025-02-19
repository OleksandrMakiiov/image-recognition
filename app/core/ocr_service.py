from typing import List

import cv2
import easyocr
import numpy as np
import torch
from fastapi import File

from app.core.models import BoundingBox
from app.core.models import ImageElement
from app.core.models import Word
from app.core.utils import center_calc
from app.core.utils import int_box_area
from app.core.utils import load_gray_image


# EasyOCR
def recognize_text(image, text_threshold=0.8, languages=None):
    if languages is None:
        languages = ['en']

    w, h = image.size
    image_np = np.array(image)

    reader = easyocr.Reader(languages)
    easyocr_args = {'text_threshold': text_threshold}

    result = reader.readtext(image_np, **easyocr_args)
    coord = [item[0] for item in result]
    text = [item[1] for item in result]

    bbox = torch.tensor([get_xyxy(item) for item in coord]) / torch.Tensor([w, h, w, h])
    bbox = bbox.tolist()

    elements = [ImageElement('text', box, False, txt, ['ocr'], i) for i, (box, txt) in enumerate(zip(bbox, text)) if
                int_box_area(box, w, h) > 0]

    return elements


async def find_text_in_image(image_file: File, search_text: str) -> List[BoundingBox]:
    # read file
    image = await load_gray_image(image_file)

    # PreProcess
    processed_image = __preprocess_image(image)
    boxes = __find_text_regions(processed_image, search_text)

    return boxes


def __find_text_regions(processed_image, target_text, text_threshold=0.8):
    languages = ['en']

    image_np = np.array(processed_image)

    reader = easyocr.Reader(languages)
    easyocr_args = {'text_threshold': text_threshold}

    result = reader.readtext(image_np, **easyocr_args)
    coord = [item[0] for item in result]
    text = [item[1] for item in result]

    words = __convert_to_words(coord, text)

    matched_words = __find_text_coordinates(words, target_text)

    matched_phrases = []
    for phrase in matched_words:
        word_coords = __get_bounding_box(phrase)
        center_x, center_y = center_calc((word_coords[0], word_coords[1], word_coords[2], word_coords[3]))
        matched_phrases.append(
            BoundingBox(x1=word_coords[0], y1=word_coords[1], x2=word_coords[2], y2=word_coords[3], center_x=center_x,
                        center_y=center_y))

    return matched_phrases


def __convert_to_words(coords, text):
    words = []
    for coord, text in zip(coords, text):
        text = text.strip()
        if text:
            x1, y1 = coord[0]
            x2, y2 = coord[2]
            center_x, center_y = center_calc((x1, y1, x2, y2))
            bounding_box = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, center_x=center_x, center_y=center_y)
            word = Word(text, bounding_box)
            words.append(word)
    return words


def __find_text_coordinates(words, search_text):
    splitted_search_text = search_text.split(" ")

    matched_phrase = []

    i = 0
    while i < len(words):
        start_index = i
        matched_words = []
        is_matched = True

        for target_word in splitted_search_text:
            if i >= len(words):
                is_matched = False
                break
            current_word = words[i]

            if current_word.text.lower() != target_word.lower():
                is_matched = False
                break
            matched_words.append(current_word)
            i += 1

        if is_matched:
            matched_phrase.append(matched_words)
        else:
            i = start_index + 1

    return matched_phrase


def __get_bounding_box(coords):
    min_x = min(coords, key=lambda w: w.bounding_box.x1).bounding_box.x1
    min_y = min(coords, key=lambda w: w.bounding_box.y1).bounding_box.y1
    max_x = max(coords, key=lambda w: w.bounding_box.x2).bounding_box.x2
    max_y = max(coords, key=lambda w: w.bounding_box.y2).bounding_box.y2

    return (min_x, min_y, max_x, max_y)


def __preprocess_image(image):
    gray = cv2.bitwise_not(image)
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def get_xyxy(input):
    x, y, xp, yp = input[0][0], input[0][1], input[2][0], input[2][1]
    x, y, xp, yp = int(x), int(y), int(xp), int(yp)
    return x, y, xp, yp
