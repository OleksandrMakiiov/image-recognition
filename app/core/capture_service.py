import cv2
from torchvision.transforms import ToPILImage
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image
import time
import os

MODELS_PATH = os.getenv("MODELS_PATH", "./../models")

model_name_or_path = f"{MODELS_PATH}/icon_caption_florence"
# model_name_or_path = "microsoft/Florence-2-base-ft"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)

@torch.inference_mode()
def capture(elements, image_source, batch_size=128):
    to_pil = ToPILImage()
    start_time = time.time()

    non_ocr_elements = [el for el in elements if el.content is None]

    non_ocr_boxes = torch.tensor([box.bbox for box in non_ocr_elements])
    croped_pil_image = []
    for i, coord in enumerate(non_ocr_boxes):
        try:
            xmin, xmax = int(coord[0] * image_source.shape[1]), int(coord[2] * image_source.shape[1])
            ymin, ymax = int(coord[1] * image_source.shape[0]), int(coord[3] * image_source.shape[0])
            cropped_image = image_source[ymin:ymax, xmin:xmax, :]
            cropped_image = cv2.resize(cropped_image, (32, 32))
            croped_pil_image.append(to_pil(cropped_image))
        except:
            continue

    prepare_time = time.time()

    print(f"Florence process time: image [{len(croped_pil_image)}] preparation {prepare_time - start_time}")

    prompt = "<CAPTION>"

    generated_texts = []
    device = model.device
    for i in range(0, len(croped_pil_image), batch_size):
        start_time = time.time()
        batch = croped_pil_image[i:i + batch_size]
        t1 = time.time()
        if model.device.type == 'cuda':
            inputs = processor(images=batch, text=[prompt] * len(batch), return_tensors="pt", do_resize=False).to(
                device=device, dtype=torch_dtype)
        else:
            inputs = processor(images=batch, text=[prompt] * len(batch), return_tensors="pt").to(device=device)
        t2 = time.time()

        generated_ids = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                                           max_new_tokens=30, num_beams=3, do_sample=False)
        t3 = time.time()
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_text = [gen.strip() for gen in generated_text]
        generated_texts.extend(generated_text)
        t4 = time.time()
        print(f"Florence process times[{len(batch)}]: {t1 - start_time}, {t2-t1}, {t3-t2}, {t4-t3}")

    # print(f"RESULT: {generated_texts}")

    for i, el in enumerate(non_ocr_elements):
        el.content = generated_texts.pop(0)
        el.source.append('capture')

    return non_ocr_elements