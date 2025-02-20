import os
import time

import cv2
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

MODELS_PATH = os.getenv("MODELS_PATH", "./../models")

model_name_or_path = f"{MODELS_PATH}/icon_caption_florence"
# model_name_or_path = "microsoft/Florence-2-base-ft"

device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# device = "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=True).to(
    device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)

if device == "mps":
    warm_start = time.time()
    dummy_input_ids = torch.zeros((1, 1), dtype=torch.long, device="mps")
    dummy_pixel_values = torch.rand((1, 3, 224, 224), dtype=torch.float32, device="mps")
    _ = model.generate(input_ids=dummy_input_ids, pixel_values=dummy_pixel_values)
    print(f"MPS warmed for {time.time() - warm_start} s")


@torch.inference_mode()
def capture(elements, image_source, batch_size=128):
    start_time = time.time()

    non_ocr_elements = [el for el in elements if el.content is None]

    non_ocr_boxes = torch.tensor([box.bbox for box in non_ocr_elements])
    croped_images_np = []
    for i, coord in enumerate(non_ocr_boxes):
        try:
            xmin, xmax = int(coord[0] * image_source.shape[1]), int(coord[2] * image_source.shape[1])
            ymin, ymax = int(coord[1] * image_source.shape[0]), int(coord[3] * image_source.shape[0])
            cropped_image = image_source[ymin:ymax, xmin:xmax, :]
            cropped_image = cv2.resize(cropped_image, (32, 32))
            croped_images_np.append(cropped_image)
        except:
            continue

    prepare_time = time.time()

    print(f"Florence process time: image [{len(croped_images_np)}] preparation {prepare_time - start_time}")

    prompt = "<CAPTION>"

    generated_texts = []
    device = model.device
    for i in range(0, len(croped_images_np), batch_size):
        start_time = time.time()
        batch = croped_images_np[i:i + batch_size]
        t1 = time.time()
        if model.device.type == 'cuda':
            inputs = processor(images=batch, text=[prompt] * len(batch), return_tensors="pt", do_resize=False).to(
                device=device, dtype=torch_dtype)
        else:
            inputs = processor(images=batch, text=[prompt] * len(batch), return_tensors="pt").to(device=device)
        t2 = time.time()

        generated_ids = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                                       max_new_tokens=30, num_beams=1, do_sample=False)
        t3 = time.time()
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_text = [gen.strip() for gen in generated_text]
        generated_texts.extend(generated_text)
        t4 = time.time()
        print(f"Florence process times[{len(batch)}]: {t1 - start_time}, {t2 - t1}, {t3 - t2}, {t4 - t3}")

    # print(f"RESULT: {generated_texts}")

    for i, el in enumerate(non_ocr_elements):
        el.content = generated_texts.pop(0)
        el.source.append('capture')

    if device == "mps":
        mem_clean_time = time.time()
        torch.mps.empty_cache()
        print(f"MPS memory cleared for {time.time() - mem_clean_time}")

    return non_ocr_elements
