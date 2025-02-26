# ImageRecognition

## Description

**ImageRecognition** is a project for image and object recognition. The project is written in Python and provides a REST
API for working with images.

### Technologies Used

- **EasyOCR** – for text recognition
- **YOLO** – for object detection
- **Florence-2** – for describing icons and objects
- **FastAPI** – for building the REST API
- **Docker** – for containerization

---

## Installation and Running

### Install AI Models

Follow these steps to download and setup YOLO and Florence-2 models:

1. Visit the [OmniParser 2 page on Hugging Face](https://huggingface.co/microsoft/OmniParser-v2.0).
2. Download the `icon_caption` and `icon_detect` folders.
3. Copy these folders to the `models` directory on your device (`./image-recognition/models`)

OR use CLI (run it in `image-recognition` working folder):

```commandline
rm -rf models/icon_detect models/icon_caption models/icon_caption_florence 
for folder in icon_caption icon_detect; do huggingface-cli download microsoft/OmniParser-v2.0 --local-dir models --repo-type model --include "$folder/*"; done
mv models/icon_caption models/icon_caption_florence
```

To ensure the models function correctly, specify the path to the models directory in an environment variable:

   ```sh
   export MODELS_PATH={path_to_models_directory}
   ```

Replace `{path_to_models_directory}` with the actual path to the directory where the models are stored.

### MPS Support Configuration

If you are using an older Apple device with Metal Performance Shaders (MPS) support but with limited RAM, you can
disable MPS support by setting this environment variable:

   ```sh
   export IS_MPS_AVAILABLE=False
   ```

This disables the use of MPS which can help to avoid performance issues on devices with low memory.

### Running without Docker

1. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```
2. Start the FastAPI server:
    ```sh
    uvicorn app.main:app --host 0.0.0.0 --port 8000
    ```

### Running with Docker

1. Build the Docker image:
    ```sh
    docker build -t image-recognition .
    ```
2. Run the container:
    ```sh
    docker run -d -p 8000:8000 --name image-recognition image-recognition
    ```

---

## REST API

### 1. Text Recognition

**POST** `/recognize/text`

**Parameters:**

- `file` (UploadFile) – the image to search for text
- `search_text` (Query) – the text to find

**Example Request:**

```sh
curl -X 'POST' \
  'http://127.0.0.1:8000/recognize/text?search_text=example' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@image.png'
```

**Response:**

```json
[
  {
    "x1": 451,
    "y1": 565,
    "x2": 511,
    "y2": 579,
    "center_x": 481,
    "center_y": 572
  }
]
```

---

### 2. Image Search within Another Image

**POST** `/recognize/img`

**Parameters:**

- `original_file` (UploadFile) – the original image
- `template_file` (UploadFile) – the image to find

**Example Request:**

```sh
curl -X 'POST' \
  'http://127.0.0.1:8000/recognize/img' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'original_file=@scene.png' \
  -F 'template_file=@icon.png'
```

**Response:**

```json
[
  {
    "x1": 451,
    "y1": 565,
    "x2": 511,
    "y2": 579,
    "center_x": 481,
    "center_y": 572
  }
]
```

---

### 3. Recognizing All Objects

**POST** `/recognize/all-objects`

**Parameters:**

- `image` (UploadFile) – the image
- `settings` (RecognitionSettings) – optional recognition settings

**Example Request:**

```sh
curl -X 'POST' \
  'http://127.0.0.1:8000/recognize/all-objects' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@image.png'
```

**Response:**

```json
[
  {
    "type": "text",
    "bbox": [
      211.0,
      55.0,
      253.0,
      71.0
    ],
    "center_x": 232,
    "center_y": 63,
    "interactivity": false,
    "label": "Home"
  },
  {
    "type": "object",
    "bbox": [
      1315.0,
      42.0,
      1343.0,
      78.0
    ],
    "center_x": 1329,
    "center_y": 60,
    "interactivity": true,
    "label": "Refresh or reload the page."
  }
]
```

---

## Hardware Requirements

- **CPU:** At least 8 cores processor
- **RAM:** Minimum 16GB (32GB recommended for better performance)
- **GPU:** Recommended for better YOLO model performance (NVIDIA GPU with CUDA support)
- **Storage:** At least 10GB of free disk space

## Model Licenses

-
YOLO: [https://github.com/ultralytics/yolov5/blob/master/LICENSE](https://github.com/ultralytics/yolov5/blob/master/LICENSE)
-
Florence-2: [https://huggingface.co/microsoft/Florence-2-large/blob/main/LICENSE](https://huggingface.co/docs/transformers/model_doc/florence)

## Authors

Developed for automatic image and object recognition in various scenarios.

