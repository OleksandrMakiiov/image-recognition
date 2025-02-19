# ImageRecognition

## Description
**ImageRecognition** is a project for image and object recognition. The project is written in Python and provides a REST API for working with images.

### Technologies Used
- **EasyOCR** – for text recognition
- **YOLO** – for object detection
- **Florence-2** – for describing icons and objects
- **FastAPI** – for building the REST API
- **Docker** – for containerization

---

## Installation and Running

### Running without Docker
1. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```
2. Start the FastAPI server:
    ```sh
    uvicorn main:app --host 0.0.0.0 --port 8000
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
[{"x1":451,"y1":565,"x2":511,"y2":579,"center_x":481,"center_y":572}]
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
[{"x1":451,"y1":565,"x2":511,"y2":579,"center_x":481,"center_y":572}]
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
- YOLO: [https://github.com/ultralytics/yolov5/blob/master/LICENSE](https://github.com/ultralytics/yolov5/blob/master/LICENSE)
- Florence-2: [https://huggingface.co/microsoft/Florence-2-large/blob/main/LICENSE](https://huggingface.co/docs/transformers/model_doc/florence)

## Authors
Developed for automatic image and object recognition in various scenarios.

