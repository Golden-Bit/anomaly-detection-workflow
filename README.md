# YOLOv8 FastAPI Object Detection and Segmentation API

This project provides a FastAPI application that serves as an API for object detection and segmentation using YOLOv8 models from Ultralytics. It allows users to:

- Perform object detection and segmentation on input images.
- Save outputs (annotated images, JSON data, cropped images) in specified directories.
- Process images by applying bounding boxes and segmentation masks, tinting specific pixels.
- Customize tint colors for image processing.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
  - [Object Detection Endpoint](#object-detection-endpoint)
  - [Segmentation Endpoint](#segmentation-endpoint)
  - [Image Processing Endpoint](#image-processing-endpoint)
- [Usage Examples](#usage-examples)
  - [Performing Object Detection](#performing-object-detection)
  - [Performing Segmentation](#performing-segmentation)
  - [Processing an Image](#processing-an-image)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Notes](#notes)
- [License](#license)

## Features

- **Object Detection**: Detect objects in images using a YOLOv8 detection model.
- **Segmentation**: Segment objects in images using a YOLOv8 segmentation model.
- **Output Management**: Save outputs in specified directories, with separation between detection and segmentation results.
- **Image Processing**: Apply bounding boxes and segmentation masks to images, tinting specific pixels based on user input.
- **Customization**: Users can specify output directories and tint colors for image processing.

## Prerequisites

- Python 3.7 or higher
- Trained YOLOv8 models for detection and segmentation
- Required Python packages (see Installation)

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your_username/your_repository.git
   cd your_repository
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Required Packages**

   ```bash
   pip install fastapi uvicorn pillow ultralytics opencv-python numpy
   ```

   - **Package Descriptions**:
     - `fastapi`: Web framework for building APIs.
     - `uvicorn`: ASGI server to run the FastAPI application.
     - `pillow`: Image processing library.
     - `ultralytics`: Library containing YOLOv8 models.
     - `opencv-python`: Image processing library.
     - `numpy`: Library for numerical computations.

4. **Download or Place Your Trained Models**

   - Place your trained YOLOv8 detection and segmentation models in appropriate directories.
   - Update the model paths in the script accordingly.

## Running the Application

1. **Update Model Paths**

   In the script `main.py`, update the `DETECT_MODEL_PATH` and `SEG_MODEL_PATH` variables to point to your trained models.

   ```python
   DETECT_MODEL_PATH = 'path_to_your_detection_model.pt'
   SEG_MODEL_PATH = 'path_to_your_segmentation_model.pt'
   ```

2. **Run the FastAPI Application**

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8100 --reload
   ```

   - The application will start running at `http://0.0.0.0:8100`.

## API Endpoints

### Object Detection Endpoint

- **URL**: `/predict/detect`
- **Method**: `POST`
- **Request Body**:

  ```json
  {
    "image_base64": "base64_encoded_image_string",
    "output_subdir": "optional_output_subdirectory_name"
  }
  ```

- **Description**:

  - Performs object detection on the input image.
  - Saves outputs (annotated image, detections JSON, cropped images) in the specified output subdirectory under `outputs/`.
  - Only the most centrally located object is processed if multiple detections are found.

### Segmentation Endpoint

- **URL**: `/predict/segment`
- **Method**: `POST`
- **Request Body**:

  ```json
  {
    "image_base64": "base64_encoded_image_string",
    "output_subdir": "optional_output_subdirectory_name"
  }
  ```

- **Description**:

  - Performs segmentation on the input image.
  - Saves outputs (annotated image, segmentations JSON, cropped images) in the specified output subdirectory under `outputs/`.
  - Only the most centrally located object is processed if multiple segmentations are found.

### Image Processing Endpoint

- **URL**: `/process/image`
- **Method**: `POST`
- **Request Body**:

  ```json
  {
    "image_base64": "base64_encoded_image_string",
    "output_subdir": "name_of_output_subdirectory",
    "tint_color": [R, G, B]  // Optional, RGB values between 0 and 255
  }
  ```

- **Description**:

  - Processes the input image using the previously saved detection and segmentation outputs.
  - Applies the bounding box and segmentation mask, tinting pixels inside the bounding box but not in the segmentation mask with the specified color.
  - Saves the processed image in the specified output directory.

## Usage Examples

### Performing Object Detection

```python
import requests
import base64

# Read and encode the image
with open('input_image.jpg', 'rb') as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode()

# Prepare the payload
payload = {
    'image_base64': image_base64,
    'output_subdir': 'example_directory'  # Optional
}

# Send the POST request
response = requests.post('http://127.0.0.1:8100/predict/detect', json=payload)

# Check the response
if response.status_code == 200:
    data = response.json()
    print("Message:", data['message'])
    print("Detection outputs saved in 'outputs/example_directory/detection/'")
else:
    print("Request failed with status code:", response.status_code)
    print("Detail:", response.json())
```

### Performing Segmentation

```python
import requests
import base64

# Read and encode the image
with open('input_image.jpg', 'rb') as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode()

# Prepare the payload
payload = {
    'image_base64': image_base64,
    'output_subdir': 'example_directory'  # Optional
}

# Send the POST request
response = requests.post('http://127.0.0.1:8100/predict/segment', json=payload)

# Check the response
if response.status_code == 200:
    data = response.json()
    print("Message:", data['message'])
    print("Segmentation outputs saved in 'outputs/example_directory/segmentation/'")
else:
    print("Request failed with status code:", response.status_code)
    print("Detail:", response.json())
```

### Processing an Image

```python
import requests
import base64

# Read and encode the image
with open('input_image.jpg', 'rb') as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode()

# Prepare the payload
payload = {
    'image_base64': image_base64,
    'output_subdir': 'example_directory',
    'tint_color': [0, 255, 0]  # Optional tint color (green in this example)
}

# Send the POST request
response = requests.post('http://127.0.0.1:8100/process/image', json=payload)

# Check the response
if response.status_code == 200:
    data = response.json()
    print("Message:", data['message'])
    print("Processed image saved in 'outputs/example_directory/processed_image.png'")
else:
    print("Request failed with status code:", response.status_code)
    print("Detail:", response.json())
```

## Project Structure

```
.
├── main.py                  # The main FastAPI application script
├── outputs/                 # Directory where outputs are saved
│   └── example_directory/   # Example output subdirectory
│       ├── detection/       # Detection outputs
│       ├── segmentation/    # Segmentation outputs
│       └── processed_image.png  # Processed image
```

## Customization

- **Model Paths**: Update `DETECT_MODEL_PATH` and `SEG_MODEL_PATH` in `main.py` to point to your trained models.
- **Output Directory**: The base output directory is `outputs/`. You can change this by modifying the `BASE_OUTPUT_DIR` variable in `main.py`.
- **Tint Color**: When using the `/process/image` endpoint, you can specify a custom tint color by providing the `tint_color` parameter in the request body.

## Notes

- **Single Object Processing**: The application processes and outputs only the most centrally located object in the image when multiple detections or segmentations are found.
- **Output Subdirectories**: If `output_subdir` is not specified in the request, a timestamp-based directory will be created under `outputs/`.
- **Dependencies**: Ensure all required Python packages are installed. Use the provided installation instructions.
- **Error Handling**: The API includes error handling and will return appropriate HTTP status codes and messages if issues occur.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.