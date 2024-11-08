# Install required packages
# pip install fastapi uvicorn pillow ultralytics opencv-python

import os
import time
import json
import logging
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the YOLOv8 models
DETECT_MODEL_PATH = "yolov8s.pt" #"'runs/detect/yolov8_bounding_box_experiment4/weights/best.pt'  # Replace with your trained detection model path
SEG_MODEL_PATH = "yolov8s-seg.pt"# "'runs/segment/yolov8_segmentation_experiment11/weights/best.pt'  # Replace with your trained segmentation model path

# Initialize the models
try:
    logger.info("Loading detection model...")
    detection_model = YOLO(DETECT_MODEL_PATH)
    logger.info("Detection model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load detection model: {e}")
    logger.debug(traceback.format_exc())
    raise RuntimeError(f"Failed to load detection model: {e}")

try:
    logger.info("Loading segmentation model...")
    segmentation_model = YOLO(SEG_MODEL_PATH)
    logger.info("Segmentation model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load segmentation model: {e}")
    logger.debug(traceback.format_exc())
    raise RuntimeError(f"Failed to load segmentation model: {e}")

# Ensure the base output directory exists
BASE_OUTPUT_DIR = 'outputs'
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# Define request and response models
class ImageRequest(BaseModel):
    image_base64: str
    output_subdir: Optional[str] = None  # Added field for output subdirectory name

class SimpleResponse(BaseModel):
    message: str

# Helper function to decode base64 image
def decode_base64_image(base64_string):
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        return image
    except Exception as e:
        logger.error(f"Invalid base64 image data: {e}")
        raise ValueError(f"Invalid base64 image data: {e}")

# Endpoint for object detection
@app.post("/predict/detect", response_model=SimpleResponse)
def predict_detect(request: ImageRequest):
    try:
        # Decode the base64 image
        image = decode_base64_image(request.image_base64)

        # Convert PIL Image to numpy array
        image_np = np.array(image)

        # Perform prediction
        results = detection_model.predict(source=image_np, save=False)

        # Process results
        detections = []
        for result in results:
            if result.boxes is not None:
                for idx, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = box.xyxy.squeeze().tolist()
                    confidence = box.conf.item()
                    class_id = int(box.cls.item())
                    class_name = detection_model.names.get(class_id, "Unknown")
                    detections.append({
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2]
                    })

        if detections:
            # Compute center of the image
            image_height, image_width = image_np.shape[:2]
            image_center = (image_width / 2, image_height / 2)

            # For each detection, compute the center of the bounding box and distance to image center
            distances = []
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                bbox_center_x = (x1 + x2) / 2
                bbox_center_y = (y1 + y2) / 2
                distance = ((bbox_center_x - image_center[0]) ** 2 + (bbox_center_y - image_center[1]) ** 2) ** 0.5
                distances.append(distance)

            # Find the detection with the minimal distance
            min_index = distances.index(min(distances))
            # Keep only the most central detection
            detections = [detections[min_index]]
        else:
            logger.warning("No detections found.")

        # Annotate image
        annotated_image = image_np.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated_image,
                f"{det['class_name']} {det['confidence']:.2f}",
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (36, 255, 12),
                2
            )

        # Determine the output directory
        if request.output_subdir:
            output_dir = os.path.join(BASE_OUTPUT_DIR, request.output_subdir, 'detection')
        else:
            # Use timestamp if no output_subdir is provided
            timestamp = int(time.time() * 1000)  # milliseconds since epoch
            output_dir = os.path.join(BASE_OUTPUT_DIR, str(timestamp), 'detection')
        os.makedirs(output_dir, exist_ok=True)

        # Save the annotated image
        annotated_image_pil = Image.fromarray(annotated_image)
        annotated_image_path = os.path.join(output_dir, 'annotated_image.png')
        annotated_image_pil.save(annotated_image_path)

        # Save detections as JSON
        detections_json_path = os.path.join(output_dir, 'detections.json')
        with open(detections_json_path, 'w') as f:
            json.dump(detections, f, indent=4)

        # Save cropped images
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = map(int, det['bbox'])
            cropped_image = image_np[y1:y2, x1:x2]
            cropped_image_pil = Image.fromarray(cropped_image)
            cropped_image_path = os.path.join(output_dir, f'cropped_{idx}.png')
            cropped_image_pil.save(cropped_image_path)

        logger.info(f"Detection results saved to {output_dir}")
        return SimpleResponse(message="Detection completed successfully.")

    except Exception as e:
        logger.error(f"Error in predict_detect: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for segmentation
@app.post("/predict/segment", response_model=SimpleResponse)
def predict_segment(request: ImageRequest):
    try:
        # Decode the base64 image
        image = decode_base64_image(request.image_base64)

        # Convert PIL Image to numpy array
        image_np = np.array(image)
        original_height, original_width = image_np.shape[:2]  # Store original image size

        # Perform prediction
        results = segmentation_model.predict(source=image_np, save=False, task='segment')

        # Process results
        segmentations = []
        for result in results:
            masks = result.masks
            boxes = result.boxes
            if masks is not None and boxes is not None:
                # Get the ratio of the original image to the model input size
                input_height, input_width = result.orig_shape[:2]  # Original shape before resizing
                ratio_height = original_height / input_height
                ratio_width = original_width / input_width

                for idx, (mask_data, box) in enumerate(zip(masks.data, boxes)):
                    # Rescale mask to original image size
                    mask = mask_data.cpu().numpy().astype(np.uint8)
                    mask_resized = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

                    # Rescale the box coordinates
                    x1, y1, x2, y2 = box.xyxy.squeeze().tolist()
                    x1 *= ratio_width
                    y1 *= ratio_height
                    x2 *= ratio_width
                    y2 *= ratio_height

                    # Convert coordinates to integers
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    # Find contours on the resized mask
                    contours, _ = cv2.findContours(mask_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    segmentation_coords = [cnt.squeeze(1).tolist() for cnt in contours if cnt.ndim == 3]

                    confidence = box.conf.item()
                    class_id = int(box.cls.item())
                    class_name = segmentation_model.names.get(class_id, "Unknown")
                    segmentations.append({
                        "class_name": class_name,
                        "confidence": confidence,
                        "segmentation_mask": segmentation_coords,
                        "bbox": [x1, y1, x2, y2],  # Include bbox
                        "mask": mask_resized  # Include resized mask
                    })
            else:
                logger.warning("No masks or boxes found in the result.")

        if segmentations:
            # Compute center of the image
            image_height, image_width = image_np.shape[:2]
            image_center = (image_width / 2, image_height / 2)

            # For each segmentation, compute the center of the bounding box and distance to image center
            distances = []
            for seg in segmentations:
                x1, y1, x2, y2 = seg['bbox']
                bbox_center_x = (x1 + x2) / 2
                bbox_center_y = (y1 + y2) / 2
                distance = ((bbox_center_x - image_center[0]) ** 2 + (bbox_center_y - image_center[1]) ** 2) ** 0.5
                distances.append(distance)

            # Find the segmentation with the minimal distance
            min_index = distances.index(min(distances))
            # Keep only the most central segmentation
            seg = segmentations[min_index]
            segmentations = [seg]  # Keep only the most central segmentation

            # Update the image annotation to only include the most central segmentation
            # Reset the image_np to the original image
            image_np = np.array(image)

            mask = seg['mask']
            x1, y1, x2, y2 = map(int, seg['bbox'])
            class_name = seg['class_name']
            confidence = seg['confidence']

            # Find contours on the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Annotate image
            cv2.drawContours(image_np, contours, -1, (0, 255, 0), 2)
            cv2.putText(
                image_np,
                f"{class_name} {confidence:.2f}",
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (36, 255, 12),
                2
            )
        else:
            logger.warning("No segmentations found.")

        # Determine the output directory
        if request.output_subdir:
            output_dir = os.path.join(BASE_OUTPUT_DIR, request.output_subdir, 'segmentation')
        else:
            # Use timestamp if no output_subdir is provided
            timestamp = int(time.time() * 1000)  # milliseconds since epoch
            output_dir = os.path.join(BASE_OUTPUT_DIR, str(timestamp), 'segmentation')
        os.makedirs(output_dir, exist_ok=True)

        # Save the annotated image
        annotated_image_pil = Image.fromarray(image_np)
        annotated_image_path = os.path.join(output_dir, 'annotated_image.png')
        annotated_image_pil.save(annotated_image_path)

        # Save segmentations as JSON
        segmentations_json_path = os.path.join(output_dir, 'segmentations.json')
        with open(segmentations_json_path, 'w') as f:
            # Remove the mask from the JSON data to avoid serialization issues
            seg_data = {k: v for k, v in seg.items() if k != 'mask'}
            json.dump([seg_data], f, indent=4)

        # Save cropped images
        for idx, seg in enumerate(segmentations):
            # Use the mask to extract the object
            mask = seg['mask']
            # Extract the object using the mask
            masked_image = cv2.bitwise_and(image_np, image_np, mask=mask)

            # Crop the image to the bounding rectangle of the mask
            x, y, w, h = cv2.boundingRect(mask)
            cropped_image = masked_image[y:y + h, x:x + w]

            cropped_image_pil = Image.fromarray(cropped_image)
            cropped_image_path = os.path.join(output_dir, f'cropped_{idx}.png')
            cropped_image_pil.save(cropped_image_path)

        logger.info(f"Segmentation results saved to {output_dir}")
        return SimpleResponse(message="Segmentation completed successfully.")

    except Exception as e:
        logger.error(f"Error in predict_segment: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# Function to process the image as per your requirement
def process_image(bbox_json_path, segmentation_json_path, image_base64, output_dir, tint_color=(0, 255, 0)):
    """
    Processes the input image by applying the bounding box and segmentation mask.
    Pixels inside the bounding box but not in the segmentation mask are colored with the specified tint color.

    Parameters:
    - bbox_json_path: Path to the bounding box JSON file.
    - segmentation_json_path: Path to the segmentation JSON file.
    - image_base64: Base64-encoded string of the input image.
    - output_dir: Directory where the output image will be saved.
    - tint_color: Tuple specifying the RGB color to tint pixels (default is green (0, 255, 0)).
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Decode base64 image
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        image_np = np.array(image)
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        return False

    # Load bbox JSON
    try:
        with open(bbox_json_path, 'r') as f:
            bbox_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading bbox JSON: {e}")
        return False

    # Load segmentation JSON
    try:
        with open(segmentation_json_path, 'r') as f:
            segmentation_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading segmentation JSON: {e}")
        return False

    # Check if detections are available
    if not bbox_data:
        logger.warning("No detections found in bbox JSON.")
        return False
    if not segmentation_data:
        logger.warning("No segmentations found in segmentation JSON.")
        return False

    # Get bbox coordinates
    bbox = bbox_data[0]['bbox']
    x1, y1, x2, y2 = map(int, bbox)

    # Create a mask for the segmentation
    mask = np.zeros(image_np.shape[:2], dtype=np.uint8)

    # Draw the segmentation contours onto the mask
    segmentation = segmentation_data[0]
    segmentation_masks = segmentation['segmentation_mask']
    for contour in segmentation_masks:
        contour_np = np.array(contour, dtype=np.int32)
        cv2.drawContours(mask, [contour_np], -1, color=255, thickness=-1)

    # Crop the image and mask using the bbox coordinates
    cropped_image = image_np[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]

    # Create an output image initialized to the cropped image
    output_image = cropped_image.copy()

    # Create a boolean mask of the segmentation within the cropped area
    segmentation_mask_cropped = cropped_mask > 0  # True where mask is 255

    # Pixels not in the segmentation mask
    not_in_mask = np.logical_not(segmentation_mask_cropped)

    # Color pixels not in mask with the specified tint color
    output_image[not_in_mask] = tint_color  # RGB format

    # Save the output image
    output_image_pil = Image.fromarray(output_image)
    output_image_path = os.path.join(output_dir, 'processed_image.png')
    output_image_pil.save(output_image_path)
    logger.info(f"Processed image saved to {output_image_path}")
    return True


# New endpoint to process the image
class ProcessRequest(BaseModel):
    image_base64: str
    output_subdir: str  # Directory where bbox and segmentation outputs are located
    tint_color: Optional[List[int]] = None  # RGB color as a list of three integers


@app.post("/process/image", response_model=SimpleResponse)
def process_image_endpoint(request: ProcessRequest):
    try:
        # Validate tint_color
        if request.tint_color:
            if len(request.tint_color) != 3 or not all(0 <= c <= 255 for c in request.tint_color):
                raise ValueError("tint_color must be a list of three integers between 0 and 255")
            tint_color = tuple(request.tint_color)
        else:
            tint_color = (0, 255, 0)  # Default green color

        # Determine the output directory
        output_dir = os.path.join(BASE_OUTPUT_DIR, request.output_subdir)
        if not os.path.exists(output_dir):
            raise HTTPException(status_code=400, detail=f"Output directory '{output_dir}' does not exist.")

        # Paths to bbox and segmentation JSON files
        bbox_json_path = os.path.join(output_dir, 'detection', 'detections.json')
        segmentation_json_path = os.path.join(output_dir, 'segmentation', 'segmentations.json')

        # Check if the files exist
        if not os.path.exists(bbox_json_path):
            raise HTTPException(status_code=400, detail=f"Bounding box JSON file not found at '{bbox_json_path}'")
        if not os.path.exists(segmentation_json_path):
            raise HTTPException(status_code=400,
                                detail=f"Segmentation JSON file not found at '{segmentation_json_path}'")

        # Call the process_image function
        success = process_image(
            bbox_json_path=bbox_json_path,
            segmentation_json_path=segmentation_json_path,
            image_base64=request.image_base64,
            output_dir=output_dir,
            tint_color=tint_color
        )

        if success:
            return SimpleResponse(message="Image processing completed successfully.")
        else:
            raise HTTPException(status_code=500, detail="Image processing failed.")

    except Exception as e:
        logger.error(f"Error in process_image_endpoint: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# Run the app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8100)
