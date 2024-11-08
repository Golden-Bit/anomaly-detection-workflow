# Install required packages
# pip install fastapi uvicorn pillow tensorflow scikit-learn opencv-python

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Add this line to force TensorFlow to use CPU

import base64
from io import BytesIO
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

# Additional imports for clustering
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.cluster import KMeans
import traceback

app = FastAPI()

# Load the pre-trained ResNet50 model (without the top classification layer)
try:
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
except Exception as e:
    raise RuntimeError(f"Failed to load ResNet50 model: {e}")


# Define request models
class TileRequest(BaseModel):
    image_base64: str
    tile_height_percent: Optional[float] = None  # Percentage of image height
    tile_width_percent: Optional[float] = None  # Percentage of image width
    vertical_overlap_percent: float  # Overlap in percentage of tile height
    horizontal_overlap_percent: float  # Overlap in percentage of tile width
    output_subdir: str  # Output directory name


class ClusterRequest(BaseModel):
    output_subdir: str  # Name of the output subdirectory where tiles are stored
    n_clusters: int = 5  # Default number of clusters


# Helper function to decode base64 image
def decode_base64_image(base64_string):
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {e}")


# Function to load and preprocess image
def load_image(image):
    # Resize the image to 224x224 as required by ResNet50
    img = image.resize((224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


# Function to extract features from images
def extract_features(images):
    features = []
    for image in images:
        img = load_image(image)
        feature = model.predict(img)
        features.append(feature.flatten())
    return np.array(features)


# Function to save images into cluster subdirectories
def save_cluster_images(images, image_names, labels, n_clusters, output_dir):
    # Create the main output directory
    os.makedirs(output_dir, exist_ok=True)
    for cluster in range(n_clusters):
        cluster_dir = os.path.join(output_dir, f"cluster_{cluster}")
        os.makedirs(cluster_dir, exist_ok=True)
        for idx, label in enumerate(labels):
            if label == cluster:
                # Save the image in the cluster directory
                image = images[idx]
                image_name = image_names[idx]
                image_path = os.path.join(cluster_dir, image_name)
                image.save(image_path)


# Endpoint to split image into tiles
@app.post("/split/image")
def split_image(request: TileRequest):
    try:
        # Decode the base64 image
        image = decode_base64_image(request.image_base64)
        image_width, image_height = image.size

        # Default tile percentage if both are None
        default_tile_percent = 25.0  # Default tile size as 25% of image width

        # Determine tile_height_percent and tile_width_percent
        tile_height_percent = request.tile_height_percent
        tile_width_percent = request.tile_width_percent

        # Handle null values to make square tiles in pixels
        if tile_height_percent is None and tile_width_percent is None:
            # Use default_tile_percent for tile_width_percent
            tile_width_percent = default_tile_percent
            # Calculate tile_height_percent to make tiles square in pixels
            tile_width = image_width * (tile_width_percent / 100)
            tile_height = tile_width  # Square tile in pixels
            tile_height_percent = (tile_height / image_height) * 100
        elif tile_height_percent is None:
            # tile_width_percent is provided
            tile_width = image_width * (tile_width_percent / 100)
            tile_height = tile_width  # Square tile in pixels
            tile_height_percent = (tile_height / image_height) * 100
        elif tile_width_percent is None:
            # tile_height_percent is provided
            tile_height = image_height * (tile_height_percent / 100)
            tile_width = tile_height  # Square tile in pixels
            tile_width_percent = (tile_width / image_width) * 100

        # Validate percentages
        if not (0 < tile_height_percent <= 100) or not (0 < tile_width_percent <= 100):
            raise HTTPException(status_code=400, detail="Calculated tile percentages must be between 0 and 100.")

        # Calculate tile dimensions in pixels
        tile_height = int(image_height * (tile_height_percent / 100))
        tile_width = int(image_width * (tile_width_percent / 100))

        if tile_height <= 0 or tile_width <= 0:
            raise HTTPException(status_code=400, detail="Tile dimensions must be greater than zero.")

        # Calculate overlap steps
        vertical_step = int(tile_height * (1 - request.vertical_overlap_percent / 100))
        horizontal_step = int(tile_width * (1 - request.horizontal_overlap_percent / 100))

        if vertical_step <= 0 or horizontal_step <= 0:
            raise HTTPException(status_code=400, detail="Overlap steps must result in positive movement.")

        # Ensure output directory exists
        output_dir = os.path.join('outputs', request.output_subdir, "processed_tiles")
        os.makedirs(output_dir, exist_ok=True)

        # Initialize tile indices
        tile_index_vertical = 0

        # Slide over the image
        for y in range(0, image_height - tile_height + 1, vertical_step):
            tile_index_horizontal = 0
            for x in range(0, image_width - tile_width + 1, horizontal_step):
                # Crop the tile
                box = (x, y, x + tile_width, y + tile_height)
                tile = image.crop(box)

                # Save the tile with appropriate name
                tile_filename = f"tile_{tile_index_vertical}_{tile_index_horizontal}.png"
                tile_path = os.path.join(output_dir, tile_filename)
                tile.save(tile_path)

                tile_index_horizontal += 1

            tile_index_vertical += 1

        # Handle any remaining tiles at the edges
        # For right edge
        if (image_width - tile_width) % horizontal_step != 0 and image_width > tile_width:
            x = image_width - tile_width
            tile_index_horizontal = tile_index_horizontal
            tile_index_vertical = 0
            for y in range(0, image_height - tile_height + 1, vertical_step):
                box = (x, y, x + tile_width, y + tile_height)
                tile = image.crop(box)

                tile_filename = f"tile_{tile_index_vertical}_{tile_index_horizontal}.png"
                tile_path = os.path.join(output_dir, tile_filename)
                tile.save(tile_path)

                tile_index_vertical += 1

        # For bottom edge
        if (image_height - tile_height) % vertical_step != 0 and image_height > tile_height:
            y = image_height - tile_height
            tile_index_vertical = tile_index_vertical
            tile_index_horizontal = 0
            for x in range(0, image_width - tile_width + 1, horizontal_step):
                box = (x, y, x + tile_width, y + tile_height)
                tile = image.crop(box)

                tile_filename = f"tile_{tile_index_vertical}_{tile_index_horizontal}.png"
                tile_path = os.path.join(output_dir, tile_filename)
                tile.save(tile_path)

                tile_index_horizontal += 1

        # Handle the bottom-right corner tile
        if ((image_width - tile_width) % horizontal_step != 0 and image_width > tile_width) and \
                ((image_height - tile_height) % vertical_step != 0 and image_height > tile_height):
            x = image_width - tile_width
            y = image_height - tile_height
            box = (x, y, x + tile_width, y + tile_height)
            tile = image.crop(box)

            tile_filename = f"tile_{tile_index_vertical}_{tile_index_horizontal}.png"
            tile_path = os.path.join(output_dir, tile_filename)
            tile.save(tile_path)

        return {"message": f"Image split into tiles and saved in '{output_dir}'."}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint to cluster tiles
@app.post("/cluster/tiles")
def cluster_tiles(request: ClusterRequest):
    try:
        # Path to the directory containing the tiles
        tiles_dir = os.path.join('outputs', request.output_subdir, 'processed_tiles')
        if not os.path.exists(tiles_dir):
            raise HTTPException(status_code=400, detail=f"Tiles directory '{tiles_dir}' does not exist.")

        # Get list of tile image paths
        image_paths = [os.path.join(tiles_dir, img) for img in os.listdir(tiles_dir) if img.endswith('.png')]
        if not image_paths:
            raise HTTPException(status_code=400, detail=f"No tile images found in '{tiles_dir}'.")

        # Load images and extract features
        images = []
        image_names = []
        for image_path in image_paths:
            image = Image.open(image_path).convert('RGB')
            images.append(image)
            image_name = os.path.basename(image_path)
            image_names.append(image_name)

        # Extract features
        features = extract_features(images)

        # Perform KMeans clustering
        n_clusters = request.n_clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(features)
        labels = kmeans.labels_.tolist()

        # Save images into cluster subdirectories
        output_dir = os.path.join('outputs', request.output_subdir, 'clusters')
        save_cluster_images(images, image_names, labels, n_clusters, output_dir)

        return {"message": f"Tiles clustered and saved in '{output_dir}'."}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Run the app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8101)
