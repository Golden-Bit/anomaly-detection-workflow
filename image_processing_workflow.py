import requests
import base64
import os

# Configuration
API_URL_1 = 'http://127.0.0.1:8100'  # URL of your first FastAPI application
API_URL_2 = 'http://127.0.0.1:8101'  # URL of your second FastAPI application

def workflow(image_path,
             output_subdir):

    # Read and encode the image
    with open(image_path, 'rb') as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode()

    # Prepare the payload with the output_subdir
    payload = {
        'image_base64': image_base64,
        'output_subdir': output_subdir
    }

    # Step 1: Send request to the segmentation endpoint
    print("Sending request to the segmentation endpoint...")
    response_seg = requests.post(f'{API_URL_1}/predict/segment', json=payload)

    if response_seg.status_code == 200:
        data_seg = response_seg.json()
        print("Segmentation completed successfully.")
    else:
        print(f"Segmentation request failed with status code: {response_seg.status_code}")
        print("Detail:", response_seg.json())
        exit()

    # Step 2: Send request to the detection endpoint
    print("Sending request to the detection endpoint...")
    response_det = requests.post(f'{API_URL_1}/predict/detect', json=payload)

    if response_det.status_code == 200:
        data_det = response_det.json()
        print("Detection completed successfully.")
    else:
        print(f"Detection request failed with status code: {response_det.status_code}")
        print("Detail:", response_det.json())
        exit()

    # Step 3: Send request to the processing endpoint
    print("Sending request to the processing endpoint...")
    process_payload = {
        'image_base64': image_base64,
        'output_subdir': output_subdir,
        # 'tint_color': [0, 255, 0]  # Optional: specify the tint color
    }

    response_process = requests.post(f'{API_URL_1}/process/image', json=process_payload)

    if response_process.status_code == 200:
        data_process = response_process.json()
        print("Image processing completed successfully.")
        print(f"Processed image saved in 'outputs/{output_subdir}/processed_image.png'")
    else:
        print(f"Processing request failed with status code: {response_process.status_code}")
        print("Detail:", response_process.json())
        exit()

    # Step 4: Send request to the tile extractor endpoint
    print("Sending request to the tile extractor endpoint...")

    # Read the processed image
    processed_image_path = os.path.join('outputs', output_subdir, 'processed_image.png')
    if not os.path.exists(processed_image_path):
        print(f"Processed image not found at '{processed_image_path}'")
        exit()

    with open(processed_image_path, 'rb') as image_file:
        processed_image_base64 = base64.b64encode(image_file.read()).decode()

    # Prepare the payload for the tile extractor
    tile_payload = {
        'image_base64': processed_image_base64,
        'tile_height_percent': None,  # Adjust as needed
        'tile_width_percent': 25.0,   # Adjust as needed
        'vertical_overlap_percent': 0.0,  # Adjust as needed
        'horizontal_overlap_percent': 0.0,  # Adjust as needed
        'output_subdir': output_subdir  # Use the same output subdir
    }

    response_tiles = requests.post(f'{API_URL_2}/split/image', json=tile_payload)

    if response_tiles.status_code == 200:
        data_tiles = response_tiles.json()
        print("Tile extraction completed successfully.")
        print(f"Tiles saved in 'outputs/{output_subdir}/processed_tiles/'")
        print("Message:", data_tiles['message'])
    else:
        print(f"Tile extraction request failed with status code: {response_tiles.status_code}")
        print("Detail:", response_tiles.json())
        exit()

    # Step 5: Send request to the clustering endpoint
    print("Sending request to the clustering endpoint...")

    # Prepare the payload for clustering
    cluster_payload = {
        'output_subdir': output_subdir,  # Same output subdir where tiles are stored
        'n_clusters': 10  # Adjust the number of clusters as needed
    }

    response_cluster = requests.post(f'{API_URL_2}/cluster/tiles', json=cluster_payload)

    if response_cluster.status_code == 200:
        data_cluster = response_cluster.json()
        print("Clustering completed successfully.")
        print(f"Clustered tiles saved in 'outputs/{output_subdir}/clusters/'")
        print("Message:", data_cluster['message'])
    else:
        print(f"Clustering request failed with status code: {response_cluster.status_code}")
        print("Detail:", response_cluster.json())
        exit()


if __name__ == "__main__":
    OUTPUT_DIR = 'outputs'
    #IMAGE_PATH = 'frame_0001.png'  # Path to your input image
    #OUTPUT_SUBDIR = 'test_output'  # Subdirectory name for outputs

    while True:
        for subdir in os.listdir(OUTPUT_DIR):
            try:
                output_subdir = f"{OUTPUT_DIR}\\{subdir}"
                image_path = f"{output_subdir}\\input_frame.png"
                tiles_subdir = f"{output_subdir}\\processed_tiles"
                #predictions_subdir = f"{output_subdir}\\predicted_tiles"

                if not os.path.isdir(tiles_subdir) and os.path.exists(image_path):

                    workflow(
                        image_path=image_path,
                        output_subdir=subdir
                    )
            except Exception as e:
                print(f"[ERROR]: {e}")
