import time

import requests
import os
import pickle
from pathlib import Path


API_URL = 'http://127.0.0.1:8102/inference'  # URL dell'endpoint di inferenza


def predict(image_path,
            root_dir,
            output_dir,
            output_image,
            output_file):

    output_image_path = f'{output_dir}\\{output_image}'  # Percorso per salvare l'immagine di output (opzionale)
    output_file_path = f'{output_dir}\\{output_file}'  # Percorso per salvare i risultati in formato pickle (opzionale)

    # Assicurarsi che la directory di output esista
    os.makedirs(output_dir, exist_ok=True)

    # Preparazione dei parametri della richiesta
    payload = {
        'image_path': image_path,
        'root_dir': root_dir,
        #'openvino_model_path': MODEL_PATH,
        #'metadata': METADATA_PATH,
        'output_image_path': output_image_path,
        'output_file_path': output_file_path
    }

    # Invio della richiesta all'endpoint di inferenza
    print("Invio della richiesta di inferenza...")
    response = requests.post(API_URL, json=payload)

    # Gestione della risposta
    if response.status_code == 200:
        # Stampa i risultati di inferenza
        data = response.json()
        print("Inferenza completata con successo.")
        print("Risultati dell'inferenza:")
        print(data)

        # Salva i risultati in formato pickle se specificato
        if output_file_path:
            with open(output_file_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Risultati salvati in '{output_file_path}'")

    else:
        # In caso di errore, stampa il messaggio
        print(f"Inferenza fallita con codice di stato: {response.status_code}")
        print("Dettaglio:", response.json())


def workflow():

    #IMAGE_PATH = 'C:\\Users\\Hairemi_1\\Desktop\\anomalib_proj\\anomaly-detection-workflow\outputs\\test_output\clusters\cluster_0\\tile_1_1.png'  # Percorso dell'immagine per l'inferenza
    #ROOT_DIR = 'results/Padim/hazelnut_toy_train_1/latest'  # Directory radice per i risultati di inferenza
    OUTPUT_DIR = 'C:\\Users\\Hairemi_1\\Desktop\\anomalib_proj\\anomaly-detection-workflow\\outputs'

    for subdir in os.listdir(OUTPUT_DIR):
        try:
            output_subdir = f"{OUTPUT_DIR}\\{subdir}"
            tiles_subdir = f"{output_subdir}\\processed_tiles"
            predictions_subdir = f"{output_subdir}\\predicted_tiles"

            if os.path.isdir(tiles_subdir) and os.listdir(tiles_subdir):
                if not os.path.isdir(predictions_subdir):
                    for input_tile in os.listdir(tiles_subdir):

                        input_tile_path = f"{tiles_subdir}\\{input_tile}"
                        output_image = input_tile
                        output_file = f"{input_tile.split('.')[0]}.json"
                        output_dir = f"{output_subdir}\\predicted_tiles"

                        tile_name = input_tile.split('.')[0]
                        model_subdirectories = os.listdir('results/Padim')

                        model_subdirectory = [item for item in model_subdirectories if tile_name in item][0]

                        root_dir = f"results/Padim/{model_subdirectory}/latest"

                        predict(
                            image_path=input_tile_path,
                            #root_dir=ROOT_DIR,
                            root_dir=root_dir,
                            output_dir=output_dir,
                            output_image=output_image,
                            output_file=output_file,
                        )
                else:
                    print("skip to next tiles dir...")
            else:
                print("skip to next tiles dir...")
        except Exception as e:
            print(f"[ERROR]: {e}")


if __name__ == "__main__":

    while True:
        time.sleep(1)
        workflow()