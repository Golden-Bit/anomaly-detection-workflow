import cv2
import os
import time
import base64
import requests
import numpy as np

def estrai_frame(api_url, save_dir, frame_interval=30):
    """
    Estrae frame da un'API, crea una sottodirectory per ogni frame
    con il nome in epoca di millisecondi, e salva il frame in quella directory.

    :param api_url: URL dell'API per ottenere il frame in formato Base64
    :param save_dir: Directory locale dove creare le sottodirectory
    :param frame_interval: Numero di frame tra ogni estrazione
    """
    # Crea la directory principale se non esiste
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    frame_count = 0

    while True:
        # Richiede il frame all'API
        try:
            response = requests.get(api_url)
            if response.status_code != 200:
                print("Errore nella richiesta all'API")
                break
            # Ottiene il frame in formato Base64 e lo decodifica
            frame_data = response.json().get("frame_base64")
            if not frame_data:
                print("Frame non trovato nella risposta API")
                break
            frame_bytes = base64.b64decode(frame_data)
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

            # Salva il frame solo se è il momento giusto (ogni frame_interval)
            if frame_count % frame_interval == 0:
                # Ottieni l'epoca in millisecondi
                epoch_millis = int(time.time() * 1000)

                # Crea una sottodirectory con l'epoca come nome
                frame_dir = os.path.join(save_dir, str(epoch_millis))
                os.makedirs(frame_dir, exist_ok=True)

                # Salva il frame all'interno della sottodirectory
                frame_path = os.path.join(frame_dir, f"input_frame.png")
                cv2.imwrite(frame_path, frame)
                print(f"Salvato: {frame_path}")

            frame_count += 1

        except Exception as e:
            print(f"Errore durante l'estrazione del frame: {e}")
            break

    print("Estrazione completata")


# Esempio di utilizzo
api_url = "http://192.168.200.161:5000/get_frame"  # URL dell'API che restituisce il frame in Base64
save_dir = "outputs"
frame_interval = 1  # salva un frame ogni 30
estrai_frame(api_url, save_dir, frame_interval)
