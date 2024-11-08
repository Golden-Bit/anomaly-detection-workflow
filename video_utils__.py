import cv2
import os
import time


def estrai_frame(rtsp_url, save_dir, frame_interval=30):
    """
    Estrae frame da un flusso RTSP, crea una sottodirectory per ogni frame
    con il nome in epoca di millisecondi, e salva il frame in quella directory.

    :param rtsp_url: URL RTSP dello stream
    :param save_dir: Directory locale dove creare le sottodirectory
    :param frame_interval: Numero di frame tra ogni estrazione
    """
    # Crea la directory principale se non esiste
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Apri lo stream
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    #cap.set(cv2.CAP_PROP_FOCUS, 50)
    if not cap.isOpened():
        print("Impossibile aprire il flusso RTSP")
        return

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

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

    cap.release()
    print("Estrazione completata")


# Esempio di utilizzo
rtsp_url = "rtsp://192.168.200.205:554/H264?ch=1"
save_dir = "outputs"
frame_interval = 30  # salva un frame ogni 30
estrai_frame(rtsp_url, save_dir, frame_interval)
