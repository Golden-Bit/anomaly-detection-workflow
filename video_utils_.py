import cv2
import os
import time
from onvif import ONVIFCamera


def regola_messa_a_fuoco(ip, porta, utente, password, posizione_fuoco):
    """
    Regola la messa a fuoco della telecamera utilizzando ONVIF.

    :param ip: Indirizzo IP della telecamera
    :param porta: Porta ONVIF della telecamera
    :param utente: Nome utente della telecamera
    :param password: Password della telecamera
    :param posizione_fuoco: Valore di messa a fuoco tra 0.0 e 1.0
    """
    # Connetti alla telecamera tramite ONVIF
    cam = ONVIFCamera(ip, porta, utente, password)

    # Ottieni i servizi di imaging e media
    media_service = cam.create_media_service()
    imaging_service = cam.create_imaging_service()

    # Ottieni il primo profilo della telecamera
    profiles = media_service.GetProfiles()
    token = profiles[0].token

    # Imposta la messa a fuoco
    focus_request = imaging_service.create_type('Move')
    focus_request.VideoSourceToken = token
    focus_request.Focus = {'Absolute': {'Position': posizione_fuoco}}

    imaging_service.Move(focus_request)
    print(f"Messa a fuoco impostata a {posizione_fuoco}")


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


# Configurazione telecamera ONVIF
ip_camera = "192.168.200.205"
porta_camera = 80  # Porta ONVIF standard, controlla le impostazioni della tua telecamera
utente_camera = "user"  # Inserisci il tuo nome utente
password_camera = "password"  # Inserisci la tua password
posizione_fuoco = 5  # Valore tra 0.0 (minimo) e 1.0 (massimo)

# Regola la messa a fuoco della telecamera
regola_messa_a_fuoco(ip_camera, porta_camera, utente_camera, password_camera, posizione_fuoco)

# Estrai i frame dallo stream RTSP
rtsp_url = "rtsp://192.168.200.205:554/H264?ch=1"
save_dir = "outputs"
frame_interval = 30  # salva un frame ogni 30
estrai_frame(rtsp_url, save_dir, frame_interval)
