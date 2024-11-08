import cv2
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

app = FastAPI()


def cattura_frame():
    """
    Cattura un singolo frame dalla telecamera USB.
    """
    cap = cv2.VideoCapture(0)  # Assumi che la telecamera USB sia sul dispositivo 0
    if not cap.isOpened():
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    # Codifica il frame in formato JPEG
    _, buffer = cv2.imencode('.jpg', frame)

    # Converti il frame codificato in Base64
    frame_base64 = base64.b64encode(buffer).decode('utf-8')

    return frame_base64


class FrameResponse(BaseModel):
    frame_base64: str


@app.get("/get_frame", response_model=FrameResponse)
async def get_frame():
    """
    Endpoint API che restituisce il frame della telecamera in formato Base64.
    """
    frame_base64 = cattura_frame()
    if frame_base64 is None:
        raise HTTPException(status_code=500, detail="Impossibile catturare il frame dalla telecamera")

    return {"frame_base64": frame_base64}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
