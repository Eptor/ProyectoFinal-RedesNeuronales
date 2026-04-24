import time
import csv
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# 1. CONFIGURACIÓN DEL MODELO Y DATASET
# ==========================================
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path("hand_landmarker.task")
CSV_FILE = "dataset_senas.csv"

# Diccionario de tus 29 clases
CLASES_MAP = {i: chr(65 + i) for i in range(26)}
CLASES_MAP[26] = "CH"
CLASES_MAP[27] = "INICIO"
CLASES_MAP[28] = "FIN"


def descargar_modelo_google():
    if not MODEL_PATH.exists():
        print(f"[INFO] Descargando modelo oficial de Google desde {MODEL_URL}...")
        urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Modelo descargado con éxito.")


def iniciar_csv():
    if not Path(CSV_FILE).exists():
        with open(CSV_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            header = ["clase"]
            for i in range(21):
                header.extend([f"x{i}", f"y{i}"])
            writer.writerow(header)
        print("[INFO] Archivo dataset_senas.csv creado.")


# ==========================================
# 2. DIBUJO ANTI-CRASH (Bypass M1)
# ==========================================
def dibujar_landmarks_seguro(frame, detection_result):
    """Dibuja los puntos usando OpenCV puro y conexiones estáticas"""
    annotated_frame = frame.copy()
    height, width, _ = annotated_frame.shape

    if not detection_result.hand_landmarks:
        return annotated_frame

    # Lista oficial de conexiones
    CONEXIONES_MANO = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),  # Pulgar
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),  # Índice
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),  # Medio
        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),  # Anular
        (13, 17),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),  # Meñique
    ]

    for hand_landmarks in detection_result.hand_landmarks:
        # Dibujar conexiones
        for start_idx, end_idx in CONEXIONES_MANO:
            if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                punto_inicio = (
                    int(hand_landmarks[start_idx].x * width),
                    int(hand_landmarks[start_idx].y * height),
                )
                punto_fin = (
                    int(hand_landmarks[end_idx].x * width),
                    int(hand_landmarks[end_idx].y * height),
                )
                cv2.line(annotated_frame, punto_inicio, punto_fin, (255, 255, 255), 2)

        # Dibujar landmarks
        for landmark in hand_landmarks:
            punto = (int(landmark.x * width), int(landmark.y * height))
            cv2.circle(annotated_frame, punto, 4, (0, 0, 255), -1)

    return annotated_frame


# ==========================================
# 3. NÚCLEO DEL PROGRAMA (API Tasks)
# ==========================================
def main():
    descargar_modelo_google()
    iniciar_csv()

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(1)  # Cambia a 0 si falla en tu Mac
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    clase_actual = 0
    grabando = False
    muestras = 0

    print("[INFO] Cámara iniciada. Presiona 'r' grabar, 'n' siguiente, 'q' salir.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)

            detection_result = detector.detect_for_video(mp_image, timestamp_ms)
            frame_anotado = dibujar_landmarks_seguro(frame, detection_result)

            # LÓGICA DE EXTRACCIÓN CON REFERENCIA A LA MUÑECA
            if grabando and detection_result.hand_landmarks:
                hand = detection_result.hand_landmarks[0]
                fila = [clase_actual]

                wrist_x = hand[0].x
                wrist_y = hand[0].y

                for landmark in hand:
                    rel_x = landmark.x - wrist_x
                    rel_y = landmark.y - wrist_y
                    fila.extend([rel_x, rel_y])

                with open(CSV_FILE, mode="a", newline="") as f:
                    csv.writer(f).writerow(fila)
                muestras += 1

            # Interfaz
            nombre_clase = CLASES_MAP.get(clase_actual, "?")
            estado = "GRABANDO" if grabando else "PAUSA"
            color_estado = (0, 0, 255) if grabando else (255, 0, 0)

            cv2.rectangle(frame_anotado, (0, 0), (640, 80), (0, 0, 0), -1)
            cv2.putText(
                frame_anotado,
                f"Clase: {clase_actual} ({nombre_clase})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame_anotado,
                f"Estado: {estado} | Muestras: {muestras}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color_estado,
                2,
            )

            cv2.imshow("Recolector de Dataset", frame_anotado)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            elif tecla == ord("r"):
                grabando = not grabando
            elif tecla == ord("n"):
                clase_actual = min(28, clase_actual + 1)
                grabando = False
                muestras = 0
            elif tecla == ord("p"):
                clase_actual = max(0, clase_actual - 1)
                grabando = False
                muestras = 0

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Proceso finalizado.")


if __name__ == "__main__":
    main()
