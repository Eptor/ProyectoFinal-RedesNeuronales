import time
import csv
from pathlib import Path
from urllib.request import urlretrieve
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURACIÓN ---
MODEL_PATH = Path("hand_landmarker.task")
CSV_FILE = "dataset_secuencias.csv"
FRAMES_POR_SECUENCIA = 20  # El tamaño de nuestro "clip" de video

CLASES_MAP = {i: chr(65 + i) for i in range(26)}
CLASES_MAP[26] = "CH"
CLASES_MAP[27] = "INICIO"
CLASES_MAP[28] = "FIN"


def iniciar_csv():
    if not Path(CSV_FILE).exists():
        with open(CSV_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            header = ["clase"]
            # 20 frames * 21 puntos * 2 coordenadas (x,y) = 840 columnas
            for f_idx in range(FRAMES_POR_SECUENCIA):
                for i in range(21):
                    header.extend([f"f{f_idx}_x{i}", f"f{f_idx}_y{i}"])
            writer.writerow(header)


def dibujar_landmarks_seguro(frame, detection_result):
    annotated_frame = frame.copy()
    height, width, _ = annotated_frame.shape
    if not detection_result.hand_landmarks:
        return annotated_frame

    CONEXIONES_MANO = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (13, 17),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    ]

    for hand_landmarks in detection_result.hand_landmarks:
        for start_idx, end_idx in CONEXIONES_MANO:
            if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                p1 = (
                    int(hand_landmarks[start_idx].x * width),
                    int(hand_landmarks[start_idx].y * height),
                )
                p2 = (
                    int(hand_landmarks[end_idx].x * width),
                    int(hand_landmarks[end_idx].y * height),
                )
                cv2.line(annotated_frame, p1, p2, (255, 255, 255), 2)
        for landmark in hand_landmarks:
            cv2.circle(
                annotated_frame,
                (int(landmark.x * width), int(landmark.y * height)),
                4,
                (0, 0, 255),
                -1,
            )
    return annotated_frame


def main():
    if not MODEL_PATH.exists():
        urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            MODEL_PATH,
        )
    iniciar_csv()

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options, num_hands=1, running_mode=vision.RunningMode.VIDEO
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(1)

    clase_actual = 0
    grabando = False
    muestras = 0
    buffer_secuencia = []  # Aquí guardaremos los frames temporalmente

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = detector.detect_for_video(mp_image, int(time.time() * 1000))
        frame_anotado = dibujar_landmarks_seguro(frame, detection_result)

        if grabando:
            if detection_result.hand_landmarks:
                hand = detection_result.hand_landmarks[0]
                wrist_x, wrist_y = hand[0].x, hand[0].y

                coords_frame = []
                for lm in hand:
                    coords_frame.extend([lm.x - wrist_x, lm.y - wrist_y])

                buffer_secuencia.append(coords_frame)

                # Si ya juntamos 20 frames, guardamos la secuencia completa
                if len(buffer_secuencia) == FRAMES_POR_SECUENCIA:
                    fila = [clase_actual]
                    for frame_coords in buffer_secuencia:
                        fila.extend(frame_coords)

                    with open(CSV_FILE, mode="a", newline="") as f:
                        csv.writer(f).writerow(fila)

                    muestras += 1
                    buffer_secuencia = []  # Vaciamos la caja para el siguiente clip
            else:
                # Si se pierde la mano, limpiamos el buffer para no guardar secuencias rotas
                buffer_secuencia = []

        # Interfaz
        estado = f"GRABANDO (Frame {len(buffer_secuencia)}/20)" if grabando else "PAUSA"
        color = (0, 0, 255) if grabando else (255, 0, 0)
        cv2.rectangle(frame_anotado, (0, 0), (640, 80), (0, 0, 0), -1)
        cv2.putText(
            frame_anotado,
            f"Clase: {clase_actual} ({CLASES_MAP.get(clase_actual, '?')})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame_anotado,
            f"{estado} | Muestras: {muestras}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
        cv2.imshow("Recolector LSTM", frame_anotado)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        elif tecla == ord("r"):
            grabando = not grabando
            buffer_secuencia = []
        elif tecla == ord("n"):
            clase_actual = min(28, clase_actual + 1)
            grabando, buffer_secuencia, muestras = False, [], 0
        elif tecla == ord("p"):
            clase_actual = max(0, clase_actual - 1)
            grabando, buffer_secuencia, muestras = False, [], 0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
