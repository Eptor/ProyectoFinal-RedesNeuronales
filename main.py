import time
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

# ==========================================
# 1. DEFINICIÓN DEL MODELO
# ==========================================
NUM_CLASES = 29


def crear_modelo(num_clases: int = NUM_CLASES) -> tf.keras.Model:
    model = Sequential(
        [
            Input(shape=(42,)),
            Dense(128, activation="relu"),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(num_clases, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


CLASES_MAP = {i: chr(65 + i) for i in range(26)}
CLASES_MAP[26] = "CH"
CLASES_MAP[27] = "INICIO"
CLASES_MAP[28] = "FIN"


# ==========================================
# 2. DIBUJO ANTI-CRASH (Bypass M1)
# ==========================================
def dibujar_resultados(frame, hand_landmarks, prediccion_texto):
    height, width, _ = frame.shape

    CONEXIONES = [
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

    for start_idx, end_idx in CONEXIONES:
        p1 = (
            int(hand_landmarks[start_idx].x * width),
            int(hand_landmarks[start_idx].y * height),
        )
        p2 = (
            int(hand_landmarks[end_idx].x * width),
            int(hand_landmarks[end_idx].y * height),
        )
        cv2.line(frame, p1, p2, (255, 255, 255), 2)

    for lm in hand_landmarks:
        cv2.circle(frame, (int(lm.x * width), int(lm.y * height)), 5, (0, 0, 255), -1)

    cv2.rectangle(frame, (0, 0), (400, 60), (0, 0, 0), -1)
    cv2.putText(
        frame, prediccion_texto, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )


# ==========================================
# 3. LÓGICA PRINCIPAL DE TRADUCCIÓN
# ==========================================
def main():
    print("[INFO] Cargando modelo de TensorFlow...")
    model = crear_modelo()
    try:
        # Cargar los pesos con la extensión requerida por Keras 3
        model.load_weights("pesos_modelo.weights.h5")
        print("[OK] Pesos cargados correctamente.")
    except Exception as e:
        print(f"[ERROR] No se pudo cargar 'pesos_modelo.weights.h5': {e}")
        return

    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options, num_hands=1, running_mode=vision.RunningMode.VIDEO
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(1)  # Cambia a 0 si falla
    print("[INFO] Iniciando traductor. Presiona 'q' para salir.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = detector.detect_for_video(mp_image, int(time.time() * 1000))
        texto_display = "Buscando mano..."

        if result.hand_landmarks:
            hand_lms = result.hand_landmarks[0]

            # EXTRACCIÓN CON REFERENCIA A LA MUÑECA PARA LA IA
            coords = []
            wrist_x = hand_lms[0].x
            wrist_y = hand_lms[0].y

            for lm in hand_lms:
                rel_x = lm.x - wrist_x
                rel_y = lm.y - wrist_y
                coords.extend([rel_x, rel_y])

            # Predicción
            input_data = np.array([coords])
            pred = model.predict(input_data, verbose=0)
            clase_idx = np.argmax(pred)
            confianza = np.max(pred)

            if confianza > 0.7:
                letra = CLASES_MAP.get(clase_idx, "?")
                texto_display = f"Letra: {letra} ({confianza*100:.1f}%)"
            else:
                texto_display = "Incierto..."

            dibujar_resultados(frame, hand_lms, texto_display)
        else:
            cv2.putText(
                frame,
                texto_display,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Traductor", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
