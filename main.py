import time
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model
from entrenar import crear_modelo_lstm

NUM_CLASES = 29
FRAMES_POR_SECUENCIA = 20
COORDENADAS_POR_FRAME = 42


def cargar_scaler_stats(base_dir: Path):
    scaler_path = base_dir / "scaler_lstm.npz"
    if not scaler_path.exists():
        print(f"[ADVERTENCIA] No se encontró scaler en: {scaler_path}")
        print("[ADVERTENCIA] Las predicciones se harán sin normalización.")
        return None

    data = np.load(scaler_path)
    print(f"[OK] Scaler cargado desde: {scaler_path}")
    return data["mean"], data["scale"]


def normalizar_secuencia(secuencia, scaler_stats):
    secuencia_array = np.asarray(secuencia, dtype=np.float32)

    if scaler_stats is None:
        return secuencia_array

    mean, scale = scaler_stats

    scale = np.where(scale == 0, 1.0, scale)

    return (secuencia_array - mean) / scale



def cargar_modelo(base_dir: Path):
    modelo_path = base_dir / "modelo_lstm.keras"
    pesos_path = base_dir / "pesos_lstm.weights.h5"

    print(f"[INFO] Buscando modelo completo en: {modelo_path}")
    print(f"[INFO] Buscando pesos en: {pesos_path}")

    if modelo_path.exists():
        try:
            model = load_model(str(modelo_path))
            print("[OK] Modelo completo cargado desde modelo_lstm.keras.")
            return model
        except Exception as e:
            print("[ERROR] El modelo completo existe, pero no se pudo cargar.")
            print(f"[ERROR] Detalle: {e}")
            print("[INFO] Intentando cargar pesos manualmente...")

    if not pesos_path.exists():
        print(f"[ERROR] No se encontró el archivo de modelo: {modelo_path}")
        print(f"[ERROR] No se encontró el archivo de pesos: {pesos_path}")
        return None

    model = crear_modelo_lstm()

    try:
        model.load_weights(str(pesos_path))
        print("[OK] Pesos LSTM cargados.")
        return model
    except Exception as e:
        print("[ERROR] El archivo de pesos existe, pero no se pudo cargar.")
        print(f"[ERROR] Detalle: {e}")
        print(
            "[AYUDA] Tus pesos no coinciden con la arquitectura actual del modelo. "
            "Vuelve a ejecutar entrenar.py para generar un modelo_lstm.keras nuevo, "
            "o elimina pesos_lstm.weights.h5 si corresponde a una arquitectura vieja."
        )
        return None


CLASES_MAP = {i: chr(65 + i) for i in range(26)}
CLASES_MAP[26] = "CH"
CLASES_MAP[28] = "FIN"


def dibujar_resultados(frame, hand_landmarks, prediccion_texto):
    height, width, _ = frame.shape

    conexiones = [
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

    for start_idx, end_idx in conexiones:
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
        cv2.circle(
            frame,
            (int(lm.x * width), int(lm.y * height)),
            5,
            (0, 0, 255),
            -1,
        )

    cv2.rectangle(frame, (0, 0), (500, 60), (0, 0, 0), -1)
    cv2.putText(
        frame,
        prediccion_texto,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )


def main():
    base_dir = Path(__file__).resolve().parent
    scaler_stats = cargar_scaler_stats(base_dir)

    print(f"[INFO] Directorio de ejecución: {Path.cwd()}")
    print(f"[INFO] Directorio del script: {base_dir}")

    model = cargar_modelo(base_dir)

    if model is None:
        return

    base_options = python.BaseOptions(
        model_asset_path=str(base_dir / "hand_landmarker.task")
    )
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(1)

    secuencia_buffer = deque(maxlen=FRAMES_POR_SECUENCIA)

    word_buffer = []
    current_candidate = None
    candidate_start_time = None
    cooldown_until = 0.0

    fin_candidate_start = None

    LETTER_CONFIRM_SECONDS = 3.0
    FIN_CONFIRM_SECONDS = 1.5
    COOLDOWN_SECONDS = 1.0
    CONFIDENCE_THRESHOLD = 0.7

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

            coords = []
            wrist_x, wrist_y = hand_lms[0].x, hand_lms[0].y

            for lm in hand_lms:
                coords.extend([lm.x - wrist_x, lm.y - wrist_y])

            if len(coords) == COORDENADAS_POR_FRAME:
                secuencia_buffer.append(coords)

            if len(secuencia_buffer) == FRAMES_POR_SECUENCIA:
                secuencia_normalizada = normalizar_secuencia(
                    secuencia_buffer,
                    scaler_stats,
                )

                input_data = np.expand_dims(secuencia_normalizada, axis=0)

                pred = model.predict(input_data, verbose=0)
                clase_idx = int(np.argmax(pred))
                confianza = float(np.max(pred))

                clase_text = CLASES_MAP.get(clase_idx, "?")

                now = time.time()

                if clase_idx == 28 and confianza > CONFIDENCE_THRESHOLD:
                    if fin_candidate_start is None:
                        fin_candidate_start = now
                        print(f"Iniciando conteo para: {clase_text}")
                    elif now - fin_candidate_start >= FIN_CONFIRM_SECONDS:
                        print("Seña FIN detectada: Borrando palabra.")
                        word_buffer = []
                        fin_candidate_start = None
                        current_candidate = None
                        candidate_start_time = None
                        cooldown_until = now + COOLDOWN_SECONDS

                    texto_display = f"Seña: {clase_text} ({confianza * 100:.1f}%)"

                else:
                    fin_candidate_start = None

                    if confianza > CONFIDENCE_THRESHOLD and (0 <= clase_idx <= 26):
                        if now < cooldown_until:
                            texto_display = (
                                f"Letra cooldown: {clase_text} "
                                f"({confianza * 100:.1f}%)"
                            )
                        else:
                            if current_candidate != clase_idx:
                                current_candidate = clase_idx
                                candidate_start_time = now
                                print(f"Iniciando conteo para: {clase_text}")
                                texto_display = (
                                    f"Contando: {clase_text} "
                                    f"({confianza * 100:.1f}%)"
                                )
                            else:
                                elapsed = now - (candidate_start_time or now)
                                texto_display = (
                                    f"Letra: {clase_text} "
                                    f"({confianza * 100:.1f}%) - {elapsed:.1f}s"
                                )

                                if elapsed >= LETTER_CONFIRM_SECONDS:
                                    letra_confirmada = clase_text
                                    print(f"Letra {letra_confirmada} confirmada.")
                                    word_buffer.append(letra_confirmada)
                                    print(f"Palabra actual: {''.join(word_buffer)}")

                                    current_candidate = None
                                    candidate_start_time = None
                                    cooldown_until = now + COOLDOWN_SECONDS
                    else:
                        texto_display = f"Incierto... ({confianza * 100:.1f}%)"
                        current_candidate = None
                        candidate_start_time = None

            dibujar_resultados(frame, hand_lms, texto_display)

        else:
            secuencia_buffer.clear()
            current_candidate = None
            candidate_start_time = None
            fin_candidate_start = None

            cv2.putText(
                frame,
                "Buscando mano...",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        palabra_actual = "".join(word_buffer)
        height, width, _ = frame.shape

        cv2.putText(
            frame,
            f"Palabra: {palabra_actual}",
            (10, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            3,
        )

        cv2.imshow("Traductor Dinamico LSTM", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
