#!/usr/bin/env python3
"""
Hand landmarks detection with MediaPipe Tasks using webcam input.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Silencia un warning conocido de protobuf disparado internamente por MediaPipe.
warnings.filterwarnings(
	"ignore",
	message=r"SymbolDatabase.GetPrototype\(\) is deprecated.*",
	category=UserWarning,
)


MODEL_URL = (
	"https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
	"hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path("hand_landmarker.task")

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)


def log(message: str) -> None:
	print(f"[INFO] {message}")


def ensure_file(file_path: Path, url: str, label: str) -> None:
	if file_path.exists():
		log(f"{label} encontrado: {file_path}")
		return

	log(f"{label} no existe. Descargando desde: {url}")
	urlretrieve(url, file_path)
	log(f"{label} descargado en: {file_path}")


def draw_landmarks_on_frame(frame_bgr, detection_result, mirrored: bool = False):
	handedness_list = detection_result.handedness
	hand_landmarks_list = detection_result.hand_landmarks
	annotated_frame = frame_bgr.copy()

	for idx, hand_landmarks in enumerate(hand_landmarks_list):
		hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
		hand_landmarks_proto.landmark.extend(
			[
				landmark_pb2.NormalizedLandmark(
					x=landmark.x,
					y=landmark.y,
					z=landmark.z,
				)
				for landmark in hand_landmarks
			]
		)

		mp.solutions.drawing_utils.draw_landmarks(
			annotated_frame,
			hand_landmarks_proto,
			mp.solutions.hands.HAND_CONNECTIONS,
			mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
			mp.solutions.drawing_styles.get_default_hand_connections_style(),
		)

		height, width, _ = annotated_frame.shape
		x_coordinates = [landmark.x for landmark in hand_landmarks]
		y_coordinates = [landmark.y for landmark in hand_landmarks]
		text_x = int(min(x_coordinates) * width)
		text_y = int(min(y_coordinates) * height) - MARGIN

		label = handedness_list[idx][0].category_name
		if mirrored:
			if label == "Left":
				label = "Right"
			elif label == "Right":
				label = "Left"
		cv2.putText(
			annotated_frame,
			label,
			(text_x, text_y),
			cv2.FONT_HERSHEY_DUPLEX,
			FONT_SIZE,
			HANDEDNESS_TEXT_COLOR,
			FONT_THICKNESS,
			cv2.LINE_AA,
		)

	return annotated_frame


def main() -> None:
	log("Iniciando deteccion de hand landmarks en camara...")
	ensure_file(MODEL_PATH, MODEL_URL, "Modelo")

	log("Creando detector (modo VIDEO)...")
	base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
	options = vision.HandLandmarkerOptions(
		base_options=base_options,
		num_hands=1,
		running_mode=vision.RunningMode.VIDEO,
	)
	detector = vision.HandLandmarker.create_from_options(options)

	log("Abriendo camara por defecto (indice 1)...")
	cap = cv2.VideoCapture(1)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
	if not cap.isOpened():
		raise RuntimeError("No se pudo abrir la camara. Verifica permisos y dispositivo.")

	log("Camara abierta. Presiona 'q' para salir.")
	frame_count = 0
	start_time = time.time()

	try:
		while True:
			ok, frame_bgr = cap.read()
			if not ok:
				log("No se pudo leer un frame de la camara. Finalizando.")
				break

			# Efecto espejo para comportamiento tipo camara frontal.
			frame_bgr = cv2.flip(frame_bgr, 1)

			rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
			mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
			timestamp_ms = int(time.time() * 1000)

			detection_result = detector.detect_for_video(mp_image, timestamp_ms)
			annotated_frame = draw_landmarks_on_frame(
				frame_bgr,
				detection_result,
				mirrored=True,
			)

			frame_count += 1
			elapsed = max(time.time() - start_time, 1e-6)
			fps = frame_count / elapsed
			cv2.putText(
				annotated_frame,
				f"FPS: {fps:.1f}",
				(10, 30),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.8,
				(0, 255, 255),
				2,
				cv2.LINE_AA,
			)

			cv2.imshow("Hand Landmarks - Camara", annotated_frame)

			if cv2.waitKey(1) & 0xFF == ord("q"):
				log("Tecla 'q' detectada. Cerrando aplicacion.")
				break
	finally:
		cap.release()
		cv2.destroyAllWindows()
		log("Recursos liberados. Proceso completado.")


if __name__ == "__main__":
	main()
