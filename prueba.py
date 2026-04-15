#!/usr/bin/env python3
"""
Hand landmarks detection with MediaPipe Tasks.

Converted from notebook format to a local Python script.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_URL = (
	"https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
	"hand_landmarker/float16/1/hand_landmarker.task"
)
IMAGE_URL = (
	"https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/"
	"woman_hands.jpg"
)

MODEL_PATH = Path("hand_landmarker.task")
IMAGE_PATH = Path("image.jpg")
OUTPUT_PATH = Path("image_annotated.jpg")


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


def draw_landmarks_on_image(image_array: np.ndarray, detection_result) -> np.ndarray:
	handedness_list = detection_result.handedness
	hand_landmarks_list = detection_result.hand_landmarks

	# MediaPipe Image often provides RGBA; drawing_utils expects 3-channel BGR.
	if image_array.ndim == 3 and image_array.shape[2] == 4:
		annotated_image = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
	elif image_array.ndim == 3 and image_array.shape[2] == 3:
		annotated_image = np.copy(image_array)
	else:
		raise ValueError(f"Formato de imagen no soportado: shape={image_array.shape}")

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
			annotated_image,
			hand_landmarks_proto,
			mp.solutions.hands.HAND_CONNECTIONS,
			mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
			mp.solutions.drawing_styles.get_default_hand_connections_style(),
		)

		height, width, _ = annotated_image.shape
		x_coordinates = [landmark.x for landmark in hand_landmarks]
		y_coordinates = [landmark.y for landmark in hand_landmarks]
		text_x = int(min(x_coordinates) * width)
		text_y = int(min(y_coordinates) * height) - MARGIN

		label = handedness_list[idx][0].category_name
		cv2.putText(
			annotated_image,
			label,
			(text_x, text_y),
			cv2.FONT_HERSHEY_DUPLEX,
			FONT_SIZE,
			HANDEDNESS_TEXT_COLOR,
			FONT_THICKNESS,
			cv2.LINE_AA,
		)

	return annotated_image


def main() -> None:
	log("Iniciando script de deteccion de hand landmarks...")

	ensure_file(MODEL_PATH, MODEL_URL, "Modelo")
	ensure_file(IMAGE_PATH, IMAGE_URL, "Imagen de prueba")

	log("Creando detector de manos...")
	base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
	options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
	detector = vision.HandLandmarker.create_from_options(options)

	log(f"Cargando imagen: {IMAGE_PATH}")
	image = mp.Image.create_from_file(str(IMAGE_PATH))

	log("Ejecutando inferencia...")
	detection_result = detector.detect(image)

	total_hands = len(detection_result.hand_landmarks)
	log(f"Inferencia terminada. Manos detectadas: {total_hands}")

	log("Dibujando landmarks sobre la imagen...")
	annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

	bgr_image = annotated_image
	cv2.imwrite(str(OUTPUT_PATH), bgr_image)
	log(f"Resultado guardado en: {OUTPUT_PATH}")

	# Intento de visualizacion local (puede fallar en entornos sin GUI).
	try:
		cv2.imshow("Hand Landmarks", bgr_image)
		log("Mostrando ventana. Presiona cualquier tecla para cerrar...")
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	except cv2.error:
		log("No se pudo abrir una ventana GUI. Revisa el archivo generado.")

	log("Proceso completado.")


if __name__ == "__main__":
	main()
