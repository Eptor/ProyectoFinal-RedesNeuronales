# Proyecto - Traductor de Lengua de Señas (Manual rápido)

Este repositorio contiene scripts para recolectar datos de manos, entrenar un clasificador simple y ejecutar un traductor en tiempo real usando MediaPipe y TensorFlow.

## Estructura relevante

- `main.py` : Ejecuta el traductor en tiempo real (usa el archivo de pesos `pesos_modelo.weights.h5`).
- `entrenar.py` : Script para entrenar la red neuronal y generar `pesos_modelo.weights.h5`.
- `recolector_google.py` : Herramienta para recolectar muestras de manos y generar un CSV de entrenamiento.
- `dataset_senas.csv` : CSV con las muestras recolectadas (recolector_google.py usa este nombre).
- `hand_landmarker.task` : Modelo de MediaPipe (debe estar en la raíz o en la misma carpeta que los scripts).

---

## Requisitos

- macOS (probado en entornos con Python 3.11)
- Cámara web
- Entorno virtual (recomendado)

Dependencias (instálalas en el entorno virtual):

```bash
python -m venv venv_final
source venv_final/bin/activate
pip install -r requirements.txt
```

Si ya existe `venv_final`, actívalo con `source venv_final/bin/activate`.

## Preparar el modelo de MediaPipe

- `main.py` y `recolector_google.py` usan `hand_landmarker.task`. Si no tienes ese archivo en la carpeta del proyecto, puedes:

- Ejecutar `recolector_google.py` (él descargará `hand_landmarker.task` automáticamente si falta).

Si trabajas en otra rama que contiene `recolector_lstm.py`, puedes usar ese script en su lugar; de lo contrario, usa `recolector_google.py` o descarga el archivo manualmente.
- O descargar manualmente desde:

https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

y colocarlo en la raíz del proyecto.

---

## Recolectar datos (opcional — sólo si quieres entrenar tú)

1. Ejecuta:

```bash
python recolector_google.py
```

2. Controles durante la captura:
- `r` : Toggle grabación (comienza / pausa).
- `n` : Clase siguiente.
- `p` : Clase anterior.
- `q` : Salir.

3. Observaciones:
-- `recolector_google.py` guarda las muestras en `dataset_senas.csv` (revisa la constante `CSV_FILE` dentro del script).
-- `entrenar.py` busca `dataset_senas.csv` por defecto. Si usas otro nombre, renómbralo o abre `entrenar.py` y ajusta la ruta.

---

## Entrenar el modelo

1. Asegúrate de tener el CSV con las muestras (`dataset_senas.csv` o renombra `dataset_secuencias.csv` a `dataset_senas.csv`).

2. Ejecuta:

```bash
python entrenar.py
```

3. Resultado:
- Al finalizar, `entrenar.py` guardará los pesos en `pesos_modelo.weights.h5` (este archivo es el que usa `main.py`).

Nota: Puedes ajustar `epochs`, `batch_size` y otras opciones dentro de `entrenar.py`.

---

## Ejecutar el traductor (inferencia en tiempo real)

1. Activa el entorno virtual y asegúrate de tener:
- `hand_landmarker.task` en la raíz
- `pesos_modelo.weights.h5` (generado por `entrenar.py`)

2. Ejecuta:

```bash
python main.py
```

3. Cámara:
- `main.py` abre `cv2.VideoCapture(1)` por defecto (línea `cap = cv2.VideoCapture(1)`). Si no se inicia la cámara, cambia el índice a `0` dentro de `main.py`.

4. Salir: presiona `q` en la ventana de OpenCV.

---

## Problemas comunes

- Error al cargar pesos: verifica que `pesos_modelo.weights.h5` exista y sea compatible con el modelo definido en `main.py` / `entrenar.py`.
- `hand_landmarker.task` no encontrado: ejecuta `recolector_lstm.py` o descárgalo manualmente.
- Cámara no detectada: prueba `cap = cv2.VideoCapture(0)` o verifica permisos de cámara en macOS (Preferencias del Sistema → Seguridad y Privacidad → Cámara).

---

## Recursos y notas

- Si quieres recolectar más datos, ajusta `FRAMES_POR_SECUENCIA` en `recolector_lstm.py` y adapta la arquitectura del modelo si cambias la dimensión de entrada.
- Archivos importantes: [main.py](main.py), [entrenar.py](entrenar.py), [recolector_lstm.py](recolector_lstm.py)

¿Deseas que añada instrucciones para crear un paquete `requirements.txt` mínimo o que ejecute una prueba rápida aquí? 
