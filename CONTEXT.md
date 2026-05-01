# Contexto Maestro — Proyecto: Traductor de Lengua de Señas

Fecha: 2026-04-30

Este documento resume el propósito, arquitectura, dependencias y flujos principales del repositorio.

**1. Resumen del Proyecto**

- **Descripción:** Aplicación monolítica en Python para recolectar señales de mano, entrenar una red LSTM y realizar inferencia en tiempo real usando MediaPipe y OpenCV. El objetivo es traducir gestos de mano (alfabeto y marcas especiales) a letras/texto en tiempo real.
- **Objetivo principal:** Proveer un pipeline completo: captura (webcam) → extracción de landmarks (MediaPipe) → recolección y almacenado (CSV) → entrenamiento de LSTM → inferencia en tiempo real con visualización.

**2. Stack Tecnológico**

- **Lenguaje:** Python 3.11 (orientado a ejecución local en macOS con venv)
- **Visión / captura:** OpenCV (`opencv-contrib-python`)
- **Detección de manos:** MediaPipe (`mediapipe`, `mediapipe.tasks.python.vision.HandLandmarker`)
- **ML / Entrenamiento:** TensorFlow / Keras (uso de `tf.keras` y objetos `Sequential`, `LSTM`, `Dense`)
- **Dependencias principales (según `requirements.txt`):** mediapipe, numpy, opencv-contrib-python, matplotlib, pandas (no listado en requirements.txt pero usado), torch (presente), jax, scipy, h5py, etc.

Nota: `requirements.txt` no incluye explícitamente `tensorflow` ni `pandas`, aunque son usados por el código; esto es un punto ciego (ver sección 6).

**3. Arquitectura y Estructura**

- Tipo: Monolito de scripts Python organizados por responsabilidad (recolección, entrenamiento, inferencia). No hay servidor web, base de datos relacional ni microservicios.
- Archivos/clases clave:
  - `recolector_lstm.py` — Interfaz de captura en tiempo real con webcam, anotación de landmarks y guardado de secuencias en `dataset_secuencias.csv`.
  - `entrenar.py` — Lee CSV de secuencias (filas ya aplanadas), reconstruye shape (N, 20, 42), entrena LSTM y guarda pesos en `pesos_lstm.weights.h5`.
  - `main.py` — Carga pesos, crea detector MediaPipe (`hand_landmarker.task`), captura cámara y realiza inferencia sobre ventanas de 20 frames mostrando resultado en GUI OpenCV.
  - `dataset_secuencias.csv`, `dataset_senas.csv` — CSVs con datos recolectados (formato explicado más abajo).
- Patrón de diseño: Procedural/modular mínimo. Cada script actúa como una pequeña aplicación CLI con su `main()`.

**4. Modelos de Datos / Base de Datos**

- No hay base de datos; los datos se almacenan en CSVs planos.
- **Formato CSV (`dataset_secuencias.csv` / `dataset_senas.csv`):**
  - Primera columna: `clase` (entero 0..28)
  - Resto: 840 columnas que representan 20 frames × 21 puntos × 2 coordenadas (x,y) = 840 valores por muestra.
  - En los scripts, las coordenadas se almacenan como valores relativos al `wrist` (restando la posición de la muñeca) para cada landmark.
- **Entidad principal:** Secuencia de mano (una muestra) — atributos: clase (label) y matriz de forma (20, 42) con coordenadas normalizadas.
- **Mapeo de clases:** 29 clases: 0..25 → `A`..`Z`, 26 → `CH`, 27 → `INICIO`, 28 → `FIN`.

**5. Flujos Principales**

- Flujo de recolección (UX):
  1. Ejecutar `recolector_lstm.py` (captura cámara). Si `hand_landmarker.task` falta, el script lo descarga automáticamente.
  2. Navegar entre clases con `n`/`p`, presionar `r` para grabar secuencias de 20 frames; las secuencias completas se agregan al CSV.

- Flujo de entrenamiento:
  1. `entrenar.py` lee `dataset_secuencias.csv` (espera filas con 840 features + label).
  2. Reconstruye `X` con shape `(N, 20, 42)` y usa `train_test_split` para particionar.
  3. Entrena modelo LSTM (2 capas LSTM: 64 → 32, dropout, dense final softmax) y guarda pesos en `pesos_lstm.weights.h5`.

- Flujo de inferencia (tiempo real):
  1. `main.py` crea la misma arquitectura LSTM y carga pesos desde `pesos_lstm.weights.h5`.
  2. Captura frames, usa MediaPipe `HandLandmarker` para obtener landmarks, calcula vector de 42 features por frame (coords relativas), alimenta buffer deque de 20 frames.
  3. Cuando el buffer tiene 20 frames, forma un batch (1,20,42) y ejecuta `model.predict()`; si la confianza > 0.7, muestra la letra mapeada.

**6. Puntos Ciegos / Riesgos / Inconsistencias**

- Nombres inconsistentes entre README y código:
  - `README.md` menciona `recolector_google.py` y `pesos_modelo.weights.h5`, pero el repositorio contiene `recolector_lstm.py` y el código usa `pesos_lstm.weights.h5`.
  - Hay dos CSV presentes: `dataset_secuencias.csv` y `dataset_senas.csv`. Los scripts (`entrenar.py` y `recolector_lstm.py`) usan `dataset_secuencias.csv`, mientras que README alude a `dataset_senas.csv`.
- Dependencias faltantes/discordantes:
  - `requirements.txt` no incluye `tensorflow` ni `pandas`, pero ambos módulos son importados en `main.py`/`entrenar.py`. Esto puede causar errores al reproducir el entorno desde `requirements.txt`.
  - Keras aparece en el entorno del venv, pero la especificación en `requirements.txt` no es coherente con lo instalado en `venv_final`.
- Requisitos de ejecución/permiso:
  - Requiere cámara y permisos de cámara en macOS; índice por defecto de cámara en `main.py` es `1` (puede necesitar cambiarse a `0`).
- Archivos de modelo y assets:
  - `hand_landmarker.task` no está incluido en el repo (pero `recolector_lstm.py` descarga el archivo si falta). Si se desea ejecución offline sin descarga, hay que almacenar el `.task` en la raíz.
- Consideraciones de reproducibilidad:
  - No hay script de instalación reproducible (p. ej. `pip freeze > requirements.txt` actualizado). Falta `tensorflow` en `requirements.txt` y la versión exacta de TF (crítico para compatibilidad de pesos `.h5`).

**7. Recomendaciones rápidas (próximos pasos)**

- Corregir y unificar nombres: elegir `pesos_lstm.weights.h5` o `pesos_modelo.weights.h5` y actualizar `README.md` y scripts para que coincidan.
- Actualizar `requirements.txt` para incluir `tensorflow` (version compatible con `tf.keras`) y `pandas`. Ejemplo: `tensorflow==2.12.0` (ver compatibilidad con mediapipe en el entorno).
- Añadir un pequeño `scripts/setup.sh` o actualizar `README.md` con comandos reproducibles para crear y activar `venv` e instalar dependencias.
- Añadir tests básicos o un script `check_env.py` que valide la presencia de `hand_landmarker.task`, `pesos_lstm.weights.h5` y la cámara.

---

Archivo principal creado a partir del análisis de: `main.py`, `entrenar.py`, `recolector_lstm.py`, `requirements.txt`, `README.md`.

Si quieres, puedo:
- Actualizar `requirements.txt` con las dependencias faltantes y versiones recomendadas.
- Unificar nombres e insertar una pequeña comprobación al inicio de `main.py` para manejar la diferencia de nombre de pesos.

Archivo guardado: CONTEXT.md
