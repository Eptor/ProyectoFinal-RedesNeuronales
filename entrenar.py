import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

# ==========================================
# 1. DEFINICIÓN DEL MODELO (Tu código)
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
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ==========================================
# 2. LÓGICA DE ENTRENAMIENTO
# ==========================================
def entrenar():
    print("Cargando dataset...")
    # Asegúrate de tener tu archivo CSV en la misma carpeta
    try:
        df = pd.read_csv("dataset_senas.csv")
    except FileNotFoundError:
        print(
            "Error: No se encontró 'dataset_senas.csv'. Necesitas recolectar datos primero."
        )
        return

    # Separar etiquetas (Y) y características (X)
    # Suponemos que la columna 0 es la clase y de la 1 a la 42 son las coordenadas
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    # Dividir en datos de entrenamiento (80%) y prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Datos listos: {len(X_train)} para entrenar, {len(X_test)} para validar.")

    # Crear el modelo
    model = crear_modelo()

    # Entrenar la red neuronal
    # Epochs es la cantidad de veces que la red verá todos los datos
    print("Iniciando el entrenamiento de la red neuronal...")
    historial = model.fit(
        X_train,
        y_train,
        epochs=50,  # Puedes subir esto a 100 si la precisión es baja
        batch_size=32,  # Procesa 32 ejemplos a la vez
        validation_data=(X_test, y_test),
    )

    # Evaluar el modelo final
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\n--- Resultados Finales ---")
    print(f"Precisión en datos desconocidos (prueba): {accuracy * 100:.2f}%")

    # Guardar los pesos para usarlos en tu script de traducción
    model.save_weights("pesos_modelo.weights.h5")
    print("\n¡Listo! Los pesos se han guardado como 'pesos_modelo.h5'.")


if __name__ == "__main__":
    entrenar()
