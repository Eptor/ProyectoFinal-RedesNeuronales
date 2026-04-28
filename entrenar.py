import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM

NUM_CLASES = 29
FRAMES_POR_SECUENCIA = 20


def crear_modelo_lstm(num_clases: int = NUM_CLASES) -> tf.keras.Model:
    model = Sequential(
        [
            # La entrada ahora es una matriz (20 frames, 42 coordenadas)
            Input(shape=(FRAMES_POR_SECUENCIA, 42)),
            # Primera capa LSTM (devuelve secuencias para que la segunda LSTM las lea)
            LSTM(64, return_sequences=True, activation="tanh"),
            Dropout(0.2),
            # Segunda capa LSTM (procesa la info final y la aplana)
            LSTM(32, return_sequences=False, activation="tanh"),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(num_clases, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def entrenar():
    print("Cargando dataset de secuencias...")
    try:
        df = pd.read_csv("dataset_secuencias.csv")
    except FileNotFoundError:
        print("Error: No se encontró 'dataset_secuencias.csv'.")
        return

    # Extraer características
    X_aplanado = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    # RECONSTRUIR LAS SECUENCIAS:
    # Convertimos las filas de 840 valores de vuelta a bloques de (20, 42)
    num_muestras = X_aplanado.shape[0]
    X = X_aplanado.reshape((num_muestras, FRAMES_POR_SECUENCIA, 42))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Datos listos: {len(X_train)} secuencias para entrenar.")

    model = crear_modelo_lstm()
    model.summary()

    print("Iniciando el entrenamiento LSTM...")
    model.fit(
        X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test)
    )

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nPrecisión final en secuencias desconocidas: {accuracy * 100:.2f}%")

    model.save_weights("pesos_lstm.weights.h5")
    print("Pesos guardados como 'pesos_lstm.weights.h5'.")


if __name__ == "__main__":
    entrenar()
