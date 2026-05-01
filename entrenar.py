import os
import json
import random

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

NUM_CLASES = 29
FRAMES_POR_SECUENCIA = 20
COORDENADAS_POR_FRAME = 42

DATASET_PATH = "dataset_secuencias.csv"
MODELO_PATH = "modelo_lstm.keras"
PESOS_PATH = "pesos_lstm.weights.h5"
SCALER_PATH = "scaler_lstm.npz"
CONFIG_PATH = "config_lstm.json"

SEED = 42


def fijar_semillas(seed: int = SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def crear_modelo_lstm(num_clases: int = NUM_CLASES) -> tf.keras.Model:
    model = Sequential(
        [
            Input(shape=(FRAMES_POR_SECUENCIA, COORDENADAS_POR_FRAME)),
            LSTM(
                128,
                return_sequences=True,
                activation="tanh",
                recurrent_dropout=0.15,
                kernel_regularizer=l2(0.0005),
            ),
            BatchNormalization(),
            Dropout(0.35),
            LSTM(
                64,
                return_sequences=False,
                activation="tanh",
                recurrent_dropout=0.15,
                kernel_regularizer=l2(0.0005),
            ),
            BatchNormalization(),
            Dropout(0.35),
            Dense(64, activation="relu", kernel_regularizer=l2(0.0005)),
            BatchNormalization(),
            Dropout(0.30),
            Dense(32, activation="relu", kernel_regularizer=l2(0.0005)),
            Dropout(0.20),
            Dense(num_clases, activation="softmax"),
        ]
    )

    optimizer = Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def validar_dataset(df: pd.DataFrame):
    columnas_esperadas = 1 + (FRAMES_POR_SECUENCIA * COORDENADAS_POR_FRAME)

    if df.shape[1] != columnas_esperadas:
        raise ValueError(
            f"El dataset tiene {df.shape[1]} columnas, pero se esperaban "
            f"{columnas_esperadas}: 1 columna de clase + "
            f"{FRAMES_POR_SECUENCIA * COORDENADAS_POR_FRAME} características."
        )

    if df.isnull().sum().sum() > 0:
        raise ValueError("El dataset contiene valores vacíos o NaN.")

    clases_detectadas = sorted(df.iloc[:, 0].unique())
    clases_esperadas = list(range(NUM_CLASES))
    clases_faltantes = sorted(set(clases_esperadas) - set(clases_detectadas))

    print(f"Clases detectadas: {clases_detectadas}")
    print(f"Total de clases detectadas: {len(clases_detectadas)}")

    if clases_faltantes:
        print(f"[ADVERTENCIA] Faltan datos para estas clases: {clases_faltantes}")

    conteo_clases = df.iloc[:, 0].value_counts().sort_index()

    print("\nDistribución de clases:")
    print(conteo_clases)

    clase_menor = conteo_clases.min()
    clase_mayor = conteo_clases.max()

    print(f"\nMuestras en clase menor: {clase_menor}")
    print(f"Muestras en clase mayor: {clase_mayor}")
    print(f"Relación de desbalance: {clase_mayor / clase_menor:.2f}x")


def normalizar_secuencias(X_train, X_val, X_test):
    scaler = StandardScaler()

    X_train_2d = X_train.reshape(-1, COORDENADAS_POR_FRAME)
    X_val_2d = X_val.reshape(-1, COORDENADAS_POR_FRAME)
    X_test_2d = X_test.reshape(-1, COORDENADAS_POR_FRAME)

    scaler.fit(X_train_2d)

    X_train_norm = scaler.transform(X_train_2d).reshape(X_train.shape)
    X_val_norm = scaler.transform(X_val_2d).reshape(X_val.shape)
    X_test_norm = scaler.transform(X_test_2d).reshape(X_test.shape)

    np.savez(
        SCALER_PATH,
        mean=scaler.mean_,
        scale=scaler.scale_,
    )

    return X_train_norm, X_val_norm, X_test_norm


def crear_class_weight(y_train):
    clases_presentes = np.unique(y_train)

    pesos = compute_class_weight(
        class_weight="balanced",
        classes=clases_presentes,
        y=y_train,
    )

    class_weight = {
        int(clase): float(peso) for clase, peso in zip(clases_presentes, pesos)
    }

    print("\nPesos por clase:")
    for clase, peso in class_weight.items():
        print(f"Clase {clase}: {peso:.4f}")

    return class_weight


def entrenar():
    fijar_semillas()

    print("Cargando dataset de secuencias...")

    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: No se encontró '{DATASET_PATH}'.")
        return

    try:
        validar_dataset(df)
    except ValueError as error:
        print(f"Error en el dataset: {error}")
        return

    X_aplanado = df.iloc[:, 1:].values.astype("float32")
    y = df.iloc[:, 0].values.astype("int32")

    num_muestras = X_aplanado.shape[0]

    X = X_aplanado.reshape((num_muestras, FRAMES_POR_SECUENCIA, COORDENADAS_POR_FRAME))

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=SEED,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=SEED,
        stratify=y_temp,
    )

    X_train, X_val, X_test = normalizar_secuencias(X_train, X_val, X_test)

    class_weight = crear_class_weight(y_train)

    print("\nDatos listos:")
    print(f"- Entrenamiento: {len(X_train)} secuencias")
    print(f"- Validación: {len(X_val)} secuencias")
    print(f"- Prueba final: {len(X_test)} secuencias")

    model = crear_modelo_lstm(NUM_CLASES)
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=6,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            MODELO_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
    ]

    print("\nIniciando el entrenamiento LSTM...")

    history = model.fit(
        X_train,
        y_train,
        epochs=120,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weight,
        shuffle=True,
        verbose=1,
    )

    print("\nEvaluando con el conjunto de prueba final...")

    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

    print(f"\nPérdida final en prueba: {loss:.4f}")
    print(f"Precisión final en prueba: {accuracy * 100:.2f}%")

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nReporte por clase:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    model.save(MODELO_PATH)
    model.save_weights(PESOS_PATH)

    config = {
        "num_clases": NUM_CLASES,
        "frames_por_secuencia": FRAMES_POR_SECUENCIA,
        "coordenadas_por_frame": COORDENADAS_POR_FRAME,
        "modelo_path": MODELO_PATH,
        "pesos_path": PESOS_PATH,
        "scaler_path": SCALER_PATH,
        "dataset_path": DATASET_PATH,
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4, ensure_ascii=False)

    print(f"\nModelo completo guardado como '{MODELO_PATH}'.")
    print(f"Pesos guardados como '{PESOS_PATH}'.")
    print(f"Scaler guardado como '{SCALER_PATH}'.")
    print(f"Configuración guardada como '{CONFIG_PATH}'.")


if __name__ == "__main__":
    entrenar()
