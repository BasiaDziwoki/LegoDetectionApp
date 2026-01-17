import os
import json
import tensorflow as tf
from tensorflow import keras # type: ignore[reportMissingImports]
from tensorflow.keras import layers # type: ignore[reportMissingImports]

DATA_DIR = r"C:\Users\bdziw\Desktop\Studia\Praca_Inzynierska\Program\colors_dataset"

IMG_SIZE = 128
BATCH_SIZE = 32
VAL_SPLIT = 0.2
SEED = 42
EPOCHS = 30

COLOR_MODEL_PATH = "lego_color_model.keras"
COLOR_TFLITE_PATH = "lego_color_128.tflite"
COLOR_NAMES_PATH = "color_names.json"

print("TensorFlow:", tf.__version__)
print("Folder z kolorami:", DATA_DIR)

train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="int",
    validation_split=VAL_SPLIT,
    subset="training",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
)

val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="int",
    validation_split=VAL_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Klasy kolorow:", class_names)

with open(COLOR_NAMES_PATH, "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)
print(f"Zapisano etykiety do: {COLOR_NAMES_PATH}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(0.10, 0.10),
        layers.RandomZoom((-0.20, 0.20)),
        layers.RandomBrightness(factor=0.25),
        layers.RandomContrast(0.25),
    ],
    name="data_augmentation",
)

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
x = data_augmentation(inputs)
x = layers.Rescaling(1.0 / 255.0)(x)
x = layers.GaussianNoise(0.03)(x)

x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D()(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(num_classes, activation="softmax", name="probs")(x)


model = keras.Model(inputs=inputs, outputs=outputs, name="lego_color_cnn")
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        COLOR_MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1,
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

print("\nNajlepszy model zapisany jako:", COLOR_MODEL_PATH)

print("\nKonwersja do TFLite…")
best_model = keras.models.load_model(COLOR_MODEL_PATH)
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
tflite_model = converter.convert()

tflite_model = converter.convert()

with open(COLOR_TFLITE_PATH, "wb") as f:
    f.write(tflite_model) # type: ignore

print("Zapisano TFLite:", COLOR_TFLITE_PATH)
print("Gotowe.")
