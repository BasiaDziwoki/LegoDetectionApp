from __future__ import annotations

import os, json, collections, warnings
from typing import Any, Tuple, List, cast

import numpy as np
import tensorflow as tf
from tensorflow import keras  # type: ignore[reportMissingImports]

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning, module="keras")
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)

stage1_epochs = 15        
stage2_epochs = 20         
top_layers_to_finetune = 60

use_class_weight_stage1 = True
use_class_weight_stage2 = False

use_mixup_in_ft = False   
mixup_alpha = 0.10


best_stage1_path = "best_tl_stage1.keras"
best_stage2_path = "best_tl_stage2.keras"
final_model_path = "lego_mobilenetv2_finetuned_288.keras"
final_model_ft_path = "lego_mobilenetv2_finetuned_288_ft_targeted.keras"
class_names_path = "class_names.json"

tflite_ft_path = "lego_mobilenetv2_288_ft.tflite"
tflite_stage2_path = "lego_mobilenetv2_288.tflite"


data_dir = "./dataset"
batch_size = 32
img_size = (288, 288)
seed = 123

data_augmentation = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(
            factor=0.05,
            fill_mode="constant",
            fill_value=0.0,
            interpolation="nearest",
            name="rand_rot",
        ),
        keras.layers.RandomZoom(
            height_factor=(-0.15, 0.15),
            width_factor=(-0.15, 0.15),
            fill_mode="constant",
            fill_value=0.0,
            interpolation="nearest",
            name="rand_zoom",
        ),
        keras.layers.RandomTranslation(
            height_factor=0.04,
            width_factor=0.04,
            fill_mode="constant",
            fill_value=0.0,
            interpolation="nearest",
            name="rand_trans",
        ),
        keras.layers.RandomContrast(0.10, name="rand_contrast"),
    ],
    name="augment",
)

def aug_map(x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = tf.cast(x, tf.float32) # type: ignore
    x = data_augmentation(x, training=True)
    return x, y

full_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False,
)

class_names = full_ds.class_names
num_classes = len(class_names)
print("Klasy:", class_names)
full_ds = full_ds.cache()
full_ds = full_ds.shuffle(10000, seed=seed, reshuffle_each_iteration=False)

with open(class_names_path, "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

num_batches = full_ds.cardinality().numpy()
print("Liczba batchy w full_ds:", num_batches)

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

train_batches = int(num_batches * train_ratio)
val_batches = int(num_batches * val_ratio)
test_batches = num_batches - train_batches - val_batches

print("Podzial batchy: train =", train_batches, "val =", val_batches, "test =", test_batches)

full_ds = full_ds.shuffle(10000, seed=seed, reshuffle_each_iteration=False)

train_ds = full_ds.take(train_batches)
rest_ds = full_ds.skip(train_batches)
val_ds = rest_ds.take(val_batches)
test_ds = rest_ds.skip(val_batches)

raw_train_ds = train_ds 

def oversample_classes(ds: tf.data.Dataset, target_ids: list[int], factor: int = 3) -> tf.data.Dataset:
    base = ds.shuffle(2048, reshuffle_each_iteration=True)
    streams = [base]
    for tid in target_ids:
        filt = base.unbatch().filter(lambda x, y: tf.equal(y, tid)).batch(batch_size)
        for _ in range(factor - 1):
            streams.append(filt)
    return tf.data.Dataset.sample_from_datasets(streams)

idx_3005 = class_names.index("3005 brick 1x1") if "3005 brick 1x1" in class_names else None
idx_3004 = class_names.index("3004 brick 1x2") if "3004 brick 1x2" in class_names else None

if idx_3005 is not None and idx_3004 is not None:
    train_ds = oversample_classes(train_ds, [idx_3005, idx_3004], factor=3)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = (
    train_ds
    .map(aug_map, num_parallel_calls=AUTOTUNE)
    .shuffle(2000)
    .prefetch(1)
)

val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

def _sample_beta(alpha: float, shape):
    a = tf.random.gamma(shape=shape, alpha=alpha, beta=1.0)
    b = tf.random.gamma(shape=shape, alpha=alpha, beta=1.0)
    lam = a / (a + b)
    return tf.cast(lam, tf.float32)

def mixup_in_batch(alpha: float, num_classes: int):
    def _fn(x, y):
        shp = tf.shape(x, out_type=tf.int32)
        bsz = tf.gather(shp, 0)

        idx = tf.random.shuffle(tf.range(bsz))
        x2 = tf.gather(x, idx)
        y2 = tf.gather(y, idx)

        lam = _sample_beta(alpha, tf.reshape(bsz, [1]))
        lam_x = tf.reshape(lam, [-1, 1, 1, 1])
        lam_y = tf.reshape(lam, [-1, 1])

        x = tf.cast(x, tf.float32)
        x2 = tf.cast(x2, tf.float32)

        x_mix = x * lam_x + x2 * (1.0 - lam_x)

        y1 = tf.one_hot(y, depth=num_classes, dtype=tf.float32)
        y2 = tf.one_hot(y2, depth=num_classes, dtype=tf.float32)
        y_mix = y1 * lam_y + y2 * (1.0 - lam_y)
        return x_mix, y_mix
    return _fn

class MixupAccuracy(tf.keras.metrics.Metric): # type: ignore
    def __init__(self, name="mixup_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_idx = tf.argmax(y_pred, axis=-1, output_type=tf.int64)
        rank = tf.rank(y_true)

        def from_soft():
            yt = tf.cast(y_true, tf.float32)
            return tf.argmax(yt, axis=-1, output_type=tf.int64)

        def from_sparse():
            return tf.cast(tf.reshape(y_true, [-1]), tf.int64)

        y_true_idx = tf.cond(tf.equal(rank, 2), from_soft, from_sparse)
        matches = tf.cast(tf.equal(y_true_idx, y_pred_idx), tf.float32)

        self.total.assign_add(tf.reduce_sum(matches))
        self.count.assign_add(tf.cast(tf.size(matches), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

base = keras.applications.MobileNetV2(
    include_top=False,
    input_shape=img_size + (3,),
    weights="imagenet",
)
base.trainable = False

inputs = keras.Input(shape=img_size + (3,), name="image")

x = keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="preproc")(inputs)

x = base(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.4)(x)
logits = keras.layers.Dense(num_classes, name="logits")(x)

model = keras.Model(inputs, logits, name="mobilenetv2_lego_288")

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[MixupAccuracy()],
)

callbacks_stage1 = [
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-6),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
    keras.callbacks.ModelCheckpoint(best_stage1_path, monitor="val_loss", save_best_only=True, verbose=1),
]

class_weight = None
if use_class_weight_stage1:
    counts = collections.Counter()
    for _, y in raw_train_ds.unbatch():
        counts[int(y.numpy())] += 1
    total = sum(counts.values())
    class_weight = {i: total / (num_classes * counts[i]) for i in counts}
    print("Class weights (stage1):", class_weight)

steps_per_epoch = train_batches

history1 = model.fit(
    train_ds.repeat(),
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    epochs=stage1_epochs,
    callbacks=callbacks_stage1,
    class_weight=class_weight,
)

def focal_ce_from_logits_bimodal(gamma: float = 2.0, alpha: float | list[float] = 0.25):
    alpha_vec = tf.constant(alpha, tf.float32) if isinstance(alpha, list) else None
    alpha_scalar = tf.constant(alpha, tf.float32) if not isinstance(alpha, list) else None

    def loss_fn(y_true, logits):
        y_true = tf.convert_to_tensor(y_true)
        is_soft = tf.equal(tf.rank(y_true), 2)

        def soft_branch():
            y = tf.cast(y_true, tf.float32)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            ce = -tf.reduce_sum(y * log_probs, axis=-1) # type: ignore
            p_t = tf.exp(-ce)
            weight = tf.pow(1.0 - p_t, gamma)
            if alpha_vec is not None:
                a = tf.reduce_sum(y * alpha_vec, axis=-1) # type: ignore
                ce2 = a * weight * ce
            else:
                ce2 = alpha_scalar * weight * ce  # type: ignore
            return tf.reduce_mean(ce2)

        def hard_branch():
            yi = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yi, logits=logits)
            p_t = tf.exp(-ce) # type: ignore
            weight = tf.pow(1.0 - p_t, gamma)
            if alpha_vec is not None:
                a = tf.gather(alpha_vec, yi)
                ce2 = a * weight * ce
            else:
                ce2 = alpha_scalar * weight * ce  # type: ignore
            return tf.reduce_mean(ce2)

        return tf.cond(is_soft, soft_branch, hard_branch)

    return loss_fn

base.trainable = True

for layer in base.layers:
    if isinstance(layer, keras.layers.BatchNormalization):
        layer.trainable = False

for layer in base.layers[:-top_layers_to_finetune]:
    if not isinstance(layer, keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=focal_ce_from_logits_bimodal(gamma=2.0, alpha=0.25),
    metrics=[MixupAccuracy()],
)

callbacks_stage2 = [
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-6),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
    keras.callbacks.ModelCheckpoint(best_stage2_path, monitor="val_loss", save_best_only=True, verbose=1),
]

class_weight_ft = class_weight if use_class_weight_stage2 else None

train_ds_ft_base = train_ds.repeat()
if use_mixup_in_ft:
    train_ds_ft = (
        train_ds_ft_base
        .map(mixup_in_batch(mixup_alpha, num_classes), num_parallel_calls=AUTOTUNE)
        .prefetch(1)
    )
    y_is_soft = True
else:
    train_ds_ft = train_ds_ft_base
    y_is_soft = False

history2 = model.fit(
    train_ds_ft,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    epochs=stage2_epochs,
    callbacks=callbacks_stage2,
    class_weight=None if y_is_soft else class_weight_ft,
)

model.save(final_model_path)
print(f"Zapisano: {final_model_path}")

tta_aug = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.05),
        keras.layers.RandomZoom(0.05),
        keras.layers.RandomTranslation(0.05, 0.05),
    ],
    name="tta_aug",
)

def _ensure_tensor(x: Any) -> tf.Tensor:
    if isinstance(x, (tuple, list)):
        x = x[0]
    return tf.convert_to_tensor(x)

def tta_predict(model_: keras.Model, ds: tf.data.Dataset, repeats: int = 5) -> np.ndarray:
    all_probs: List[np.ndarray] = []
    for batch in iter(ds):
        x_batch, _ = cast(Tuple[tf.Tensor, tf.Tensor], batch)
        probs_list: List[tf.Tensor] = []
        for _ in range(repeats):
            x_aug = tta_aug(tf.cast(x_batch, tf.float32), training=True)
            out = model_(x_aug, training=False)
            logits_ = _ensure_tensor(out)
            probs = tf.nn.softmax(logits_, axis=-1)
            probs_list.append(probs) # type: ignore
        stacked = tf.stack(probs_list, axis=0)
        probs_mean = tf.reduce_mean(stacked, axis=0)
        all_probs.append(cast(np.ndarray, probs_mean.numpy()))
    return np.vstack(all_probs)

def evaluate_and_report(ds: tf.data.Dataset, name: str, use_tta: bool = False) -> None:
    print(f"\n=== Ewaluacja {name} ({'TTA' if use_tta else 'bez TTA'}) ===")

    y_true_all = []
    y_pred_all = []
    top2 = keras.metrics.SparseTopKCategoricalAccuracy(k=2)

    for x_batch, y_batch in ds: # type: ignore
        if use_tta:
            probs_list = []
            for _ in range(5):
                x_aug = tta_aug(tf.cast(x_batch, tf.float32), training=True)
                logits = model(x_aug, training=False)
                probs_list.append(tf.nn.softmax(logits, axis=-1))
            probs = tf.reduce_mean(tf.stack(probs_list, axis=0), axis=0)
            logits_like = tf.math.log(tf.maximum(probs, 1e-9))
            y_pred = tf.argmax(probs, axis=-1, output_type=tf.int64)
        else:
            logits = model(tf.cast(x_batch, tf.float32), training=False)
            logits_like = logits
            y_pred = tf.argmax(logits, axis=-1, output_type=tf.int64)

        top2.update_state(y_batch, logits_like)
        y_true_all.append(y_batch.numpy())
        y_pred_all.append(y_pred.numpy())

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)

    print("Top-2 acc:", float(top2.result().numpy()))

    from sklearn.metrics import classification_report, confusion_matrix
    rep = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    print("\n=== Classification report ===\n", rep)
    print("=== Confusion matrix ===\n", cm)

    tag = f"{name}_{'tta' if use_tta else 'no_tta'}"
    with open(f"report_{tag}.txt", "w", encoding="utf-8") as f:
        f.write(rep) # type: ignore
    np.savetxt(f"confusion_{tag}.csv", cm, fmt="%d", delimiter=",")


print("\n=== Ewaluacja na walidacji (bez TTA) ===")
evaluate_and_report(val_ds, "val", use_tta=False)
print("\n=== Ewaluacja TTA (walidacja) ===")
evaluate_and_report(val_ds, "val", use_tta=True)
print("\n=== Test (bez TTA) ===")
evaluate_and_report(test_ds, "test", use_tta=False)
print("\n=== Test (TTA) ===")
evaluate_and_report(test_ds, "test", use_tta=True)

print("\n=== Krotki fine-tuning na wszystkie klocki ===")

augment_stronger = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.06),
        keras.layers.RandomZoom(
            height_factor=(-0.25, 0.10),
            width_factor=(-0.25, 0.10),
            fill_mode="constant",
            fill_value=0.0,
        ),
        keras.layers.RandomTranslation(
            height_factor=0.06,
            width_factor=0.06,
            fill_mode="constant",
            fill_value=0.0,
        ),
        keras.layers.RandomContrast(0.15),
        keras.layers.GaussianNoise(0.08),
    ],
    name="augment_stronger",
)

def with_strong_aug(ds: tf.data.Dataset) -> tf.data.Dataset:
    def _map(x, y):
        x = tf.cast(x, tf.float32)
        x = augment_stronger(x, training=True)
        return x, y
    return ds.map(_map, num_parallel_calls=1).prefetch(1)

train_ds_ft_aug = with_strong_aug(train_ds)

for layer in model.layers:
    if isinstance(layer, keras.layers.BatchNormalization):
        layer.trainable = False

for layer in model.layers:
    name = getattr(layer, "name", "")
    if any(k in name for k in ["block_15", "block_16", "Conv_1"]):
        if not isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss=focal_ce_from_logits_bimodal(gamma=2.0, alpha=0.25),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc"), MixupAccuracy()],

)

callbacks_ft_targeted = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6, verbose=1),
]

base_weights = class_weight.copy() if class_weight is not None else {i: 1.0 for i in range(num_classes)}
class_weight_ft_targeted = base_weights
print("Class weights (FT all):", class_weight_ft_targeted)

history_ft = model.fit(
    train_ds_ft_aug.repeat(),
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    epochs=5,
    class_weight=class_weight_ft_targeted,
    callbacks=callbacks_ft_targeted,
)

print("\n=== Test po FT (bez TTA) ===")
evaluate_and_report(test_ds, "test_after_ft", use_tta=False)
print("\n=== Test po FT (TTA) ===")
evaluate_and_report(test_ds, "test_after_ft", use_tta=True)

model.save(final_model_ft_path)
print(f"\nZapisano: {final_model_ft_path}")

print("\nKonwersja do TFLite (FT)…")

inputs_inf = keras.Input(shape=img_size + (3,), name="image")
x = keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="preproc")(inputs_inf)
x = base(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.4)(x)
logits_inf = keras.layers.Dense(num_classes, name="logits")(x)
model_inf = keras.Model(inputs_inf, logits_inf, name="mobilenetv2_lego_288_infer")

for layer in model_inf.layers:
    if layer.name in [l.name for l in model.layers]:
        try:
            layer.set_weights(model.get_layer(layer.name).get_weights())
        except Exception:
            pass

try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model_inf)
    converter.optimizations = [tf.lite.Optimize.DEFAULT] # type: ignore
    tflite_model = converter.convert()
    with open(tflite_ft_path, "wb") as f:
        f.write(tflite_model) # type: ignore
    print("Zapisano:", tflite_ft_path)
except Exception as e:
    print("TFLite export (FT) pominiety:", e)

print("Gotowe.")
