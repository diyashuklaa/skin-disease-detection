"""
Train a MobileNet transfer-learning classifier on HAM10000.

Expected dataset layout (create it from the Kaggle HAM10000 download):
    data/HAM10000_images/*.jpg           (all images)
    data/HAM10000_metadata.csv           (columns: image_id, dx, ...)

Run from the project root:
    python training/train_cnn_improved.py

Outputs:
    model/skin_model.h5
    outputs/accuracy_curve.png, loss_curve.png,
    outputs/confusion_matrix.png, classification_report.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras import layers, models, optimizers, callbacks

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "data", "HAM10000_images")
META = os.path.join(ROOT, "data", "HAM10000_metadata.csv")
OUT_DIR = os.path.join(ROOT, "outputs")
MODEL_DIR = os.path.join(ROOT, "model")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH = 32
EPOCHS_HEAD = 8
EPOCHS_FINE = 12
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]  # must match app.py
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# ---------------------------------------------------------------- data
df = pd.read_csv(META)
df["path"] = df["image_id"].apply(lambda i: os.path.join(IMG_DIR, f"{i}.jpg"))
df["label"] = df["dx"].map(CLASS_TO_IDX)

# Split by lesion_id so the same lesion never appears in train AND test (prevents leakage)
lesions = df.drop_duplicates("lesion_id")[["lesion_id", "label"]]
train_l, test_l = train_test_split(lesions, test_size=0.15, stratify=lesions["label"], random_state=42)
train_l, val_l = train_test_split(train_l, test_size=0.15, stratify=train_l["label"], random_state=42)
train_df = df[df["lesion_id"].isin(train_l["lesion_id"])]
val_df = df[df["lesion_id"].isin(val_l["lesion_id"])]
test_df = df[df["lesion_id"].isin(test_l["lesion_id"])]
print(len(train_df), len(val_df), len(test_df))


def load(path, label, augment=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, 0.15)
        img = tf.image.random_contrast(img, 0.85, 1.15)
    img = preprocess_input(img)  # MobileNet expects [-1, 1]
    return img, tf.one_hot(label, len(CLASSES))


def make_ds(frame, augment=False, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((frame["path"].values, frame["label"].values))
    if shuffle:
        ds = ds.shuffle(len(frame), seed=42)
    ds = ds.map(lambda p, l: load(p, l, augment), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


train_ds = make_ds(train_df, augment=True, shuffle=True)
val_ds = make_ds(val_df)
test_ds = make_ds(test_df)

# ---------------------------------------------------------------- class weights
weights = compute_class_weight("balanced", classes=np.arange(len(CLASSES)), y=train_df["label"].values)
class_weight = dict(enumerate(weights))
print("Class weights:", class_weight)

# ---------------------------------------------------------------- model
base = MobileNet(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(len(CLASSES), activation="softmax"),
])

cbs = [
    callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6),
]

# Phase 1: train only the new head
model.compile(optimizer=optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
h1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD,
               class_weight=class_weight, callbacks=cbs)

# Phase 2: fine-tune the top MobileNet layers at a low learning rate
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
model.compile(optimizer=optimizers.Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
h2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE,
               class_weight=class_weight, callbacks=cbs)

model.save(os.path.join(MODEL_DIR, "skin_model.h5"))

# ---------------------------------------------------------------- curves
def join(key):
    return h1.history[key] + h2.history[key]

for key, title, fname in [("accuracy", "Accuracy", "accuracy_curve.png"), ("loss", "Loss", "loss_curve.png")]:
    plt.figure(figsize=(7, 4))
    plt.plot(join(key), label="train")
    plt.plot(join("val_" + key), label="validation")
    plt.axvline(len(h1.history[key]) - 1, color="gray", ls="--", label="fine-tuning starts")
    plt.title(f"Training and validation {title.lower()}")
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=150)
    plt.close()

# ---------------------------------------------------------------- evaluation
y_true = np.concatenate([np.argmax(y, axis=1) for _, y in test_ds])
y_pred = np.argmax(model.predict(test_ds), axis=1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

report = classification_report(y_true, y_pred, target_names=CLASSES, digits=3)
print(report)
plt.figure(figsize=(8, 4))
plt.axis("off")
plt.text(0, 1, report, family="monospace", fontsize=10, va="top")
plt.savefig(os.path.join(OUT_DIR, "classification_report.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved model and evaluation outputs.")
