
# -----------------------------------------------------
# Cell 1: Original computation block
# -----------------------------------------------------

# --- Reproducibility: set random seeds ---
import os, random, numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
print("Seeds set to", SEED)


# -----------------------------------------------------
# Cell 2: Original computation block
# -----------------------------------------------------

from pathlib import Path

BASE_DIR = Path('/mnt/data')
MODEL_DIR = BASE_DIR / 'model_output'
MODEL_DIR.mkdir(exist_ok=True, parents=True)


# -----------------------------------------------------
# Cell 3: Original computation block
# -----------------------------------------------------

import os, shutil, random, re
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models


# -----------------------------------------------------
# Cell 4: Original computation block
# -----------------------------------------------------

from google.colab import drive
drive.mount('/content/drive')


# -----------------------------------------------------
# Cell 5: Original computation block
# -----------------------------------------------------

!unzip "/content/drive/MyDrive/NEU-CLS.zip" -d /content/NEU_Images


# -----------------------------------------------------
# Cell 6: Original computation block
# -----------------------------------------------------

!find /content/NEU_Images -maxdepth 5 -type d


# -----------------------------------------------------
# Cell 7: Original computation block
# -----------------------------------------------------

!find /content/NEU_Images -maxdepth 5 -type f | head -n 20


# -----------------------------------------------------
# Cell 8: Original computation block
# -----------------------------------------------------

RAW_TRAIN = "/content/NEU_Images/train/train/images"
RAW_VALID = "/content/NEU_Images/valid/valid/images"

WORKDIR = "/content/NEU_Workdir"
NEU_DIR = f"{WORKDIR}/NEU_by_class"
E_NEU_DIR = f"{WORKDIR}/E_NEU"
COMBINED_DIR = f"{WORKDIR}/COMBINED"
MODEL_DIR = f"{WORKDIR}/models"

classes = ["crazing","inclusion","patches","pitted_surface","rolled-in_scale","scratches"]

os.makedirs(NEU_DIR, exist_ok=True)
os.makedirs(E_NEU_DIR, exist_ok=True)
os.makedirs(COMBINED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("Directories ready")


# -----------------------------------------------------
# Cell 9: Original computation block
# -----------------------------------------------------

import os
import shutil
from pathlib import Path

# Correct paths (based on your find output)
RAW_TRAIN = "/content/NEU_Images/train/train/images"
RAW_VALID = "/content/NEU_Images/valid/valid/images"

NEU_DIR = "/content/NEU_Workdir/NEU_by_class"

classes = ["crazing","inclusion","patches","pitted_surface","rolled-in_scale","scratches"]

# Create class directories
for cls in classes:
    os.makedirs(os.path.join(NEU_DIR, cls), exist_ok=True)

def collect_images(src):
    src = Path(src)
    if not src.exists():
        raise RuntimeError(f"Source folder missing: {src}")

    for fname in os.listdir(src):
        low = fname.lower()
        for cls in classes:
            if low.startswith(cls):
                shutil.copy(src / fname, os.path.join(NEU_DIR, cls, fname))
                break

collect_images(RAW_TRAIN)
collect_images(RAW_VALID)

print("NEU dataset organized by class!")

# Verify counts
for cls in classes:
    count = len(os.listdir(os.path.join(NEU_DIR, cls)))
    print(cls, ":", count)


# -----------------------------------------------------
# Cell 10: Original computation block
# -----------------------------------------------------

def motion_blur(img, degree=8, angle=45):
    image = np.array(img)

    # generate motion blur kernel
    M = np.zeros((degree, degree))
    M[int((degree-1)/2), :] = np.ones(degree)
    M = Image.fromarray(M)
    M = M.rotate(angle)
    kernel = np.array(M) / degree

    # convolution
    from scipy.signal import convolve2d
    blurred = np.zeros_like(image)
    for c in range(3):
        blurred[:,:,c] = convolve2d(image[:,:,c], kernel, mode='same')

    return Image.fromarray(np.uint8(np.clip(blurred,0,255)))


# -----------------------------------------------------
# Cell 11: Original computation block
# -----------------------------------------------------

from PIL import Image, ImageFilter
import os
import cv2
import numpy as np
from tqdm import tqdm

# FAST motion blur
def motion_blur(img, degree=10, angle=0):
    image = np.array(img)
    kernel = np.zeros((degree, degree))
    kernel[int((degree-1)/2), :] = np.ones(degree)
    M = cv2.getRotationMatrix2D((degree/2, degree/2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (degree, degree))
    kernel = kernel / degree
    blurred = cv2.filter2D(image, -1, kernel)
    return Image.fromarray(blurred)

# Generate E-NEU
for cls in classes:
    src = os.path.join(NEU_DIR, cls)
    dst = os.path.join(E_NEU_DIR, cls)
    os.makedirs(dst, exist_ok=True)

    files = [f for f in os.listdir(src) if f.lower().endswith(("jpg","png","bmp","jpeg"))]

    for fname in tqdm(files, desc=f"Processing {cls}"):
        img = Image.open(os.path.join(src, fname))

        g1 = img.filter(ImageFilter.GaussianBlur(radius=1))
        g2 = img.filter(ImageFilter.GaussianBlur(radius=2))

        m1 = motion_blur(img, degree=10, angle=30)
        m2 = motion_blur(img, degree=15, angle=60)

        base = fname.split(".")[0]
        g1.save(f"{dst}/{base}_gb1.jpg")
        g2.save(f"{dst}/{base}_gb2.jpg")
        m1.save(f"{dst}/{base}_mb1.jpg")
        m2.save(f"{dst}/{base}_mb2.jpg")

print("FAST E-NEU generation done!")


# -----------------------------------------------------
# Cell 12: Original computation block
# -----------------------------------------------------

for cls in classes:
    count = len(os.listdir(os.path.join(E_NEU_DIR, cls)))
    print(cls, ":", count)


# -----------------------------------------------------
# Cell 13: Original computation block
# -----------------------------------------------------

IMG_SIZE = (224, 224)
BATCH = 32

neu_train = tf.keras.preprocessing.image_dataset_from_directory(
    NEU_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH)

neu_val = tf.keras.preprocessing.image_dataset_from_directory(
    NEU_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH)


# -----------------------------------------------------
# Cell 14: Original computation block
# -----------------------------------------------------

e_train = tf.keras.preprocessing.image_dataset_from_directory(
    E_NEU_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH)

e_val = tf.keras.preprocessing.image_dataset_from_directory(
    E_NEU_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH)


# -----------------------------------------------------
# Cell 15: Original computation block
# -----------------------------------------------------

for cls in classes:
    os.makedirs(os.path.join(COMBINED_DIR, cls), exist_ok=True)

def merge(cls):
    for fname in os.listdir(os.path.join(NEU_DIR, cls)):
        shutil.copy(os.path.join(NEU_DIR, cls, fname), os.path.join(COMBINED_DIR, cls, fname))

    for fname in os.listdir(os.path.join(E_NEU_DIR, cls)):
        shutil.copy(os.path.join(E_NEU_DIR, cls, fname), os.path.join(COMBINED_DIR, cls, fname))

for cls in classes:
    merge(cls)

print("Combined dataset created")


# -----------------------------------------------------
# Cell 16: Original computation block
# -----------------------------------------------------

for cls in classes:
    print(cls, len(os.listdir(os.path.join(NEU_DIR, cls))), len(os.listdir(os.path.join(E_NEU_DIR, cls))))


# -----------------------------------------------------
# Cell 17: Original computation block
# -----------------------------------------------------

# --- Dataset summary (for Methods / Experiments) ---
import pandas as pd

print("Notebook run: dataset summary")

def print_dataset_summary(df, label_col='class', name='combined'):
    counts = df[label_col].value_counts().sort_index()
    total = len(df)
    print(f"\n{name} total samples: {total}")
    print(counts.to_string())

    summary = pd.DataFrame({
        'class': counts.index.astype(str),
        'count': counts.values,
        'pct': (counts.values/total*100).round(2)
    })
    display(summary)

# Use your correct dataset variable name: _20
print_dataset_summary(_20, label_col='class', name='Combined E-NEU dataset')


# -----------------------------------------------------
# Cell 18: Original computation block
# -----------------------------------------------------

# CELL A
import os, sys, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import entropy as kl_divergence
import seaborn as sns

# Paths (adjust if different)
NEU_DIR = Path("/content/NEU_Workdir/NEU_by_class")
E_NEU_DIR = Path("/content/NEU_Workdir/E_NEU")
COMBINED_DIR = Path("/content/NEU_Workdir/COMBINED")
MODEL_DIR = Path("/content/NEU_Workdir/models")
OUT_DIR = MODEL_DIR / "exp_1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("NEU_DIR:", NEU_DIR.exists(), NEU_DIR)
print("E_NEU_DIR:", E_NEU_DIR.exists(), E_NEU_DIR)
print("COMBINED_DIR:", COMBINED_DIR.exists(), COMBINED_DIR)
print("OUT_DIR:", OUT_DIR)


# -----------------------------------------------------
# Cell 19: Original computation block
# -----------------------------------------------------

# CELL B
def image_paths_in_dir(d:Path):
    p = Path(d)
    files = []
    for c in sorted([x for x in p.iterdir() if x.is_dir()]):
        imgs = [f for f in c.iterdir() if f.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.tif','.tiff')]
        files.append((c.name, imgs))
    return files

def compute_gray_hist(img, bins=256):
    arr = np.array(img.convert('L')).ravel()
    hist, edges = np.histogram(arr, bins=bins, range=(0,255), density=True)
    cdf = np.cumsum(hist)
    return hist, cdf, edges

def compute_color_stats(img):
    arr = np.array(img).astype(np.float32)/255.0
    means = arr.mean(axis=(0,1))
    stds = arr.std(axis=(0,1))
    return means.tolist(), stds.tolist()

# storage
hist_dir = OUT_DIR / "histograms"
hist_dir.mkdir(exist_ok=True)

summary_rows = []

for cls, imgs in image_paths_in_dir(NEU_DIR):
    neuhist_all = []
    neucdf_all = []
    color_means = []
    color_stds = []
    for p in imgs:
        try:
            im = Image.open(p).convert('RGB')
        except:
            continue
        h, cdf, edges = compute_gray_hist(im)
        neuhist_all.append(h)
        neucdf_all.append(cdf)
        means, stds = compute_color_stats(im)
        color_means.append(means)
        color_stds.append(stds)
    if len(neuhist_all)==0:
        continue
    hist_mean = np.mean(neuhist_all, axis=0)
    cdf_mean = np.mean(neucdf_all, axis=0)
    mean_color = np.mean(color_means, axis=0)
    std_color = np.mean(color_stds, axis=0)

    np.save(hist_dir / f"NEU_hist_{cls}.npy", hist_mean)
    np.save(hist_dir / f"NEU_cdf_{cls}.npy", cdf_mean)

    summary_rows.append({
        "class": cls,
        "dataset": "NEU",
        "n_images": len(neuhist_all),
        "mean_r": float(mean_color[0]),
        "mean_g": float(mean_color[1]),
        "mean_b": float(mean_color[2]),
        "std_r": float(std_color[0]),
        "std_g": float(std_color[1]),
        "std_b": float(std_color[2])
    })

# Repeat for E-NEU
for cls, imgs in image_paths_in_dir(E_NEU_DIR):
    enehist_all = []
    enecdf_all = []
    color_means = []
    color_stds = []
    for p in imgs:
        try:
            im = Image.open(p).convert('RGB')
        except:
            continue
        h, cdf, edges = compute_gray_hist(im)
        enehist_all.append(h)
        enecdf_all.append(cdf)
        means, stds = compute_color_stats(im)
        color_means.append(means)
        color_stds.append(stds)
    if len(enehist_all)==0:
        continue
    hist_mean = np.mean(enehist_all, axis=0)
    cdf_mean = np.mean(enecdf_all, axis=0)
    np.save(hist_dir / f"ENEU_hist_{cls}.npy", hist_mean)
    np.save(hist_dir / f"ENEU_cdf_{cls}.npy", cdf_mean)

    mean_color = np.mean(color_means, axis=0)
    std_color = np.mean(color_stds, axis=0)
    summary_rows.append({
        "class": cls,
        "dataset": "E-NEU",
        "n_images": len(enehist_all),
        "mean_r": float(mean_color[0]),
        "mean_g": float(mean_color[1]),
        "mean_b": float(mean_color[2]),
        "std_r": float(std_color[0]),
        "std_g": float(std_color[1]),
        "std_b": float(std_color[2])
    })

# Save summary
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR/"per_class_color_stats.csv", index=False)
print("Saved per-class color stats to:", OUT_DIR/"per_class_color_stats.csv")

# Plot example histograms (NEU vs E-NEU) for each class
for cls,_ in image_paths_in_dir(NEU_DIR):
    nfile = hist_dir / f"NEU_hist_{cls}.npy"
    efile = hist_dir / f"ENEU_hist_{cls}.npy"
    if not nfile.exists() or not efile.exists(): continue
    h1 = np.load(nfile)
    h2 = np.load(efile)
    plt.figure(figsize=(6,3))
    plt.plot(h1, label='NEU')
    plt.plot(h2, label='E-NEU', alpha=0.7)
    plt.title(f"Gray hist: {cls}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR/f"hist_{cls}.png")
    plt.close()
print("Histogram plots saved to:", OUT_DIR)


# -----------------------------------------------------
# Cell 20: Original computation block
# -----------------------------------------------------

# CELL C
from scipy.stats import entropy
classes = sorted([d.name for d in NEU_DIR.iterdir() if d.is_dir()])

kl_rows = []
for cls in classes:
    nfile = hist_dir / f"NEU_hist_{cls}.npy"
    efile = hist_dir / f"ENEU_hist_{cls}.npy"
    if not nfile.exists() or not efile.exists():
        print("Missing hist for", cls); continue
    h1 = np.load(nfile).astype(np.float64)
    h2 = np.load(efile).astype(np.float64)
    # stabilize
    eps = 1e-10
    h1 = h1 + eps
    h2 = h2 + eps
    h1 = h1 / h1.sum()
    h2 = h2 / h2.sum()
    # KL (h1 || h2)
    kl = entropy(h1, h2)
    # histogram intersection
    inter = np.sum(np.minimum(h1, h2))
    kl_rows.append({"class":cls, "kl_NEU_to_ENEU": float(kl), "hist_intersection": float(inter)})
kl_df = pd.DataFrame(kl_rows)
kl_df.to_csv(OUT_DIR/"kl_hist_intersection.csv", index=False)
print("Saved KL/intersection:", OUT_DIR/"kl_hist_intersection.csv")
kl_df


# -----------------------------------------------------
# Cell 21: Original computation block
# -----------------------------------------------------

# CELL D
IMG_SIZE = (224,224)
BATCH = 32
AUTOTUNE = tf.data.AUTOTUNE

def build_datasets_from_dirs(neu_dir=NEU_DIR, eneu_dir=E_NEU_DIR, combined_dir=COMBINED_DIR):

    # ----- NEU -----
    neu_train_raw = tf.keras.preprocessing.image_dataset_from_directory(
        str(neu_dir),
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH)

    neu_val_raw = tf.keras.preprocessing.image_dataset_from_directory(
        str(neu_dir),
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH)

    # Save class names BEFORE prefetch
    class_names = neu_train_raw.class_names

    # ----- E-NEU -----
    eneu_train_raw = tf.keras.preprocessing.image_dataset_from_directory(
        str(eneu_dir),
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH)

    eneu_val_raw = tf.keras.preprocessing.image_dataset_from_directory(
        str(eneu_dir),
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH)

    # ----- COMBINED -----
    combined_train_raw = tf.keras.preprocessing.image_dataset_from_directory(
        str(combined_dir),
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH)

    combined_val_raw = tf.keras.preprocessing.image_dataset_from_directory(
        str(combined_dir),
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH)

    # Now apply PREFETCH
    neu_train = neu_train_raw.prefetch(AUTOTUNE)
    neu_val = neu_val_raw.prefetch(AUTOTUNE)
    eneu_train = eneu_train_raw.prefetch(AUTOTUNE)
    eneu_val = eneu_val_raw.prefetch(AUTOTUNE)
    combined_train = combined_train_raw.prefetch(AUTOTUNE)
    combined_val = combined_val_raw.prefetch(AUTOTUNE)

    return {
        "neu_train": neu_train,
        "neu_val": neu_val,
        "eneu_train": eneu_train,
        "eneu_val": eneu_val,
        "comb_train": combined_train,
        "comb_val": combined_val,
        "class_names": class_names
    }


dsets = build_datasets_from_dirs()

# Print cardinality
for k in dsets:
    if "train" in k or "val" in k:
        print(k, dsets[k].cardinality().numpy())

print("Class names:", dsets["class_names"])


# -----------------------------------------------------
# Cell 22: Original computation block
# -----------------------------------------------------

# CELL E
def build_model(num_classes, input_shape=(224,224,3)):
    base = EfficientNetB0(include_top=False, input_shape=input_shape, weights="imagenet", pooling='avg')
    x = base.output
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=base.input, outputs=out)
    return model, base

def train_two_stage(model, base, train_ds, val_ds, out_dir,
                    epochs_stage1=6, epochs_stage2=6, fine_tune_at=None,
                    lr_stage1=1e-3, lr_stage2=1e-5):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: freeze base
    base.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(lr_stage1),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    ck1 = ModelCheckpoint(out_dir/"model_stage1.h5", save_best_only=True, monitor='val_accuracy', mode='max')
    csv1 = CSVLogger(out_dir/"train_stage1.csv")
    r1 = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    hist1 = model.fit(train_ds, validation_data=val_ds, epochs=epochs_stage1, callbacks=[ck1,csv1,r1])

    # Stage 2: optional fine-tune
    hist2 = None
    if fine_tune_at is not None:
        # unfreeze layers from fine_tune_at
        base.trainable = True
        # freeze earlier layers
        for i,layer in enumerate(base.layers):
            layer.trainable = (i >= fine_tune_at)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_stage2),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        ck2 = ModelCheckpoint(out_dir/"model_stage2.h5", save_best_only=True, monitor='val_accuracy', mode='max')
        csv2 = CSVLogger(out_dir/"train_stage2.csv")
        r2 = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-8)
        hist2 = model.fit(train_ds, validation_data=val_ds, epochs=epochs_stage2, callbacks=[ck2,csv2,r2])
    return model, hist1, hist2


# -----------------------------------------------------
# Cell 23: Original computation block
# -----------------------------------------------------

# --- Hyperparameters (for the paper, include this block) ---
HYPERPARAMS = {
    "model": "YourModelName",        # e.g. ResNet50 / CustomCNN
    "optimizer": "Adam",
    "lr": 1e-4,
    "batch_size": 32,
    "epochs": 50,
    "weight_decay": 1e-5,
    "augmentation": "E-NEU (describe below)",
    "train_val_test_split": "70/15/15 (stratified)",
    "seed": 42
}
import pandas as pd
display(pd.DataFrame.from_dict(HYPERPARAMS, orient='index', columns=['value']).rename_axis('hyperparam'))


# -----------------------------------------------------
# Cell 24: Original computation block
# -----------------------------------------------------

# CELL F
train_on = 'comb'      # options: 'neu','eneu','comb'
fine_tune_at = 200     # set to None to skip stage2; otherwise layer index to unfreeze
epochs_stage1 = 6
epochs_stage2 = 6
lr_stage1 = 1e-3
lr_stage2 = 1e-5

# select datasets
if train_on == 'neu':
    train_ds = dsets['neu_train']; val_ds = dsets['neu_val']
elif train_on == 'eneu':
    train_ds = dsets['eneu_train']; val_ds = dsets['eneu_val']
else:
    train_ds = dsets['comb_train']; val_ds = dsets['comb_val']

# Get class names from dsets
class_names = dsets["class_names"]
num_classes = len(class_names)

# Build model
model, base = build_model(num_classes)
model.summary()

# Train
model, hist1, hist2 = train_two_stage(
    model, base, train_ds, val_ds, OUT_DIR,
    epochs_stage1=epochs_stage1,
    epochs_stage2=epochs_stage2,
    fine_tune_at=fine_tune_at,
    lr_stage1=lr_stage1,
    lr_stage2=lr_stage2
)

# Save final model
model.save(OUT_DIR/"model_final.h5")
print("Saved final model to:", OUT_DIR/"model_final.h5")


# -----------------------------------------------------
# Cell 25: Original computation block
# -----------------------------------------------------

# CELL G
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import itertools

def dataset_to_preds(model, dataset):
    # returns y_true (flat), y_pred (int), y_prob (N x C)
    y_true = []
    y_prob = []
    for batch_x, batch_y in dataset:
        probs = model.predict(batch_x, verbose=0)
        y_prob.append(probs)
        y_true.append(batch_y.numpy())
    y_prob = np.vstack(y_prob)
    y_true = np.concatenate(y_true)
    y_pred = np.argmax(y_prob, axis=1)
    return y_true, y_pred, y_prob

def save_confusion_and_report(model, dataset, class_names, out_prefix):
    y_true, y_pred, y_prob = dataset_to_preds(model, dataset)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    # Save cm csv
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(f"{out_prefix}_confusion.csv")
    # Save report csv
    pd.DataFrame(report).transpose().to_csv(f"{out_prefix}_classification_report.csv")
    # Plot cm
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.title(f"Confusion matrix: {Path(out_prefix).stem}")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_confusion.png")
    plt.close()
    print(f"Saved {out_prefix}_confusion.csv and _classification_report.csv and png")

# Evaluate on NEU val if available
if 'neu_val' in dsets:
    print("Evaluating on NEU val")
    save_confusion_and_report(model, dsets['neu_val'], class_names, str(OUT_DIR/'metrics_neu'))

if 'eneu_val' in dsets:
    print("Evaluating on E-NEU val")
    save_confusion_and_report(model, dsets['eneu_val'], class_names, str(OUT_DIR/'metrics_eneu'))


# -----------------------------------------------------
# Cell 26: Original computation block
# -----------------------------------------------------

# CELL H
def save_training_curves(hist, name_prefix):
    if hist is None:
        return
    pd.DataFrame(hist.history).to_csv(OUT_DIR/f"{name_prefix}_history.csv", index=False)
    # plot
    plt.figure(figsize=(6,4))
    if 'accuracy' in hist.history:
        plt.plot(hist.history['accuracy'], label='train_acc')
    if 'val_accuracy' in hist.history:
        plt.plot(hist.history['val_accuracy'], label='val_acc')
    if 'loss' in hist.history:
        plt.plot(hist.history['loss'], label='train_loss')
    if 'val_loss' in hist.history:
        plt.plot(hist.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title(name_prefix)
    plt.tight_layout()
    plt.savefig(OUT_DIR/f"{name_prefix}_curve.png")
    plt.close()

save_training_curves(hist1, "stage1")
save_training_curves(hist2, "stage2")
print("Saved training histories & plots to:", OUT_DIR)


# -----------------------------------------------------
# Cell 27: Original computation block
# -----------------------------------------------------

# CELL I
print("Outputs in:", OUT_DIR)
!ls -la {OUT_DIR}


# -----------------------------------------------------
# Cell 28: Original computation block
# -----------------------------------------------------

import pandas as pd

# Correct paths (NO SPACE at beginning)
neu = pd.read_csv("/content/NEU_Workdir/models/exp_1/metrics_neu_classification_report.csv", index_col=0)
eneu = pd.read_csv("/content/NEU_Workdir/models/exp_1/metrics_eneu_classification_report.csv", index_col=0)

# Keep only per-class rows
neu = neu.loc[neu.index.isin(class_names)]
eneu = eneu.loc[eneu.index.isin(class_names)]

# Add prefixes
neu.columns = [f"NEU_{c}" for c in neu.columns]
eneu.columns = [f"ENEU_{c}" for c in eneu.columns]

# Combine
combined = pd.concat([neu, eneu], axis=1)

# Save
combined.to_csv("/content/NEU_Workdir/models/exp_1/combined_class_metrics_neu_eneu.csv")

print("Saved combined metrics to combined_class_metrics_neu_eneu.csv")


# -----------------------------------------------------
# Cell 29: Original computation block
# -----------------------------------------------------

from pathlib import Path, PurePath
def counts_by_dir(d):
    d = Path(d)
    rows=[]
    for c in sorted([p.name for p in d.iterdir() if p.is_dir()]):
        n = len(list((d/c).glob("*.*")))
        rows.append((c,n))
    return pd.DataFrame(rows, columns=['class','n_images'])
neu_counts = counts_by_dir("/content/NEU_Workdir/NEU_by_class")
eneu_counts = counts_by_dir("/content/NEU_Workdir/E_NEU")
comb_counts = counts_by_dir("/content/NEU_Workdir/COMBINED")
pd.concat([neu_counts.set_index('class'), eneu_counts.set_index('class'), comb_counts.set_index('class')], axis=1).fillna(0).to_csv("/content/NEU_Workdir/models/exp_1/per_class_counts.csv")
print("Saved per_class_counts.csv")


# -----------------------------------------------------
# Cell 30: Original computation block
# -----------------------------------------------------

!zip -r /content/exp_1_results.zip /content/NEU_Workdir/models/exp_1


# -----------------------------------------------------
# Cell 31: Original computation block
# -----------------------------------------------------

from google.colab import files
files.download('/content/exp_1_results.zip')


# -----------------------------------------------------
# Cell 32: Original computation block
# -----------------------------------------------------

import pkg_resources, json
packages = ["tensorflow","numpy","opencv-python","scipy","pillow","scikit-learn","pandas","matplotlib","seaborn"]
reqs = {p: pkg_resources.get_distribution(p).version for p in packages if pkg_resources.get_distribution(p)}
with open("/content/NEU_Workdir/models/exp_1/requirements_used.json","w") as f:
    json.dump(reqs, f, indent=2)
print(reqs)


# -----------------------------------------------------
# Cell 33: Original computation block
# -----------------------------------------------------

pd.read_csv("/content/NEU_Workdir/models/exp_1/train_stage1.csv").tail()
pd.read_csv("/content/NEU_Workdir/models/exp_1/train_stage2.csv").tail()


# -----------------------------------------------------
# Cell 34: Original computation block
# -----------------------------------------------------

import tensorflow as tf
import numpy as np
import pandas as pd
import os, random
from pathlib import Path
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import shutil
import matplotlib.pyplot as plt

# =============================
# CONFIGURATION
# =============================
NEU_DIR = Path("/content/NEU_Workdir/NEU_by_class")
MODEL_PATH = Path("/content/NEU_Workdir/models/exp_1/model_final.h5")
OUT_DIR = Path("/content/NEU_Workdir/models/exp_1/classwise_predictions")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224,224)

# Class names (same order as during training)
class_names = ['crazing','inclusion','patches','pitted_surface','rolled-in_scale','scratches']

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
print("Model Loaded Successfully!")

# =============================
# Collect predictions
# =============================
rows = []

for cls in class_names:
    cls_path = NEU_DIR / cls
    images = [f for f in cls_path.iterdir() if f.suffix.lower() in [".jpg",".png",".jpeg",".bmp"]]

    # pick 5 random images per class
    selected = random.sample(images, 5)

    for img_path in selected:
        # Load & preprocess image
        img = load_img(img_path, target_size=IMG_SIZE)
        arr = img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        # Predict
        pred = model.predict(arr, verbose=0)
        pred_idx = np.argmax(pred[0])
        pred_cls = class_names[pred_idx]
        confidence = float(pred[0][pred_idx])

        # Save image with prediction name
        save_name = f"{cls}_{img_path.name}_pred_{pred_cls}.jpg"
        save_path = OUT_DIR / save_name
        shutil.copy(img_path, save_path)

        # Add row
        rows.append({
            "true_class": cls,
            "image": img_path.name,
            "predicted_class": pred_cls,
            "confidence": confidence
        })

print("Finished predictions for all classes!")

# =============================
# Save CSV
# =============================
df = pd.DataFrame(rows)
csv_path = OUT_DIR / "classwise_predictions.csv"
df.to_csv(csv_path, index=False)
print("Saved CSV:", csv_path)

# =============================
# Show sample output
# =============================
df.head(10)


# -----------------------------------------------------
# Cell 35: Original computation block
# -----------------------------------------------------

test_ds = val_ds  # or neu_val, or e_val depending on which you want to evaluate


# -----------------------------------------------------
# Cell 36: Original computation block
# -----------------------------------------------------

import numpy as np
import tensorflow as tf

y_true = []
y_pred = []

for images, labels in test_ds:
    outputs = model(images, training=False)
    predicted = tf.argmax(outputs, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("y_true shape:", y_true.shape)
print("y_pred shape:", y_pred.shape)


# -----------------------------------------------------
# Cell 37: Original computation block
# -----------------------------------------------------

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print(classification_report(y_true, y_pred, digits=4))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()
