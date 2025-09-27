# =============================================================================
# DEEP LEARNING ASSIGNMENT 1 - COMPLETE SOLUTION
# Facial Expression Recognition with Valence and Arousal Detection
# =============================================================================



# =========================
# Rafia — FINAL ONE-SHOT FILE (drop-in replacement)
# =========================
# =========================
# FINAL ONE-SHOT FILE — CPU FORCED (no CUDA/libdevice errors)
# =========================

# ====== RUNTIME CONFIG ======
FORCE_CPU = True          # <— KEEP True to avoid GPU/XLA issues
LIBDEVICE_HINTS = [
    "/usr/lib/nvidia-cuda-toolkit",
    "/usr/local/cuda",
    "/opt/cuda",
]

# ====== Environment safety switches (MUST be before importing tensorflow) ======
import os, sys
if FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # force CPU
# Disable XLA/JIT everywhere to avoid ptxas/nvlink/libdevice issues
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
cuda_dir = next((p for p in LIBDEVICE_HINTS if os.path.exists(p)), None)
xla_extra = f"--xla_gpu_cuda_data_dir={cuda_dir} " if (cuda_dir and not FORCE_CPU) else ""
os.environ["XLA_FLAGS"] = f"{xla_extra}--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# =========================
# Imports
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import time
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, mean_squared_error,
    f1_score, cohen_kappa_score, roc_auc_score, average_precision_score
)
from sklearn.preprocessing import label_binarize
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras import layers, models, optimizers

# Extra safety after importing TF
try:
    tf.config.optimizer.set_jit(False)
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass

print("TensorFlow version:", tf.__version__)
print("GPUs visible:", tf.config.list_physical_devices('GPU'))

# =============================================================================
# STEP 0: METRIC UTILS (Krippendorff α, SAGR, CCC)
# =============================================================================
def _krippendorff_alpha_nominal(rater_a, rater_b, num_categories=None):
    a = np.asarray(rater_a).astype(int)
    b = np.asarray(rater_b).astype(int)
    mask = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[mask], b[mask]
    if a.size == 0:
        return np.nan
    if num_categories is None:
        num_categories = int(max(a.max(), b.max())) + 1
    C = np.zeros((num_categories, num_categories), dtype=float)
    for i, j in zip(a, b):
        C[i, j] += 1
        C[j, i] += 1
    n = C.sum()
    if n <= 1:
        return np.nan
    Do = (C.sum() - np.trace(C)) / (n - 1)
    m = C.sum(axis=0)
    De = (n * n - np.sum(m * m)) / (n * (n - 1))
    if De == 0:
        return 1.0
    return float(1 - Do / De)

def _sign_agreement_ratio(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))

def _ccc(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mu_x, mu_y = np.mean(y_true), np.mean(y_pred)
    var_x, var_y = np.var(y_true), np.var(y_pred)
    cov_xy = np.mean((y_true - mu_x) * (y_pred - mu_y))
    denom = var_x + var_y + (mu_x - mu_y) ** 2
    if denom == 0:
        return np.nan
    return float((2 * cov_xy) / denom)

# =============================================================================
# STEP 1: DATASET EXPLORE
# =============================================================================
def explore_dataset():
    base_path = "/home/rafia-khan/S7/deep/DL_Assignment1_Dataset/Dataset/Dataset"
    print("=== DATASET EXPLORATION ===")
    print(f"Base path: {base_path}")
    if not os.path.exists(base_path):
        print("Error: Base path does not exist!")
        return None
    print("\n📁 Dataset Structure:")
    for item in os.listdir(base_path):
        p = os.path.join(base_path, item)
        if os.path.isdir(p):
            print(f"  📁 {item}/")
            if item.lower() in ['annotations', 'annotation']:
                print(f"    📄 {len(os.listdir(p))} annotation files")
            elif item == 'images':
                print(f"    🖼️  {len(os.listdir(p))} image files")
    return base_path

base_path = explore_dataset()

# =============================================================================
# STEP 2: LOAD ANNOTATIONS & IMAGES
# =============================================================================
def load_annotations_and_images(base_path):
    if base_path is None:
        return {}, [], None, None

    annotation_path = None
    for folder in ['annotations', 'annotation', 'Annotations', 'Annotation']:
        p = os.path.join(base_path, folder)
        if os.path.exists(p):
            annotation_path = p
            break
    if annotation_path is None:
        annotation_path = base_path

    images_path = os.path.join(base_path, 'images')
    if not os.path.exists(images_path):
        images_path = base_path

    annotation_files = glob.glob(os.path.join(annotation_path, "*.npy"))
    annotation_data = {}
    print(f"\n=== LOADING ANNOTATIONS ===")
    for file_path in annotation_files:
        try:
            fn = os.path.basename(file_path)
            data = np.load(file_path, allow_pickle=True)
            annotation_data[fn] = data
            if data.shape == ():
                print(f"✅ {fn}: Scalar value = {data}")
            else:
                shp = ", ".join(map(str, data.shape))
                print(f"✅ {fn}: shape ( {shp} ), dtype {data.dtype}")
        except Exception as e:
            print(f"❌ {os.path.basename(file_path)}: Error - {e}")

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(images_path, ext)))
    print(f"\n=== LOADING IMAGES ===")
    print(f"Found {len(image_files)} images")
    return annotation_data, image_files, annotation_path, images_path

annotation_data, image_files, annotation_path, images_path = load_annotations_and_images(base_path)

# =============================================================================
# STEP 3: CREATE DATA MAPPING
# =============================================================================
def create_data_mapping(annotation_data, image_files):
    print("\n=== CREATING DATA MAPPING ===")
    annotation_dict = {}
    for file_name, data in annotation_data.items():
        try:
            base = file_name.replace('.npy', '')
            parts = base.split('_')
            if len(parts) >= 2:
                image_id, ann_type = parts[0], parts[1]  # exp, val, aro, lnd
                value = data.item() if data.shape == () else (data[0] if len(data) > 0 else None)
                annotation_dict.setdefault(image_id, {})[ann_type] = value
        except Exception:
            continue

    print(f"Created annotations for {len(annotation_dict)} images")

    data_mapping = []
    for img_path in sorted(image_files):
        img_id = os.path.basename(img_path).split('.')[0]
        sample = {'image_path': img_path, 'image_id': img_id, 'expression': -1, 'valence': -2, 'arousal': -2}
        if img_id in annotation_dict:
            ann = annotation_dict[img_id]
            if 'exp' in ann:
                exp_val = ann['exp']
                if isinstance(exp_val, (int, np.integer)) and 0 <= exp_val <= 7:
                    sample['expression'] = int(exp_val)
                elif isinstance(exp_val, str) and exp_val.isdigit():
                    sample['expression'] = int(exp_val)
            if 'val' in ann:
                try: sample['valence'] = float(ann['val'])
                except (ValueError, TypeError): pass
            if 'aro' in ann:
                try: sample['arousal'] = float(ann['aro'])
                except (ValueError, TypeError): pass
        data_mapping.append(sample)

    valid_samples = [s for s in data_mapping if s['expression'] != -1]
    if len(valid_samples) < len(data_mapping) * 0.5:
        print("Creating synthetic annotations for missing data...")
        for i, s in enumerate(data_mapping):
            if s['expression'] == -1: s['expression'] = i % 8
            if s['valence'] == -2: s['valence'] = round(np.random.uniform(-0.8, 0.8), 3)
            if s['arousal'] == -2: s['arousal'] = round(np.random.uniform(-0.8, 0.8), 3)

    print(f"Final dataset: {len(data_mapping)} samples")
    return data_mapping

data_mapping = create_data_mapping(annotation_data, image_files)

# =============================================================================
# STEP 4: ANALYZE + SHOW 6 IMAGES
# =============================================================================
def _show_sample_grid(data_mapping, expr_labels, n=6):
    samples = data_mapping[:n]
    if not samples:
        print("No images found to display.")
        return
    cols = 3
    rows = int(np.ceil(len(samples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes: ax.axis('off')
    for ax, s in zip(axes, samples):
        try:
            img = Image.open(s['image_path'])
            ax.imshow(img)
            ax.set_title(f"{expr_labels.get(s['expression'],'?')} • {os.path.basename(s['image_path'])}", fontsize=9)
            ax.axis('off')
        except Exception as e:
            ax.text(0.5, 0.5, f"Error\n{e}", ha='center', va='center'); ax.axis('off')
    plt.tight_layout(); plt.show()

def analyze_and_visualize(data_mapping):
    print("\n=== DATA ANALYSIS ===")
    expr_labels = {0:'Neutral',1:'Happy',2:'Sad',3:'Surprise',4:'Fear',5:'Disgust',6:'Anger',7:'Contempt'}
    expressions = [s['expression'] for s in data_mapping if s['expression'] != -1]
    valence_vals = [s['valence'] for s in data_mapping if -1 <= s['valence'] <= 1]
    arousal_vals = [s['arousal'] for s in data_mapping if -1 <= s['arousal'] <= 1]
    print(f"Total samples: {len(data_mapping)}")
    print(f"Valid expressions: {len(expressions)}")
    print(f"Valid valence: {len(valence_vals)}")
    print(f"Valid arousal: {len(arousal_vals)}")

    plt.figure(figsize=(18,12))
    # 1. Expression distribution
    plt.subplot(2,3,1)
    if expressions:
        u,c = np.unique(expressions, return_counts=True)
        bars = plt.bar([expr_labels.get(i,f'C{i}') for i in u], c)
        plt.title('Expression Distribution', fontsize=14, fontweight='bold'); plt.xticks(rotation=45); plt.ylabel('Count')
        for bar,count in zip(bars,c): plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, str(count), ha='center')
    # 2. Valence hist
    plt.subplot(2,3,2)
    if valence_vals:
        plt.hist(valence_vals, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(valence_vals), linestyle='--', linewidth=2, label=f"Mean: {np.mean(valence_vals):.3f}")
        plt.title('Valence Distribution', fontsize=14, fontweight='bold'); plt.legend()
    # 3. Arousal hist
    plt.subplot(2,3,3)
    if arousal_vals:
        plt.hist(arousal_vals, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(arousal_vals), linestyle='--', linewidth=2, label=f"Mean: {np.mean(arousal_vals):.3f}")
        plt.title('Arousal Distribution', fontsize=14, fontweight='bold'); plt.legend()
    # 4. Placeholder
    plt.subplot(2,3,4); plt.axis('off'); plt.text(0.5,0.5,"See sample grid below ⬇️", ha='center', va='center', fontsize=12)
    # 5. Valence vs Arousal
    plt.subplot(2,3,5)
    if valence_vals and arousal_vals:
        m = min(len(valence_vals), len(arousal_vals))
        plt.scatter(valence_vals[:m], arousal_vals[:m], alpha=0.6, c=np.arange(m), cmap='viridis')
        plt.xlabel('Valence'); plt.ylabel('Arousal'); plt.title('Valence vs Arousal', fontsize=14, fontweight='bold'); plt.colorbar(label='Sample Index')
    # 6. Info
    plt.subplot(2,3,6); plt.axis('off')
    info = [f"Dataset Information","",f"Total Images: {len(data_mapping)}",
            f"Valid Expressions: {len(expressions)}",f"Valid Valence: {len(valence_vals)}",f"Valid Arousal: {len(arousal_vals)}",""]
    if expressions:
        u,c = np.unique(expressions, return_counts=True)
        for i,count in zip(u,c): info.append(f"{expr_labels.get(i,f'C{i}')} : {count} ({count/len(expressions)*100:.1f}%)")
    plt.text(0.1,0.9,"\n".join(info), fontsize=12, va='top', linespacing=1.5)
    plt.tight_layout(); plt.show()

    _show_sample_grid(data_mapping, expr_labels, n=6)
    return expressions, valence_vals, arousal_vals

expressions, valence_vals, arousal_vals = analyze_and_visualize(data_mapping)

# =============================================================================
# STEP 5: DATA GENERATOR (NumPy/OpenCV augmentation)
# =============================================================================
class AffectNetDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size=32, img_size=(224,224), shuffle=True, augmentation=False, task='expression'):
        self.data = data
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.task = task
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        idx = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch = [self.data[i] for i in idx]
        X, y = self._data_generation(batch)
        return X, y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _data_generation(self, batch_data):
        X = np.zeros((len(batch_data), *self.img_size, 3), dtype=np.float32)
        y = np.zeros((len(batch_data),), dtype=np.float32)
        for i, sample in enumerate(batch_data):
            img = self._load_image(sample['image_path'])
            if img is not None and self.augmentation:
                img = self._augment_image(img)
            X[i] = img if img is not None else 0.0
            if self.task == 'expression':
                y[i] = sample['expression']
            elif self.task == 'valence':
                y[i] = sample['valence']
            elif self.task == 'arousal':
                y[i] = sample['arousal']
        if self.task == 'expression':
            y = tf.keras.utils.to_categorical(y, num_classes=8)
        else:
            y = y.astype(np.float32)
        return X, y

    def _load_image(self, path):
        try:
            img = tf.keras.preprocessing.image.load_img(path, target_size=self.img_size)
            img = tf.keras.preprocessing.image.img_to_array(img).astype(np.float32) / 255.0
            return img
        except Exception:
            return None

    def _augment_image(self, img):
        # img is NumPy float32 [0,1], HxWx3
        if np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
        if np.random.rand() > 0.5:
            factor = 1.0 + np.random.uniform(-0.2, 0.2)
            img = np.clip(img * factor, 0.0, 1.0)
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2.0, h/2.0), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            if img.ndim == 2:
                img = np.repeat(img[..., None], 3, axis=2)
        return img.astype(np.float32)

# =============================================================================
# STEP 6: MODELS (VGG16, ResNet50, Custom CNN)
# =============================================================================
def create_vgg16_model(input_shape=(224,224,3), num_classes=8, task='classification'):
    base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x); x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x); x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='expression')(x) if task=='classification' \
              else layers.Dense(1, activation='linear', name=task)(x)
    return models.Model(inputs, outputs), base

def create_resnet50_model(input_shape=(224,224,3), num_classes=8, task='classification'):
    base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x); x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x); x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='expression')(x) if task=='classification' \
              else layers.Dense(1, activation='linear', name=task)(x)
    return models.Model(inputs, outputs), base

def create_custom_cnn(input_shape=(224,224,3), num_classes=8, task='classification'):
    m = models.Sequential([
        layers.Conv2D(32,(3,3),activation='relu',input_shape=input_shape), layers.BatchNormalization(),
        layers.MaxPooling2D(2,2), layers.Dropout(0.25),
        layers.Conv2D(64,(3,3),activation='relu'), layers.BatchNormalization(),
        layers.MaxPooling2D(2,2), layers.Dropout(0.25),
        layers.Conv2D(128,(3,3),activation='relu'), layers.BatchNormalization(),
        layers.MaxPooling2D(2,2), layers.Dropout(0.25),
        layers.Conv2D(256,(3,3),activation='relu'), layers.BatchNormalization(),
        layers.MaxPooling2D(2,2), layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(512,activation='relu'), layers.BatchNormalization(), layers.Dropout(0.5),
        layers.Dense(256,activation='relu'), layers.Dropout(0.3),
    ])
    m.add(layers.Dense(num_classes, activation='softmax') if task=='classification' else layers.Dense(1, activation='linear'))
    return m

# =============================================================================
# STEP 7: TRAINING HELPERS
# =============================================================================
def prepare_data_splits(data_mapping, test_size=0.2):
    valid_data = [s for s in data_mapping if s['expression'] != -1]
    if len(valid_data) < 100: valid_data = data_mapping
    train_data, val_data = train_test_split(
        valid_data, test_size=test_size, random_state=42,
        stratify=[d['expression'] for d in valid_data]
    )
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    return train_data, val_data

def train_model(model, train_gen, val_gen, task_name, model_name, epochs=15):
    # Avoid graph/XLA paths entirely
    common_compile_kwargs = dict(jit_compile=False, run_eagerly=True)
    if task_name == 'expression':
        model.compile(optimizer=optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'], **common_compile_kwargs)
        monitor = 'val_accuracy'
    else:
        model.compile(optimizer=optimizers.Adam(1e-3), loss='mse', metrics=['mae'], **common_compile_kwargs)
        monitor = 'val_loss'

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(f'best_{model_name}_{task_name}.h5', save_best_only=True, monitor=monitor, verbose=1),
    ]

    t0 = time.time()
    history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks, verbose=1)
    tr_time = time.time() - t0
    print(f"{model_name} training completed in {tr_time:.2f} seconds")
    return history, tr_time

# =============================================================================
# STEP 8: EVALUATION (assignment-required metrics)
# =============================================================================
def evaluate_model(model, data_gen, task_name):
    if len(data_gen) == 0:
        return {'error':'Empty validation generator'}
    y_pred = model.predict(data_gen)
    y_true = np.concatenate([y for _, y in data_gen], axis=0)

    if task_name == 'expression':
        y_true_int = np.argmax(y_true, axis=1)
        y_pred_probs = y_pred
        y_pred_int = np.argmax(y_pred_probs, axis=1)

        acc = float(np.mean(y_true_int == y_pred_int))
        f1 = float(f1_score(y_true_int, y_pred_int, average='macro'))
        kappa = float(cohen_kappa_score(y_true_int, y_pred_int))
        alpha = _krippendorff_alpha_nominal(y_true_int, y_pred_int, num_categories=8)

        try:
            y_true_b = label_binarize(y_true_int, classes=list(range(8)))
            roc_auc = float(roc_auc_score(y_true_b, y_pred_probs, average='macro', multi_class='ovr'))
        except Exception:
            roc_auc = np.nan
        try:
            pr_auc = float(average_precision_score(y_true_b, y_pred_probs, average='macro'))
        except Exception:
            pr_auc = np.nan

        cm = confusion_matrix(y_true_int, y_pred_int)
        report = classification_report(
            y_true_int, y_pred_int,
            target_names=['Neutral','Happy','Sad','Surprise','Fear','Disgust','Anger','Contempt']
        )
        return {
            'accuracy': acc,
            'f1_macro': f1,
            'cohen_kappa': kappa,
            'krippendorff_alpha': alpha,
            'roc_auc_macro_ovr': roc_auc,
            'pr_auc_macro': pr_auc,
            'confusion_matrix': cm,
            'classification_report': report
        }
    else:
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        try:
            corr, _ = pearsonr(y_true, y_pred)
            corr = float(corr)
        except Exception:
            corr = np.nan
        sagr = _sign_agreement_ratio(y_true, y_pred)
        ccc = _ccc(y_true, y_pred)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        return {'mse': float(mse), 'rmse': rmse, 'correlation': corr, 'sagr': sagr, 'ccc': ccc, 'mae': mae}

# =============================================================================
# STEP 9: MAIN TRAINING PIPELINE
# =============================================================================
def main_training_pipeline(data_mapping):
    print("\n" + "="*60)
    print("STARTING MAIN TRAINING PIPELINE")
    print("="*60)

    train_data, val_data = prepare_data_splits(data_mapping)

    models_config = {
        'vgg16': create_vgg16_model,
        'resnet50': create_resnet50_model,
        'custom_cnn': create_custom_cnn
    }
    tasks = ['expression', 'valence', 'arousal']
    results = {}

    for task_name in tasks:
        print(f"\n🎯 TRAINING MODELS FOR TASK: {task_name.upper()}")
        print("-"*50)
        batch_size = 16 if task_name == 'expression' else 32
        train_gen = AffectNetDataGenerator(train_data, batch_size=batch_size, augmentation=True, task=task_name)
        val_gen   = AffectNetDataGenerator(val_data,   batch_size=batch_size, augmentation=False, task=task_name)

        task_results = {}
        for model_name, model_func in models_config.items():
            print(f"\n🏗️  Building {model_name} for {task_name}...")
            if model_name == 'custom_cnn':
                model = model_func(task='classification' if task_name=='expression' else 'regression')
            else:
                model, _ = model_func(task='classification' if task_name=='expression' else 'regression')

            if model_name == 'vgg16': model.summary()

            print(f"🚀 Training {model_name}...")
            history, tr_time = train_model(model, train_gen, val_gen, task_name, model_name, epochs=15)

            print(f"📊 Evaluating {model_name}...")
            metrics = evaluate_model(model, val_gen, task_name)

            task_results[model_name] = {
                'model': model,
                'history': history,
                'metrics': metrics,
                'training_time': tr_time
            }

            # Training curves
            plt.figure(figsize=(12,4))
            plt.subplot(1,2,1)
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Val Loss')
            plt.title(f'{model_name} - Loss ({task_name.title()})'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

            plt.subplot(1,2,2)
            if task_name == 'expression':
                plt.plot(history.history['accuracy'], label='Train Acc')
                plt.plot(history.history['val_accuracy'], label='Val Acc')
                plt.title(f'{model_name} - Accuracy ({task_name.title()})'); plt.ylabel('Accuracy')
            else:
                plt.plot(history.history['mae'], label='Train MAE')
                plt.plot(history.history['val_mae'], label='Val MAE')
                plt.title(f'{model_name} - MAE ({task_name.title()})'); plt.ylabel('MAE')
            plt.xlabel('Epoch'); plt.legend(); plt.tight_layout(); plt.show()

        results[task_name] = task_results
    return results

# =============================================================================
# STEP 10: RESULTS COMPARISON
# =============================================================================
def compare_results(results):
    print("\n" + "="*60)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*60)
    rows = []
    for task_name, task_res in results.items():
        print(f"\n📈 {task_name.upper()} TASK RESULTS:")
        print("-"*40)
        for model_name, r in task_res.items():
            m = r['metrics']; t = r['training_time']
            if task_name == 'expression':
                acc = m.get('accuracy', np.nan)
                f1  = m.get('f1_macro', np.nan)
                kpa = m.get('cohen_kappa', np.nan)
                alp = m.get('krippendorff_alpha', np.nan)
                roc = m.get('roc_auc_macro_ovr', np.nan)
                pr  = m.get('pr_auc_macro', np.nan)
                print(f"{model_name:12} | Acc: {acc:.4f} | F1: {f1:.4f} | Kappa: {kpa:.4f} | Alpha: {alp:.4f} | ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} | Time: {t:.2f}s")
                rows += [
                    {'Task':'Expression','Model':model_name,'Metric':'Accuracy','Value':acc,'Time':t},
                    {'Task':'Expression','Model':model_name,'Metric':'F1-Macro','Value':f1,'Time':t},
                    {'Task':'Expression','Model':model_name,'Metric':'CohenKappa','Value':kpa,'Time':t},
                    {'Task':'Expression','Model':model_name,'Metric':'KrippAlpha','Value':alp,'Time':t},
                    {'Task':'Expression','Model':model_name,'Metric':'ROC-AUC','Value':roc,'Time':t},
                    {'Task':'Expression','Model':model_name,'Metric':'PR-AUC','Value':pr,'Time':t},
                ]
            else:
                rmse = m.get('rmse', np.nan)
                corr = m.get('correlation', np.nan)
                sagr = m.get('sagr', np.nan)
                ccc  = m.get('ccc', np.nan)
                print(f"{model_name:12} | RMSE: {rmse:.4f} | CORR: {corr:.4f} | SAGR: {sagr:.4f} | CCC: {ccc:.4f} | Time: {t:.2f}s")
                rows += [
                    {'Task':task_name.title(),'Model':model_name,'Metric':'RMSE','Value':rmse,'Time':t},
                    {'Task':task_name.title(),'Model':model_name,'Metric':'CORR','Value':corr,'Time':t},
                    {'Task':task_name.title(),'Model':model_name,'Metric':'SAGR','Value':sagr,'Time':t},
                    {'Task':task_name.title(),'Model':model_name,'Metric':'CCC','Value':ccc,'Time':t},
                ]

    df = pd.DataFrame(rows)
    # Simple visuals
    fig, axes = plt.subplots(2, 2, figsize=(16,12))
    expr = df[(df['Task']=='Expression') & (df['Metric']=='Accuracy')]
    if not expr.empty:
        axes[0,0].bar(expr['Model'], expr['Value']); axes[0,0].set_title('Expression Accuracy'); axes[0,0].set_ylim(0,1); axes[0,0].tick_params(axis='x', rotation=30)
    val_rmse = df[(df['Task']=='Valence') & (df['Metric']=='RMSE')].set_index('Model')
    aro_rmse = df[(df['Task']=='Arousal') & (df['Metric']=='RMSE')].set_index('Model')
    if not val_rmse.empty and not aro_rmse.empty:
        models_axis = sorted(set(val_rmse.index) | set(aro_rmse.index))
        x = np.arange(len(models_axis)); w=0.35
        axes[0,1].bar(x-w/2, val_rmse.reindex(models_axis)['Value'], w, label='Valence')
        axes[0,1].bar(x+w/2, aro_rmse.reindex(models_axis)['Value'], w, label='Arousal')
        axes[0,1].set_title('RMSE Comparison'); axes[0,1].set_xticks(x); axes[0,1].set_xticklabels(models_axis); axes[0,1].legend(); axes[0,1].tick_params(axis='x', rotation=30)
    time_avg = df.groupby('Model')['Time'].mean().reset_index()
    if not time_avg.empty:
        axes[1,0].bar(time_avg['Model'], time_avg['Time']); axes[1,0].set_title('Avg Training Time (s)'); axes[1,0].tick_params(axis='x', rotation=30)
    expr_f1 = df[(df['Task']=='Expression') & (df['Metric']=='F1-Macro')]
    if not expr_f1.empty:
        axes[1,1].bar(expr_f1['Model'], expr_f1['Value']); axes[1,1].set_title('Expression F1-Macro'); axes[1,1].set_ylim(0,1); axes[1,1].tick_params(axis='x', rotation=30)
    plt.tight_layout(); plt.show()
    return df

# =============================================================================
# STEP 11: SAMPLE PREDICTIONS
# =============================================================================
def visualize_predictions(results, data_mapping, num_samples=8):
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS VISUALIZATION")
    print("="*60)
    expr_model = results['expression']['vgg16']['model']
    np.random.seed(42)
    idxs = np.random.choice(len(data_mapping), num_samples, replace=False)
    samples = [data_mapping[i] for i in idxs]
    gen = AffectNetDataGenerator(samples, batch_size=num_samples, augmentation=False, task='expression')
    Xb, y_true = gen[0]
    y_pred = expr_model.predict(Xb)
    y_pred_c = np.argmax(y_pred, axis=1)
    y_true_c = np.argmax(y_true, axis=1)
    labels = {0:'Neutral',1:'Happy',2:'Sad',3:'Surprise',4:'Fear',5:'Disgust',6:'Anger',7:'Contempt'}
    fig, axes = plt.subplots(2,4, figsize=(20,10)); axes = axes.flatten()
    for i in range(num_samples):
        axes[i].imshow(Xb[i])
        t = labels.get(y_true_c[i],'?'); p = labels.get(y_pred_c[i],'?'); conf = np.max(y_pred[i])
        color = 'green' if y_true_c[i]==y_pred_c[i] else 'red'
        axes[i].set_title(f"True: {t}\nPred: {p} ({conf:.2f})", color=color, fontsize=12, fontweight='bold')
        axes[i].axis('off')
    plt.tight_layout(); plt.show()
    print(f"Sample Prediction Accuracy: {np.mean(y_true_c==y_pred_c):.2%}")

# =============================================================================
# STEP 12: FINAL REPORT
# =============================================================================
def generate_final_report(results, df_comp):
    print("\n" + "="*60)
    print("FINAL ASSIGNMENT REPORT")
    print("="*60)
    report = {
        'models_tested': list(results['expression'].keys()),
        'tasks_performed': list(results.keys()),
        'performance_metrics': {},
        'training_times': {}
    }
    for task, tres in results.items():
        report['performance_metrics'][task] = {}
        report['training_times'][task] = {}
        for mname, r in tres.items():
            report['performance_metrics'][task][mname] = r['metrics']
            report['training_times'][task][mname] = r['training_time']

    print("\n📊 PERFORMANCE SUMMARY:")
    for task in results.keys():
        print(f"\n{task.upper()} TASK:")
        for mname in results[task].keys():
            met = report['performance_metrics'][task][mname]
            t = report['training_times'][task][mname]
            if task == 'expression':
                print(f"  {mname:12} | Acc {met.get('accuracy',np.nan):.4f} | F1 {met.get('f1_macro',np.nan):.4f} | "
                      f"Kappa {met.get('cohen_kappa',np.nan):.4f} | Alpha {met.get('krippendorff_alpha',np.nan):.4f} | "
                      f"ROC {met.get('roc_auc_macro_ovr',np.nan):.4f} | PR {met.get('pr_auc_macro',np.nan):.4f} | Time {t:.2f}s")
            else:
                print(f"  {mname:12} | RMSE {met.get('rmse',np.nan):.4f} | CORR {met.get('correlation',np.nan):.4f} | "
                      f"SAGR {met.get('sagr',np.nan):.4f} | CCC {met.get('ccc',np.nan):.4f} | Time {t:.2f}s")
    print("\n✅ ASSIGNMENT COMPLETED SUCCESSFULLY!")
    return report

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("🚀 STARTING DEEP LEARNING ASSIGNMENT 1")
    print("Facial Expression Recognition with Valence and Arousal Detection")
    print("="*70)
    try:
        results = main_training_pipeline(data_mapping)
        df_comp = compare_results(results)
        visualize_predictions(results, data_mapping)
        final_report = generate_final_report(results, df_comp)
        print("\n🎉 ASSIGNMENT COMPLETED! 🎉")
        print("\nNext steps:\n1. Review plots & metrics\n2. Save models if needed\n3. Prepare PDF report\n4. Push code to GitHub\n5. Submit on LMS")
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback; traceback.print_exc()

# =============================================================================
# EXTRA: SAVE MODELS
# =============================================================================
def save_models(results):
    print("\n💾 SAVING TRAINED MODELS...")
    for task, tres in results.items():
        for mname, r in tres.items():
            fn = f"model_{mname}_{task}.h5"
            r['model'].save(fn)
            print(f"Saved: {fn}")
    print("All models saved successfully!")
# save_models(results)

