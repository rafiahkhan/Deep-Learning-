
**Facial Expression Recognition + Valence/Arousal**

---

## 1) Overview

This repo contains the code and results for training three CNN baselines on the **Facial Expression** task (8 classes) and **Valence** regression (Arousal optional). The models compared:

* **Custom CNN (from scratch)**
* **VGG16 (transfer learning, ImageNet)**
* **ResNet50 (transfer learning, ImageNet)**

Key outputs include: training curves, validation metrics, and a short discussion of results & next steps.

---

## 2) Dataset

* **Total images:** 3,999
* **Expression classes (balanced ~500 each):** Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt
* **Valence mean:** ≈ −0.19 **Arousal mean:** ≈ 0.25
* **Split:** Train 3,199 / Val 800
* **Inputs:** 224×224 RGB

> EDA figures (class histograms, valence/arousal histograms, V-A scatter) are included in the screenshots.

---

## 3) Environment

* Python 3.x, TensorFlow/Keras
* Suggested packages: `tensorflow`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `albumentations`/`imgaug` (optional)

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib
```

---

## 4) How to Run

> The main script is `ass1.py`. Paths inside the script expect folders:

```
Dataset/
 ├─ images/
 └─ annotations/   # labels for expression, valence, arousal
```

Typical usage:

```bash
python ass1.py                # trains baselines (expression + valence), saves logs/plots
```

Common training setup:

* **Optimizer:** Adam (initial LR 1e-3) + ReduceLROnPlateau
* **Epochs:** up to 15 with EarlyStopping(patience=5), best weights restored
* **Batch size:** 16 (expression) / 32 (valence & arousal)
* **Augmentations:** horizontal flip, small rotations, brightness/contrast jitter
* **Losses:** Categorical Cross-Entropy (expression), MSE (valence/arousal)
* **Metrics tracked:** Accuracy (expr), MAE (val/ar). Code supports F1, Kappa, RMSE, CORR, SAGR, CCC.

---

## 5) Model Architectures

### Transfer Learning (VGG16 / ResNet50)

```
Backbone (ImageNet, frozen)
→ GlobalAveragePooling2D
→ Dense(512, ReLU) + Dropout(0.5)
→ Dense(256, ReLU) + Dropout(0.3)
→ Head:
   • Expression: Dense(8, softmax)
   • Valence/Arousal: Dense(1, linear)
```

### Custom CNN (from scratch)

```
4× [Conv2D → BatchNorm → MaxPool → Dropout]
→ Flatten
→ Dense(512) + BN + Dropout(0.5)
→ Dense(256) + Dropout(0.3)
→ Head (as above)
```

**Baseline rationale:** compare strong frozen ImageNet features vs. a small, regularised task-specific model for speed/accuracy trade-offs.

---

## 6) Results

### 6.1 Expression (8-class) — Validation Accuracy & Training Time

| Model             | Val Accuracy |     ~Train Time |
| ----------------- | -----------: | --------------: |
| **Custom CNN**    |   **0.3250** | **≈ 2,655.9 s** |
| **VGG16 (TL)**    |       0.2837 |     ≈ 9,157.7 s |
| **ResNet50 (TL)** |       0.1250 |     ≈ 9,044.3 s |

**Summary:** Custom CNN performed best and was ~3.4× faster than VGG16. ResNet50 under-fit with a frozen base.

---

### 6.2 Valence (Regression) — Validation MAE/MSE & Training Time

| Model             |    Val MAE |    Val MSE |     ~Train Time | Notes                  |
| ----------------- | ---------: | ---------: | --------------: | ---------------------- |
| **VGG16 (TL)**    | **0.3628** | **0.1936** | **≈ 8,882.3 s** | Best error             |
| **ResNet50 (TL)** |     0.3860 |     0.2200 |     ≈ 4,087.1 s | Below VGG16            |
| **Custom CNN**    |     0.4486 |     0.3358 |     ≈ 1,571.0 s | Fastest, highest error |

**Observation:** VGG16 generalises best for valence. Custom CNN is fast but less accurate (early-epoch spikes then stabilisation).

> **Arousal:** not fully captured in the screenshots; compute via the script to fill RMSE/CORR/SAGR/CCC.

---

## 7) Training Curves & Samples 

* Expression: loss↓ & accuracy↑ for **Custom CNN**, **VGG16**, **ResNet50**
* Valence: loss/MAE curves for **VGG16**, **ResNet50**, **Custom CNN** (new)
* EDA dashboard and a sample image grid

---

## 8) Discussion & Recommendations

* **Expression:** Custom CNN likely matches data scale and regularisation needs → best accuracy.
  *Next:* try label smoothing, longer training, or unfreeze TL top blocks to see if VGG16 can surpass.
* **Valence:** VGG16 > ResNet50 > Custom CNN in MAE.
  *Next for Custom CNN:* Huber (Smooth-L1) loss, weight decay, slightly wider network, cosine/one-cycle LR; more aggressive aug.
  *Next for TL:* unfreeze top 1–3 blocks with low LR (1e-5–3e-5), stronger augmentation (MixUp/CutMix), longer schedules.

---

## 9) License / Acknowledgements

* Backbones: `tf.keras.applications` (ImageNet weights)
* This README summarises experiments conducted by **Rafia (i222054)** for Assignment-1.
