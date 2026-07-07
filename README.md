# Multimodal fall detection system

Real-time automatic fall detection system based on accelerometer and gyroscope data. It uses machine learning on the SisFall dataset and consists of three components: an ML pipeline, a FastAPI backend, and an Android application.

---

## Structure

```
fall-detection/
├── ri/
│   ├── SisFall_dataset/
│   └── src/
│       ├── data_loader.py     # dataset loader, ADC conversion, EDA
│       ├── features.py        # sliding window, features extraction, GroupKFold
│       ├── models.py          # GNB, kNN, MLP
│       ├── evaluation.py      # metrics, graphs, ROC
│       ├── projekat.ipynb     # Jupyter notebook - pipeline
│       └── requirements.txt
├── backend/
│   ├── main.py                # FastAPI server
│   ├── requirements.txt
│   └── models/                # trained models (not in Git, can be generated with notebook)
│       ├── gnb.pkl
│       ├── knn.pkl
│       ├── scaler.pkl
│       ├── mlp.keras
│       └── feature_names.json
└── android/                   # Android app for demo (Java)
```

---

## Dataset

SisFall dataset — 38 subjects (23 young + 15 older), 200 Hz, 9-axis sensor (ADXL345 + ITG3200 + MMA8451Q), 19 ADL types i 15 fall types.

**Download:**
```
https://www.kaggle.com/datasets/nvnikhil0001/sis-fall-original-dataset
```

Extract it in `ri/SisFall_dataset/`.

---

## ML Pipeline

### Installation

```bash
cd ri/src
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Pokretanje

```bash
jupyter notebook projekat.ipynb
```

### Ključne tehničke odluke

**Weak Labelling** - windows inside Fall files get `label=1` only if `vm_max ≥ 2.0g`. Fall files contain walking before falling and lying down after - without this fix all those windows would be incorrectly marked as falling.

**GroupKFold division by subjects** - the same subject cannot be in both the train and the test set. Prevents Data Leakage that occurs because each person has recognizable individual movement patterns.

**17 features per window** - VM statistics (mean, max, std, range, energy, ZCR), axis statistics, VM gyro and complex features (SMA, tilt_post).

### Models and results

| Model | Accuracy | Recall | F1 | Inference |
|---|---|---|---|---|
| Gaussian Naive Bayes | 0.8789 | 0.8902 | 0.5241 | 0.003 ms |
| kNN (k=9) | 0.9870 | 0.8845 | 0.9104 | 0.090 ms |
| MLP Neural Network | 0.9620 | 0.9686 | 0.7923 | 0.998 ms |

The primary metric is **Recall** - in the medical system, a missed real fall (False Negative) is more dangerous than a false alarm (False Positive). MLP threshold is lowered to 0.3 for higher Recall.

---

## Backend

### Installation

```bash
cd back
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy the trained models from the notebook to `back/models/`:
```
gnb.pkl, knn.pkl, scaler.pkl, mlp.keras, feature_names.json
```

### Launch

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### API

```
GET  /health
     → { "status": "ok", "models_loaded": [...] }

POST /predict
     Body: { "model": "gnb"|"knn"|"mlp", "features": [17 floats] }
     → { "label": 0|1, "label_text": "ADL"|"FALL",
         "confidence": float, "model_used": str, "inference_ms": float }
```

---

## Android app

The application registers the accelerometer and gyroscope at 200 Hz, accumulates samples in a ring buffer of 600 samples (3 seconds), and for every 300 new samples it calculates 17 features and sends a POST request to the backend.

### Settings

In `android/main/java/com/falldetection/api/ApiClient.java` change the IP address:

```java
public static final String BASE_URL = "http://192.168.X.X:8000";
```

Find the laptop's IP address with `ipconfig` (Windows) or `ip addr` (Linux). Phone and laptop must be on the same WiFi network.

### Dependencies (app/build.gradle)

```
com.squareup.okhttp3:okhttp:4.12.0
com.google.code.gson:gson:2.10.1
androidx.appcompat:appcompat:1.7.0
com.google.android.material:material:1.12.0
```