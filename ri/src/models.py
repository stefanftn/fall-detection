"""
Modeli:
  1. Gaussian Naive Bayes (GNB)
     - Verovatnosni model, brz baseline
     - Pretpostavlja normalnu raspodelu obeležja unutar svake klase
     - Prednost: izuzetno brz trening i inferencija, interpretabilan

  2. k-Nearest Neighbors (kNN)
     - Metrički model: klasifikuje na osnovu k najbližih suseda u prostoru obeležja
     - Grid search za optimalno k prioritizujući Recall (medicinska primena)
     - Prednost: nema pretpostavki o raspodeli, dobro sa nelinearnim granicama

  3. Multilayer Perceptron (MLP)
     - Arhitektura: Input → Dense(128) → BN → Dropout(0.3) → Dense(64) → BN →
                    Dropout(0.2) → Dense(32) → Dense(1, sigmoid)
     - L2 regularizacija, class weights za imbalanced dataset
     - Custom threshold 0.3: snižen radi višeg Recall-a (bolje je lažna uzbuna
       nego propušten pravi pad u medicinskoj primeni)
     - EarlyStopping + ReduceLROnPlateau za stabilan trening
"""

import time
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score, confusion_matrix

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')


# ─────────────────────────────────────────────────────────────────
# POMOĆNA FUNKCIJA: EVALUACIJA
# ─────────────────────────────────────────────────────────────────

def _evaluate(name, model, X_test, y_test, train_time, threshold=0.5):
    """
    Računa sve metrike i meri inference vreme za jedan model.

    Inference vreme se meri kao prosek 10 prolaza nad prvih 200 uzoraka,
    pa se deli sa brojem uzoraka → ms po jednom prozoru.

    Povratne vrednosti:
        dict sa svim metrikama i dodatnim podacima (y_pred, y_prob, cm, ...)
    """
    n_measure = min(200, len(X_test))
    sample = X_test[:n_measure]

    # Meri inference vreme
    t0 = time.perf_counter()
    for _ in range(10):
        if hasattr(model, 'predict_proba'):
            model.predict_proba(sample)
        else:
            model.predict(sample)
    inference_ms = (time.perf_counter() - t0) / 10 / n_measure * 1000

    # Predikcija verovatnoća
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test).flatten()

    y_pred = (y_prob >= threshold).astype(int)

    return {
        'name': name,
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'inference_ms': float(inference_ms),
        'train_time_s': float(train_time),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_pred': y_pred,
        'y_prob': y_prob,
        'threshold': threshold,
    }


# ─────────────────────────────────────────────────────────────────
# MODEL 1: GAUSSIAN NAIVE BAYES
# ─────────────────────────────────────────────────────────────────

def train_gnb(X_train, y_train, X_test, y_test):
    """
    Pretpostavke modela:
      - p(x_i | klasa) ~ N(μ, σ²) za svako obeležje i
      - Obeležja su uslovno nezavisna (Naive pretpostavka)
      - Prior verovatnoće se uče iz podataka (var_smoothing=1e-9 stabilizuje varijancu)
    """
    print("\n  [1/3] Gaussian Naive Bayes")
    print("        Pretpostavka: p(x_i | klasa) ~ Gaussian, obeležja nezavisna")

    model = GaussianNB(var_smoothing=1e-9)

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    result = _evaluate('Gaussian Naive Bayes', model, X_test, y_test, train_time)
    result['model'] = model

    print(f"        ✓ Trening: {train_time:.4f}s  |  "
          f"Recall={result['recall']:.4f}  F1={result['f1']:.4f}")
    return result


# ─────────────────────────────────────────────────────────────────
# MODEL 2: k-NEAREST NEIGHBORS
# ─────────────────────────────────────────────────────────────────

def train_knn(X_train, y_train, X_val, y_val, X_test, y_test, k_candidates=None):
    """
    Grid search kriterijum: maksimalni Recall, uz F1 kao tiebreaker.
    Razlog: u medicinskom sistemu za detekciju pada Recall (Sensitivity) je
    primarna metrika — propušteni pravi pad (False Negative) je opasno,
    dok je lažna uzbuna (False Positive) samo neugodnost.

    Konfiguracija:
      weights='distance' — bliži susedi imaju veći uticaj (bolje od uniform za ove podatke)
      metric='euclidean' — standardni L2 u standardizovanom prostoru obeležja
      n_jobs=-1          — paralelno računanje na svim CPU jezgrima
    """
    if k_candidates is None:
        k_candidates = [3, 5, 7, 9, 11, 15]

    print("\n  [2/3] k-Nearest Neighbors")
    print(f"        Grid search za k ∈ {k_candidates}  "
          f"(kriterijum: max Recall, tiebreaker: F1)")

    best_k, best_recall, best_f1 = k_candidates[0], -1.0, -1.0
    k_search = {}

    for k in k_candidates:
        tmp = KNeighborsClassifier(n_neighbors=k, weights='distance',
                                   metric='euclidean', n_jobs=-1)
        tmp.fit(X_train, y_train)
        y_pred = tmp.predict(X_val)
        rec = float(recall_score(y_val, y_pred, zero_division=0))
        f1 = float(f1_score(y_val, y_pred, zero_division=0))
        k_search[k] = {'recall': rec, 'f1': f1}

        if rec > best_recall or (rec == best_recall and f1 > best_f1):
            best_k, best_recall, best_f1 = k, rec, f1

        print(f"          k={k:2d}: Recall={rec:.4f}  F1={f1:.4f}")

    print(f"        ✓ Optimalno: k={best_k}  (Recall={best_recall:.4f})")

    # Treniraj finalni model sa optimalnim k
    model = KNeighborsClassifier(n_neighbors=best_k, weights='distance',
                                 metric='euclidean', n_jobs=-1)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    result = _evaluate(f'kNN (k={best_k})', model, X_test, y_test, train_time)
    result['model'] = model
    result['k_search'] = k_search
    result['best_k'] = best_k

    print(f"        ✓ Trening: {train_time:.4f}s  |  "
          f"Recall={result['recall']:.4f}  F1={result['f1']:.4f}")
    return result


# ─────────────────────────────────────────────────────────────────
# MODEL 3: MLP NEURONSKA MREŽA
# ─────────────────────────────────────────────────────────────────

def build_mlp(n_features):
    """
    MLP arhitektura za binarnu klasifikaciju.

    Arhitektura:
        Input(n_features)
        → Dense(128, ReLU, L2=1e-4)
        → BatchNormalization          ← stabilizuje distribuciju aktivacija
        → Dropout(0.3)                ← regularizacija, sprečava overfitting
        → Dense(64, ReLU, L2=1e-4)
        → BatchNormalization
        → Dropout(0.2)
        → Dense(32, ReLU)             ← finalna reprezentacija
        → Dense(1, Sigmoid)           ← izlaz: p(Fall | prozor)

    Regularizacione tehnike:
      - L2 (weight decay): penalizuje velike težine → glaðja odlučna granica
      - BatchNorm: normalizuje aktivacije → brža konvergencija, manja osetljivost na lr
      - Dropout: slučajno isključuje neurone → bolja generalizacija

    Povratne vrednosti:
        tf.keras.Model — nekompajlirani model
    """
    tf.random.set_seed(42)
    model = Sequential([
        Dense(128, activation='relu', input_shape=(n_features,),
              kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid'),
    ], name='MLP_FallDetection')
    return model


def train_mlp(X_train, y_train, X_val, y_val, X_test, y_test, epochs=100, batch_size=32, mlp_threshold=0.3):
    """
    Posebnosti treninga:
      class_weight: automatski izračunat iz raspodele train seta
        → Fall klasa dobija veći weight ako je manje zastupljena
        → kompenzuje class imbalance bez undersamplinga

      EarlyStopping (patience=15): zaustavlja trening ako val_loss ne opada
        → čuva best weights automatski

      ReduceLROnPlateau (patience=7): prepolovi lr ako val_loss stagnira
        → fine-tuning kad se model približi optimumu

      custom threshold=0.3 (default):
        p(Fall) >= 0.3 → predikcija = 1
        Snižen sa standardnih 0.5 jer nam je važniji Recall od Precision.
        Svaki Fall koji model oceni sa ≥30% verovatnoćom biće prijavljen.

    Povratne vrednosti:
        dict — result_dict + 'history' + 'keras_model'
    """
    n_features = X_train.shape[1]
    print(f"\n  [3/3] MLP Neuronska Mreža")
    print(f"        Arhitektura: {n_features}→128(BN,D0.3)→64(BN,D0.2)→32→1(σ)")
    print(f"        Threshold: {mlp_threshold}  |  Epochs: {epochs}  |  Batch: {batch_size}")

    model = build_mlp(n_features)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    # Class weights
    n_adl = int(np.sum(y_train == 0))
    n_fall = int(np.sum(y_train == 1))
    total = n_adl + n_fall
    w_adl = total / (2 * n_adl) if n_adl > 0 else 1.0
    w_fall = total / (2 * n_fall) if n_fall > 0 else 1.0
    class_weight = {0: w_adl, 1: w_fall}
    print(f"        Class weights: ADL={w_adl:.2f}, Fall={w_fall:.2f}")

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=7, min_lr=1e-6, verbose=0),
    ]

    t0 = time.perf_counter()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=0,
    )
    train_time = time.perf_counter() - t0
    actual_epochs = len(history.history['loss'])

    result = _evaluate('MLP Neural Network', model, X_test, y_test,
                       train_time, threshold=mlp_threshold)
    result['model'] = model
    result['history'] = history.history
    result['keras_model'] = model

    print(f"        ✓ Epohe: {actual_epochs}/{epochs}  |  "
          f"Trening: {train_time:.1f}s  |  "
          f"Recall={result['recall']:.4f}  F1={result['f1']:.4f}")
    return result
