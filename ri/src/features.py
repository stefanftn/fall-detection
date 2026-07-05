import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
from collections import defaultdict
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from data_loader import (
    FS, WINDOW_SAMPLES, STEP_SAMPLES, WINDOW_SEC, OVERLAP_PCT,
    FALL_IMPACT_THRESHOLD_G, PALETTE
)

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────
# EKSTRAKCIJA OBELEŽJA
# ─────────────────────────────────────────────────────────────────

def extract_features(window_df):
    """
    17 obeležja iz jednog prozora (DataFrame [WINDOW_SAMPLES × 9]).

    Grupe obeležja:
    ┌─────────────────────────────────────────────────────────────┐
    │ Vector Magnitude (VM) — ADXL345                             │
    │   vm_mean    prosečna VM → nivo aktivnosti                  │
    │   vm_max     maksimalna VM → intenzitet udarca (key!)       │
    │   vm_std     std VM → haotičnost/nepravilnost kretanja      │
    │   vm_range   max−min VM → amplituda promene                 │
    │   vm_energy  Σ(vm²)/N → ukupna energija kretanja            │
    │   vm_zcr     Zero Crossing Rate → učestalost smene smera    │
    ├─────────────────────────────────────────────────────────────┤
    │ Statistike po osama — ADXL345                               │
    │   ax_std, ay_std, az_std  → varijabilnost po osama          │
    │   ax_max, ay_max, az_max  → peak apsolutnih vrednosti       │
    ├─────────────────────────────────────────────────────────────┤
    │ Žiroskop (ITG3200)                                          │
    │   gyro_vm_mean  prosečna rotaciona VM                       │
    │   gyro_vm_max   peak rotacije → nagla promena ugla          │
    │   gyro_vm_std   varijabilnost rotacije                      │
    ├─────────────────────────────────────────────────────────────┤
    │ Složena obeležja                                            │
    │   sma        Signal Magnitude Area = Σ|x|+|y|+|z| / N       │
    │   tilt_post  srednja z-osa u poslednjoj 1/3 prozora         │
    │              (detect. ležećeg položaja posle pada)          │
    └─────────────────────────────────────────────────────────────┘

    Parametri:
        window_df : pd.DataFrame oblika [WINDOW_SAMPLES × 9]

    Povratne vrednosti:
        dict — 17 numeričkih obeležja
    """
    x = window_df['adxl_x'].values
    y = window_df['adxl_y'].values
    z = window_df['adxl_z'].values
    gx = window_df['itg_x'].values
    gy = window_df['itg_y'].values
    gz = window_df['itg_z'].values

    N = len(x)
    vm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    gyro_vm = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)

    # Zero Crossing Rate — koliko puta VM prelazi svoju srednju vrednost
    vm_c = vm - vm.mean()
    zcr = float(np.sum(np.abs(np.diff(np.sign(vm_c)))) / (2 * N))

    # Nagnutost u poslednjoj 1/3 prozora (stanje posle pada)
    post = z[N * 2 // 3:]
    tilt_post = float(np.mean(post))

    return {
        # VM obeležja
        'vm_mean': float(np.mean(vm)),
        'vm_max': float(np.max(vm)),
        'vm_std': float(np.std(vm)),
        'vm_range': float(np.max(vm) - np.min(vm)),
        'vm_energy': float(np.sum(vm ** 2) / N),
        'vm_zcr': zcr,
        # Statistike po osama
        'ax_std': float(np.std(x)),
        'ay_std': float(np.std(y)),
        'az_std': float(np.std(z)),
        'ax_max': float(np.max(np.abs(x))),
        'ay_max': float(np.max(np.abs(y))),
        'az_max': float(np.max(np.abs(z))),
        # Žiroskop
        'gyro_vm_mean': float(np.mean(gyro_vm)),
        'gyro_vm_max': float(np.max(gyro_vm)),
        'gyro_vm_std': float(np.std(gyro_vm)),
        # Složena obeležja
        'sma': float((np.sum(np.abs(x)) + np.sum(np.abs(y)) + np.sum(np.abs(z))) / N),
        'tilt_post': tilt_post,
    }


def _label_window(vm_max, file_label, impact_threshold=FALL_IMPACT_THRESHOLD_G):
    """
    Određuje labelu jednog prozora.
        ADL fajl  (file_label=0): uvek 0 — čiste ADL sekvence
        Fall fajl (file_label=1):
            vm_max >= impact_threshold → 1 (PAD detektovan)
            vm_max <  impact_threshold → 0 (hodanje/stajanje pre/posle pada)
    """
    if file_label == 0:
        return 0
    return 1 if vm_max >= impact_threshold else 0


# ─────────────────────────────────────────────────────────────────
# KLIZEĆI PROZOR
# ─────────────────────────────────────────────────────────────────

def apply_sliding_window(records, impact_threshold=FALL_IMPACT_THRESHOLD_G):
    """
    Povratne vrednosti:
        X          : pd.DataFrame [N_prozora × 17]  — matrica obeležja
        y          : np.ndarray [N_prozora]         — labele (0/1)
        groups     : np.ndarray [N_prozora]         — subject ID po prozoru (za GroupKFold)
        win_stats  : dict — statistike po tipu labele
    """
    print(f"\n  Parametri segmentacije:")
    print(f"    FS={FS}Hz  |  Prozor={WINDOW_SEC}s={WINDOW_SAMPLES} uzoraka  "
          f"|  Overlap={int(OVERLAP_PCT * 100)}%  |  Korak={STEP_SAMPLES} uzoraka")
    print(f"    Weak Labelling threshold: vm_max ≥ {impact_threshold}g → label=1")

    all_features = []
    all_labels = []
    all_groups = []  # subject_id po prozoru — za GroupKFold
    win_stats = defaultdict(int)

    for rec in records:
        df = rec['df']
        file_label = rec['file_label']
        subject_id = rec['subject_id']
        n_samples = len(df)

        for start in range(0, n_samples - WINDOW_SAMPLES + 1, STEP_SAMPLES):
            end = start + WINDOW_SAMPLES
            window = df.iloc[start:end]

            if len(window) < WINDOW_SAMPLES:
                break

            feats = extract_features(window)
            win_label = _label_window(feats['vm_max'], file_label, impact_threshold)

            all_features.append(feats)
            all_labels.append(win_label)
            all_groups.append(subject_id)

            # Statistike za ispis
            if file_label == 0:
                win_stats['ADL (iz ADL fajla)'] += 1
            elif win_label == 1:
                win_stats['Fall (udarac detektovan)'] += 1
            else:
                win_stats['ADL (mirovanje u Fall fajlu)'] += 1

    X = pd.DataFrame(all_features)
    y = np.array(all_labels, dtype=int)
    groups = np.array(all_groups)

    # Provera kvaliteta podataka
    nan_c = int(X.isna().sum().sum())
    inf_c = int(np.isinf(X.values).sum())
    if nan_c > 0 or inf_c > 0:
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        print(f"  [UPOZORENJE] Zamenjeno {nan_c} NaN i {inf_c} Inf vrednosti")

    print(f"\n  Statistike prozora po tipu:")
    for tip, count in win_stats.items():
        pct = 100 * count / len(y) if len(y) > 0 else 0
        print(f"    {tip:<38}: {count:5d}  ({pct:.1f}%)")
    print(f"\n  UKUPNO prozora: {len(y)}  "
          f"(ADL={np.sum(y == 0)}, Fall={np.sum(y == 1)})")
    print(f"  Obeležja: {X.shape[1]}  →  {list(X.columns)}")

    return X, y, groups, win_stats


# ─────────────────────────────────────────────────────────────────
# PODELA PODATAKA — GroupKFold po subjektima
# ─────────────────────────────────────────────────────────────────

def split_by_subjects(X, y, groups, test_subjects=None, n_splits=5, random_state=42):
    """
    Deli dataset na train/test skupove tako da isti subjekt ne može biti u oba.

    Dve strategije:
    A) Eksplicitna lista test subjekata (test_subjects != None):
       Direktno razdvajanje po imenima subjekata — maksimalna kontrola.
       Npr. test_subjects=['SA04', 'SA05', 'SE03'] za leave-out evaluaciju.

    B) GroupKFold (test_subjects=None):
       n_splits foldova; uzima poslednji fold kao test skup.
       Ekvivalentno Leave-One-Group-Out za mali broj subjekata.

    Povratne vrednosti:
        X_train_sc, X_test_sc : np.ndarray — standardizovani setovi
        y_train, y_test       : np.ndarray — labele
        groups_train          : np.ndarray — grupe train seta (za CV)
        scaler                : StandardScaler — fitovan na train setu
        split_info            : dict — detalji o podeli
    """
    X_vals = X.values if isinstance(X, pd.DataFrame) else X

    if test_subjects is not None:
        # Strategija A: eksplicitna lista
        test_mask = np.isin(groups, test_subjects)
        train_mask = ~test_mask
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        method = f"Eksplicitni test subjekti: {test_subjects}"
    else:
        # Strategija B: GroupKFold — poslednji fold kao test
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(X_vals, y, groups))
        train_idx, test_idx = splits[-1]
        test_subs = list(set(groups[test_idx]))
        method = f"GroupKFold (fold {n_splits}/{n_splits}), test subjecti: {sorted(test_subs)}"

    X_train_raw = X_vals[train_idx]
    X_test_raw = X_vals[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    groups_train = groups[train_idx]

    # Standardizacija: fit samo na train setu, transform na oba
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_raw)
    X_test_sc = scaler.transform(X_test_raw)

    split_info = {
        'method': method,
        'train_size': len(y_train),
        'test_size': len(y_test),
        'train_adl': int(np.sum(y_train == 0)),
        'train_fall': int(np.sum(y_train == 1)),
        'test_adl': int(np.sum(y_test == 0)),
        'test_fall': int(np.sum(y_test == 1)),
        'test_subjects': sorted(set(groups[test_idx])),
    }

    print(f"\n  Podela podataka: {method}")
    print(f"  Train: {split_info['train_size']} prozora  "
          f"(ADL={split_info['train_adl']}, Fall={split_info['train_fall']})")
    print(f"  Test:  {split_info['test_size']} prozora  "
          f"(ADL={split_info['test_adl']}, Fall={split_info['test_fall']})")
    print(f"  Test subjecti: {split_info['test_subjects']}")
    print(f"  Scaler: StandardScaler (fit samo na train setu)")

    return X_train_sc, X_test_sc, y_train, y_test, groups_train, scaler, split_info


# ─────────────────────────────────────────────────────────────────
# VIZUELIZACIJA
# ─────────────────────────────────────────────────────────────────

def plot_feature_distributions(X, y, output_dir):
    """
    Histogram distribucija 6 ključnih obeležja po klasama.
      - Zeleni histogram: ADL klasa
      - Crveni histogram: Fall klasa
      - Legenda sa prosečnom vrednošću svake klase
    """
    P = PALETTE
    key_feats = ['vm_max', 'vm_std', 'vm_energy', 'gyro_vm_max', 'sma', 'vm_range']
    # Filtriranje feature-a koji stvarno postoje u X
    key_feats = [f for f in key_feats if f in X.columns]

    n_cols = 3
    n_rows = (len(key_feats) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.patch.set_facecolor(P['bg'])
    axes = np.array(axes).flatten()

    for i, feat in enumerate(key_feats):
        ax = axes[i]
        vals_adl = X[feat][y == 0].values
        vals_fall = X[feat][y == 1].values
        ax.set_facecolor(P['ax_bg'])
        ax.hist(vals_adl, bins=50, alpha=0.65, color=P['green'],
                label=f'ADL  (μ={vals_adl.mean():.2f})', density=True)
        ax.hist(vals_fall, bins=50, alpha=0.65, color=P['red'],
                label=f'Fall (μ={vals_fall.mean():.2f})', density=True)
        ax.set_title(feat, color=P['text'], fontsize=10, fontweight='bold')
        ax.tick_params(colors=P['muted'], labelsize=8)
        ax.spines[:].set_edgecolor(P['border'])
        ax.grid(True, color=P['border'], linewidth=0.4, alpha=0.6)
        ax.legend(fontsize=7, facecolor=P['header'], edgecolor=P['border'], labelcolor=P['text'])

    # Sakrij neiskorišćene ose
    for j in range(len(key_feats), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Distribucija Obeležja po Klasama (ADL vs Fall)',
                 color=P['text'], fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, '2_feature_distributions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=P['bg'])
    plt.close()
    print(f"  ✓ Feature distribucije sačuvane: {save_path}")
