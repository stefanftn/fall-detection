import os
import re
import glob
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# GLOBAL CONSTS
# ─────────────────────────────────────────────────────────────────

FS = 200  # frekvencija uzorkovanja (Hz)
WINDOW_SEC = 3.0  # dužina prozora (sekunde)
OVERLAP_PCT = 0.5  # preklapanje prozora (50%)
WINDOW_SAMPLES = int(WINDOW_SEC * FS)  # 600 uzoraka
STEP_SAMPLES = int(WINDOW_SAMPLES * (1 - OVERLAP_PCT))  # 300 uzoraka

# ADC → fizičke jedinice
# ADXL345: opseg ±16g, 13-bitni ADC  →  1 LSB = (2×16) / 2^13 = 0.00390625 g
ADXL345_SCALE = (2 * 16) / (2 ** 13)  # = 0.00390625 g/LSB

# Prag za detekciju udarca unutar Fall fajla (Weak Labelling korekcija)
# Prozori u Fall fajlovima čiji vm_max prelazi ovaj prag dobijaju label=1 (PAD),
# dok ostali (hodanje/stajanje pre/posle pada) dobijaju label=0 (ADL).
# Vrednost 2.0g je konzervativna: slobodan pad bez rotacije ≈ 0g,
# normalno hodanje ≈ 1.2–1.8g peak, udarac pri padu ≈ 3–8g.
FALL_IMPACT_THRESHOLD_G = 2.0

# Nazivi kolona u .txt fajlovima
COL_NAMES = [
    'adxl_x', 'adxl_y', 'adxl_z',  # ADXL345 tro-osni akcelerometar
    'itg_x', 'itg_y', 'itg_z',  # ITG3200 tro-osni žiroskop
    'mma_x', 'mma_y', 'mma_z',  # MMA8451Q tro-osni akcelerometar (rezerva)
]

# Paleta boja za grafike (GitHub Dark tema)
PALETTE = {
    'bg': '#0d1117',
    'ax_bg': '#161b22',
    'border': '#30363d',
    'header': '#21262d',
    'text': '#e6edf3',
    'muted': '#8b949e',
    'green': '#3fb950',
    'red': '#f85149',
    'blue': '#58a6ff',
    'orange': '#ffa657',
    'yellow': '#f0e68c',
    'coral': '#f78166',
    'green_bg': '#1a3a1a',
    'red_bg': '#3a1a1a',
}


# ─────────────────────────────────────────────────────────────────
# PARSE
# ─────────────────────────────────────────────────────────────────

def parse_filename(filepath):
    """
    Format:  <ActivityType><ActivityNumber>_<SubjectID>_R<TrialNumber>.txt
    Example:  D16_SA23_R03.txt  →  subject='SA23', tip='D'(ADL=0)
             F05_SA23_R05.txt  →  subject='SA23', tip='F'(Fall=1)

    Return:
        subject_id  : str  — 'SA01', 'SE03'
        file_label  : int  — 0 za ADL, 1 za Fall, -1 for unknown
    """
    basename = os.path.splitext(os.path.basename(filepath))[0]

    m = re.match(r'^([DF])\d+_([A-Z]{2}\d{2})_R\d+$', basename)
    if not m:
        return None, -1

    file_label = 0 if m.group(1) == 'D' else 1
    subject_id = m.group(2)
    return subject_id, file_label


def load_file(filepath):
    """
    Returns:
        pd.DataFrame with COL_NAMES, or None if there is load error.
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        cleaned = []
        for line in lines:
            line = line.strip().rstrip(';').strip()
            if line:
                cleaned.append(line)

        if not cleaned:
            raise ValueError("Empty file")

        import io
        content = '\n'.join(cleaned)
        df = pd.read_csv(io.StringIO(content), header=None,
                         names=COL_NAMES, sep=',', engine='python')

        if df.shape[1] == 9 and df.shape[0] > 10:
            for col in ['adxl_x', 'adxl_y', 'adxl_z']:
                df[col] = df[col] * ADXL345_SCALE
            return df
    except Exception:
        pass

    print(f"  [ERROR] Cannot load: {os.path.basename(filepath)}")
    return None


# ─────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────

def load_dataset(data_path):
    """
    Returns:
        records : list of dict
            every element: {
                'df'         : pd.DataFrame,   # raw sensor signal
                'file_label' : int,            # 0=ADL, 1=Fall (file label)
                'subject_id' : str,            # 'SA01'
                'filepath'   : str,
            }
        label_counts : dict — {'ADL': N, 'Fall': M}
    """
    all_files = glob.glob(os.path.join(data_path, '**', '*.txt'), recursive=True)
    if not all_files:
        print(f"[ERROR] There is no .txt files in: {data_path}")
        sys.exit(1)

    print(f"  Found {len(all_files)} .txt files in: {data_path}")

    records = []
    label_counts = defaultdict(int)

    for fp in sorted(all_files):
        subject_id, file_label = parse_filename(fp)
        if file_label == -1:
            continue

        df = load_file(fp)
        if df is None or len(df) < WINDOW_SAMPLES:
            continue

        records.append({
            'df': df,
            'file_label': file_label,
            'subject_id': subject_id,
            'filepath': fp,
        })
        label_counts['ADL' if file_label == 0 else 'Fall'] += 1

    subjects = sorted(set(r['subject_id'] for r in records))
    print(f"  ✓ Loaded files:  {len(records)}  "
          f"(ADL: {label_counts['ADL']}, Fall: {label_counts['Fall']})")
    print(f"  ✓ Subjects ({len(subjects)}): {subjects}")

    return records, label_counts


# ─────────────────────────────────────────────────────────────────
# EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────

def _style_ax(ax, title, palette=PALETTE):
    ax.set_facecolor(palette['ax_bg'])
    ax.tick_params(colors=palette['muted'], labelsize=8)
    ax.title.set_color(palette['text'])
    ax.xaxis.label.set_color(palette['muted'])
    ax.yaxis.label.set_color(palette['muted'])
    ax.spines[:].set_edgecolor(palette['border'])
    ax.grid(True, color=palette['border'], linewidth=0.4, alpha=0.6)
    ax.set_title(title, fontsize=10, pad=8, fontweight='bold')


def plot_eda(records, label_counts, output_dir, n_vm_samples=20):
    """
    Generate EDA graph with 6 sub-graphs and save as PNG.

      [0,0] ADL signal — XYZ ose (3 sec)
      [0,1] Fall signal — XYZ ose (3 sec around fall)
      [1,0] VM ceo ADL signal + horizontal line (mean)
      [1,1] VM ceo Fall signal + vertical line (impact peak)
      [2,0] Bar chart: file number per class (class imbalance)
      [2,1] Histogram VM values per class
    """
    P = PALETTE

    adl_rec = next(r for r in records if r['file_label'] == 0)
    fall_rec = next(r for r in records if r['file_label'] == 1)
    df_adl = adl_rec['df']
    df_fall = fall_rec['df']

    vm_adl = np.sqrt(df_adl['adxl_x'] ** 2 + df_adl['adxl_y'] ** 2 + df_adl['adxl_z'] ** 2)
    vm_fall = np.sqrt(df_fall['adxl_x'] ** 2 + df_fall['adxl_y'] ** 2 + df_fall['adxl_z'] ** 2)
    t_adl = np.arange(len(df_adl)) / FS
    t_fall = np.arange(len(df_fall)) / FS

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor(P['bg'])
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.3)

    # --- [0,0] ADL XYZ ---
    ax1 = fig.add_subplot(gs[0, 0])
    n_show = min(600, len(df_adl))
    ax1.plot(t_adl[:n_show], df_adl['adxl_x'][:n_show], color=P['blue'], lw=0.7, label='X', alpha=0.9)
    ax1.plot(t_adl[:n_show], df_adl['adxl_y'][:n_show], color=P['green'], lw=0.7, label='Y', alpha=0.9)
    ax1.plot(t_adl[:n_show], df_adl['adxl_z'][:n_show], color=P['coral'], lw=0.7, label='Z', alpha=0.9)
    ax1.set_xlabel('Vreme (s)');
    ax1.set_ylabel('Ubrzanje (g)')
    ax1.legend(fontsize=8, facecolor=P['header'], edgecolor=P['border'], labelcolor=P['text'])
    _style_ax(ax1, '🟢 ADL — Svakodnevna aktivnost (3 sec)')

    # --- [0,1] Fall XYZ ---
    ax2 = fig.add_subplot(gs[0, 1])
    n_show_f = min(600, len(df_fall))
    ax2.plot(t_fall[:n_show_f], df_fall['adxl_x'][:n_show_f], color=P['blue'], lw=0.7, label='X', alpha=0.9)
    ax2.plot(t_fall[:n_show_f], df_fall['adxl_y'][:n_show_f], color=P['green'], lw=0.7, label='Y', alpha=0.9)
    ax2.plot(t_fall[:n_show_f], df_fall['adxl_z'][:n_show_f], color=P['coral'], lw=0.7, label='Z', alpha=0.9)
    ax2.set_xlabel('Vreme (s)');
    ax2.set_ylabel('Ubrzanje (g)')
    ax2.legend(fontsize=8, facecolor=P['header'], edgecolor=P['border'], labelcolor=P['text'])
    _style_ax(ax2, '🔴 PAD — Signal tokom pada (3 sec)')

    # --- [1,0] VM ADL ceo signal ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t_adl, vm_adl, color=P['blue'], lw=0.8, alpha=0.9)
    ax3.axhline(y=vm_adl.mean(), color=P['yellow'], lw=1.2, linestyle='--', alpha=0.8,
                label=f'Mean = {vm_adl.mean():.2f} g')
    ax3.set_xlabel('Vreme (s)');
    ax3.set_ylabel('VM (g)')
    ax3.legend(fontsize=8, facecolor=P['header'], edgecolor=P['border'], labelcolor=P['text'])
    _style_ax(ax3, 'Vector Magnitude — ADL (ceo signal)')

    # --- [1,1] VM Fall ceo signal ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t_fall, vm_fall, color=P['red'], lw=0.8, alpha=0.9)
    vm_peak_idx = int(np.argmax(vm_fall))
    vm_peak_g = float(vm_fall.iloc[vm_peak_idx])
    ax4.axvline(x=t_fall[vm_peak_idx], color=P['orange'], lw=1.5, linestyle='--', alpha=0.9,
                label=f'Impact @ {t_fall[vm_peak_idx]:.1f}s  ({vm_peak_g:.1f} g)')
    ax4.axhline(y=FALL_IMPACT_THRESHOLD_G, color=P['yellow'], lw=1.0, linestyle=':', alpha=0.7,
                label=f'Threshold = {FALL_IMPACT_THRESHOLD_G} g')
    ax4.legend(fontsize=8, facecolor=P['header'], edgecolor=P['border'], labelcolor=P['text'])
    ax4.set_xlabel('Vreme (s)');
    ax4.set_ylabel('VM (g)')
    _style_ax(ax4, 'Vector Magnitude — PAD (impact peak + threshold)')

    # --- [2,0] Class distribution ---
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_facecolor(P['ax_bg'])
    bars = ax5.bar(['ADL (0)', 'Fall (1)'],
                   [label_counts['ADL'], label_counts['Fall']],
                   color=[P['green'], P['red']], edgecolor=P['border'], width=0.5)
    for bar, val in zip(bars, [label_counts['ADL'], label_counts['Fall']]):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(val), ha='center', va='bottom',
                 color=P['text'], fontsize=11, fontweight='bold')
    ax5.set_ylabel('Broj fajlova')
    total = label_counts['ADL'] + label_counts['Fall']
    imbalance = label_counts['ADL'] / max(label_counts['Fall'], 1)
    _style_ax(ax5, f'Distribucija klasa  (ukupno: {total},  ADL:Fall = {imbalance:.1f}:1)')

    # --- [2,1] VM Histogram ---
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.set_facecolor(P['ax_bg'])
    all_vm_adl, all_vm_fall = [], []
    for rec in records[:n_vm_samples]:
        df = rec['df']
        vm = np.sqrt(df['adxl_x'] ** 2 + df['adxl_y'] ** 2 + df['adxl_z'] ** 2).values
        if rec['file_label'] == 0:
            all_vm_adl.extend(vm[:500].tolist())
        else:
            all_vm_fall.extend(vm[:500].tolist())

    ax6.hist(all_vm_adl, bins=80, alpha=0.6, color=P['green'], label='ADL', density=True)
    ax6.hist(all_vm_fall, bins=80, alpha=0.6, color=P['red'], label='Fall', density=True)
    ax6.axvline(x=FALL_IMPACT_THRESHOLD_G, color=P['yellow'], lw=1.2, linestyle='--',
                alpha=0.8, label=f'Threshold = {FALL_IMPACT_THRESHOLD_G} g')
    ax6.set_xlabel('Vector Magnitude (g)');
    ax6.set_ylabel('Gustina')
    ax6.legend(fontsize=8, facecolor=P['header'], edgecolor=P['border'], labelcolor=P['text'])
    _style_ax(ax6, 'Distribucija VM vrednosti (ADL vs Fall)')

    # VM statistike
    print(f"\n  VM statistike ({n_vm_samples} uzoraka):")
    if all_vm_adl:
        print(f"    ADL  — mean={np.mean(all_vm_adl):.3f}g  "
              f"max={np.max(all_vm_adl):.3f}g  std={np.std(all_vm_adl):.3f}g")
    if all_vm_fall:
        print(f"    Fall — mean={np.mean(all_vm_fall):.3f}g  "
              f"max={np.max(all_vm_fall):.3f}g  std={np.std(all_vm_fall):.3f}g")
    print(f"    FALL_IMPACT_THRESHOLD_G = {FALL_IMPACT_THRESHOLD_G} g "
          f"(vm_max iznad praga → label=1 za prozore u Fall fajlovima)")

    fig.suptitle(
        'SisFall Dataset — Exploratory Data Analysis\n'
        'Multimodelni sistem za detekciju pada',
        fontsize=14, color=P['text'], fontweight='bold', y=0.98,
    )

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, '1_eda.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=P['bg'])
    plt.close()
    print(f"  ✓ EDA grafik sačuvan: {save_path}")

    return vm_peak_g
