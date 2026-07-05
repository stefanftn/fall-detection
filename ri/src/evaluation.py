import os
import warnings
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc

from data_loader import PALETTE

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────
# GLAVNI EVALUATION GRAFIK
# ─────────────────────────────────────────────────────────────────

def plot_model_results(results_list, output_dir):
    """
    Red 1: Confusion matrica za svaki model (normalizovana) + Bar chart inference vremena
    Red 2: Tabela metrika (Accuracy, Recall, Precision, F1, Inference, Train vreme)
    Red 3: MLP learning krive — Loss i Accuracy tokom treninga

    Parametri:
        results_list : lista dict-ova iz models.py (jedan po modelu)
        output_dir   : putanja za čuvanje PNG-a
    """
    P = PALETTE
    n_models = len(results_list)

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor(P['bg'])
    gs = gridspec.GridSpec(3, n_models + 1, figure=fig,
                           hspace=0.55, wspace=0.35,
                           height_ratios=[1.2, 0.8, 0.9])

    # ── Red 1: Confusion matrice ──────────────────────────────────
    for i, res in enumerate(results_list):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(P['ax_bg'])
        cm = res['confusion_matrix']
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        ax.imshow(cm_norm, interpolation='nearest', cmap='Greys', vmin=0, vmax=1)
        ax.set_xticks([0, 1]);
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['ADL (0)', 'Fall (1)'], color=P['muted'], fontsize=8)
        ax.set_yticklabels(['ADL (0)', 'Fall (1)'], color=P['muted'], fontsize=8, rotation=45)
        ax.set_xlabel('Predviđeno', color=P['muted'], fontsize=8)
        ax.set_ylabel('Stvarno', color=P['muted'], fontsize=8)

        for r in range(2):
            for c in range(2):
                v_abs = cm[r, c]
                v_norm = cm_norm[r, c]
                color = P['bg'] if v_norm > 0.5 else P['text']
                ax.text(c, r, f'{v_abs}\n({v_norm:.1%})',
                        ha='center', va='center',
                        color=color, fontsize=9, fontweight='bold')

        rec = res['recall']
        f1 = res['f1']
        ax.set_title(f"{res['name']}\nRecall={rec:.4f}  F1={f1:.4f}",
                     color=P['text'], fontsize=9, fontweight='bold', pad=8)
        ax.spines[:].set_edgecolor(P['border'])
        ax.tick_params(colors=P['muted'])

    # ── Inference Bar Chart ───────────────────────────────────────
    ax_inf = fig.add_subplot(gs[0, n_models])
    ax_inf.set_facecolor(P['ax_bg'])
    names = [r['name'].replace(' (', '\n(').replace('Neural Network', 'NN')
             for r in results_list]
    times = [r['inference_ms'] for r in results_list]
    colors = [P['blue'], P['green'], P['orange']]
    bars = ax_inf.bar(range(len(names)), times,
                      color=colors[:n_models], edgecolor=P['border'], width=0.5)
    for bar, t in zip(bars, times):
        ax_inf.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(times) * 0.02,
                    f'{t:.3f}ms', ha='center',
                    color=P['text'], fontsize=9, fontweight='bold')
    ax_inf.set_xticks(range(len(names)))
    ax_inf.set_xticklabels(names, color=P['muted'], fontsize=8)
    ax_inf.set_ylabel('Vreme (ms)', color=P['muted'], fontsize=9)
    ax_inf.set_title('Inference Time\n(po prozoru)', color=P['text'], fontsize=9, fontweight='bold')
    ax_inf.tick_params(colors=P['muted'])
    ax_inf.spines[:].set_edgecolor(P['border'])
    ax_inf.grid(True, axis='y', color=P['border'], linewidth=0.4, alpha=0.6)

    # ── Red 2: Tabela metrika ──────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[1, :])
    ax_tbl.set_facecolor(P['bg'])
    ax_tbl.axis('off')

    col_headers = ['Model', 'Accuracy', 'Recall\n(Sensitivity)',
                   'Precision', 'F1-Score',
                   'Inference\n(ms/prozor)', 'Train\nVreme (s)']
    table_data = []
    for res in results_list:
        table_data.append([
            res['name'],
            f"{res['accuracy']:.4f}",
            f"{res['recall']:.4f}",
            f"{res['precision']:.4f}",
            f"{res['f1']:.4f}",
            f"{res['inference_ms']:.4f}",
            f"{res['train_time_s']:.2f}",
        ])

    tbl = ax_tbl.table(cellText=table_data, colLabels=col_headers,
                       loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.2)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(P['border'])
        if row == 0:
            cell.set_facecolor(P['header'])
            cell.set_text_props(color=P['text'], fontweight='bold')
        else:
            cell.set_facecolor(P['ax_bg'])
            txt = cell.get_text().get_text()
            if col == 2:  # Recall kolona
                try:
                    v = float(txt)
                    if v >= 0.85:
                        cell.set_facecolor(P['green_bg'])
                        cell.set_text_props(color=P['green'], fontweight='bold')
                    elif v < 0.70:
                        cell.set_facecolor(P['red_bg'])
                        cell.set_text_props(color=P['red'], fontweight='bold')
                    else:
                        cell.set_text_props(color=P['text'])
                except ValueError:
                    cell.set_text_props(color=P['text'])
            else:
                cell.set_text_props(color=P['text'])

    ax_tbl.set_title('Komparativna Tabela Metrika', color=P['text'],
                     fontsize=11, fontweight='bold', pad=15)

    # ── Red 3: MLP Learning Krive ─────────────────────────────────
    mlp_res = next((r for r in results_list if 'history' in r), None)
    if mlp_res:
        hist = mlp_res['history']
        epochs = range(1, len(hist['loss']) + 1)

        ax_loss = fig.add_subplot(gs[2, :2])
        ax_loss.set_facecolor(P['ax_bg'])
        ax_loss.plot(epochs, hist['loss'], color=P['red'], lw=1.5, label='Train Loss')
        ax_loss.plot(epochs, hist['val_loss'], color=P['blue'], lw=1.5,
                     linestyle='--', label='Val Loss')
        ax_loss.set_xlabel('Epoha', color=P['muted'], fontsize=9)
        ax_loss.set_ylabel('Loss (Binary Crossentropy)', color=P['muted'], fontsize=9)
        ax_loss.set_title('MLP — Learning Krive (Loss)',
                          color=P['text'], fontsize=10, fontweight='bold')
        ax_loss.legend(fontsize=9, facecolor=P['header'],
                       edgecolor=P['border'], labelcolor=P['text'])
        ax_loss.tick_params(colors=P['muted'])
        ax_loss.spines[:].set_edgecolor(P['border'])
        ax_loss.grid(True, color=P['border'], linewidth=0.4, alpha=0.6)

        ax_acc = fig.add_subplot(gs[2, 2:n_models + 1])
        ax_acc.set_facecolor(P['ax_bg'])
        ax_acc.plot(epochs, hist['accuracy'], color=P['green'], lw=1.5, label='Train Acc')
        ax_acc.plot(epochs, hist['val_accuracy'], color=P['orange'], lw=1.5,
                    linestyle='--', label='Val Acc')
        ax_acc.set_xlabel('Epoha', color=P['muted'], fontsize=9)
        ax_acc.set_ylabel('Accuracy', color=P['muted'], fontsize=9)
        ax_acc.set_title('MLP — Learning Krive (Accuracy)',
                         color=P['text'], fontsize=10, fontweight='bold')
        ax_acc.legend(fontsize=9, facecolor=P['header'],
                      edgecolor=P['border'], labelcolor=P['text'])
        ax_acc.tick_params(colors=P['muted'])
        ax_acc.spines[:].set_edgecolor(P['border'])
        ax_acc.grid(True, color=P['border'], linewidth=0.4, alpha=0.6)

    fig.suptitle(
        'Evaluacija Modela — Multimodelni sistem za detekciju pada (SisFall)\n'
        'Stefan Ilić SV12/2023 | Računarska inteligencija',
        fontsize=13, color=P['text'], fontweight='bold', y=0.99,
    )

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, '3_model_evaluation.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=P['bg'])
    plt.close()
    print(f"  ✓ Evaluation grafik sačuvan: {save_path}")


# ─────────────────────────────────────────────────────────────────
# ROC KRIVE
# ─────────────────────────────────────────────────────────────────

def plot_roc_curves(results_list, y_test, output_dir):
    """
    Prikazuje ROC krive sva tri modela na jednom grafiku.
    AUC (Area Under Curve): vrednost 1.0 = savršeni model, 0.5 = nasumično.

    Parametri:
        results_list : lista dict-ova iz models.py
        y_test       : np.ndarray — stvarne labele test seta
        output_dir   : putanja za čuvanje PNG-a
    """
    P = PALETTE
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(P['bg'])
    ax.set_facecolor(P['ax_bg'])

    line_colors = [P['blue'], P['green'], P['orange']]

    for res, color in zip(results_list, line_colors):
        if res.get('y_prob') is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{res['name']}  (AUC = {roc_auc:.4f})")

    # Dijagonala (nasumični classifier)
    ax.plot([0, 1], [0, 1], color=P['muted'], lw=1, linestyle='--', alpha=0.6,
            label='Nasumični (AUC = 0.5)')

    ax.set_xlabel('False Positive Rate (1 - Specificity)', color=P['muted'], fontsize=10)
    ax.set_ylabel('True Positive Rate (Recall)', color=P['muted'], fontsize=10)
    ax.set_title('ROC Krive — Sva 3 modela', color=P['text'], fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, facecolor=P['header'], edgecolor=P['border'], labelcolor=P['text'])
    ax.tick_params(colors=P['muted'])
    ax.spines[:].set_edgecolor(P['border'])
    ax.grid(True, color=P['border'], linewidth=0.4, alpha=0.6)
    ax.set_xlim([0, 1]);
    ax.set_ylim([0, 1.02])

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, '4_roc_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=P['bg'])
    plt.close()
    print(f"  ✓ ROC krive sačuvane: {save_path}")


# ─────────────────────────────────────────────────────────────────
# FINALNI TEKSTUALNI IZVEŠTAJ
# ─────────────────────────────────────────────────────────────────

def print_final_report(results_list, split_info=None, output_dir=None):
    """
    Rang-lista je sortirana po Recall-u (primarno), pa F1 (sekundarno).

    Parametri:
        results_list : lista dict-ova iz models.py
        split_info   : dict iz features.split_by_subjects() (opciono, za ispis podele)
        output_dir   : putanja za ispis liste fajlova (opciono)
    """
    print("\n" + "=" * 62)
    print("  FINALNI IZVEŠTAJ")
    print("=" * 62)

    if split_info:
        print(f"\n  Podela podataka: {split_info['method']}")
        print(f"  Test subjecti:   {split_info['test_subjects']}")

    ranked = sorted(results_list, key=lambda r: (r['recall'], r['f1']), reverse=True)
    best = ranked[0]

    header = f"  {'Model':<26} {'Acc':>7} {'Recall':>8} {'F1':>8} {'Inf(ms)':>9}"
    print(f"\n{header}")
    print("  " + "─" * 60)
    for res in ranked:
        marker = "  ← BEST" if res is best else ""
        print(f"  {res['name']:<26} {res['accuracy']:>7.4f} "
              f"{res['recall']:>8.4f} {res['f1']:>8.4f} "
              f"{res['inference_ms']:>9.4f}{marker}")
    print("  " + "─" * 60)

    print(f"\n  Preporučeni model: {best['name']}")
    rt_ok = "✓ real-time OK" if best['inference_ms'] < 50 else "⚠ sporo za real-time"
    print(f"     Recall      = {best['recall']:.4f}   (% stvarnih padova detektovano)")
    print(f"     Precision   = {best['precision']:.4f}   (% lažnih uzbuna)")
    print(f"     F1-Score    = {best['f1']:.4f}")
    print(f"     Threshold   = {best['threshold']}")
    print(f"     Inference   = {best['inference_ms']:.4f} ms/prozor  ({rt_ok})")

    print(f"\n  Napomena o thresholdu MLP-a ({best.get('threshold', 0.5)}):")
    print(f"    p(Fall) ≥ threshold → predviđeno je pad.")
    print(f"    Threshold 0.3 (umesto standardnih 0.5) povećava Recall")
    print(f"    jer je u medicinskom kontekstu propušteni pad (FN) opasniji")
    print(f"    od lažne uzbune (FP). Trade-off: niža Precision.")

    if output_dir and os.path.isdir(output_dir):
        print(f"\n  Sačuvani fajlovi ({output_dir}):")
        for fname in sorted(os.listdir(output_dir)):
            fpath = os.path.join(output_dir, fname)
            size = os.path.getsize(fpath) / 1024
            print(f"    {fname:<44} ({size:.1f} KB)")
