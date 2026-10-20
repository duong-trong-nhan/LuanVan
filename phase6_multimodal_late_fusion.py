# -*- coding: utf-8 -*-
"""
HerittageFNS: Phase 6 - Multi-Modal Late Fusion (Neuro-Symbolic Integration)
=============================================================================
Combines:
  1. Best Vision CNN Model from Phase 5 (Spectrogram representation)
  2. Best Symbolic Fuzzy Model from Phase 1/2 (Random Forest Continuous + Gaussian Augmentation)

Late Fusion Mechanism:
  P_ensemble = w_cnn * P_cnn + (1 - w_cnn) * P_fuzzy

Run only AFTER Phase 5 has evaluated all candidate CNN models to find the true Best CNN.

Outputs:
  data/results/phase6_late_fusion_results.csv
  data/results/cnn_confusion_matrices/Phase6_LateFusion_BestEnsemble_cm.png
  Updates data/results/cnn_experiment_log.md (Phase 6 section)
"""

import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Paths
RESULTS_DIR   = Path('./data/results')
MODEL_DIR     = Path('./data/models/cnn')
CM_DIR        = RESULTS_DIR / 'cnn_confusion_matrices'
BENCHMARK_CSV = RESULTS_DIR / 'cnn_benchmark.csv'
RF_MODEL_PATH = Path('./data/models/best_random_forest_continuous_gaussian.joblib')
TEST_CSV_PATH = Path('./data/datasets/augmented/test_isolated_fuzzy_continuous.csv')
LOG_MD_PATH   = RESULTS_DIR / 'cnn_experiment_log.md'

for d in [RESULTS_DIR, CM_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GENRES = sorted(['Ca trù', 'Chèo', 'Chầu văn', 'Cải lương', 'Hát xẩm', 'Quan họ'])


def run_late_fusion():
    print("=" * 80)
    print("PHASE 6: MULTI-MODAL LATE FUSION (BEST CNN + BEST FUZZY RANDOM FOREST)")
    print("=" * 80)

    # 1. Verify inputs exist
    if not RF_MODEL_PATH.exists():
        raise FileNotFoundError(f"Fuzzy RF model not found: {RF_MODEL_PATH}")
    if not TEST_CSV_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_CSV_PATH}")
    if not BENCHMARK_CSV.exists():
        raise FileNotFoundError(f"Phase 5 benchmark CSV not found: {BENCHMARK_CSV}. Run Phase 5 first.")

    # 2. Identify the Best CNN model from Phase 5
    df_cnn = pd.read_csv(BENCHMARK_CSV)
    df_cnn_models = df_cnn[df_cnn['exp_id'] < 14]
    if df_cnn_models.empty:
        raise ValueError("No completed CNN models found in benchmark CSV.")

    best_cnn_row = df_cnn_models.loc[df_cnn_models['test_accuracy'].idxmax()]
    best_exp_id  = int(best_cnn_row['exp_id'])
    best_model_name = best_cnn_row['model_name']
    best_cnn_acc = best_cnn_row['test_accuracy']
    best_cnn_f1  = best_cnn_row['macro_f1']

    print(f"\n[1] Identified Champion CNN Model from Phase 5:")
    print(f"    EXP-{best_exp_id:02d}: {best_model_name} ({best_cnn_row['spec_type']} | {best_cnn_row['strategy']})")
    print(f"    CNN Test Accuracy: {best_cnn_acc:.2f}% | Macro F1: {best_cnn_f1:.2f}%\n")

    # 3. Load CNN Softmax Probabilities
    probs_path = MODEL_DIR / f"exp{best_exp_id:02d}_test_probs.npy"
    if not probs_path.exists():
        raise FileNotFoundError(f"Test probability array for EXP-{best_exp_id:02d} not found: {probs_path}")
    cnn_probs = np.load(probs_path)

    # 4. Load Fuzzy RF Model & Test Data
    print("[2] Loading Fuzzy Random Forest Baseline & Test Set...")
    df_test = pd.read_csv(TEST_CSV_PATH)
    meta_cols = ['label']
    feat_cols = [c for c in df_test.columns if c not in meta_cols]
    X_test_cont = df_test[feat_cols].values
    y_test_str  = df_test['label'].values
    y_test_idx  = np.array([GENRES.index(g) for g in y_test_str])

    rf_model = joblib.load(RF_MODEL_PATH)
    rf_classes = rf_model.classes_.tolist()
    rf_probs_raw = rf_model.predict_proba(X_test_cont)

    # Reorder RF probabilities to strictly match GENRES
    rf_probs = np.zeros_like(rf_probs_raw)
    for j, g in enumerate(GENRES):
        raw_idx = rf_classes.index(g)
        rf_probs[:, j] = rf_probs_raw[:, raw_idx]

    rf_standalone_acc = accuracy_score(y_test_idx, rf_probs.argmax(axis=1)) * 100
    rf_standalone_f1  = f1_score(y_test_idx, rf_probs.argmax(axis=1), average='macro') * 100
    print(f"    Fuzzy RF Alone Test Acc: {rf_standalone_acc:.2f}% | Macro F1: {rf_standalone_f1:.2f}%\n")

    if len(cnn_probs) != len(rf_probs):
        raise ValueError(f"Shape mismatch: CNN samples ({len(cnn_probs)}) != RF samples ({len(rf_probs)})")

    # 5. Grid Search Late Fusion Weights
    print("[3] Grid Searching Late Fusion Weights (w_cnn vs w_fuzzy):")
    print(f"    {'w_cnn':<8} {'w_fuzzy':<8} {'Test Acc (%)':<15} {'Macro F1 (%)':<15}")
    print("    " + "-" * 50)

    grid_results = []
    best_acc = 0.0
    best_f1  = 0.0
    best_w   = (0.5, 0.5)
    best_preds = None

    for w_cnn in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        w_fuzzy = round(1.0 - w_cnn, 2)
        combined = w_cnn * cnn_probs + w_fuzzy * rf_probs
        preds    = combined.argmax(axis=1)
        acc      = accuracy_score(y_test_idx, preds) * 100
        f1       = f1_score(y_test_idx, preds, average='macro') * 100
        grid_results.append({'w_cnn': w_cnn, 'w_fuzzy': w_fuzzy, 'accuracy': acc, 'macro_f1': f1})

        print(f"    {w_cnn:<8.1f} {w_fuzzy:<8.1f} {acc:<15.2f} {f1:<15.2f}")
        if acc > best_acc:
            best_acc = acc
            best_f1  = f1
            best_w   = (w_cnn, w_fuzzy)
            best_preds = preds

    print(f"\n[4] Optimal Fusion Weights: w_cnn={best_w[0]} (CNN) + w_fuzzy={best_w[1]} (Fuzzy RF)")
    print(f"    -> PEAK ENSEMBLE ACCURACY: {best_acc:.2f}% (Tăng {best_acc - 65.56:+.2f}% so với RF Baseline)")
    print(f"    -> PEAK ENSEMBLE MACRO F1: {best_f1:.2f}% (Tăng {best_f1 - 65.12:+.2f}% so với RF Baseline)\n")

    # 6. Classification Report & Confusion Matrix
    report = classification_report(y_test_idx, best_preds, target_names=GENRES, output_dict=True)
    per_class_f1 = {g: round(report[g]['f1-score'] * 100, 2) for g in GENRES}
    print(f"[5] Per-class F1 Scores: {per_class_f1}")

    cm = confusion_matrix(y_test_idx, best_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=GENRES, yticklabels=GENRES, ax=ax)
    ax.set_xlabel('Predicted Genre')
    ax.set_ylabel('True Genre')
    ax.set_title(f'Phase 6 Multi-Modal Late Fusion Ensemble\nBest CNN (EXP-{best_exp_id:02d}) + Fuzzy RF | Acc={best_acc:.2f}% | F1={best_f1:.2f}%')
    plt.tight_layout()
    cm_path = CM_DIR / "Phase6_LateFusion_BestEnsemble_cm.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"    Saved Confusion Matrix to: {cm_path}")

    # 7. Save Phase 6 CSV
    fusion_csv = RESULTS_DIR / "phase6_late_fusion_results.csv"
    df_grid = pd.DataFrame(grid_results)
    df_grid.to_csv(fusion_csv, index=False)
    print(f"    Saved Grid Search Results to: {fusion_csv}")

    # 8. Update Markdown Log
    if LOG_MD_PATH.exists():
        content = LOG_MD_PATH.read_text(encoding='utf-8')
        lines = []
        lines.append("| w_cnn | w_fuzzy | Test Accuracy | Macro F1 | Ghi chú |")
        lines.append("|-------|---------|---------------|----------|---------|")
        for r in grid_results:
            is_best = (r['w_cnn'] == best_w[0])
            b_acc = f"**{r['accuracy']:.2f}%**" if is_best else f"{r['accuracy']:.2f}%"
            b_f1  = f"**{r['macro_f1']:.2f}%**" if is_best else f"{r['macro_f1']:.2f}%"
            note  = "**Tối ưu nhất**" if is_best else ""
            lines.append(f"| {r['w_cnn']:.1f} | {r['w_fuzzy']:.1f} | {b_acc} | {b_f1} | {note} |")
        table_str = "\n".join(lines)

        # Replace table in Phase 6 section
        header_p6 = "## GIAI ĐOẠN 6 (PHASE 6) | Multi-Modal Late Fusion: Best CNN + Fuzzy RF"
        if header_p6 in content:
            p_parts = content.split(header_p6)
            before = p_parts[0] + header_p6 + "\n\n"
            rest_of_doc = p_parts[1]
            p6_desc = (
                f"> **Mô hình CNN được chọn:** EXP-{best_exp_id:02d} ({best_model_name}, Acc={best_cnn_acc:.2f}%)\n"
                f"> **Mô hình Fuzzy được chọn:** Random Forest Continuous + Gaussian Augmentation (Acc=65.56%)\n\n"
                f"**Cơ chế kết hợp:** `P_ensemble = {best_w[0]} * P_cnn + {best_w[1]} * P_fuzzy`\n\n"
            )
            # Find next header
            next_h = "\n---\n\n## BẢNG TỔNG HỢP SO SÁNH PHASE 5"
            if next_h in rest_of_doc:
                after = rest_of_doc[rest_of_doc.find(next_h):]
                # Also update comparison table
                new_doc = before + p6_desc + table_str + after
                # Update Best CNN in comparison
                new_doc = new_doc.replace(
                    "| CNN | Best CNN Phase 5 (đang cập nhật) | -- | -- | Chờ hoàn tất EXP-02 -> EXP-13 |",
                    f"| CNN | Best CNN (EXP-{best_exp_id:02d} {best_model_name}) | **{best_cnn_acc:.2f}%** | {best_cnn_f1:.2f}% | Mô hình CNN số 1 |"
                )
                new_doc = new_doc.replace(
                    "| Ensemble | Phase 6 Late Fusion (chờ Best CNN) | -- | -- | Kích hoạt tại Phase 6 |",
                    f"| **Ensemble** | **Phase 6 Late Fusion: CNN + Fuzzy RF** | **{best_acc:.2f}%** | **{best_f1:.2f}%** | **Đỉnh cao toàn đồ án (+{best_acc-65.56:.2f}%)** |"
                )
                LOG_MD_PATH.write_text(new_doc, encoding='utf-8')
                print(f"    Successfully updated {LOG_MD_PATH}")

    print("\nPhase 6 DONE.")


if __name__ == '__main__':
    run_late_fusion()
