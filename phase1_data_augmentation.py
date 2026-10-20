# -*- coding: utf-8 -*-
"""
HerittageFNS: Phase 1 - Feature-Level Data Augmentation on Train Set
=====================================================================
- Preserves 80/20 Stratified train/test split (random_state=42)
- Test set (720 samples) is strictly untouched (ZERO data leakage)
- Train set (2,880 samples) is augmented via:
    1. SMOTE (Synthetic Minority Over-sampling Technique)
    2. Gaussian Jittering / Noise (preserving [0, 1] bounds and Ruspini partition)
    3. Combined SMOTE + Gaussian Noise
- Evaluates Decision Tree & Random Forest (50 trees, max_depth=5, min_samples_leaf=10)
- Saves benchmark comparison results and augmented datasets
"""

import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE

# Ensure output directories exist
output_dir = Path("./data/datasets/augmented")
output_dir.mkdir(parents=True, exist_ok=True)
plots_dir = Path("./data/plots")
plots_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Load Data
# -----------------------------------------------------------------------------
df_crisp = pd.read_csv('./data/datasets/features_fuzzy_crisp.csv')
df_cont  = pd.read_csv('./data/datasets/features_fuzzy_continuous.csv')

print(f"Loaded features_fuzzy_crisp.csv: {df_crisp.shape}")
print(f"Loaded features_fuzzy_continuous.csv: {df_cont.shape}")

crisp_feat_cols = ['Energy', 'Harmony', 'Pitch', 'Spectral', 'Tempo', 'Timbre', 'Vibrato']
ORDINAL_MAPS = {
    'Energy':   {'Rất yếu': 0, 'Yếu': 1, 'Vừa': 2, 'Mạnh': 3, 'Rất mạnh': 4,
                 'Very Weak': 0, 'Weak': 1, 'Moderate': 2, 'Strong': 3, 'Very Strong': 4},
    'Harmony':  {'Thấp': 0, 'Trung bình': 1, 'Cao': 2,
                 'Low': 0, 'Medium': 1, 'High': 2},
    'Pitch':    {'Rất trầm': 0, 'Trầm': 1, 'Trung': 2, 'Cao': 3, 'Rất cao': 4,
                 'Very Low': 0, 'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4},
    'Spectral': {'Thấp': 0, 'Trung bình': 1, 'Cao': 2,
                 'Low': 0, 'Medium': 1, 'High': 2},
    'Tempo':    {'Chậm': 0, 'Vừa': 1, 'Nhanh': 2,
                 'Slow': 0, 'Moderate': 1, 'Fast': 2},
    'Timbre':   {'Rất thấp': 0, 'Thấp': 1, 'Trung bình': 2, 'Cao': 3, 'Rất cao': 4,
                 'Very Low': 0, 'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4},
    'Vibrato':  {'Ít luyến': 0, 'Vừa': 1, 'Nhiều luyến': 2,
                 'Light Ornament': 0, 'Moderate': 1, 'Heavy Ornament': 2}
}

df_crisp_encoded = df_crisp.copy()
for col in crisp_feat_cols:
    df_crisp_encoded[col] = df_crisp[col].map(ORDINAL_MAPS[col])

meta_cols = ['sample_idx', 'label', 'audio_path', 'segment_id', 'segment_time']
cont_feat_cols = [c for c in df_cont.columns if c not in meta_cols]

# Map each continuous feature to its parent axis
axis_groups = {
    'Energy':   [c for c in cont_feat_cols if c.startswith('Energy_')],
    'Harmony':  [c for c in cont_feat_cols if c.startswith('Harmony_')],
    'Pitch':    [c for c in cont_feat_cols if c.startswith('Pitch_')],
    'Spectral': [c for c in cont_feat_cols if c.startswith('Spectral_')],
    'Tempo':    [c for c in cont_feat_cols if c.startswith('Tempo_')],
    'Timbre':   [c for c in cont_feat_cols if c.startswith('Timbre_')],
    'Vibrato':  [c for c in cont_feat_cols if c.startswith('Vibrato_')],
}

X_cont = df_cont[cont_feat_cols].values
y_labels = df_cont['label'].values
class_names = sorted(list(set(y_labels)))
label_to_id = {cls: idx for idx, cls in enumerate(class_names)}
id_to_label = {idx: cls for idx, cls in enumerate(class_names)}
y_ids = np.array([label_to_id[l] for l in y_labels])

# -----------------------------------------------------------------------------
# 2. Stratified 80/20 Train / Test Split (random_state=42)
# -----------------------------------------------------------------------------
indices = np.arange(len(y_labels))
train_idx, test_idx = train_test_split(
    indices, test_size=0.20, random_state=42, stratify=y_labels
)

X_train_orig = X_cont[train_idx]
y_train_orig = y_labels[train_idx]
y_train_orig_ids = y_ids[train_idx]

X_test = X_cont[test_idx]
y_test = y_labels[test_idx]
y_test_ids = y_ids[test_idx]

# Crisp datasets for the exact same split
X_crisp = df_crisp_encoded[crisp_feat_cols].values
X_c_train_orig = X_crisp[train_idx]
X_c_test = X_crisp[test_idx]

print(f"\nTrain set: {len(train_idx)} samples (480/class)")
print(f"Test set:  {len(test_idx)} samples (120/class) [UNTOUCHED]")

# Helper: Convert continuous memberships back to crisp ordinal encoded features
def cont_to_crisp(X_matrix, cont_cols=cont_feat_cols, axes=axis_groups):
    col_idx_map = {col: i for i, col in enumerate(cont_cols)}
    crisp_matrix = np.zeros((X_matrix.shape[0], len(crisp_feat_cols)), dtype=int)
    for ax_idx, ax_name in enumerate(crisp_feat_cols):
        cols_for_ax = axes[ax_name]
        indices_for_ax = [col_idx_map[c] for c in cols_for_ax]
        sub_matrix = X_matrix[:, indices_for_ax]
        crisp_matrix[:, ax_idx] = np.argmax(sub_matrix, axis=1)
    return crisp_matrix

# Helper: Normalize continuous memberships to preserve Ruspini partition of unity (sum to 1.0)
def normalize_ruspini(X_matrix, cont_cols=cont_feat_cols, axes=axis_groups):
    col_idx_map = {col: i for i, col in enumerate(cont_cols)}
    X_norm = np.clip(X_matrix, 0.0, 1.0).copy()
    for ax_name, cols_for_ax in axes.items():
        indices_for_ax = [col_idx_map[c] for c in cols_for_ax]
        row_sums = np.sum(X_norm[:, indices_for_ax], axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        X_norm[:, indices_for_ax] = X_norm[:, indices_for_ax] / row_sums
    return np.round(X_norm, 4)

# -----------------------------------------------------------------------------
# 3. Model Training & Evaluation Function
# -----------------------------------------------------------------------------
def evaluate_models(X_tr_cont, y_tr, X_te_cont, y_te, X_tr_crisp, X_te_crisp, desc=""):
    # 1. DT Crisp
    dt_c = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
    dt_c.fit(X_tr_crisp, y_tr)
    p_dt_c = dt_c.predict(X_te_crisp)
    acc_dt_c = accuracy_score(y_te, p_dt_c)
    f1_dt_c  = f1_score(y_te, p_dt_c, average='macro')
    
    # 2. RF Crisp
    rf_c = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_leaf=10, random_state=42)
    rf_c.fit(X_tr_crisp, y_tr)
    p_rf_c = rf_c.predict(X_te_crisp)
    acc_rf_c = accuracy_score(y_te, p_rf_c)
    f1_rf_c  = f1_score(y_te, p_rf_c, average='macro')
    
    # 3. DT Continuous
    dt_cnt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
    dt_cnt.fit(X_tr_cont, y_tr)
    p_dt_cnt = dt_cnt.predict(X_te_cont)
    acc_dt_cnt = accuracy_score(y_te, p_dt_cnt)
    f1_dt_cnt  = f1_score(y_te, p_dt_cnt, average='macro')
    
    # 4. RF Continuous
    rf_cnt = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_leaf=10, random_state=42)
    rf_cnt.fit(X_tr_cont, y_tr)
    p_rf_cnt = rf_cnt.predict(X_te_cont)
    acc_rf_cnt = accuracy_score(y_te, p_rf_cnt)
    f1_rf_cnt  = f1_score(y_te, p_rf_cnt, average='macro')
    
    res = {
        'Description': desc,
        'Train_Samples': len(y_tr),
        'DT_Crisp_Acc': acc_dt_c * 100,
        'DT_Crisp_F1': f1_dt_c * 100,
        'RF_Crisp_Acc': acc_rf_c * 100,
        'RF_Crisp_F1': f1_rf_c * 100,
        'DT_Cont_Acc': acc_dt_cnt * 100,
        'DT_Cont_F1': f1_dt_cnt * 100,
        'RF_Cont_Acc': acc_rf_cnt * 100,
        'RF_Cont_F1': f1_rf_cnt * 100,
    }
    return res, (rf_cnt, p_rf_cnt)

results = []

# --- Experiment 1: Baseline (No Augmentation) ---
print("\n" + "="*70)
print("1. Running Baseline (Original 2,880 Train samples)...")
res_baseline, (best_rf, best_pred) = evaluate_models(
    X_train_orig, y_train_orig, X_test, y_test,
    X_c_train_orig, X_c_test, desc="Baseline (No Augmentation)"
)
results.append(res_baseline)

# --- Experiment 2: Gaussian Jittering / Noise (preserving Ruspini partition) ---
# Add mild noise N(0, 0.02 * std) to each feature, double train size (480 -> 960 per class)
print("\n" + "="*70)
print("2. Running Gaussian Jittering (+100%, 5,760 Train samples)...")
np.random.seed(42)
feat_stds = np.std(X_train_orig, axis=0, keepdims=True)
noise = np.random.normal(0, 0.05 * (feat_stds + 1e-4), size=X_train_orig.shape)
X_train_jitter = normalize_ruspini(X_train_orig + noise)
X_train_gauss = np.vstack([X_train_orig, X_train_jitter])
y_train_gauss = np.hstack([y_train_orig, y_train_orig])
X_c_train_gauss = cont_to_crisp(X_train_gauss)

res_gauss, _ = evaluate_models(
    X_train_gauss, y_train_gauss, X_test, y_test,
    X_c_train_gauss, X_c_test, desc="Gaussian Jittering (+100%)"
)
results.append(res_gauss)

# --- Experiment 3: SMOTE (+50% to 720/class = 4,320 Train samples) ---
print("\n" + "="*70)
print("3. Running SMOTE (+50%, 4,320 Train samples)...")
target_dict_50 = {cls: 720 for cls in class_names}
smote_50 = SMOTE(sampling_strategy=target_dict_50, k_neighbors=5, random_state=42)
X_train_smote50, y_train_smote50 = smote_50.fit_resample(X_train_orig, y_train_orig)
X_train_smote50 = normalize_ruspini(X_train_smote50)
X_c_train_smote50 = cont_to_crisp(X_train_smote50)

res_smote50, _ = evaluate_models(
    X_train_smote50, y_train_smote50, X_test, y_test,
    X_c_train_smote50, X_c_test, desc="SMOTE (+50% to 720/class)"
)
results.append(res_smote50)

# --- Experiment 4: SMOTE (+100% to 960/class = 5,760 Train samples) ---
print("\n" + "="*70)
print("4. Running SMOTE (+100%, 5,760 Train samples)...")
target_dict_100 = {cls: 960 for cls in class_names}
smote_100 = SMOTE(sampling_strategy=target_dict_100, k_neighbors=5, random_state=42)
X_train_smote100, y_train_smote100 = smote_100.fit_resample(X_train_orig, y_train_orig)
X_train_smote100 = normalize_ruspini(X_train_smote100)
X_c_train_smote100 = cont_to_crisp(X_train_smote100)

res_smote100, _ = evaluate_models(
    X_train_smote100, y_train_smote100, X_test, y_test,
    X_c_train_smote100, X_c_test, desc="SMOTE (+100% to 960/class)"
)
results.append(res_smote100)

# --- Experiment 5: Combined SMOTE (+50%) + Gaussian Jittering (+50%) = 5,760 Train samples ---
print("\n" + "="*70)
print("5. Running Combined SMOTE + Gaussian Noise (5,760 Train samples)...")
noise_smote = np.random.normal(0, 0.03 * (feat_stds + 1e-4), size=X_train_smote50.shape)
X_train_comb = normalize_ruspini(X_train_smote50 + noise_smote)
X_train_comb = np.vstack([X_train_orig, X_train_comb[len(X_train_orig):], X_train_jitter])
y_train_comb = np.hstack([y_train_orig, y_train_smote50[len(X_train_orig):], y_train_orig])
X_c_train_comb = cont_to_crisp(X_train_comb)

res_comb, (best_rf_aug, best_pred_aug) = evaluate_models(
    X_train_comb, y_train_comb, X_test, y_test,
    X_c_train_comb, X_c_test, desc="Combined SMOTE + Gaussian Noise"
)
results.append(res_comb)

# -----------------------------------------------------------------------------
# 4. Display & Save Results
# -----------------------------------------------------------------------------
df_res = pd.DataFrame(results)
print("\n" + "="*95)
print("BẢNG TỔNG HỢP SO SÁNH HIỆU NĂNG TĂNG CƯỜNG DỮ LIỆU TRÊN TẬP HUẤN LUYỆN (TEST SET CỐ ĐỊNH 720 MẪU)")
print("="*95)
print(df_res.to_string(index=False))
print("="*95)

# Save comparison table
csv_out = output_dir / "augmentation_benchmark_results.csv"
df_res.to_csv(csv_out, index=False)
print(f"\n-> Saved benchmark table to: {csv_out}")

# Save the best augmented training sets for subsequent use
df_train_aug_cont = pd.DataFrame(X_train_smote100, columns=cont_feat_cols)
df_train_aug_cont['label'] = y_train_smote100
df_train_aug_cont.to_csv(output_dir / "train_augmented_fuzzy_continuous.csv", index=False)

df_train_aug_crisp = pd.DataFrame(X_c_train_smote100, columns=crisp_feat_cols)
df_train_aug_crisp['label'] = y_train_smote100
df_train_aug_crisp.to_csv(output_dir / "train_augmented_fuzzy_crisp.csv", index=False)

# Save the isolated test set for consistent evaluation across all experiments
df_test_cont = pd.DataFrame(X_test, columns=cont_feat_cols)
df_test_cont['label'] = y_test
df_test_cont.to_csv(output_dir / "test_isolated_fuzzy_continuous.csv", index=False)

df_test_crisp = pd.DataFrame(X_c_test, columns=crisp_feat_cols)
df_test_crisp['label'] = y_test
df_test_crisp.to_csv(output_dir / "test_isolated_fuzzy_crisp.csv", index=False)

print("-> Saved augmented train sets and isolated test sets to: ./data/datasets/augmented/")
print("\nHoàn tất Giai đoạn 1 (Data Augmentation)!")
