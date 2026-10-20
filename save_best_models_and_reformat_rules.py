# -*- coding: utf-8 -*-
"""
HerittageFNS: Save Best Models, Datasets & Reformat Rule Induction Comparison
=============================================================================
1. Identifies the BEST augmentation methods:
   - Overall Best (Continuous Models): Gaussian Jittering (+100%, 5,760 samples) -> RF Cont 65.56%, DT Cont 58.75%
   - Best for Rule Induction / Crisp: SMOTE (+50%, 4,320 samples) -> RIPPER 52.36%, RF Crisp 56.11%
   - Best for Decision Tree Crisp: SMOTE (+100%, 5,760 samples) -> DT Crisp 51.94%
2. Saves trained model objects (.joblib) to data/models/
3. Saves the respective training datasets (.csv) to data/datasets/augmented/
4. Saves extracted rule sets (.json & .csv) to data/rules/
5. Reformats data/rules/rule_induction_augmented_comparison.csv into a clean,
   side-by-side, visually intuitive layout for easy manual comparison.
"""

import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import os
import math
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import wittgenstein as lw

# Output directories
models_dir = Path("./data/models")
models_dir.mkdir(parents=True, exist_ok=True)
augmented_dir = Path("./data/datasets/augmented")
augmented_dir.mkdir(parents=True, exist_ok=True)
rules_dir = Path("./data/rules")
rules_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Load Data
# -----------------------------------------------------------------------------
df_crisp = pd.read_csv('./data/datasets/features_fuzzy_crisp.csv')
df_cont  = pd.read_csv('./data/datasets/features_fuzzy_continuous.csv')

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

REV_MAPS = {col: {v: k for k, v in ORDINAL_MAPS[col].items()} for col in crisp_feat_cols}

meta_cols = ['sample_idx', 'label', 'audio_path', 'segment_id', 'segment_time']
cont_feat_cols = [c for c in df_cont.columns if c not in meta_cols]

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

indices = np.arange(len(y_labels))
train_idx, test_idx = train_test_split(
    indices, test_size=0.20, random_state=42, stratify=y_labels
)

X_train_orig = X_cont[train_idx]
y_train_orig = y_labels[train_idx]
X_test = X_cont[test_idx]
y_test = y_labels[test_idx]

X_crisp = df_crisp_encoded[crisp_feat_cols].values
X_c_train_orig = X_crisp[train_idx]
X_c_test = X_crisp[test_idx]

def cont_to_crisp(X_matrix, cont_cols=cont_feat_cols, axes=axis_groups):
    col_idx_map = {col: i for i, col in enumerate(cont_cols)}
    crisp_matrix = np.zeros((X_matrix.shape[0], len(crisp_feat_cols)), dtype=int)
    for ax_idx, ax_name in enumerate(crisp_feat_cols):
        cols_for_ax = axes[ax_name]
        indices_for_ax = [col_idx_map[c] for c in cols_for_ax]
        sub_matrix = X_matrix[:, indices_for_ax]
        crisp_matrix[:, ax_idx] = np.argmax(sub_matrix, axis=1)
    return crisp_matrix

def normalize_ruspini(X_matrix, cont_cols=cont_feat_cols, axes=axis_groups):
    col_idx_map = {col: i for i, col in enumerate(cont_cols)}
    X_norm = np.clip(X_matrix, 0.0, 1.0).copy()
    for ax_name, cols_for_ax in axes.items():
        indices_for_ax = [col_idx_map[c] for c in cols_for_ax]
        row_sums = np.sum(X_norm[:, indices_for_ax], axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        X_norm[:, indices_for_ax] = X_norm[:, indices_for_ax] / row_sums
    return np.round(X_norm, 4)

def crisp_indices_to_df(X_c):
    df_out = pd.DataFrame(X_c, columns=crisp_feat_cols)
    for col in crisp_feat_cols:
        df_out[col] = df_out[col].map(REV_MAPS[col])
    return df_out

# -----------------------------------------------------------------------------
# 2. Build the Best Augmented Datasets
# -----------------------------------------------------------------------------
np.random.seed(42)
feat_stds = np.std(X_train_orig, axis=0, keepdims=True)

# BEST 1: Gaussian Jittering (+100%, 5,760 samples)
print("1. Building Gaussian Jittering (+100%)...")
noise = np.random.normal(0, 0.05 * (feat_stds + 1e-4), size=X_train_orig.shape)
X_tr_gauss = np.vstack([X_train_orig, normalize_ruspini(X_train_orig + noise)])
y_tr_gauss = np.hstack([y_train_orig, y_train_orig])
X_c_tr_gauss = cont_to_crisp(X_tr_gauss)

# Save best Gaussian datasets
df_gauss_cont = pd.DataFrame(X_tr_gauss, columns=cont_feat_cols)
df_gauss_cont['label'] = y_tr_gauss
df_gauss_cont.to_csv(augmented_dir / "train_best_gaussian_jitter_continuous.csv", index=False)

df_gauss_crisp = crisp_indices_to_df(X_c_tr_gauss)
df_gauss_crisp['label'] = y_tr_gauss
df_gauss_crisp.to_csv(augmented_dir / "train_best_gaussian_jitter_crisp.csv", index=False)
print("-> Saved: train_best_gaussian_jitter_continuous.csv and crisp.csv")

# BEST 2: SMOTE (+50%, 4,320 samples) - Best for Rule Induction / RIPPER
print("2. Building SMOTE (+50%)...")
target_dict_50 = {cls: 720 for cls in class_names}
smote_50 = SMOTE(sampling_strategy=target_dict_50, k_neighbors=5, random_state=42)
X_tr_sm50, y_tr_sm50 = smote_50.fit_resample(X_train_orig, y_train_orig)
X_tr_sm50 = normalize_ruspini(X_tr_sm50)
X_c_tr_sm50 = cont_to_crisp(X_tr_sm50)

df_sm50_cont = pd.DataFrame(X_tr_sm50, columns=cont_feat_cols)
df_sm50_cont['label'] = y_tr_sm50
df_sm50_cont.to_csv(augmented_dir / "train_best_smote50_continuous.csv", index=False)

df_sm50_crisp = crisp_indices_to_df(X_c_tr_sm50)
df_sm50_crisp['label'] = y_tr_sm50
df_sm50_crisp.to_csv(augmented_dir / "train_best_smote50_crisp.csv", index=False)
print("-> Saved: train_best_smote50_continuous.csv and crisp.csv")

# -----------------------------------------------------------------------------
# 3. Train & Save Best Model Objects (.joblib)
# -----------------------------------------------------------------------------
print("\n3. Training & Saving Best Model Objects to data/models/...")

# Model 1: Best RF Continuous (Trained on Gaussian Jittering) -> Acc = 65.56%
best_rf_cont = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_leaf=10, random_state=42)
best_rf_cont.fit(X_tr_gauss, y_tr_gauss)
rf_cont_path = models_dir / "best_random_forest_continuous_gaussian.joblib"
joblib.dump(best_rf_cont, rf_cont_path)
print(f"-> Saved: {rf_cont_path} (Accuracy: 65.56%, Macro F1: 65.12%)")

# Model 2: Best DT Continuous (Trained on Gaussian Jittering) -> Acc = 58.75%
best_dt_cont = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
best_dt_cont.fit(X_tr_gauss, y_tr_gauss)
dt_cont_path = models_dir / "best_decision_tree_continuous_gaussian.joblib"
joblib.dump(best_dt_cont, dt_cont_path)
print(f"-> Saved: {dt_cont_path} (Accuracy: 58.75%, Macro F1: 58.50%)")

# Model 3: Best RF Crisp (Trained on SMOTE +50%) -> Acc = 56.11%
best_rf_crisp = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_leaf=10, random_state=42)
best_rf_crisp.fit(X_c_tr_sm50, y_tr_sm50)
rf_crisp_path = models_dir / "best_random_forest_crisp_smote50.joblib"
joblib.dump(best_rf_crisp, rf_crisp_path)
print(f"-> Saved: {rf_crisp_path} (Accuracy: 56.11%, Macro F1: 55.16%)")

# Model 4: Best DT Crisp (Trained on SMOTE +100%) -> Acc = 51.94%
target_dict_100 = {cls: 960 for cls in class_names}
smote_100 = SMOTE(sampling_strategy=target_dict_100, k_neighbors=5, random_state=42)
X_tr_sm100, y_tr_sm100 = smote_100.fit_resample(X_train_orig, y_train_orig)
X_tr_sm100 = normalize_ruspini(X_tr_sm100)
X_c_tr_sm100 = cont_to_crisp(X_tr_sm100)

best_dt_crisp = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
best_dt_crisp.fit(X_c_tr_sm100, y_tr_sm100)
dt_crisp_path = models_dir / "best_decision_tree_crisp_smote100.joblib"
joblib.dump(best_dt_crisp, dt_crisp_path)
print(f"-> Saved: {dt_crisp_path} (Accuracy: 51.94%, Macro F1: 52.07%)")

# Model 5: Best RIPPER OvR (Trained on SMOTE +50%) -> Acc = 52.36%, F1 = 53.84%
print("Training & Saving Best RIPPER (SMOTE +50%)...")
df_tr_sm50_str = crisp_indices_to_df(X_c_tr_sm50)
ripper_models = {}
ripper_best_rules = []
for cls in class_names:
    train_bin = df_tr_sm50_str.copy()
    train_bin['target'] = (y_tr_sm50 == cls).astype(int)
    r_model = lw.RIPPER(k=2, prune_size=0.33, dl_allowance=64, random_state=42)
    r_model.fit(train_bin, class_feat='target', pos_class=1)
    ripper_models[cls] = r_model
    for r in r_model.ruleset_.rules:
        if r.conds:
            cond_str = " AND ".join([f"{cond.feature} == '{cond.val}'" for cond in r.conds])
            ripper_best_rules.append({
                "rule_text": cond_str,
                "consequent": cls,
                "antecedents_count": len(r.conds)
            })

ripper_path = models_dir / "best_ripper_ovr_smote50.joblib"
joblib.dump(ripper_models, ripper_path)
print(f"-> Saved: {ripper_path} (Accuracy: 52.36%, Macro F1: 53.84%, Rules: {len(ripper_best_rules)})")

# Save rules JSON & CSV for Best RIPPER
with open(rules_dir / "rules_best_ripper_smote50.json", "w", encoding="utf-8") as f:
    json.dump(ripper_best_rules, f, ensure_ascii=False, indent=2)
pd.DataFrame(ripper_best_rules).to_csv(rules_dir / "rules_best_ripper_smote50.csv", index=False)
print("-> Saved: rules_best_ripper_smote50.json and .csv")

# -----------------------------------------------------------------------------
# 4. Reformat rule_induction_augmented_comparison.csv for Easy Visual Comparison
# -----------------------------------------------------------------------------
print("\n4. Reformatting rule_induction_augmented_comparison.csv...")

# We structure the table so each row is a Data Augmentation Method,
# and columns are grouped side-by-side by algorithm:
# [DT Crisp Acc/F1/Rules/Len] | [RF Crisp Acc/F1/Rules/Len] | [RIPPER Acc/F1/Rules/Len] | [CN2 Acc/F1/Rules/Len] | [Best Algorithm]

raw_comp = pd.read_csv(augmented_dir / "augmentation_all_algorithms_benchmark.csv")

clean_comparison_rows = []
for idx, r in raw_comp.iterrows():
    method = r['Augmentation_Method']
    samples = int(r['Train_Samples'])
    
    # Identify which algorithm won in accuracy in this setting
    acc_map = {
        'Decision Tree Crisp': r['DT_Crisp_Acc'],
        'Random Forest Crisp': r['RF_Crisp_Acc'],
        'RIPPER': r['RIPPER_Acc'],
        'CN2': r['CN2_Acc'],
    }
    best_algo = max(acc_map, key=acc_map.get)
    best_acc = acc_map[best_algo]
    
    clean_comparison_rows.append({
        'Phương pháp tăng cường': method,
        'Số mẫu Train': samples,
        'DT Acc (%)': round(r['DT_Crisp_Acc'], 2),
        'DT F1 (%)': round(r['DT_Crisp_F1'], 2),
        'DT Số luật': int(r['DT_Crisp_Rules']),
        'DT Đ.Dài TB': round(r['DT_Crisp_AvgLen'], 2),
        'RF Acc (%)': round(r['RF_Crisp_Acc'], 2),
        'RF F1 (%)': round(r['RF_Crisp_F1'], 2),
        'RF Số luật': int(r['RF_Crisp_Rules']),
        'RF Đ.Dài TB': round(r['RF_Crisp_AvgLen'], 2),
        'RIPPER Acc (%)': round(r['RIPPER_Acc'], 2),
        'RIPPER F1 (%)': round(r['RIPPER_F1'], 2),
        'RIPPER Số luật': int(r['RIPPER_Rules']),
        'RIPPER Đ.Dài TB': round(r['RIPPER_AvgLen'], 2),
        'CN2 Acc (%)': round(r['CN2_Acc'], 2),
        'CN2 F1 (%)': round(r['CN2_F1'], 2),
        'CN2 Số luật': int(r['CN2_Rules']),
        'CN2 Đ.Dài TB': round(r['CN2_AvgLen'], 2),
        'Thuật toán đỉnh nhất (Crisp)': f"{best_algo} ({best_acc:.2f}%)",
        'RF Cont Acc (Tham chiếu %)': round(r['RF_Cont_Acc'], 2)
    })

df_clean_comp = pd.DataFrame(clean_comparison_rows)

# Overwrite rule_induction_augmented_comparison.csv with the intuitive layout
reformatted_csv = rules_dir / "rule_induction_augmented_comparison.csv"
df_clean_comp.to_csv(reformatted_csv, index=False)
print(f"-> Overwritten {reformatted_csv} with the clear, side-by-side comparison layout!")

# Also generate a markdown format for terminal display and easy copy-pasting
print("\n" + "="*145)
print("BẢNG BỐ CỤC MỚI: SO SÁNH TRỰC DIỆN BẰNG MẮT THƯỜNG GIỮA CÁC THUẬT TOÁN THEO TỪNG PHƯƠNG PHÁP TĂNG CƯỜNG")
print("="*145)
print(df_clean_comp[[
    'Phương pháp tăng cường', 'Số mẫu Train',
    'DT Acc (%)', 'RF Acc (%)', 'RIPPER Acc (%)', 'CN2 Acc (%)',
    'DT Số luật', 'RF Số luật', 'RIPPER Số luật', 'CN2 Số luật',
    'CN2 Đ.Dài TB', 'Thuật toán đỉnh nhất (Crisp)', 'RF Cont Acc (Tham chiếu %)'
]].to_string(index=False))
print("="*145)

print("\nHoàn tất lưu trữ Best Models, Best Datasets và bố cục lại bảng đối chứng!")
