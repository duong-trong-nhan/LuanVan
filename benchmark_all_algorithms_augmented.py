# -*- coding: utf-8 -*-
"""
HerittageFNS: Unified Benchmark - 4 Algorithms Across Data Augmentation Methods
================================================================================
Algorithms evaluated:
1. Decision Tree (Crisp & Continuous)
2. Random Forest (Crisp & Continuous)
3. RIPPER (Repeated Incremental Pruning to Produce Error Reduction)
4. CN2 (Beam Search Sequential Covering with Laplace Heuristic)

Data Augmentation methods evaluated on Train set (Test set fixed at 720 samples):
- Baseline (No Augmentation, 2,880 samples)
- Gaussian Jittering (+100%, 5,760 samples)
- SMOTE (+50%, 4,320 samples)
- SMOTE (+100%, 5,760 samples)
- Combined SMOTE + Gaussian Noise (7,200 samples)
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
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import wittgenstein as lw

# Output directories
output_dir = Path("./data/datasets/augmented")
output_dir.mkdir(parents=True, exist_ok=True)
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

# Reverse maps for readable rules
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

# -----------------------------------------------------------------------------
# 2. Stratified 80/20 Train / Test Split (random_state=42)
# -----------------------------------------------------------------------------
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
# 3. CN2 Rule Classifier Implementation
# -----------------------------------------------------------------------------
class CN2Classifier:
    def __init__(self, beam_width=5, min_significance=0.01, min_support=10, max_antecedents=3):
        self.beam_width = beam_width
        self.min_significance = min_significance
        self.min_support = min_support
        self.max_antecedents = max_antecedents
        self.rules = []
        self.default_class = None
        self.classes_ = []
        
    def _laplace_score(self, y_covered, target_class):
        n = len(y_covered)
        if n == 0: return 0.0
        n_correct = np.sum(y_covered == target_class)
        k = len(self.classes_)
        return (n_correct + 1.0) / (n + k)
    
    def _g_statistic(self, y_covered, y_all, target_class):
        n = len(y_covered)
        if n == 0: return 0.0
        N = len(y_all)
        p_prior = np.sum(y_all == target_class) / N
        f_pos = np.sum(y_covered == target_class)
        f_neg = n - f_pos
        e_pos = n * p_prior
        e_neg = n * (1.0 - p_prior)
        g = 0.0
        if f_pos > 0 and e_pos > 0:
            g += f_pos * math.log(f_pos / e_pos)
        if f_neg > 0 and e_neg > 0:
            g += f_neg * math.log(f_neg / e_neg)
        return 2.0 * g

    def fit(self, X_df, y_series):
        self.classes_ = sorted(list(set(y_series)))
        self.default_class = y_series.mode()[0]
        
        primitives = []
        for col in X_df.columns:
            for v in X_df[col].dropna().unique():
                primitives.append((col, v))
                
        self.rules = []
        for target_cls in self.classes_:
            uncovered_mask = pd.Series([True] * len(X_df), index=X_df.index)
            while True:
                curr_X = X_df[uncovered_mask]
                curr_y = y_series[uncovered_mask]
                if np.sum(curr_y == target_cls) < self.min_support:
                    break
                    
                beam = [[]]
                best_rule_cond = None
                best_rule_score = -1.0
                
                for step in range(self.max_antecedents):
                    candidates = []
                    for cond in beam:
                        cond_features = [f for (f, v) in cond]
                        for (p_feat, p_val) in primitives:
                            if p_feat in cond_features: continue
                            new_cond = cond + [(p_feat, p_val)]
                            
                            mask = pd.Series([True] * len(curr_X), index=curr_X.index)
                            for f, v in new_cond:
                                mask = mask & (curr_X[f] == v)
                            cov_y = curr_y[mask]
                            if len(cov_y) < self.min_support: continue
                            
                            score = self._laplace_score(cov_y.values, target_cls)
                            g_stat = self._g_statistic(cov_y.values, y_series.values, target_cls)
                            candidates.append((score, g_stat, new_cond))
                    if not candidates: break
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    beam = [item[2] for item in candidates[:self.beam_width]]
                    
                    top_score, top_g, top_cond = candidates[0]
                    if top_g >= self.min_significance and top_score > best_rule_score:
                        best_rule_score = top_score
                        best_rule_cond = top_cond
                        
                if best_rule_cond is not None and best_rule_score > (1.0 / len(self.classes_)):
                    full_mask = pd.Series([True] * len(X_df), index=X_df.index)
                    for f, v in best_rule_cond:
                        full_mask = full_mask & (X_df[f] == v)
                    supp_full = int(full_mask.sum())
                    prec_full = float((y_series[full_mask] == target_cls).mean()) if supp_full > 0 else 0.0
                    rule_text = " AND ".join([f"{f} == '{v}'" for f, v in best_rule_cond])
                    
                    self.rules.append({
                        "rule_text": rule_text,
                        "consequent": target_cls,
                        "conditions": best_rule_cond,
                        "confidence": round(prec_full, 4),
                        "support": supp_full,
                        "antecedents_count": len(best_rule_cond)
                    })
                    pos_covered = full_mask & (y_series == target_cls)
                    uncovered_mask = uncovered_mask & (~pos_covered)
                    if pos_covered.sum() == 0: break
                else:
                    break
        return self

    def predict(self, X_df):
        preds = []
        for _, row in X_df.iterrows():
            matched = False
            for r in self.rules:
                m = True
                for f, v in r["conditions"]:
                    if row[f] != v:
                        m = False
                        break
                if m:
                    preds.append(r["consequent"])
                    matched = True
                    break
            if not matched:
                preds.append(self.default_class)
        return np.array(preds)

# -----------------------------------------------------------------------------
# 4. Helper for Tree Rule Extraction (Rule count & Avg antecedents)
# -----------------------------------------------------------------------------
def get_tree_rule_stats(dt_model):
    tree = dt_model.tree_
    leaf_depths = []
    def recurse(node, depth):
        if tree.feature[node] != -2:
            recurse(tree.children_left[node], depth + 1)
            recurse(tree.children_right[node], depth + 1)
        else:
            leaf_depths.append(depth)
    recurse(0, 0)
    rule_count = len(leaf_depths)
    avg_len = np.mean(leaf_depths) if leaf_depths else 0.0
    return rule_count, round(float(avg_len), 2)

def get_rf_rule_stats(rf_model):
    total_rules = 0
    all_depths = []
    for est in rf_model.estimators_:
        rc, avg_d = get_tree_rule_stats(est)
        total_rules += rc
        all_depths.append(avg_d)
    return total_rules, round(float(np.mean(all_depths)), 2)

# -----------------------------------------------------------------------------
# 5. Master Evaluation Loop
# -----------------------------------------------------------------------------
def evaluate_all_algorithms(X_tr_cont, y_tr, X_te_cont, y_te, X_tr_c, X_te_c, desc):
    print(f"\n--- Running: {desc} (Train samples: {len(y_tr)}) ---")
    
    # 1. Decision Tree Crisp
    dt_c = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
    dt_c.fit(X_tr_c, y_tr)
    p_dt_c = dt_c.predict(X_te_c)
    acc_dt_c = accuracy_score(y_te, p_dt_c) * 100
    f1_dt_c  = f1_score(y_te, p_dt_c, average='macro') * 100
    dt_c_rc, dt_c_len = get_tree_rule_stats(dt_c)
    
    # 2. Random Forest Crisp
    rf_c = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_leaf=10, random_state=42)
    rf_c.fit(X_tr_c, y_tr)
    p_rf_c = rf_c.predict(X_te_c)
    acc_rf_c = accuracy_score(y_te, p_rf_c) * 100
    f1_rf_c  = f1_score(y_te, p_rf_c, average='macro') * 100
    rf_c_rc, rf_c_len = get_rf_rule_stats(rf_c)
    
    # 3. Decision Tree Continuous
    dt_cnt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
    dt_cnt.fit(X_tr_cont, y_tr)
    p_dt_cnt = dt_cnt.predict(X_te_cont)
    acc_dt_cnt = accuracy_score(y_te, p_dt_cnt) * 100
    f1_dt_cnt  = f1_score(y_te, p_dt_cnt, average='macro') * 100
    
    # 4. Random Forest Continuous
    rf_cnt = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_leaf=10, random_state=42)
    rf_cnt.fit(X_tr_cont, y_tr)
    p_rf_cnt = rf_cnt.predict(X_te_cont)
    acc_rf_cnt = accuracy_score(y_te, p_rf_cnt) * 100
    f1_rf_cnt  = f1_score(y_te, p_rf_cnt, average='macro') * 100
    
    # Prepare Crisp string DataFrames for RIPPER & CN2
    df_tr_str = crisp_indices_to_df(X_tr_c)
    df_te_str = crisp_indices_to_df(X_te_c)
    
    # 5. RIPPER (One-vs-Rest)
    print("  Training RIPPER...")
    class_probs = []
    ripper_cond_lens = []
    for cls in class_names:
        train_bin = df_tr_str.copy()
        train_bin['target'] = (y_tr == cls).astype(int)
        r_model = lw.RIPPER(k=2, prune_size=0.33, dl_allowance=64, random_state=42)
        r_model.fit(train_bin, class_feat='target', pos_class=1)
        proba = r_model.predict_proba(df_te_str)[:, 1]
        class_probs.append(proba)
        for r in r_model.ruleset_.rules:
            if r.conds:
                ripper_cond_lens.append(len(r.conds))
    prob_mat = np.column_stack(class_probs)
    p_ripper = [class_names[i] for i in np.argmax(prob_mat, axis=1)]
    acc_ripper = accuracy_score(y_te, p_ripper) * 100
    f1_ripper  = f1_score(y_te, p_ripper, average='macro') * 100
    ripper_rc = len(ripper_cond_lens)
    ripper_len = round(float(np.mean(ripper_cond_lens)), 2) if ripper_cond_lens else 0.0
    
    # 6. CN2 Rule Classifier
    print("  Training CN2...")
    cn2 = CN2Classifier(beam_width=5, min_significance=0.01, min_support=10, max_antecedents=3)
    cn2.fit(df_tr_str, pd.Series(y_tr))
    p_cn2 = cn2.predict(df_te_str)
    acc_cn2 = accuracy_score(y_te, p_cn2) * 100
    f1_cn2  = f1_score(y_te, p_cn2, average='macro') * 100
    cn2_rc = len(cn2.rules)
    cn2_len = round(float(np.mean([r["antecedents_count"] for r in cn2.rules])), 2) if cn2.rules else 0.0
    
    # Record Master Row
    row = {
        'Augmentation_Method': desc,
        'Train_Samples': len(y_tr),
        # DT Crisp
        'DT_Crisp_Acc': acc_dt_c,
        'DT_Crisp_F1': f1_dt_c,
        'DT_Crisp_Rules': dt_c_rc,
        'DT_Crisp_AvgLen': dt_c_len,
        # RF Crisp
        'RF_Crisp_Acc': acc_rf_c,
        'RF_Crisp_F1': f1_rf_c,
        'RF_Crisp_Rules': rf_c_rc,
        'RF_Crisp_AvgLen': rf_c_len,
        # DT Cont
        'DT_Cont_Acc': acc_dt_cnt,
        'DT_Cont_F1': f1_dt_cnt,
        # RF Cont
        'RF_Cont_Acc': acc_rf_cnt,
        'RF_Cont_F1': f1_rf_cnt,
        # RIPPER
        'RIPPER_Acc': acc_ripper,
        'RIPPER_F1': f1_ripper,
        'RIPPER_Rules': ripper_rc,
        'RIPPER_AvgLen': ripper_len,
        # CN2
        'CN2_Acc': acc_cn2,
        'CN2_F1': f1_cn2,
        'CN2_Rules': cn2_rc,
        'CN2_AvgLen': cn2_len,
    }
    return row

# -----------------------------------------------------------------------------
# 6. Generate the 5 Augmented Train Sets
# -----------------------------------------------------------------------------
np.random.seed(42)
feat_stds = np.std(X_train_orig, axis=0, keepdims=True)

# 1. Baseline
data_settings = []
data_settings.append((
    "Baseline (No Augmentation)",
    X_train_orig, y_train_orig, X_c_train_orig
))

# 2. Gaussian Jittering (+100%)
noise = np.random.normal(0, 0.05 * (feat_stds + 1e-4), size=X_train_orig.shape)
X_tr_gauss = np.vstack([X_train_orig, normalize_ruspini(X_train_orig + noise)])
y_tr_gauss = np.hstack([y_train_orig, y_train_orig])
X_c_tr_gauss = cont_to_crisp(X_tr_gauss)
data_settings.append((
    "Gaussian Jittering (+100%)",
    X_tr_gauss, y_tr_gauss, X_c_tr_gauss
))

# 3. SMOTE (+50% to 720/class)
target_dict_50 = {cls: 720 for cls in class_names}
smote_50 = SMOTE(sampling_strategy=target_dict_50, k_neighbors=5, random_state=42)
X_tr_sm50, y_tr_sm50 = smote_50.fit_resample(X_train_orig, y_train_orig)
X_tr_sm50 = normalize_ruspini(X_tr_sm50)
X_c_tr_sm50 = cont_to_crisp(X_tr_sm50)
data_settings.append((
    "SMOTE (+50% to 720/class)",
    X_tr_sm50, y_tr_sm50, X_c_tr_sm50
))

# 4. SMOTE (+100% to 960/class)
target_dict_100 = {cls: 960 for cls in class_names}
smote_100 = SMOTE(sampling_strategy=target_dict_100, k_neighbors=5, random_state=42)
X_tr_sm100, y_tr_sm100 = smote_100.fit_resample(X_train_orig, y_train_orig)
X_tr_sm100 = normalize_ruspini(X_tr_sm100)
X_c_tr_sm100 = cont_to_crisp(X_tr_sm100)
data_settings.append((
    "SMOTE (+100% to 960/class)",
    X_tr_sm100, y_tr_sm100, X_c_tr_sm100
))

# 5. Combined SMOTE + Gaussian Noise
noise_sm = np.random.normal(0, 0.03 * (feat_stds + 1e-4), size=X_tr_sm50.shape)
X_tr_comb = normalize_ruspini(X_tr_sm50 + noise_sm)
X_tr_comb = np.vstack([X_train_orig, X_tr_comb[len(X_train_orig):], X_tr_gauss[len(X_train_orig):]])
y_tr_comb = np.hstack([y_train_orig, y_tr_sm50[len(X_train_orig):], y_train_orig])
X_c_tr_comb = cont_to_crisp(X_tr_comb)
data_settings.append((
    "Combined SMOTE + Gaussian Noise",
    X_tr_comb, y_tr_comb, X_c_tr_comb
))

# -----------------------------------------------------------------------------
# 7. Execute Benchmark Across All Settings
# -----------------------------------------------------------------------------
master_rows = []
for desc, X_tr_cnt, y_tr, X_tr_c in data_settings:
    row = evaluate_all_algorithms(
        X_tr_cnt, y_tr, X_test, y_test, X_tr_c, X_c_test, desc
    )
    master_rows.append(row)

df_master = pd.DataFrame(master_rows)

# Save Master Table
master_csv = output_dir / "augmentation_all_algorithms_benchmark.csv"
df_master.to_csv(master_csv, index=False)
print(f"\n-> Saved master benchmark table to: {master_csv}")

# Print Accuracy & F1 Comparison
print("\n" + "="*115)
print("BẢNG TỔNG HỢP HIỆU NĂNG TẤT CẢ 4 THUẬT TOÁN QUA CÁC PHƯƠNG PHÁP TĂNG CƯỜNG DỮ LIỆU (TEST SET = 720 MẪU)")
print("="*115)
perf_cols = [
    'Augmentation_Method', 'Train_Samples',
    'DT_Crisp_Acc', 'RF_Crisp_Acc', 'RIPPER_Acc', 'CN2_Acc',
    'DT_Cont_Acc', 'RF_Cont_Acc'
]
print(df_master[perf_cols].to_string(index=False))

# Build and Print Rule Induction Comparison Table (Rules, Length, Acc, F1)
rule_rows = []
for r in master_rows:
    aug = r['Augmentation_Method']
    samples = r['Train_Samples']
    # DT Crisp
    rule_rows.append({
        'Augmentation': aug,
        'Train_Samples': samples,
        'Algorithm': 'Decision Tree Crisp',
        'Rule_Count': r['DT_Crisp_Rules'],
        'Avg_Antecedents': r['DT_Crisp_AvgLen'],
        'Accuracy (%)': round(r['DT_Crisp_Acc'], 2),
        'Macro_F1 (%)': round(r['DT_Crisp_F1'], 2),
    })
    # RF Crisp
    rule_rows.append({
        'Augmentation': aug,
        'Train_Samples': samples,
        'Algorithm': 'Random Forest Crisp',
        'Rule_Count': r['RF_Crisp_Rules'],
        'Avg_Antecedents': r['RF_Crisp_AvgLen'],
        'Accuracy (%)': round(r['RF_Crisp_Acc'], 2),
        'Macro_F1 (%)': round(r['RF_Crisp_F1'], 2),
    })
    # RIPPER
    rule_rows.append({
        'Augmentation': aug,
        'Train_Samples': samples,
        'Algorithm': 'RIPPER (wittgenstein)',
        'Rule_Count': r['RIPPER_Rules'],
        'Avg_Antecedents': r['RIPPER_AvgLen'],
        'Accuracy (%)': round(r['RIPPER_Acc'], 2),
        'Macro_F1 (%)': round(r['RIPPER_F1'], 2),
    })
    # CN2
    rule_rows.append({
        'Augmentation': aug,
        'Train_Samples': samples,
        'Algorithm': 'CN2 Rule Induction',
        'Rule_Count': r['CN2_Rules'],
        'Avg_Antecedents': r['CN2_AvgLen'],
        'Accuracy (%)': round(r['CN2_Acc'], 2),
        'Macro_F1 (%)': round(r['CN2_F1'], 2),
    })

df_rule_comp = pd.DataFrame(rule_rows)
rule_comp_csv = rules_dir / "rule_induction_augmented_comparison.csv"
df_rule_comp.to_csv(rule_comp_csv, index=False)
print(f"\n-> Saved rule induction comparison to: {rule_comp_csv}")

print("\n" + "="*100)
print("BẢNG SO SÁNH ĐẶC TÍNH HỆ LUẬT (RULE COUNT, ĐỘ DÀI TIỀN ĐỀ, ACCURACY, F1) CỦA 4 THUẬT TOÁN")
print("="*100)
print(df_rule_comp.to_string(index=False))
print("\nHoàn tất Benchmark tổng hợp 4 thuật toán trên tất cả các phương pháp tăng cường dữ liệu!")
