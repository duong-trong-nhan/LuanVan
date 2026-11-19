import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path

# Paths
feat_csv = Path('data/datasets/features_all.csv')
meta_csv = Path('data/spectrograms/metadata_split.csv')
out_csv = Path('data/datasets/split_group_v2.csv')

df_feat = pd.read_csv(feat_csv, encoding='utf-8')
df_meta = pd.read_csv(meta_csv, encoding='utf-8')

# Ensure sample_idx matches
assert (df_feat['sample_idx'] == df_meta['sample_idx']).all()

# Unique songs
song_df = df_feat[['audio_path', 'label']].drop_duplicates().reset_index(drop=True)
print(f"Tổng số bài hát duy nhất: {len(song_df)}")
print("Phân bố bài hát theo thể loại:")
for label, count in song_df['label'].value_counts().items():
    print(f"  - {label}: {count} bài")

# Split 1: 80% train_val, 20% test (stratified by label on song level)
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_val_idx, test_idx = next(sss1.split(song_df, song_df['label']))

train_val_songs = song_df.iloc[train_val_idx].reset_index(drop=True)
test_songs = song_df.iloc[test_idx].reset_index(drop=True)

# Split 2: Trong 80% train_val, tách 10% tổng (10/80 = 0.125) làm val, 70% tổng làm train
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=42)
train_idx, val_idx = next(sss2.split(train_val_songs, train_val_songs['label']))

train_songs = train_val_songs.iloc[train_idx].reset_index(drop=True)
val_songs = train_val_songs.iloc[val_idx].reset_index(drop=True)

train_set = set(train_songs['audio_path'])
val_set = set(val_songs['audio_path'])
test_set = set(test_songs['audio_path'])

# Đảm bảo hoàn toàn rời nhau về bài hát
assert len(train_set & val_set) == 0, "Lỗi rò rỉ train & val!"
assert len(train_set & test_set) == 0, "Lỗi rò rỉ train & test!"
assert len(val_set & test_set) == 0, "Lỗi rò rỉ val & test!"

split_map = {}
for s in train_set: split_map[s] = 'train'
for s in val_set: split_map[s] = 'val'
for s in test_set: split_map[s] = 'test'

# Tạo dataframe split_group_v2
split_df = pd.DataFrame({
    'sample_idx': df_feat['sample_idx'],
    'segment_id': df_feat['segment_id'],
    'label': df_feat['label'],
    'audio_path': df_feat['audio_path'],
    'split': df_feat['audio_path'].map(split_map),
    'mel_rel_path': df_meta['mel_rel_path'],
    'cqt_rel_path': df_meta['cqt_rel_path']
})

print("\n--- PHÂN BỐ PHÂN ĐOẠN THEO SPLIT PROTOCOL B ---")
for split_name, count in split_df['split'].value_counts().items():
    print(f"  - {split_name}: {count} đoạn ({count/len(split_df)*100:.1f}%)")

print("\n--- CHI TIẾT TỪNG THỂ LOẠI (SỐ ĐOẠN) ---")
ct = pd.crosstab(split_df['label'], split_df['split'])
for genre in ct.index:
    print(f"  {genre:<12}: Train = {ct.loc[genre, 'train']} | Val = {ct.loc[genre, 'val']} | Test = {ct.loc[genre, 'test']}")

# Xuất ra file CSV
split_df.to_csv(out_csv, index=False, encoding='utf-8')
print(f"\n[THÀNH CÔNG] Đã lưu file split mới tại: {out_csv.resolve()}")
