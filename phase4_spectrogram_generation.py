# -*- coding: utf-8 -*-
"""
HerittageFNS: Phase 4 - Spectrogram Generation from Audio Segments
===================================================================
Converts 3,600 audio segments (12s, 22050Hz) into spectrogram images:
  1. Log-Mel Spectrogram  (128 mel bins, 224x224 PNG)
  2. CQT Spectrogram      (84 bins / 7 octaves, 224x224 PNG)

Output structure:
  data/spectrograms/
    mel/
      Ca tru/   *.png
      Cai luong/ *.png
      ...
    cqt/
      Ca tru/   *.png
      ...

Requires:
  pip install torch torchvision librosa soundfile matplotlib pillow tqdm
"""

import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import os
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 22050
DURATION      = 12          # seconds (12s segments from phase 0)
N_FFT         = 2048
HOP_LENGTH    = 512         # ~23ms hop → 516 frames on 12s
N_MELS        = 128         # mel bins
FMIN          = 20          # Hz
FMAX          = 8000        # Hz
CQT_N_BINS    = 84          # 7 octaves × 12 bins/octave
CQT_HOP       = 512
CQT_FMIN      = librosa.note_to_hz('C1')   # ~32.70 Hz

TARGET_SIZE   = (224, 224)  # CNN input size
DPI           = 100         # figure DPI for saving

# Paths
DATA_CSV      = Path('./data/datasets/features_all.csv')    # has audio_path + label
MEL_DIR       = Path('./data/spectrograms/mel')
CQT_DIR       = Path('./data/spectrograms/cqt')
PLOTS_PREVIEW = Path('./data/plots/spectrogram_previews')

# ── Audio root resolution ─────────────────────────────────────────────────────
# audio_path in CSV is relative like 'ca_tru/audio_17_017.wav'
# We resolve it against known root folders in order of preference.
_HF_SNAPSHOT = None
_hf_base = Path('~/.cache/huggingface/hub').expanduser()
_hf_datasets = _hf_base / 'datasets--NTQAI--Vietnamese-Traditional-Music'
if _hf_datasets.exists():
    _snap_dirs = list((_hf_datasets / 'snapshots').glob('*/audios'))
    if _snap_dirs:
        _HF_SNAPSHOT = _snap_dirs[0]  # e.g. .../snapshots/<hash>/audios

AUDIO_ROOTS = [r for r in [
    _HF_SNAPSHOT,                                    # HF cache (snapshots/hash/audios)
    Path('./dataset'),                               # local dataset/
    Path('./dataset/audio'),
    Path('.'),
] if r is not None and r.exists()]

print(f"Audio roots to search: {AUDIO_ROOTS}")


def resolve_audio_path(rel_path: str) -> str:
    """
    Try to resolve a relative audio path (e.g. 'ca_tru/audio_17_017.wav')
    against known audio root directories. Returns the first match or rel_path if none found.
    """
    p = Path(rel_path)
    if p.is_absolute() and p.exists():
        return str(p)
    # Normalize genre folder name variations (ca_tru -> catru, Ca tru, etc.)
    filename = p.name
    genre_rel = p.parent.name  # e.g. 'ca_tru'
    for root in AUDIO_ROOTS:
        # Try direct
        candidate = root / p
        if candidate.exists():
            return str(candidate)
        # Try with just filename under genre subdirs
        for sub in root.iterdir() if root.is_dir() else []:
            if sub.is_dir():
                candidate2 = sub / filename
                if candidate2.exists():
                    return str(candidate2)
    # Fallback: return as-is (will generate silence)
    return rel_path

MEL_DIR.mkdir(parents=True, exist_ok=True)
CQT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_PREVIEW.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load metadata CSV
# ─────────────────────────────────────────────────────────────────────────────
# features_all.csv must have columns: audio_path, label, segment_id, segment_time
# If features_all.csv is unavailable, fall back to features_fuzzy_continuous.csv
if DATA_CSV.exists():
    df_meta = pd.read_csv(DATA_CSV)
    print(f"Loaded {DATA_CSV}: {df_meta.shape}")
else:
    # Fallback: use fuzzy continuous which has audio_path
    fallback = Path('./data/datasets/features_fuzzy_continuous.csv')
    df_meta = pd.read_csv(fallback)
    print(f"Loaded fallback {fallback}: {df_meta.shape}")

# Keep only needed columns
required_cols = {'audio_path', 'label', 'segment_id', 'sample_idx'}
missing = required_cols - set(df_meta.columns)
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}. "
                     f"Available: {list(df_meta.columns)}")

df_meta = df_meta[list(required_cols)].copy()
df_meta['segment_id'] = df_meta['segment_id'].fillna(0).astype(int)
print(f"Total segments: {len(df_meta)}")
print(f"Labels: {df_meta['label'].value_counts().to_dict()}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def load_segment(audio_path: str, segment_id: int,
                 sr: int = SAMPLE_RATE, duration: float = DURATION) -> np.ndarray:
    """Load a 12s segment. Segment 0 = [5s, 17s], Segment 1 = [25s, 37s]."""
    offset = 5.0 if segment_id == 0 else 25.0
    try:
        y, _ = librosa.load(audio_path, sr=sr, offset=offset, duration=duration, mono=True)
    except Exception as e:
        print(f"  [WARN] Cannot load {audio_path}: {e}. Using silence.")
        y = np.zeros(int(sr * duration))
    # Pad or trim to exact length
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y


def audio_to_log_mel(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Compute Log-Mel spectrogram, shape (N_MELS, T), values in dB."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        power=2.0
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel


def audio_to_cqt(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Compute CQT magnitude spectrogram, shape (CQT_N_BINS, T), values in dB."""
    cqt = np.abs(librosa.cqt(
        y=y, sr=sr,
        hop_length=CQT_HOP,
        fmin=CQT_FMIN,
        n_bins=CQT_N_BINS,
        bins_per_octave=12
    ))
    log_cqt = librosa.amplitude_to_db(cqt, ref=np.max)
    return log_cqt


def spec_to_image(spec: np.ndarray, target_size=(224, 224)) -> Image.Image:
    """
    Convert a 2D spectrogram (H x W, dB-scaled) to a 3-channel PIL Image.
    Uses 'magma' colormap for better contrast on academic prints.
    """
    # Normalize to [0, 1]
    s_min, s_max = spec.min(), spec.max()
    if s_max > s_min:
        spec_norm = (spec - s_min) / (s_max - s_min)
    else:
        spec_norm = np.zeros_like(spec)

    # Apply colormap
    cmap = plt.get_cmap('magma')
    rgba = cmap(spec_norm)           # (H, W, 4) float [0,1]
    rgb  = (rgba[:, :, :3] * 255).astype(np.uint8)  # drop alpha

    img = Image.fromarray(rgb, mode='RGB')
    img = img.resize(target_size, Image.LANCZOS)
    return img


def save_spectrogram_image(spec: np.ndarray, out_path: Path,
                           target_size=(224, 224)):
    """Save spectrogram as PNG image."""
    img = spec_to_image(spec, target_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format='PNG')


# ─────────────────────────────────────────────────────────────────────────────
# 3. Generate all spectrograms
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PHASE 4: GENERATING LOG-MEL & CQT SPECTROGRAMS")
print("="*70)

skip_existing = True   # Set False to regenerate all
errors = []

for idx, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Generating"):
    audio_path = str(row['audio_path'])
    label      = str(row['label'])
    seg_id     = int(row['segment_id'])
    sample_idx = int(row['sample_idx'])

    stem = f"{sample_idx:04d}_seg{seg_id}"

    mel_out = MEL_DIR / label / f"{stem}.png"
    cqt_out = CQT_DIR / label / f"{stem}.png"

    mel_done = mel_out.exists() and skip_existing
    cqt_done = cqt_out.exists() and skip_existing

    if mel_done and cqt_done:
        continue

    # Resolve relative audio path → absolute path
    resolved_path = resolve_audio_path(audio_path)
    # Load audio
    y = load_segment(resolved_path, seg_id)

    # Log-Mel
    if not mel_done:
        try:
            log_mel = audio_to_log_mel(y)
            save_spectrogram_image(log_mel, mel_out, TARGET_SIZE)
        except Exception as e:
            errors.append(('mel', audio_path, str(e)))

    # CQT
    if not cqt_done:
        try:
            log_cqt = audio_to_cqt(y)
            save_spectrogram_image(log_cqt, cqt_out, TARGET_SIZE)
        except Exception as e:
            errors.append(('cqt', audio_path, str(e)))

print(f"\nGeneration complete. Errors: {len(errors)}")
if errors:
    for err in errors[:10]:
        print(f"  [{err[0]}] {err[1]}: {err[2]}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Verify counts
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Verification ---")
genres = sorted([d.name for d in MEL_DIR.iterdir() if d.is_dir()])
for genre in genres:
    mel_count = len(list((MEL_DIR / genre).glob('*.png')))
    cqt_count = len(list((CQT_DIR / genre).glob('*.png')))
    print(f"  {genre:<20s}: Mel={mel_count:4d}  CQT={cqt_count:4d}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Preview: save 1 sample per genre (Mel + CQT side-by-side)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Generating preview plots ---")

for genre in genres:
    mel_files = sorted((MEL_DIR / genre).glob('*.png'))
    cqt_files = sorted((CQT_DIR / genre).glob('*.png'))
    if not mel_files or not cqt_files:
        continue

    mel_img = np.array(Image.open(mel_files[0]))
    cqt_img = np.array(Image.open(cqt_files[0]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(mel_img, aspect='auto', origin='lower')
    axes[0].set_title(f'{genre} — Log-Mel (128 bins)', fontsize=11)
    axes[0].set_xlabel('Time frames')
    axes[0].set_ylabel('Mel bins')
    axes[0].axis('off')

    axes[1].imshow(cqt_img, aspect='auto', origin='lower')
    axes[1].set_title(f'{genre} — CQT (84 bins, 7 oct)', fontsize=11)
    axes[1].set_xlabel('Time frames')
    axes[1].set_ylabel('CQT bins (C1 to C8)')
    axes[1].axis('off')

    fig.suptitle(f'Spectrogram Preview: {genre}', fontsize=13, fontweight='bold')
    plt.tight_layout()

    out_path = PLOTS_PREVIEW / f"preview_{genre.replace(' ', '_')}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved preview: {out_path}")

print("\nPhase 4 DONE.")
print(f"  Log-Mel images: {MEL_DIR}")
print(f"  CQT images:     {CQT_DIR}")
print(f"  Previews:       {PLOTS_PREVIEW}")
