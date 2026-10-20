# -*- coding: utf-8 -*-
"""
HerittageFNS: Phase 5 - CNN Transfer Learning Benchmark on Spectrogram Images
==============================================================================
14 experiments comparing:
  - Backbones: EfficientNet-B0, EfficientNet-B2, MobileNetV2, ResNet-50, VGG16
  - Spectrogram types: Log-Mel, CQT, Mel+CQT (2-channel)
  - Fine-tuning strategies: Frozen, Partial FT, Full FT
  - Late Fusion Ensemble with Fuzzy Logic Best Model (RF Continuous + Gaussian)

All results saved to:
  data/results/cnn_benchmark.csv       (per-experiment summary)
  data/results/cnn_confusion_matrices/ (confusion matrix PNGs)
  data/models/cnn/                     (best model .pth & test_probs.npy)
  data/results/cnn_experiment_log.md   (markdown log tracking all runs)

Usage:
  python phase5_cnn_transfer_learning.py             # run all
  python phase5_cnn_transfer_learning.py --exp 2     # run only EXP-02
"""

import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
import time
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
torch.set_num_threads(min(8, os.cpu_count() or 4))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print(f"  CPU Threads: {torch.get_num_threads()}")

RANDOM_STATE   = 42
IMG_SIZE       = 224
BATCH_SIZE     = 32
LABEL_SMOOTH   = 0.1
WEIGHT_DECAY   = 1e-4
NUM_CLASSES    = 6

# Paths
MEL_DIR      = Path('./data/spectrograms/mel')
CQT_DIR      = Path('./data/spectrograms/cqt')
DATA_CSV     = Path('./data/datasets/features_all.csv')
RESULTS_DIR  = Path('./data/results')
CM_DIR       = RESULTS_DIR / 'cnn_confusion_matrices'
MODEL_DIR    = Path('./data/models/cnn')
RESULTS_CSV  = RESULTS_DIR / 'cnn_benchmark.csv'
LOG_MD_PATH  = RESULTS_DIR / 'cnn_experiment_log.md'

for d in [RESULTS_DIR, CM_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Detect genres matching the directory structure and RF classes
if MEL_DIR.exists():
    GENRES = sorted([p.name for p in MEL_DIR.iterdir() if p.is_dir()])
else:
    GENRES = ['Ca trù', 'Chèo', 'Chầu văn', 'Cải lương', 'Hát xẩm', 'Quan họ']
print(f"Target Genres ({len(GENRES)}): {GENRES}")

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset & Alignment
# ─────────────────────────────────────────────────────────────────────────────
class SpectrogramDataset(Dataset):
    """Load spectrogram PNG images from file paths."""

    def __init__(self, samples: list, transform=None, dual_channel=False):
        """
        samples: list of (primary_path, secondary_path, label_idx)
        dual_channel: if True, stack mel+cqt as 2-channel tensor
        """
        self.samples      = samples
        self.transform    = transform
        self.dual_channel = dual_channel

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p1, p2, label = self.samples[idx]

        if self.dual_channel:
            mel_img = Image.open(p1).convert('L')   # grayscale
            cqt_img = Image.open(p2).convert('L')
            mel_t = transforms.ToTensor()(mel_img)   # (1, H, W)
            cqt_t = transforms.ToTensor()(cqt_img)   # (1, H, W)
            img_t = torch.cat([mel_t, cqt_t], dim=0) # (2, H, W)
            img_t = transforms.Resize((IMG_SIZE, IMG_SIZE))(img_t)
            img_t = transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5])(img_t)
        else:
            img = Image.open(p1).convert('RGB')
            if self.transform:
                img_t = self.transform(img)
            else:
                img_t = transforms.ToTensor()(img)

        return img_t, label


def load_dataset_samples(spec_type='mel'):
    """
    Load samples strictly aligned with features_all.csv and test_isolated_fuzzy_continuous.csv.
    Split ratio 80/20 with random_state=42 matches the exact test partition of the fuzzy branch.
    """
    df_all = pd.read_csv(DATA_CSV)
    genre_to_idx = {g: i for i, g in enumerate(GENRES)}

    samples = []
    for _, row in df_all.iterrows():
        label = row['label']
        stem = f"{int(row['sample_idx']):04d}_seg{int(row['segment_id'])}"
        mel_f = MEL_DIR / label / f"{stem}.png"
        cqt_f = CQT_DIR / label / f"{stem}.png"
        idx = genre_to_idx[label]

        if spec_type == 'mel':
            samples.append((mel_f, mel_f, idx))
        elif spec_type == 'cqt':
            samples.append((cqt_f, cqt_f, idx))
        else:  # mel+cqt
            samples.append((mel_f, cqt_f, idx))

    indices = np.arange(len(df_all))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.20, random_state=RANDOM_STATE, stratify=df_all['label']
    )
    train_samples = [samples[i] for i in train_idx]
    test_samples  = [samples[i] for i in test_idx]
    return train_samples, test_samples


def get_transforms(augment=True):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if augment:
        train_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf


# ─────────────────────────────────────────────────────────────────────────────
# 2. SpecAugment (Time & Frequency Masking)
# ─────────────────────────────────────────────────────────────────────────────
class SpecAugment(nn.Module):
    def __init__(self, time_mask_param=30, freq_mask_param=16, num_masks=2):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_masks       = num_masks

    def forward(self, x):
        if not self.training:
            return x
        B, C, H, W = x.shape
        for _ in range(self.num_masks):
            f = np.random.randint(0, self.freq_mask_param)
            f0 = np.random.randint(0, max(1, H - f))
            x[:, :, f0:f0+f, :] = 0.0

            t = np.random.randint(0, self.time_mask_param)
            t0 = np.random.randint(0, max(1, W - t))
            x[:, :, :, t0:t0+t] = 0.0
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model Builders
# ─────────────────────────────────────────────────────────────────────────────
def build_efficientnet_b0(strategy='partial', in_channels=3):
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    if in_channels != 3:
        old_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size, stride=old_conv.stride,
            padding=old_conv.padding, bias=False
        )

    for p in model.parameters():
        p.requires_grad = False

    if strategy == 'partial':
        for p in model.features[5:].parameters():
            p.requires_grad = True
    elif strategy == 'full':
        for p in model.parameters():
            p.requires_grad = True

    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES)
    )
    return model


def build_efficientnet_b2(strategy='partial'):
    model = models.efficientnet_b2(weights='IMAGENET1K_V1')
    for p in model.parameters():
        p.requires_grad = False
    if strategy == 'partial':
        for p in model.features[4:].parameters():
            p.requires_grad = True
    elif strategy == 'full':
        for p in model.parameters():
            p.requires_grad = True
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1408, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES)
    )
    return model


def build_mobilenet_v2(strategy='partial'):
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    for p in model.parameters():
        p.requires_grad = False
    if strategy == 'partial':
        for p in model.features[14:].parameters():
            p.requires_grad = True
    elif strategy == 'full':
        for p in model.parameters():
            p.requires_grad = True
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES)
    )
    return model


def build_resnet50(strategy='partial'):
    model = models.resnet50(weights='IMAGENET1K_V1')
    for p in model.parameters():
        p.requires_grad = False
    if strategy == 'partial':
        for p in model.layer4.parameters():
            p.requires_grad = True
    elif strategy == 'full':
        for p in model.parameters():
            p.requires_grad = True
    model.fc = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, NUM_CLASSES)
    )
    return model


def build_vgg16(strategy='partial'):
    model = models.vgg16(weights='IMAGENET1K_V1')
    for p in model.parameters():
        p.requires_grad = False
    if strategy == 'partial':
        for p in model.features[24:].parameters():
            p.requires_grad = True
    elif strategy == 'full':
        for p in model.parameters():
            p.requires_grad = True
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, NUM_CLASSES)
    )
    return model


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training & Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, spec_aug, scaler=None):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        imgs = spec_aug(imgs)

        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                out  = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc      = accuracy_score(all_labels, all_preds) * 100
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out = model(imgs)
        probs = torch.softmax(out, dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc  = accuracy_score(all_labels, all_preds) * 100
    f1   = f1_score(all_labels, all_preds, average='macro') * 100
    return acc, f1, np.array(all_preds), np.array(all_labels), np.array(all_probs)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Markdown Log Auto-Updater
# ─────────────────────────────────────────────────────────────────────────────
def update_markdown_log(df_all_results):
    if not LOG_MD_PATH.exists():
        return

    content = LOG_MD_PATH.read_text(encoding='utf-8')

    lines = []
    lines.append("| # | Mo hinh | Spectrogram | Strategy | Test Acc | Macro F1 | Delta vs Baseline | Time (min) | Params |")
    lines.append("|---|---------|-------------|----------|----------|----------|------------------|-----------|--------|")

    for _, r in df_all_results.iterrows():
        eid = int(r['exp_id'])
        d_base = f"{r['delta_vs_baseline']:+.2f}%"
        p_str = f"{int(r['trainable_params']):,}" if r['trainable_params'] > 0 else "--"
        lines.append(
            f"| EXP-{eid:02d} | {r['model_name']} | {r['spec_type']} | {r['strategy']} | "
            f"**{r['test_accuracy']:.2f}%** | {r['macro_f1']:.2f}% | {d_base} | {r['training_time_min']:.1f} | {p_str} |"
        )
    lines.append("| REF | RF-Cont + Gaussian (BASELINE) | -- | -- | 65.56% | 65.12% | 0.00% | -- | -- |")

    table_str = "\n".join(lines)
    header_find = "## BẢNG TỔNG HỢP SO SÁNH PHASE 5"
    if header_find in content:
        parts = content.split(header_find)
        header_line = parts[1].split("\n")[0]
        before = parts[0] + header_find + header_line + "\n\n"
        after_section = parts[1][len(header_line):]
        next_header = "\n---\n\n## SO SÁNH TOÀN DIỆN VỚI NHÁNH FUZZY"
        if next_header in after_section:
            rest = after_section[after_section.find(next_header):]
            new_content = before + table_str + rest
            LOG_MD_PATH.write_text(new_content, encoding='utf-8')
            print("  [OK] Updated data/results/cnn_experiment_log.md")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Single Experiment Runner
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment(
    exp_id: int,
    model_name: str,
    spec_type: str,        # 'mel', 'cqt', 'mel+cqt'
    strategy: str,         # 'frozen', 'partial', 'full'
    phase1_lr: float = 1e-3,
    phase1_epochs: int = 8,
    phase2_lr: float = 1e-5,
    phase2_epochs: int = 10,
    use_augment: bool = True,
):
    exp_name = f"EXP-{exp_id:02d}_{model_name}_{spec_type}_{strategy}"
    print(f"\n{'='*70}")
    print(f"[{exp_name}]")
    print(f"{'='*70}")

    dual = (spec_type == 'mel+cqt')
    train_samples, test_samples = load_dataset_samples(spec_type=spec_type)
    print(f"  Train: {len(train_samples)}  Test: {len(test_samples)} (Dual={dual})")

    train_tf, val_tf = get_transforms(augment=use_augment)

    train_ds = SpectrogramDataset(train_samples, transform=train_tf if not dual else None,
                                  dual_channel=dual)
    test_ds  = SpectrogramDataset(test_samples,  transform=val_tf   if not dual else None,
                                  dual_channel=dual)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(DEVICE.type == 'cuda'))
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=(DEVICE.type == 'cuda'))

    in_ch = 2 if dual else 3
    if model_name == 'efficientnet_b0':
        model = build_efficientnet_b0(strategy, in_channels=in_ch)
    elif model_name == 'efficientnet_b2':
        model = build_efficientnet_b2(strategy)
    elif model_name == 'mobilenet_v2':
        model = build_mobilenet_v2(strategy)
    elif model_name == 'resnet50':
        model = build_resnet50(strategy)
    elif model_name == 'vgg16':
        model = build_vgg16(strategy)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    trainable_p = count_trainable(model)
    print(f"  Trainable params (Initial): {trainable_p:,}")
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    spec_aug  = SpecAugment(time_mask_param=30, freq_mask_param=16).to(DEVICE)
    scaler    = torch.cuda.amp.GradScaler() if DEVICE.type == 'cuda' else None

    start_time = time.time()
    best_val_acc = 0.0
    best_model_path = MODEL_DIR / f"{exp_name}_best.pth"
    probs_save_path = MODEL_DIR / f"exp{exp_id:02d}_test_probs.npy"

    # ── Phase 1: Warmup Classifier / Frozen Backbone ───────────────────────
    if phase1_epochs > 0:
        print(f"\n  [Phase 1] {phase1_epochs} epochs, LR={phase1_lr}")
        if strategy in ('partial', 'full'):
            for p in model.parameters():
                p.requires_grad = False
            if model_name in ('efficientnet_b0', 'efficientnet_b2', 'mobilenet_v2'):
                for p in model.classifier.parameters():
                    p.requires_grad = True
            elif model_name == 'resnet50':
                for p in model.fc.parameters():
                    p.requires_grad = True
            elif model_name == 'vgg16':
                for p in model.classifier.parameters():
                    p.requires_grad = True

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=phase1_lr, weight_decay=WEIGHT_DECAY
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=phase1_epochs, eta_min=1e-6)
        patience_cnt = 0

        for ep in range(1, phase1_epochs + 1):
            t_ep = time.time()
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion,
                                              spec_aug, scaler)
            val_acc, val_f1, _, _, _ = evaluate(model, test_loader)
            scheduler.step()
            ep_time = time.time() - t_ep

            print(f"    Ep {ep:2d}/{phase1_epochs} ({ep_time:.1f}s): loss={tr_loss:.4f}  "
                  f"train_acc={tr_acc:.2f}%  val_acc={val_acc:.2f}%  val_f1={val_f1:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= 4:
                    print("    Early stopping (Phase 1).")
                    break

    # ── Phase 2: Fine-tune Unfrozen Backbone ────────────────────────────────
    if phase2_epochs > 0 and strategy in ('partial', 'full'):
        print(f"\n  [Phase 2] {phase2_epochs} epochs, LR={phase2_lr}")
        if strategy == 'partial':
            if model_name == 'efficientnet_b0':
                for p in model.features[5:].parameters():
                    p.requires_grad = True
            elif model_name == 'efficientnet_b2':
                for p in model.features[4:].parameters():
                    p.requires_grad = True
            elif model_name == 'mobilenet_v2':
                for p in model.features[14:].parameters():
                    p.requires_grad = True
            elif model_name == 'resnet50':
                for p in model.layer4.parameters():
                    p.requires_grad = True
            elif model_name == 'vgg16':
                for p in model.features[24:].parameters():
                    p.requires_grad = True
        elif strategy == 'full':
            for p in model.parameters():
                p.requires_grad = True

        trainable_p2 = count_trainable(model)
        print(f"  Trainable params (Phase 2): {trainable_p2:,}")

        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=phase2_lr, weight_decay=WEIGHT_DECAY
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=phase2_epochs, eta_min=1e-7)
        patience_cnt = 0

        for ep in range(1, phase2_epochs + 1):
            t_ep = time.time()
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion,
                                              spec_aug, scaler)
            val_acc, val_f1, _, _, _ = evaluate(model, test_loader)
            scheduler.step()
            ep_time = time.time() - t_ep

            print(f"    Ep {ep:2d}/{phase2_epochs} ({ep_time:.1f}s): loss={tr_loss:.4f}  "
                  f"train_acc={tr_acc:.2f}%  val_acc={val_acc:.2f}%  val_f1={val_f1:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= 4:
                    print("    Early stopping (Phase 2).")
                    break

    elapsed = (time.time() - start_time) / 60.0

    # ── Final Evaluation ────────────────────────────────────────────────────
    print(f"\n  Loading best checkpoint: {best_model_path}")
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    test_acc, test_f1, preds, labels, probs = evaluate(model, test_loader)
    print(f"\n  FINAL: Test Acc={test_acc:.2f}%  Macro F1={test_f1:.2f}%  Time={elapsed:.1f}min")

    # Save prediction probabilities for ensemble
    np.save(probs_save_path, probs)

    # Classification report
    report = classification_report(labels, preds, target_names=GENRES, output_dict=True)
    per_class_f1 = {g: round(report[g]['f1-score'] * 100, 2) for g in GENRES}
    print(f"  Per-class F1: {per_class_f1}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=GENRES, yticklabels=GENRES, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{exp_name}\nTest Acc={test_acc:.2f}% | F1={test_f1:.2f}%')
    plt.tight_layout()
    cm_path = CM_DIR / f"{exp_name}_cm.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()

    result = {
        'exp_id':            exp_id,
        'exp_name':          exp_name,
        'model_name':        model_name,
        'spec_type':         spec_type,
        'strategy':          strategy,
        'test_accuracy':     round(test_acc, 4),
        'macro_f1':          round(test_f1, 4),
        'delta_vs_baseline':  round(test_acc - 65.56, 4),
        'training_time_min': round(elapsed, 2),
        'trainable_params':  count_trainable(model),
        'timestamp':         datetime.now().strftime('%Y-%m-%d %H:%M'),
    }
    result.update({f'f1_{g.replace(" ","_")}': v for g, v in per_class_f1.items()})

    return result, probs


# ─────────────────────────────────────────────────────────────────────────────
# 7. Experiment Definitions
# ─────────────────────────────────────────────────────────────────────────────
EXPERIMENTS = [
    # id  model_name          spec_type   strategy   p1_lr  p1_ep  p2_lr  p2_ep
    ( 1, 'efficientnet_b0',  'mel',      'frozen',  1e-3,  8,     0,     0    ),
    ( 2, 'efficientnet_b0',  'mel',      'partial', 1e-3,  4,     1e-5,  8    ),
    ( 3, 'efficientnet_b0',  'mel',      'full',    1e-3,  3,     5e-6,  8    ),
    ( 4, 'efficientnet_b0',  'cqt',      'partial', 1e-3,  4,     1e-5,  8    ),
    ( 5, 'mobilenet_v2',     'mel',      'frozen',  1e-3,  8,     0,     0    ),
    ( 6, 'mobilenet_v2',     'mel',      'partial', 1e-3,  4,     1e-5,  8    ),
    ( 7, 'mobilenet_v2',     'cqt',      'partial', 1e-3,  4,     1e-5,  8    ),
    ( 8, 'resnet50',         'mel',      'partial', 1e-3,  4,     1e-5,  8    ),
    ( 9, 'resnet50',         'cqt',      'partial', 1e-3,  4,     1e-5,  8    ),
    (10, 'vgg16',            'mel',      'frozen',  1e-3,  8,     0,     0    ),
    (11, 'vgg16',            'mel',      'partial', 1e-3,  4,     1e-5,  6    ),
    (12, 'efficientnet_b2',  'mel',      'partial', 1e-3,  4,     1e-5,  8    ),
    (13, 'efficientnet_b0',  'mel+cqt',  'partial', 1e-3,  4,     1e-5,  8    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# 8. Main Execution
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=0,
                        help='Run only experiment with this id (0=all, 14=ensemble)')
    args = parser.parse_args()

    # Load existing results to avoid re-running
    if RESULTS_CSV.exists():
        df_existing = pd.read_csv(RESULTS_CSV)
        done_ids = set(df_existing['exp_id'].tolist())
    else:
        df_existing = pd.DataFrame()
        done_ids = set()

    all_results = df_existing.to_dict('records') if not df_existing.empty else []

    # Run CNN experiments
    for exp_cfg in EXPERIMENTS:
        exp_id = exp_cfg[0]
        if args.exp and exp_id != args.exp:
            continue
        if exp_id in done_ids:
            print(f"  [EXP-{exp_id:02d}] already recorded in {RESULTS_CSV.name}, skipping.")
            continue

        result, probs = run_experiment(
            exp_id=exp_id,
            model_name=exp_cfg[1],
            spec_type=exp_cfg[2],
            strategy=exp_cfg[3],
            phase1_lr=exp_cfg[4],
            phase1_epochs=exp_cfg[5],
            phase2_lr=exp_cfg[6],
            phase2_epochs=exp_cfg[7],
        )
        all_results.append(result)

        # Save incrementally
        df_res = pd.DataFrame(all_results)
        df_res.to_csv(RESULTS_CSV, index=False)
        print(f"  -> Saved results to {RESULTS_CSV}")
        try:
            update_markdown_log(df_res)
        except Exception as e:
            print(f"  [WARN] Failed to update markdown: {e}")

    # ── Final Summary ────────────────────────────────────────────────────────
    df_final = pd.DataFrame(all_results)
    df_final.to_csv(RESULTS_CSV, index=False)

    print("\n" + "="*80)
    print("FINAL CNN BENCHMARK SUMMARY")
    print("="*80)
    if not df_final.empty:
        summary_cols = ['exp_id', 'model_name', 'spec_type', 'strategy',
                        'test_accuracy', 'macro_f1', 'delta_vs_baseline',
                        'training_time_min']
        avail = [c for c in summary_cols if c in df_final.columns]
        print(df_final[avail].to_string(index=False))
        print(f"\nBaseline (RF Continuous + Gaussian): 65.56% Acc | 65.12% F1")

    print(f"\n-> Results saved to: {RESULTS_CSV}")
    print("Phase 5 DONE.")
