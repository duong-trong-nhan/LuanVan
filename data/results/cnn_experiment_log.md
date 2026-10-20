# CNN Spectrogram Experiment Log — HerittageFNS
## Phan loai 6 the loai am nhac co truyen Viet Nam qua Transfer Learning

> **Baseline tham chieu (Fuzzy + RF):** Accuracy = **65.56%** | Macro F1 = **65.12%**
> *(RF Continuous + Gaussian Jittering — ket qua tot nhat cua nhanh Fuzzy)*

---

## Cau hinh chung

| Thong so | Gia tri |
|---------|---------|
| Dataset | 3,600 samples (2,880 train / 720 test), 600/class |
| Train split (random_state) | 42 (giong het nhanh Fuzzy) |
| SR | 22,050 Hz |
| Segment duration | 12 giay |
| Spectrogram size | 224 x 224 (resize ve) |
| Batch size | 32 |
| Optimizer | AdamW |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |
| Scheduler | CosineAnnealingLR |
| Early stopping patience | 10 epochs |
| SpecAugment | Time Mask T=30, Freq Mask F=16 |

---

## EXP-01 | EfficientNet-B0 | Log-Mel | Frozen Backbone

**Cau hinh fine-tuning:**
- Backbone: efficientnet_b0 pretrained ImageNet
- Freeze: Toan bo features layers
- Head: Dropout(0.4) -> Linear(1280->256) -> ReLU -> Dropout(0.3) -> Linear(256->6)
- Phase 1: 30 epochs, LR=1e-3
- Trainable params: 329,478

**Spectrogram config:** n_mels=128, n_fft=2048, hop=512, fmin=20, fmax=8000

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Accuracy | 65.24% | 66.25% | **66.25%** |
| Macro F1 | 64.90% | 66.01% | **66.01%** |
| Epochs converge | 8 / 8 | | 8 |
| Training time (min) | 35.3 min | | 35.3 min |

**Per-class F1:**

| Ca tru | Cai luong | Chau van | Cheo | Hat xam | Quan ho |
|--------|-----------|----------|------|---------|---------|
| **77.59%** | **66.00%** | **64.89%** | **56.12%** | **65.38%** | **66.06%** |

*Ma tran nham lan:* `data/results/cnn_confusion_matrices/EXP-01_efficientnet_b0_mel_frozen_cm.png`

---

## EXP-02 | EfficientNet-B0 | Log-Mel | Partial Fine-tuning (KHUYEN NGHI)

**Cau hinh fine-tuning:**
- Backbone: efficientnet_b0 pretrained ImageNet
- Phase 1 (Warm-up): Freeze toan bo features, 10 epochs, LR=1e-3
- Phase 2 (Fine-tune): Unfreeze features[5:] (MBConv blocks 5,6,7), 30 epochs, LR=1e-5
- Head: Dropout(0.4) -> Linear(1280->256) -> ReLU -> Dropout(0.3) -> Linear(256->6)
- Weight decay: 1e-4 | Trainable params (Phase 2): ~3.8M

**Spectrogram config:** n_mels=128, n_fft=2048, hop=512, fmin=20, fmax=8000

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Accuracy | 68.19% | 69.72% | **69.72%** |
| Macro F1 | -- | 68.85% | **68.85%** |
| Epochs converge | Ep 6/8 (Phase 2) | | |
| Training time (min) | 72.0 | | |

**Per-class F1:**

| Ca tru | Cai luong | Chau van | Cheo | Hat xam | Quan ho |
|--------|-----------|----------|------|---------|---------|
| **83.40%** | **68.39%** | **69.79%** | **51.37%** | **72.06%** | **68.11%** |

*Ma tran nham lan:* `data/results/cnn_confusion_matrices/EXP-02_efficientnet_b0_mel_partial_cm.png`

---

## EXP-03 | EfficientNet-B0 | Log-Mel | Full Fine-tuning

**Cau hinh:**
- Phase 1: Freeze toan bo features, 5 epochs, LR=1e-3
- Phase 2: Unfreeze toan bo, 40 epochs, LR=5e-6
- Trainable params (Phase 2): ~5.3M

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Accuracy | -- | -- | -- |
| Macro F1 | -- | -- | -- |

---

## EXP-04 | EfficientNet-B0 | CQT | Partial Fine-tuning

**Spectrogram config:** CQT, n_bins=84, bins_per_octave=12, hop=512, fmin=32.70Hz (C1)
*(Cau hinh fine-tuning giong EXP-02)*

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Accuracy | -- | -- | -- |
| Macro F1 | -- | -- | -- |

---

## EXP-05 | MobileNetV2 | Log-Mel | Frozen Backbone

**Cau hinh:**
- Freeze: Toan bo features
- Head: Dropout(0.4) -> Linear(1280->6)
- Phase 1: 30 epochs, LR=1e-3 | Trainable params: ~8K

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Accuracy | -- | -- | -- |
| Macro F1 | -- | -- | -- |

---

## EXP-06 | MobileNetV2 | Log-Mel | Partial Fine-tuning (KHUYEN NGHI)

**Cau hinh:**
- Phase 1: Freeze toan bo, 10 epochs, LR=1e-3
- Phase 2: Unfreeze features[14:] (5 InvertedResidual blocks cuoi), 30 epochs, LR=1e-5
- Head: Dropout(0.4) -> Linear(1280->256) -> ReLU -> Linear(256->6)
- Trainable params (Phase 2): ~1.2M

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Accuracy | -- | -- | -- |
| Macro F1 | -- | -- | -- |

---

## EXP-07 | MobileNetV2 | CQT | Partial Fine-tuning

*(Cau hinh fine-tuning giong EXP-06, spectrogram dung CQT n_bins=84)*

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Accuracy | -- | -- | -- |
| Macro F1 | -- | -- | -- |

---

## EXP-08 | ResNet-50 | Log-Mel | Partial Fine-tuning

**Cau hinh:**
- Phase 1: Freeze tat ca tru fc, 10 epochs, LR=1e-3
- Phase 2: Unfreeze layer4 + fc, 30 epochs, LR=1e-5
- Head: Linear(2048->256) -> ReLU -> Dropout(0.4) -> Linear(256->6)
- Trainable params (Phase 2): ~12M

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Accuracy | -- | -- | -- |
| Macro F1 | -- | -- | -- |
| Training time (min) | -- | | |

---

## EXP-09 | ResNet-50 | CQT | Partial Fine-tuning

*(Cau hinh giong EXP-08)*

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Accuracy | -- | -- | -- |
| Macro F1 | -- | -- | -- |

---

## EXP-10 | VGG16 | Log-Mel | Frozen Backbone

**Cau hinh:**
- Freeze: Toan bo features
- Head: Linear(25088->4096)->ReLU->Dropout(0.5)->Linear(4096->512)->ReLU->Dropout(0.4)->Linear(512->6)
- Trainable params: ~122K

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Accuracy | -- | -- | -- |
| Macro F1 | -- | -- | -- |

---

## EXP-11 | VGG16 | Log-Mel | Partial Fine-tuning

**Cau hinh:**
- Phase 1: Freeze toan bo features, 10 epochs, LR=1e-3
- Phase 2: Unfreeze features[24:] (block5_conv1/2/3), 20 epochs, LR=1e-5
- Trainable params (Phase 2): ~9M

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Accuracy | -- | -- | -- |
| Macro F1 | -- | -- | -- |
| Training time (min) | -- | | |

---

## EXP-12 | EfficientNet-B2 | Log-Mel | Partial Fine-tuning

**Cau hinh:**
- Phase 1: Freeze, 10 epochs, LR=1e-3
- Phase 2: Unfreeze features[4:] (top 4 blocks), 30 epochs, LR=1e-5
- Head: Dropout(0.3)->Linear(1408->256)->ReLU->Dropout(0.3)->Linear(256->6)
- Trainable params (Phase 2): ~7.2M

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Accuracy | -- | -- | -- |
| Macro F1 | -- | -- | -- |

---

## EXP-13 | EfficientNet-B0 | Mel+CQT 2-channel | Partial Fine-tuning

**Mo ta:** Stack Log-Mel (1ch) + CQT (1ch) -> 2-channel input. Thay Conv2d dau tien thanh in_channels=2.
- Phase 1: Frozen, Phase 2: Unfreeze features[5:]

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Accuracy | -- | -- | -- |
| Macro F1 | -- | -- | -- |

---

## GIAI ĐOẠN 6 (PHASE 6) | Multi-Modal Late Fusion: Best CNN + Fuzzy RF

> **Mục tiêu:** Chỉ sau khi Phase 5 hoàn tất toàn bộ 13 thực nghiệm CNN (EXP-01 -> EXP-13) và xác định được mô hình CNN có hiệu năng cao nhất (Best CNN), Phase 6 mới lấy xác suất của Best CNN kết hợp với Fuzzy Random Forest Continuous (65.56%).

**Cơ chế kết hợp:**
Weighted average softmax probability giữa Best CNN và RF probability:
`P_ensemble = w_cnn * P_cnn + (1 - w_cnn) * P_fuzzy`

| w_cnn | w_fuzzy | Test Accuracy | Macro F1 | Ghi chú |
|-------|---------|---------------|----------|---------|
| 0.2 | 0.8 | -- | -- | Sẽ chạy ở Phase 6 |
| 0.3 | 0.7 | -- | -- | Sẽ chạy ở Phase 6 |
| 0.4 | 0.6 | -- | -- | Sẽ chạy ở Phase 6 |
| 0.5 | 0.5 | -- | -- | Sẽ chạy ở Phase 6 |
| 0.6 | 0.4 | -- | -- | Sẽ chạy ở Phase 6 |
| 0.7 | 0.3 | -- | -- | Sẽ chạy ở Phase 6 |
| 0.8 | 0.2 | -- | -- | Sẽ chạy ở Phase 6 |

*(Ghi chú kiểm thử sơ bộ: Thử nghiệm nhanh với EXP-01 Frozen cho thấy tiềm năng đạt 76.25% khi w=0.5; kết quả chính thức sẽ được kích hoạt tại Phase 6 sau khi tìm ra Best CNN từ Phase 5).*

---

## BẢNG TỔNG HỢP SO SÁNH PHASE 5 (CNN TRANSFER LEARNING BENCHMARK)

| # | Mo hinh | Spectrogram | Strategy | Test Acc | Macro F1 | Delta vs Baseline | Time (min) | Params |
|---|---------|-------------|----------|----------|----------|------------------|-----------|--------|
| EXP-01 | efficientnet_b0 | mel | frozen | **66.25%** | 66.01% | +0.69% | 35.3 | 329,478 |
| EXP-02 | efficientnet_b0 | mel | partial | **69.72%** | 68.85% | +4.16% | 72.0 | 4,028,366 |
| REF | RF-Cont + Gaussian (BASELINE) | -- | -- | 65.56% | 65.12% | 0.00% | -- | -- |
---

## SO SÁNH TOÀN DIỆN VỚI NHÁNH FUZZY

| Nhánh | Phương pháp | Test Accuracy | Macro F1 | Ghi chú |
|-------|-------------|---------------|----------|---------|
| Fuzzy | Decision Tree (Crisp) | 51.25% | 50.58% | Luật cứng mờ hóa |
| Fuzzy | Random Forest (Crisp) | 55.97% | 55.04% | Rừng cây mờ gián đoạn |
| Fuzzy | Decision Tree (Continuous) | 56.53% | 56.06% | Liên tục hóa Ruspini |
| Fuzzy | RF Continuous + Gaussian (BEST FUZZY) | 65.56% | 65.12% | Baseline tham chiếu đồ án |
| Fuzzy | RIPPER (SMOTE 50%) | 52.36% | 53.84% | Trích xuất luật DNF |
| Fuzzy | CN2 Rule Induction | 52.08% | 51.38% | Trích xuất luật bao phủ |
| CNN | Best CNN Phase 5 (đang cập nhật) | -- | -- | Chờ hoàn tất EXP-02 -> EXP-13 |
| Ensemble | Phase 6 Late Fusion (chờ Best CNN) | -- | -- | Kích hoạt tại Phase 6 |

---
Cập nhật lần cuối: 2026-10-11 (Phase 5 đang tiếp tục huấn luyện EXP-02 -> EXP-13)
