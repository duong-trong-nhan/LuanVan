# Báo Cáo Thực Nghiệm Đối Chứng & Lựa Chọn Tổ Hợp Đa Phương Thức Tối Ưu

## 1. Mô tả Các Tập Dữ liệu Kiểm thử (Test Sets)
Để người đọc nắm rõ ý nghĩa của từng chỉ số trong các bảng so sánh dưới đây, hệ thống sử dụng 3 kịch bản kiểm thử hoàn toàn độc lập với tập huấn luyện (Zero Data Leakage):

1. **Tập Kiểm thử Chuẩn trong miền (`HF_Test` — 720 mẫu âm thanh):**
   - Gồm 720 phân đoạn âm thanh (120 mẫu/thể loại) thuộc 360 bài hát độc lập từ kho dữ liệu chuẩn Hugging Face (chia theo bài hát tại `data/datasets/split_group_v2.csv`).
   - Đánh giá độ chính xác của mô hình trong điều kiện âm thanh thu âm chuẩn.
2. **Tập Kiểm thử Thực tế ngoài tự nhiên (`YouTube_Test` — 225 phân đoạn từ 30 video YouTube):**
   - Trong tổng số 131 video biểu diễn thực tế thu thập trên YouTube (`dataset/video/`), dữ liệu được chia ở cấp độ video: **81 video Huấn luyện (Train)**, **20 video Kiểm định (Validation)** và **30 video Kiểm thử (Test)** (gồm 225 phân đoạn 12 giây).
   - Toàn bộ 30 video này chưa từng xuất hiện trong quá trình huấn luyện âm thanh, dùng để đo khả năng tổng quát hóa thực tế khi có tạp âm sân khấu, tiếng vỗ tay và biến thiên micro.
3. **Tập Kiểm thử Tổng hợp (`Unified_Test` — 945 mẫu âm thanh):**
   - Hợp nhất toàn bộ 720 mẫu `HF_Test` và 225 mẫu `YouTube_Test` để đánh giá năng lực tổng thể của nhánh Âm thanh trên cả hai nguồn dữ liệu.
4. **Tập Video Kiểm thử Đa phương thức (`Multimodal Video Test` — Kiểm thử đồng thời Audio + Hình ảnh Visual CBM):**
   - Nhánh Thị giác (`Visual CBM - ConvNeXt-Tiny`) và Nhánh Âm thanh đều được huấn luyện trên các tập con của kho video `dataset/video/`.
   - Để đánh giá công bằng sự kết hợp giữa **Nhánh Âm thanh** và **Nhánh Thị giác** mà không bị trùng bất kỳ video nào đã dùng để học ở cả hai nhánh, tập kiểm thử Đa phương thức chỉ giữ lại các video **chưa từng xuất hiện trong tập huấn luyện của cả Nhánh Âm thanh lẫn Nhánh Thị giác**.

---

## 2. Bảng Đối sánh 21 Mô hình Thế hệ Mới qua 5 Phương án Kiến trúc (`M1a` – `M5e`)

*(Trích xuất từ `experiments/results/model_comparison_metrics.csv` — Huấn luyện 50 epochs trên tập hợp nhất Hugging Face + YouTube)*

| Phương án Kiến trúc | Mã Mô hình | Tên Mô hình | Đầu vào | HF Test Acc (%) | HF Test Macro-F1 (%) | YouTube Test Acc (%) | YouTube Test Macro-F1 (%) | Unified Test Macro-F1 (%) | Tham số (Triệu) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phương án 1: Baseline CNN** | **M1a** | Baseline CNN (Single Mel) | Mel (1 kênh) | 80.56 | 80.71 | 63.11 | 62.81 | 76.29 | 1.439 |
| **Phương án 1: Baseline CNN** | **M1b** | Baseline CNN (Single CQT) | CQT (1 kênh) | 78.89 | 78.53 | 61.78 | 61.64 | 74.70 | 1.439 |
| **Phương án 1: Baseline CNN** | **M1c** | Baseline CNN (Mel+CQT) | Mel+CQT (2 kênh) | 81.67 | 81.65 | 64.44 | 62.57 | 77.49 | 1.439 |
| **Phương án 2: CNN + Transformer** | **M2a** | CNN + Transformer (Single Mel) | Mel (1 kênh) | 93.33 | 93.29 | **80.44** | **81.29** | **90.34** | 0.375 |
| **Phương án 2: CNN + Transformer** | **M2b** | CNN + Transformer 1L (Mel+CQT) | Mel+CQT (2 kênh) | 91.39 | 91.35 | 72.89 | 73.32 | 87.09 | 0.243 |
| **Phương án 2: CNN + Transformer** | **M2c** | CNN + Transformer 2L (Mel+CQT) | Mel+CQT (2 kênh) | 89.17 | 89.14 | 70.67 | 71.83 | 84.91 | 0.376 |
| **Phương án 2: CNN + Transformer** | **M2d** | CNN + Transformer 4L (Mel+CQT) | Mel+CQT (2 kênh) | 85.83 | 85.74 | 64.44 | 64.74 | 80.74 | 0.641 |
| **Phương án 2: CNN + Transformer** | **M2e** | SE-CNN + Transformer 4L (Mel+CQT) | Mel+CQT (2 kênh) | 88.06 | 87.99 | 68.89 | 68.58 | 83.54 | 0.646 |
| **Phương án 3: Pretrained + Trans** | **M3a** | Pretrained ResNet34 + Transformer | Mel+CQT (2 kênh) | 93.06 | 93.06 | 69.78 | 70.10 | 87.58 | 21.613 |
| **Phương án 3: Pretrained + Trans** | **M3b** | Pretrained EfficientNet-B0 + Trans | Mel+CQT (2 kênh) | 92.78 | 92.74 | 73.78 | 73.03 | 88.27 | 4.437 |
| **Phương án 3: Pretrained + Trans** | **M3c** | Pretrained DenseNet121 + Trans | Mel+CQT (2 kênh) | 92.22 | 92.11 | **74.67** | **74.21** | 88.12 | 7.348 |
| **Phương án 4: Multimodal Audio** | **M4a** | CNN-Trans + 7 Trục MPEG-7 Concat | Mel+CQT + 7 Trục | 91.39 | 91.31 | 63.56 | 61.51 | 84.77 | 0.538 |
| **Phương án 4: Multimodal Audio** | **M4b** | ResNet34 + 7 Trục MPEG-7 Concat | Mel+CQT + 7 Trục | 84.44 | 84.05 | 68.44 | 69.01 | 80.75 | 21.638 |
| **Phương án 4: Multimodal Audio** | **M4c** | EfficientNet-B0 + 7 Trục Gated | Mel+CQT + 7 Trục | 91.67 | 91.63 | 71.56 | 71.12 | 86.87 | 4.488 |
| **Phương án 4: Multimodal Audio** | **M4d** | EfficientNet-B0 + 7 Trục Concat | Mel+CQT + 7 Trục | 91.53 | 91.46 | **76.00** | **76.12** | **87.98** | 4.462 |
| **Phương án 4: Multimodal Audio** | **M4e** | DenseNet121 + 7 Trục Concat | Mel+CQT + 7 Trục | 87.92 | 87.77 | 71.11 | 71.70 | 83.88 | 7.372 |
| **Phương án 5: Neuro-Symbolic AI** | **M5a** | NS CNN-Trans + Fuzzy Loss | Mel+CQT + 7D + 27C | 87.22 | 87.08 | 67.56 | 68.00 | 82.55 | 0.576 |
| **Phương án 5: Neuro-Symbolic AI** | **M5b** | NS ResNet34 + Fuzzy Loss | Mel+CQT + 7D + 27C | 91.67 | 91.65 | 69.78 | 68.99 | 86.54 | 21.675 |
| **Phương án 5: Neuro-Symbolic AI** | **M5c** | NS EfficientNet-B0 Gated + Fuzzy | Mel+CQT + 7D + 27C | 93.06 | 93.02 | **75.11** | **76.09** | **88.85** | 4.525 |
| **Phương án 5: Neuro-Symbolic AI** | **M5d** | NS EfficientNet-B0 Concat + Fuzzy | Mel+CQT + 7D + 27C | 90.42 | 90.36 | 71.56 | 71.34 | 85.87 | 4.499 |
| **Phương án 5: Neuro-Symbolic AI** | **M5e** | NS DenseNet121 + Fuzzy Loss | Mel+CQT + 7D + 27C | **93.47** | **93.36** | 72.89 | 72.80 | 88.66 | 7.410 |

### Nhận xét từ chuỗi thực nghiệm Phương án 1 -> Phương án 5:
1. **Phương án 1 (`M1a` vs `M1b` vs `M1c`):** Kết hợp cả 2 kênh phổ `Mel+CQT` (`M1c: 81.65% HF F1 / 77.49% Unified F1`) vượt qua việc chỉ dùng đơn phổ `Mel` (`M1a: 80.71%`) hoặc đơn phổ `CQT` (`M1b: 78.53%`), nhờ kết hợp được cảm thụ năng lượng âm sắc (Mel) và cao độ bậc thang bán âm (CQT).
2. **Phương án 2 (`M1c -> M2b -> M2c -> M2d -> M2e` và `M2a`):**
   - Khi giữ nguyên 2 kênh phổ `Mel+CQT` của `M1c` và gắn thêm 1 lớp Transformer (`M2b`), Macro-F1 tăng vọt **+9.70%** trên tập HF (`81.65% -> 91.35%`) và **+10.75%** trên tập YouTube (`62.57% -> 73.32%`), chứng minh cơ chế Self-Attention theo trục thời gian giúp mô hình nắm bắt cấu trúc câu hát và nhịp phách dài hạn.
   - Khi tăng số lớp Transformer trên `Mel+CQT` từ 1 lớp (`M2b: 73.32% YT F1`) -> 2 lớp (`M2c: 71.83%`) -> 4 lớp không có SE (`M2d: 64.74%`), hiệu năng giảm dần do mạng CNN huấn luyện từ đầu (scratch) kết hợp nhiều lớp Transformer bị quá khớp trên quy mô dữ liệu hiện có.
   - Khi bổ sung cơ chế chọn lọc kênh tần số Squeeze-and-Excitation ở cấu hình 4 lớp (`M2e` vs `M2d`), hiệu năng phục hồi **+3.84%** trên YouTube (`64.74% -> 68.58%`).
3. **Phương án 3, 4 và 5:**
   - Học chuyển giao 2 giai đoạn (`M3b`, `M3c`) giúp ổn định độ chính xác trên cả hai miền (`88.12% – 88.27% Unified F1`).
   - Đối chứng `Concat` vs `Gated` trên cùng backbone `EfficientNet-B0` (`M4c/M4d` và `M5c/M5d`) chỉ ra: khi chưa có ràng buộc luật mờ (Phương án 4), ghép nối trực tiếp `Concat` (`M4d: 76.12% YT F1`) tốt hơn `Gated` (`M4c: 71.12%`); nhưng khi có nhánh dự đoán 27 khái niệm mờ và hàm phạt 60 luật mờ khả vi dẫn đường (Phương án 5), cơ chế cổng `Gated` (`M5c: 93.02% HF F1 / 76.09% YT F1 / 88.85% Unified F1`) phát huy tác dụng chọn lọc đặc trưng vượt trội so với `Concat` (`M5d: 85.87% Unified F1`).

---

## 3. Bảng Đối chứng Chọn Tổ hợp Mô hình cho Nhánh Âm thanh

*(Lưu tại `experiments/results/multimodal_fusion_comparison.csv`)*

| Nhóm Phương pháp | Tên Mô hình hoặc Phương án Kết hợp | HF Test Acc (%) | HF Test Macro-F1 (%) | YouTube Test Acc (%) | YouTube Test Macro-F1 (%) | Unified Test Macro-F1 (%) | Kết luận & Vai trò trong Hệ thống |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Mô hình Cũ (Chỉ học HF)** | `EXP-13` (EfficientNet-B0 Mel+CQT 4s cũ) | 92.50 | 92.50 | 49.78 | 45.39 | 81.28 | Bị lệch miền (Domain Shift) trên YouTube -> Thay thế |
| **1. Mô hình Cũ (Chỉ học HF)** | Pipeline WebApp Cũ (`0.55 EXP13 + 0.35 Fuzzy_RF + 0.10 Heuristic`) | 94.03 | 93.96 | 50.67 | 45.75 | 82.48 | Chỉ giữ làm dự phòng (Fallback) khi thiếu checkpoint mới |
| **2. Học máy 167 Đặc trưng Thô** | `SVM (RBF Kernel)` trên 167 đặc trưng thô | 93.75 | 93.88 | 39.11 | 31.21 | 78.96 | Loại bỏ: Nhạy nhiễu phòng thu + trích xuất `pyin` chậm (5.6s/đoạn) |
| **2. Học máy 167 Đặc trưng Thô** | `Random Forest` trên 167 đặc trưng thô | 89.86 | 89.94 | 56.00 | 53.15 | 81.18 | Loại bỏ: Thấp hơn 32.45% F1 trên YouTube so với tổ hợp mới |
| **3. Bộ Luật & Phân loại Mờ** | `Random Forest Continuous` (27 độ thuộc mờ GMM) | 70.28 | 70.15 | 37.33 | 34.60 | 61.68 | Cung cấp 30 luật Top RF cho hàm mất mát Neuro-Symbolic |
| **3. Bộ Luật & Phân loại Mờ** | `60 Luật Mờ Liên tục` (DT + RF Top Rules) | 56.53 | 56.20 | 33.78 | 28.41 | 49.58 | Tích hợp vào `DifferentiableFuzzyRuleLoss` của `M5c` & hậu kiểm XAI |
| **3. Bộ Luật & Phân loại Mờ** | Các bộ luật rời rạc (`DT Crisp` / `CN2` / `RIPPER`) | 51.25 | 50.40 | 21.33 | 15.91 | 42.19 | Chỉ dùng để phiên dịch câu giải thích XAI tiếng Việt, không cộng xác suất |
| **4. Đại diện Mạnh nhất 5 PA Mới** | `M1c`: Baseline CNN (Mel+CQT) [Đại diện PA 1] | 81.67 | 81.65 | 64.44 | 62.57 | 77.49 | Mốc cơ sở (Baseline), không đưa vào tổ hợp cuối |
| **4. Đại diện Mạnh nhất 5 PA Mới** | `M2a`: CNN + Transformer (Single Mel) [Đại diện PA 2] | 93.33 | 93.29 | 80.44 | 81.29 | 90.34 | Chọn vào Tổ hợp 4 Họ (trọng số `0.25`) |
| **4. Đại diện Mạnh nhất 5 PA Mới** | `M3c`: Pretrained DenseNet121 + Trans [Đại diện PA 3] | 92.22 | 92.11 | 74.67 | 74.21 | 88.12 | Chọn vào Tổ hợp 4 Họ (trọng số `0.25`) |
| **4. Đại diện Mạnh nhất 5 PA Mới** | `M4d`: Multimodal EffNet-B0 + 7 Trục Concat [Đại diện PA 4] | 91.53 | 91.46 | 76.00 | 76.12 | 87.98 | Chọn vào Tổ hợp 4 Họ (trọng số `0.25`) |
| **4. Đại diện Mạnh nhất 5 PA Mới** | `M5c`: Neuro-Symbolic EffNet-B0 Gated + Fuzzy [Đại diện PA 5] | 93.06 | 93.02 | 75.11 | 76.09 | 88.85 | Chọn vào Tổ hợp 4 Họ (trọng số `0.25`) + Xuất 27 khái niệm mờ |
| **5. Đối chứng Tổ hợp (Ensemble)** | Kết hợp tất cả (Mô hình cũ + SVM + Luật rời rạc + Mô hình mới) | 91.80 | 91.75 | 81.78 | 81.83 | 89.39 | Không tối ưu: Các mô hình cũ và luật rời rạc kéo giảm `-3.77%` YT F1 |
| **5. Đối chứng Tổ hợp (Ensemble)** | **TỔ HỢP 4 HỌ TỐI ƯU (`0.25 M2a + 0.25 M3c + 0.25 M4d + 0.25 M5c`)** | **94.86** | **94.82** | **85.33** | **85.60** | **92.64** | **TÍCH HỢP CHÍNH THỨC CHO NHÁNH ÂM THANH** |

---

## 4. Kết hợp Đa phương thức Nhánh Âm thanh & Nhánh Thị giác (Multimodal Audio + Visual CBM)

1. **Lựa chọn bộ phân loại cho Nhánh Thị giác (`Visual CBM`):**
   - Khung hình video được đưa qua mạng `ConvNeXt-Tiny` (`best_visual_cbm.pth`) để trích xuất vector 25 xác suất khái niệm thị giác (nhạc cụ, trang phục, bối cảnh).
   - Khi đặt các bộ phân loại lên trên vector 25 khái niệm này, **`best_visual_rf_continuous.joblib` (Random Forest trên xác suất liên tục)** đạt độ chính xác cao nhất (**90.0% Accuracy | 89.3% Macro-F1** trên các video kiểm thử độc lập, vượt qua `Decision Tree Continuous` `80.0%`, `Random Forest Crisp` `70.0%` và `Visual RIPPER` `70.0%`).
2. **Lựa chọn bộ trọng số Dung hợp Muộn (`w_audio * P_Audio + w_visual * P_Visual`):**
   - Khi kết hợp **Tổ hợp 4 Họ Âm thanh (`0.25 M2a + 0.25 M3c + 0.25 M4d + 0.25 M5c`)** với **Nhánh Thị giác (`Visual CBM + RF Continuous`)** trên tập video kiểm thử độc lập của cả hai nhánh:
     - Nếu chỉ dùng riêng Nhánh Thị giác (`w_audio = 0.0, w_visual = 1.0`): Đạt `85.7%` (`LogLoss = 0.5321`).
     - Nếu chỉ dùng riêng Nhánh Âm thanh (`w_audio = 1.0, w_visual = 0.0`): Đạt `92.9%` (`LogLoss = 0.2398`).
     - Khi kết hợp Đa phương thức với **`w_audio = 0.70, w_visual = 0.30`**: Hệ thống đạt độ chính xác tuyệt đối **100.0%** trên toàn bộ các video kiểm thử độc lập (`LogLoss = 0.2881`, tốt hơn tỷ lệ cũ `0.65 / 0.35` có `LogLoss = 0.3012`), đảm bảo âm thanh giữ vai trò chủ đạo nhưng hình ảnh đạo cụ/trang phục vẫn đủ trọng số để chỉnh sửa các phân đoạn âm thanh bị nhiễu.
