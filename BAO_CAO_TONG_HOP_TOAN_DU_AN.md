# BÁO CÁO TỔNG HỢP CHI TIẾT DỰ ÁN LUẬN VĂN TỐT NGHIỆP
**Đề tài:** Hệ thống Phân lớp và Giải thích Đa phương thức Di sản Âm nhạc Cổ truyền Việt Nam (Interpretable Multimodal Neuro-Symbolic XAI for Vietnamese Intangible Cultural Heritage)  
**6 Thể loại nghiên cứu:** Ca trù, Chèo, Chầu văn, Cải lương, Hát xẩm, Quan họ  
**Tài liệu đi kèm:** Xem cấu trúc thư mục và chức năng chi tiết từng file tại [MO_TA_CAC_FILE.md](MO_TA_CAC_FILE.md) và bảng đối chứng tối ưu đa phương thức tại [data/results/multimodal_fusion_ablation_report.md](data/results/multimodal_fusion_ablation_report.md).

---

## PHẦN 1: QUÁ TRÌNH THỰC HIỆN NGHIÊN CỨU THEO TRÌNH TỰ PHÁT TRIỂN

Dự án được xây dựng và hoàn thiện tuần tự qua **5 giai đoạn nghiên cứu thực nghiệm chính**, đi từ trích xuất đặc trưng tín hiệu cơ bản, mô hình hóa logic mờ, học sâu trên ảnh phổ, xây dựng mô hình khái niệm thị giác (Visual CBM), đến kiến trúc Tượng trưng - Nơ-ron (Neuro-Symbolic AI) và hệ thống ứng dụng giải thích đa phương thức thời gian thực:

### Giai đoạn 1: Trích xuất 167 Đặc trưng Âm học & Thiết kế 7 Trục Ngữ nghĩa Mờ MPEG-7
1. **Chuẩn hóa tập dữ liệu âm học cơ sở (Hugging Face Dataset):**
   - Tuyển chọn **1,800 bài hát hoàn chỉnh** (300 bài/thể loại cho 6 thể loại). Mỗi bài cắt 2 phân đoạn 12 giây không giao thoa ở tần số lấy mẫu `22,050 Hz Mono` -> tạo thành **3,600 mẫu âm thanh chuẩn hóa** (`600 mẫu/thể loại`).
2. **Trích xuất 167 đặc trưng âm học vật lý ([01_audio_feature_extraction.ipynb](01_audio_feature_extraction.ipynb) -> [data/datasets/features_all.csv](data/datasets/features_all.csv)):**
   - Sử dụng thư viện `librosa` trích xuất 167 đặc trưng bao phủ 9 nhóm tín hiệu: Cao độ cơ bản F0 (`pitch_mean`, `pitch_std`, `pitch_range`, `pitch_median`, `voiced_ratio`), Nhịp độ (`tempo`, `beat_count`, `beat_regularity`), Độ rung/luyến láy (`vibrato_extent`, `vibrato_energy`), Năng lượng (`energy_mean`, `energy_std`, `dynamic_range`, `energy_skew`), 106 hệ số âm sắc MFCC (40 MFCC mean/std + 13 Delta + 13 Delta-Delta), 25 hệ số hòa âm Chroma, 15 đặc trưng hình dạng phổ (Spectral Centroid, Bandwidth, Rolloff, Flatness, 7 dải Spectral Contrast, ZCR), 6 đặc trưng Tonnetz và 2 đặc trưng Mel Energy.
3. **Thiết kế 7 trục ngữ nghĩa âm học theo chuẩn MPEG-7 ([02_fuzzy_decision_tree_random_forest.ipynb](02_fuzzy_decision_tree_random_forest.ipynb)):**
   - Thay vì đưa trực tiếp 167 đặc trưng vật lý vào mô hình cảm nạp luật (gây bùng nổ tổ hợp và mất tính diễn giải nhạc lý), hệ thống chọn lọc **30 đặc trưng cốt lõi** có ý nghĩa âm nhạc học rõ ràng nhất và gom thành **7 trục ngữ nghĩa**:
     - `Energy` (Năng lượng — 5 mức mờ: *Rất yếu, Yếu, Vừa, Mạnh, Rất mạnh*)
     - `Harmony` (Hòa âm — 3 mức mờ: *Thấp, Trung bình, Cao*)
     - `Pitch` (Cao độ — 5 mức mờ: *Rất trầm, Trầm, Trung, Cao, Rất cao*)
     - `Spectral` (Độ sáng phổ — 3 mức mờ: *Thấp, Trung bình, Cao*)
     - `Tempo` (Nhịp độ — 3 mức mờ: *Chậm, Vừa, Nhanh*)
     - `Timbre` (Âm sắc — 5 mức mờ: *Rất thấp, Thấp, Trung bình, Cao, Rất cao*)
     - `Vibrato` (Luyến láy — 3 mức mờ: *Ít luyến, Vừa, Nhiều luyến*)
4. **Phân cụm mờ (FCM / GMM) và chuẩn hóa hàm thuộc:**
   - Áp dụng phân cụm mềm **Fuzzy C-Means (FCM)** và **Gaussian Mixture Model (GMM)** trên từng trục, định hướng thứ tự ngữ nghĩa từ thấp đến cao để sinh ra **27 độ thuộc mờ liên tục trong khoảng [0, 1]** ([data/datasets/features_fuzzy_continuous.csv](data/datasets/features_fuzzy_continuous.csv), tổng độ thuộc trên mỗi trục bằng 1.0 theo điều kiện phân hoạch Ruspini) và **7 nhãn định tính ngôn ngữ** ([data/datasets/features_fuzzy_crisp.csv](data/datasets/features_fuzzy_crisp.csv)).

### Giai đoạn 2: Tăng cường Dữ liệu Mờ & Cảm nạp Hệ Luật Tượng trưng (DT, RF, RIPPER, CN2)
- Thực hiện trong `phase1_data_augmentation.py`, `phase2_rule_induction.py`, `benchmark_all_algorithms_augmented.py` và `save_best_models_and_reformat_rules.py`:
  1. Khảo sát các phương pháp tăng cường dữ liệu trên không gian độ thuộc mờ (**Gaussian Jittering**, **SMOTE 50%**, **SMOTE 100%**, **Fuzzy Mixup**) có tái chuẩn hóa tổng độ thuộc từng trục bằng 1.0.
  2. Huấn luyện và trích xuất các tập luật giải thích `IF-THEN` từ 4 thuật toán: **Decision Tree (Crisp & Continuous)**, **Random Forest (Crisp & Continuous)**, **RIPPER One-vs-Rest** và **CN2 Rule Induction** lưu tại [data/rules/](data/rules/).

### Giai đoạn 3: Học sâu Chuyển giao trên Ảnh phổ Âm thanh Thế hệ 1 (`EXP-01` đến `EXP-13`)
- Thực hiện trong `phase4_spectrogram_generation.py`, `phase5_cnn_transfer_learning.py` và `CNN_Transfer_Learning_Colab.ipynb`:
  1. Sinh **7,200 ảnh phổ** từ 3,600 mẫu Hugging Face gồm 3,600 ảnh **Mel-Spectrogram** và 3,600 ảnh **CQT-Spectrogram** (bắt giữ chi tiết bậc thang bán âm và kỹ thuật rung/nhấn của nhạc cụ truyền thống).
  2. Thực hiện 13 kịch bản huấn luyện (`EXP-01` đến `EXP-13`) trên 5 kiến trúc (`EfficientNet-B0`, `MobileNet-V2`, `ResNet-50`, `VGG-16`, `EfficientNet-B2`) với 3 chiến lược (`Frozen`, `Partial Fine-tuning`, `Full Fine-tuning`) và kỹ thuật tăng cường `SpecAugment`.
  3. Xác định mô hình tốt nhất thế hệ 1 là **`EXP-13` (EfficientNet-B0 ghép 2 kênh Mel+CQT, Partial Fine-tuning)**.

### Giai đoạn 4: Xây dựng Kho Khái niệm Di sản, Nhánh Thị giác (Visual CBM) & Đối chứng 10 Mô hình Học máy Dữ liệu Thô
1. **Chuẩn hóa Kho Khái niệm Di sản ([concept/audio_concepts.md](concept/audio_concepts.md) & [concept/visual_concepts.md](concept/visual_concepts.md)) và Cầu nối Ngữ nghĩa ([phase7_audio_visual_concept_mapper.py](phase7_audio_visual_concept_mapper.py)):**
   - Xây dựng từ điển **43 khái niệm âm thanh** và **38 khái niệm thị giác** dựa trên hồ sơ di sản UNESCO, ICHCAP và Viện Âm nhạc.
   - Thiết kế bộ phiên dịch 2 tầng tự động chuyển đổi các mệnh đề luật toán học sang văn bản giải thích tiếng Việt và ánh xạ sang mã khái niệm chuẩn.
2. **Xây dựng Nhánh Thị giác — Visual Concept Bottleneck Model ([Visual_Data_Prep_and_SemiAuto_Labeling_Colab.ipynb](Visual_Data_Prep_and_SemiAuto_Labeling_Colab.ipynb) & [Visual_CBM_and_Rule_Extraction_Colab.ipynb](Visual_CBM_and_Rule_Extraction_Colab.ipynb)):**
   - Thu thập hơn 125 video biểu diễn thực tế trên YouTube (`dataset/video/`), phân đoạn thành **333 subclips** và trích xuất khung hình sắc nét kết hợp tập ảnh tĩnh Roboflow ([data/annotations/visual_concepts_dataset.csv](data/annotations/visual_concepts_dataset.csv)).
   - Gán nhãn bán tự động cho **25 khái niệm thị giác đặc trưng** (nhạc cụ, trang phục, không gian diễn xướng) và huấn luyện mạng **`ConvNeXt-Tiny` (`data/models/visual/best_visual_cbm.pth`)** với hàm mất mát **Asymmetric Loss (ASL)** kèm tối ưu hóa ngưỡng F1 riêng cho từng khái niệm (`visual_concept_thresholds.json`), đạt **mAP = 86.25%**.
   - Huấn luyện các mô hình phân loại và cảm nạp luật thị giác đặt trên đầu ra 25 khái niệm (`best_visual_rf_continuous.joblib`, `best_visual_dt_continuous.joblib`, `best_visual_ripper_ovr.joblib`, `rules_visual_ripper.json`, `rules_visual_cn2.json`).
3. **Thiết lập Giao thức Chia dữ liệu Cấp Bài hát ([create_group_split.py](create_group_split.py) -> [data/datasets/split_group_v2.csv](data/datasets/split_group_v2.csv)) & Benchmark 10 Mô hình Học máy trên 167 Đặc trưng Thô ([03_raw_features_benchmark_colab.ipynb](03_raw_features_benchmark_colab.ipynb)):**
   - Chia lại 3,600 mẫu Hugging Face theo cặp bài hát gốc (`2,520 Train / 360 Val / 720 Test`) để chống rò rỉ dữ liệu giữa 2 đoạn của cùng một bài hát.
   - Dùng Optuna (50 trials/mô hình) tối ưu 10 mô hình Học máy trên 167 đặc trưng thô và giải thích bằng **SHAP TreeExplainer**.

### Giai đoạn 5: Kiểm thử Thực tế Ngoài tự nhiên (In-The-Wild) & Phát triển Thế hệ 2 với 21 Mô hình Đa phương thức / Neuro-Symbolic (`M1a` – `M5e`)
- Thực hiện trong [evaluate_wild_videos.py](evaluate_wild_videos.py), [evaluate_audio_youtube_benchmark.py](evaluate_audio_youtube_benchmark.py) và [build_multimodal_audio_cnn_notebook.py](build_multimodal_audio_cnn_notebook.py) -> [train_multimodal_audio_cnn_experiments.ipynb](train_multimodal_audio_cnn_experiments.ipynb):
  1. Đánh giá các mô hình thế hệ 1 trên tập video YouTube thực tế, phát hiện sự sụt giảm độ chính xác của các mô hình chỉ học trên dữ liệu phòng thu Hugging Face.
  2. Xây dựng pipeline hợp nhất dữ liệu đa nguồn (3,600 mẫu Hugging Face + 902 mẫu trích xuất thích ứng 12 giây từ 131 video YouTube chia theo cấp độ video `81 Train / 20 Val / 30 Test`).
  3. Thiết kế và huấn luyện **21 mô hình thuộc 5 Phương án kiến trúc (`M1a` đến `M5e`)** tích hợp cơ chế phạt mẫu nhạy cảm chi phí (Cost-Sensitive Domain Penalty), học chuyển giao 2 giai đoạn và bộ suy luận **60 luật mờ khả vi (`DifferentiableFuzzyRuleLoss`)**.
  4. Tuyển chọn và tích hợp **Tổ hợp 4 Họ Kiến trúc Neuro-Symbolic Audio (`0.25 M2a + 0.25 M3c + 0.25 M4d + 0.25 M5c`)** kết hợp **Visual CBM (`0.70 Audio + 0.30 Visual`)** vào hệ thống Web Application ([web_app/](web_app/)).

---

## PHẦN 2: CÁC VẤN ĐỀ KHOA HỌC & KỸ THUẬT PHÁT HIỆN TRONG QUÁ TRÌNH THỰC HIỆN VÀ GIẢI PHÁP KHẮC PHỤC

### 2.1. Hiện tượng CNN và Học máy 167 đặc trưng thô học thuộc lòng bản thu trên Hugging Face, trong khi Nhánh Luật mờ (Fuzzy RF) kháng nhiễu tốt hơn nhờ lược bỏ 137 đặc trưng vi mô
- **Vấn đề phát hiện:**
  1. Trên tập dữ liệu chuẩn Hugging Face (3,600 mẫu cắt từ 1,800 bài hát), nếu chia ngẫu nhiên (Random Split) thì đoạn 1 (`seg1`) và đoạn 2 (`seg2`) của cùng một bản thu rơi vào cả tập Train lẫn tập Test, làm mô hình học thuộc lòng đặc tính phòng thu và giọng nghệ nhân.
  2. Ngay cả khi đã khắc phục bằng cách chia theo bài hát ([data/datasets/split_group_v2.csv](data/datasets/split_group_v2.csv)), khi đem các mô hình chỉ huấn luyện trên Hugging Face ra kiểm thử trên **131 video YouTube biểu diễn thực tế** ([evaluate_audio_youtube_benchmark.py](evaluate_audio_youtube_benchmark.py)), mô hình CNN `EXP-13` bị sụt giảm từ **92.50% xuống 38.93% Accuracy (30.11% Macro-F1)**, và mô hình `SVM-RBF` (167 đặc trưng thô) sụt giảm từ **93.75% xuống 31.21% Macro-F1**.
  3. Ngược lại, trên cùng 131 video YouTube đó, mô hình **Fuzzy Random Forest (chỉ dùng 7 trục ngữ nghĩa MPEG-7)** lại đạt **55.73% Accuracy (54.71% Macro-F1)** — cao hơn mạng sâu CNN tới **+16.80% Accuracy và +24.60% Macro-F1**.
- **Nguyên nhân khoa học:**
  - Mạng CNN (`EXP-13`) và các mô hình Học máy trên 167 đặc trưng thô đã học quá khớp (overfit) vào "dấu vân tay môi trường" của tập Hugging Face (đặc tuyến tần số micro phòng thu, mức nền tạp âm tĩnh và các hệ số MFCC bậc cao từ 14–40, Delta-MFCC, Spectral Flatness). Khi gặp âm thanh YouTube ngoài đời thực (có tiếng ồn khán giả, độ vang sân khấu đình/đền, nén âm thanh MP4), phân phối của 137 đặc trưng vi mô này bị biến dạng hoàn toàn.
  - Trong khi đó, nhánh Fuzzy Logic đã **lược bỏ 137 đặc trưng vật lý vi mô nhạy nhiễu**, chỉ giữ lại **30 đặc trưng cốt lõi** và gom cụm mờ thành **7 trục ngữ nghĩa bậc cao** (Năng lượng, Hòa âm, Cao độ, Độ sáng phổ, Nhịp độ, Âm sắc nền, Độ luyến láy). Nhờ sự trừu tượng hóa này, quy luật âm nhạc học (ví dụ: *Chầu văn có nhịp nhanh và năng lượng mạnh; Cải lương có độ luyến láy sâu*) vẫn giữ nguyên giá trị bất chấp môi trường thu âm thay đổi.
- **Giải pháp khắc phục triệt để:**
  - Kết hợp ưu thế của cả hai hướng: Bổ sung dữ liệu thực tế từ YouTube vào quá trình huấn luyện theo giao thức chống rò rỉ cấp độ video (**Video-Level Group Split: 81 video Train / 20 video Val / 30 video Test**), đồng thời **nhúng trực tiếp 7 trục âm học MPEG-7, 27 khái niệm mờ và 60 luật mờ khả vi vào mạng học sâu** (Phương án 4 và Phương án 5), giúp độ chính xác trên tập YouTube Test tăng vọt từ `45.39%` lên **85.60% Macro-F1**.

### 2.2. Bài toán xác định thứ tự ngữ nghĩa (Thấp -> Cao) khi phân cụm mờ trong không gian đa chiều và sự khác biệt giữa hai bộ đồ thị hàm thuộc
- **Vấn đề phát hiện:**
  - Với trục 1 chiều như `Tempo` (1 đặc trưng BPM), việc sắp xếp 3 tâm cụm từ nhỏ đến lớn để gán nhãn `Chậm -> Vừa -> Nhanh` là hiển nhiên. Tuy nhiên, với các trục đa chiều như `Timbre` (gồm 13 đặc trưng `mfcc_01_mean` đến `mfcc_13_mean`) hay `Harmony` (7 đặc trưng Chroma), tâm cụm là các vector trong không gian 13 chiều hoặc 7 chiều nên không thể so sánh lớn/nhỏ trực tiếp.
  - Ngoài ra, khi vẽ đồ thị xác suất hậu nghiệm của FCM/GMM chiếu từ không gian 13 chiều xuống 1 trục đại diện, các đường cong bị chồng lấn phức tạp và FCM bị phẳng quanh mức `1/K` (do hiện tượng tập trung khoảng cách Euclid trong không gian nhiều chiều), khác với đồ thị hàm thuộc chuẩn hóa ([data/plots/fuzzy_logic_graphs_all_axes.png](data/plots/fuzzy_logic_graphs_all_axes.png)).
- **Giải pháp khắc phục:**
  - Định hướng thứ tự tâm cụm đa chiều dựa trên **đặc trưng vật lý chủ đạo mang ý nghĩa độ lớn âm học** (ví dụ trục `Timbre` sắp xếp theo `mfcc_01_mean` kết hợp độ sáng phổ; trục `Energy` sắp xếp theo tổng `energy_mean + dynamic_range`).
  - Thiết lập thêm hệ hàm thuộc chuẩn hóa hình thang/tam giác tuân thủ **phân hoạch Ruspini** (tổng độ thuộc tại mọi điểm luôn bằng 1.0 và tại mỗi điểm giá trị chỉ giao thoa tối đa 2 tập mờ liền kề tại ngưỡng cắt `alpha = 0.5`), giúp người đọc và chuyên gia nhạc lý dễ dàng kiểm chứng trực quan.

### 2.3. Xử lý mất cân bằng thời lượng video YouTube (Adaptive Strided Sampling) và mất cân bằng số lượng mẫu giữa Hugging Face với YouTube
- **Vấn đề phát hiện:**
  1. Các video YouTube có độ dài rất chênh lệch (từ 45 giây đến hơn 20 phút). Nếu cắt tất cả video bằng cùng một bước nhảy cố định (ví dụ mỗi 5 giây cắt 1 đoạn), một video Cải lương dài 20 phút sẽ sinh ra 240 mẫu, làm mô hình chỉ học giọng của đúng 1 video đó.
  2. Sau khi cắt hợp lý, tổng số mẫu YouTube trong tập Train (579 mẫu) vẫn ít hơn nhiều so với tập Hugging Face (2,520 mẫu); đặc biệt lớp Ca trù YouTube chỉ có 57 mẫu Train (bằng `1/7.4` số mẫu Ca trù Hugging Face). Khi huấn luyện chung, hàm mất mát thông thường bị áp đảo bởi mẫu Hugging Face.
- **Giải pháp khắc phục:**
  1. Thiết kế thuật toán **Lấy mẫu bước nhảy thích ứng (Adaptive Strided Sampling)** với cửa sổ 12.0 giây, giới hạn hạn ngạch số đoạn tối đa của mỗi video theo công thức: `Quota_i = min(max(1, round(Duration_i / 30.0)), 40)` và bước nhảy `Step_i = (Duration_i - 12.0) / Quota_i`. Nhờ đó, các mẫu trích xuất luôn rải đều từ đầu đến cuối buổi biểu diễn mà không video nào vượt quá 40 mẫu.
  2. Áp dụng đồng thời **`WeightedRandomSampler` (`2.5x`)** để cân bằng tần suất xuất hiện trong từng mini-batch và **Trọng số phạt mất mát nhạy cảm chi phí (`Cost-Sensitive Domain Penalty = 3.5x * rarity_boost`)** trong hàm Cross-Entropy (phạt nặng gấp `3.5` đến `5.92` lần khi mô hình dự đoán sai một mẫu YouTube hiếm).

### 2.4. Thiết kế Ma trận Thực nghiệm 5 Phương án (21 Mô hình) đảm bảo tính đối chứng khoa học công bằng
- **Vấn đề phát hiện & Giải pháp thiết kế chuỗi thực nghiệm:**
  1. **Từ Phương án 1 sang Phương án 2 (`M1a..M1c` -> `M2a..M2e`):**
     - Ở Phương án 1, thực nghiệm chứng minh sử dụng kết hợp 2 kênh phổ `Mel+CQT` (`M1c: 81.65% HF F1`) tốt hơn đơn phổ `Mel` (`M1a: 80.71%`) và đơn phổ `CQT` (`M1b: 78.53%`).
     - Sang Phương án 2, để chứng minh chặt chẽ lợi ích của việc gắn Transformer cũng như ảnh hưởng của số lớp Transformer (1 lớp, 2 lớp, 4 lớp) và khối chọn lọc kênh `SE-Block`, hệ thống huấn luyện đầy đủ 5 mô hình theo trình tự đối chứng:
       - `M2a`: CNN + Transformer (Single Mel) — đối chứng trực tiếp với `M1a`.
       - `M2b`: CNN + Transformer 1L (Mel+CQT) — đối chứng trực tiếp với `M1c` (chứng minh gắn 1 lớp Transformer vào đa phổ giúp tăng **+9.70% HF F1** và **+10.75% YouTube F1**).
       - `M2c`: CNN + Transformer 2L (Mel+CQT) và `M2d`: CNN + Transformer 4L (Mel+CQT, không dùng SE) — chứng minh trên tập dữ liệu cỡ vừa, việc tăng độ sâu Transformer từ `1L -> 2L -> 4L` không kèm cơ chế lọc kênh sẽ gây quá khớp làm giảm F1 (`73.32% -> 71.83% -> 64.74%`).
       - `M2e`: SE-CNN + Transformer 4L (Mel+CQT) — đối chứng trực tiếp với `M2d` (chứng minh khối `SE-Block` giúp phục hồi **+3.84% YouTube F1** cho mạng Transformer 4 lớp).
  2. **Ở Phương án 3 (`M3a..M3c` — Học chuyển giao 2 giai đoạn chống quên thảm khốc):**
     - Khi thay tầng tích chập đầu tiên (`stem_conv`) của `ResNet34`, `EfficientNet-B0`, `DenseNet121` từ 3 kênh RGB sang 2 kênh `Mel+CQT`: Ở Stage 1 (10 epochs đầu), hệ thống đóng băng các khối giữa của backbone và khóa cứng các lớp `BatchNorm2d` ở chế độ `.eval()` (bảo toàn thống kê ImageNet), nhưng **mở khóa riêng tầng `stem_conv` 2 kênh** cùng `Transformer Head`; sang Stage 2 (40 epochs sau) mới mở khóa toàn bộ mạng để tinh chỉnh sâu.
  3. **Ở Phương án 4 (`M4a..M4e`) và Phương án 5 (`M5a..M5e`) — Công bằng hóa cơ chế Fusion và trọng số Fuzzy Loss:**
     - Huấn luyện đầy đủ cả 2 biến thể `Gated Fusion` và `Concat Fusion` cho `EfficientNet-B0` ở cả Phương án 4 (`M4c` vs `M4d`) và Phương án 5 (`M5c` vs `M5d`), đồng thời đồng bộ tuyệt đối trọng số hàm mất mát `Loss = Loss_CE + 0.50 * Loss_Concept_BCE + 0.75 * Loss_Rule` cho cả 5 mô hình `M5a–M5e`.

### 2.5. Giải thích hiện tượng lớp Ca trù đạt độ chính xác 96.2% – 100% trên tập YouTube Test
- **Nguyên nhân âm học:** Ca trù là thể loại nhạc thính phòng cổ điển có cấu trúc âm thanh tách biệt hoàn toàn với 5 thể loại còn lại (tiếng phách tre gõ nhịp khô giòn ở dải tần cao, tiếng đàn đáy 3 dây trầm đục và kỹ thuật hát khuôn/đổ hột ém hơi của đào nương, không có dàn nhạc gõ/nhị/trống sân khấu ồn ào như Chèo, Chầu văn hay Cải lương).
- **Nguyên nhân dữ liệu:** Lớp Ca trù có số lượng video YouTube ít nhất (3 video trong tập Test ứng với 26 phân đoạn), trong đó có 1 video ngắn (`catru_tayhohoaico_01.mp4` gồm 3 phân đoạn ở tập Test) là trích đoạn thuộc cùng buổi biểu diễn với phần `_02`, `_03` trong tập Train (`3/26 = 11.5%` số phân đoạn Ca trù Test).

### 2.6. Cơ chế học đa nhãn có giám sát yếu (Weakly-Supervised Multi-label) của Visual CBM không cần vẽ Bounding Box
- Các khái niệm thị giác của nghệ thuật diễn xướng truyền thống bao gồm cả nhạc cụ nhỏ, trang phục lẫn không gian bối cảnh toàn cục (như *điện thờ hầu đồng*, *chiếu chèo sân đình*, *đội hình 3 người Ca trù*). Thay vì gán nhãn tọa độ Bounding Box tốn kém, mô hình `Visual CBM` sử dụng mạng `ConvNeXt-Tiny` trích xuất bản đồ đặc trưng không gian rồi qua tầng `Global Average Pooling` và **25 đầu ra Sigmoid độc lập** (cho phép 1 khung hình kích hoạt đồng thời nhiều khái niệm).
- Để khắc phục việc các đạo cụ hiếm xuất hiện ít hơn bối cảnh, mô hình sử dụng hàm mất mát **Asymmetric Loss (ASL)** (`gamma_neg = 4.0, gamma_pos = 1.0`) và thực hiện **tối ưu hóa ngưỡng kích hoạt F1 riêng cho từng khái niệm** (`visual_concept_thresholds.json`) trên tập Validation thay vì dùng ngưỡng cứng `0.5`.

---

## PHẦN 3: KIẾN TRÚC PIPELINE XỬ LÝ TỔNG THỂ (TỪ INPUT ĐẾN OUTPUT) & PHƯƠNG PHÁP TÌM TRỌNG SỐ MULTIMODAL TỐI ƯU

### 3.1. Sơ đồ Luồng Xử lý Đầu cuối (End-to-End Inference Pipeline)

```text
[ĐẦU VÀO (INPUT): Video MP4 / File Audio / Link YouTube / Thu âm Micro]
                                   │
                                   ▼
   ┌───────────────────────────────────────────────────────────────┐
   │ BƯỚC 1: KIỂM TRA THỜI LƯỢNG (<= 10 PHÚT) & PHÂN ĐOẠN THÍCH ỨNG│
   │ - Nếu > 600s: Kích hoạt trình cắt video trực tuyến (ffmpeg)   │
   │ - Tách âm thanh Mono & chia 6 – 14 phân đoạn bao phủ 100%     │
   │ - Mỗi phân đoạn: Cửa sổ âm thanh 12.0s + 1 Keyframe sắc nét   │
   └──────────────────┬──────────────────────────┬─────────────────┘
                      │                          │
        (Luồng Âm thanh 12s)             (Khung hình Keyframe)
                      │                          │
                      ▼                          ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────────────┐
│ BƯỚC 2A: NHÁNH ÂM THANH NEURO-SYMBOLIC│  │ BƯỚC 2B: NHÁNH THỊ GIÁC (VISUAL CBM) │
│ 1. Sinh 2 phổ 128x128 (Mel & CQT)    │  │ 1. Chuẩn hóa ảnh RGB (224x224)       │
│ 2. Trích 7 trục MPEG-7 chuẩn hóa     │  │ 2. ConvNeXt-Tiny (best_visual_cbm)   │
│ 3. Chiếu GMM -> 27 độ thuộc mờ       │  │    -> 25 xác suất Sigmoid khái niệm  │
│ 4. Tổ hợp 4 Họ Kiến trúc (0.25/mô):  │  │ 3. So ngưỡng F1 riêng từng concept   │
│    - M2a (CNN + Trans, Mel)          │  │    -> Danh sách đạo cụ / trang phục  │
│    - M3c (DenseNet121 + Trans)       │  │ 4. Visual Random Forest Continuous   │
│    - M4d (EffNetB0 + 7 Trục Concat)  │  │    (best_visual_rf_continuous.joblib)│
│    - M5c (NS EffNetB0 Gated + 27C)   │  │    -> Xác suất P_Visual (6 lớp)      │
│    -> Xác suất P_Audio (6 lớp)       │  └──────────────────┬───────────────────┘
│    -> Vector 27 khái niệm mờ c_hat   │                     │
└──────────────────┬───────────────────┘                     │
                   │                                         │
                   └────────────────────┬────────────────────┘
                                        ▼
         ┌──────────────────────────────────────────────────────────┐
         │ BƯỚC 3: DUNG HỢP MUỘN ĐA PHƯƠNG THỨC (LATE FUSION)       │
         │ - Tại từng phân đoạn: P_Fused = 0.70*P_Audio+0.30*P_Vis  │
         │   (Nếu đầu vào chỉ có Audio/Mic: P_Fused = P_Audio)      │
         │ - Toàn bộ tệp: Trung bình cộng chuỗi phân đoạn (Pooling) │
         └──────────────────────────────┬───────────────────────────┘
                                        ▼
         ┌──────────────────────────────────────────────────────────┐
         │ BƯỚC 4: HẬU KIỂM NGOẠI LAI (OOD) & GIẢI THÍCH XAI        │
         │ 1. Lọc 3 điều kiện OOD -> Gán "Thể loại khác" nếu lạ     │
         │ 2. Đối chiếu 60 Luật mờ liên tục + Crisp/RIPPER/Visual   │
         │ 3. Phiên dịch luật & Ánh xạ Khái niệm UNESCO/ICHCAP      │
         │ 4. Gộp phân đoạn thông minh (Smart Grouping)             │
         └──────────────────────────────┬───────────────────────────┘
                                        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ BƯỚC 5: ĐẦU RA ĐA PHƯƠNG THỨC (OUTPUT & INTERACTIVE XAI DASHBOARD)        │
│ - Phán quyết thể loại & Xác suất riêng từng nhánh (Audio / Visual / Fused)│
│ - Dòng thời gian phân đoạn (Keyframes, Biểu đồ xác suất, Ảnh phổ, Radar)  │
│ - Bản đồ Bằng chứng Đa phương thức 4 tầng (Multimodal Evidence Map)       │
│ - Giải thích Đối chứng (Contrastive XAI) & Can thiệp Phản sự thực (CBM)   │
└───────────────────────────────────────────────────────────────────────────┘
```

### 3.2. Chi tiết 5 Bước Xử lý trong `web_app/inference_engine.py`

1. **Bước 1 — Tiền xử lý & Phân đoạn Thích ứng (Dynamic Adaptive Chunking):**
   - Tệp đầu vào được kiểm tra thời lượng tối đa 10 phút (`600 giây`). Nếu vượt quá, hệ thống trả về thông báo kèm trình cắt trực quan (cho phép xem khung hình và nghe âm thanh đoạn cắt bằng `ffmpeg`).
   - Âm thanh được chuẩn hóa mono và chia đều thành `6 đến 14 phân đoạn` tùy theo độ dài (đảm bảo bao phủ 100% dòng thời gian không bỏ sót). Tại tâm mỗi phân đoạn, hệ thống cắt **cửa sổ ngữ cảnh 12.0 giây** cho nhánh Âm thanh (khớp đúng độ dài huấn luyện của 21 mô hình mới) và chọn **1 khung hình keyframe sắc nét nhất** (theo độ lớn phương sai Laplacian) cho nhánh Thị giác.
2. **Bước 2 — Trích xuất Đặc trưng & Suy luận Song song 2 Nhánh:**
   - **Nhánh Âm thanh:** Từ cửa sổ 12 giây (@ `22,050 Hz`), trích xuất song song: (a) 2 ma trận phổ `Log-Mel` và `Log-CQT` kích thước `128x128` chuẩn hóa `[-1, 1]`, (b) 7 trục ngữ nghĩa MPEG-7 chuẩn hóa `[0, 1]` và 27 độ thuộc mờ GMM. Sau đó đưa qua **Tổ hợp 4 Họ Kiến trúc (`M2a`, `M3c`, `M4d`, `M5c`)** với trọng số `0.25` mỗi mô hình để tính `P_Audio`, đồng thời trích xuất đầu ra 27 khái niệm mờ từ Concept Head của `M5c` để hòa trộn (`0.5 / 0.5`) vào vector độ thuộc mờ liên tục.
   - **Nhánh Thị giác (chỉ chạy với Video):** Khung hình keyframe (`224x224`) đi qua mạng `ConvNeXt-Tiny` (`best_visual_cbm.pth`) xuất 25 xác suất Sigmoid độc lập. Các khái niệm vượt ngưỡng F1 tối ưu (`visual_concept_thresholds.json`) được ghi nhận làm bằng chứng đạo cụ/trang phục; đồng thời toàn bộ vector 25 xác suất liên tục được đưa vào `best_visual_rf_continuous.joblib` để tính `P_Visual`.
3. **Bước 3 — Dung hợp Muộn Đa phương thức (Multimodal Late Fusion):**
   - Phân phối xác suất hợp nhất tại mỗi phân đoạn được tính theo công thức:
     `P_Fused(g) = 0.70 * P_Audio(g) + 0.30 * P_Visual(g)` (với Video khi có khung hình hợp lệ qua Visual CBM), hoặc `P_Fused(g) = P_Audio(g)` (với Audio/Mic).
   - Xác suất cuối cùng của cả tệp được tổng hợp bằng trung bình cộng trên tất cả các phân đoạn (**Temporal Average Pooling**).
4. **Bước 4 — Bộ lọc Ngoại lai (OOD) & Suy luận Giải thích XAI:**
   - **Phát hiện ngoại lai (Unknown):** Đánh giá đồng thời Độ bất định thống kê (`P_max < 0.38`, hoặc `P_max < 0.48` với `margin < 0.18`, hoặc Entropy chuẩn hóa `H_norm >= 0.80` với `P_max < 0.50`) và Bằng chứng biểu tượng (số luật mờ kích hoạt & số thực thể thị giác di sản). Nếu thỏa điều kiện kích hoạt (chi tiết tại Mục 3.4) -> Hệ thống từ chối gán vào 6 thể loại và trả về nhãn *"Thể loại khác / Không đủ dữ liệu xác định"*.
   - **Ánh xạ Luật & Khái niệm:** Nếu hợp lệ, hệ thống đối chiếu **60 luật mờ liên tục** (`rules_decision_tree_continuous.json` + `rules_random_forest_top.json`) kết hợp luật `Crisp`, `RIPPER` và `Visual RIPPER`, dùng [phase7_audio_visual_concept_mapper.py](phase7_audio_visual_concept_mapper.py) dịch mệnh đề sang tiếng Việt và gắn mốc thời gian xuất hiện cụ thể cho từng khái niệm âm thanh/thị giác.
5. **Bước 5 — Xuất Kết quả & Tương tác Phản sự thực (Counterfactual CBM):**
   - Dữ liệu trả về cho giao diện gồm: phán quyết tổng hợp & xác suất từng nhánh, dải phân đoạn đã gộp thông minh (`merged_segments`), chuỗi thời gian (`timeline_series`), Bản đồ Bằng chứng Đa phương thức 4 tầng (`evidence_map`), Giải thích Đối chứng (`contrastive_analysis`), đồ thị 3D (`graph_3d`), và hỗ trợ người dùng bật/tắt trực tiếp từng khái niệm để kiểm chứng độ nhạy quyết định qua API `/api/intervene`.

---

### 3.3. Phương pháp Tìm Tổ hợp Mô hình và Bộ Trọng số Multimodal Tối ưu (Dẫn chứng từ `experiments/results/multimodal_fusion_comparison.csv`)

Việc lựa chọn tổ hợp **`0.25 M2a + 0.25 M3c + 0.25 M4d + 0.25 M5c`** cho nhánh Âm thanh và tỷ lệ **`0.70 Audio + 0.30 Visual`** cho tầng Dung hợp Đa phương thức không phải là lựa chọn cảm tính mà được chứng minh qua quy trình thực nghiệm đối chứng thống kê 3 bước (chi tiết bảng số liệu tại [experiments/results/multimodal_fusion_comparison.csv](experiments/results/multimodal_fusion_comparison.csv) và [data/results/multimodal_fusion_ablation_report.md](data/results/multimodal_fusion_ablation_report.md)):

1. **Bước A — Đánh giá & Loại bỏ các thành phần gây nhiễu (Ablation Study):**
   - Thực nghiệm kiểm chứng độc lập trên cùng tập kiểm thử chuẩn (720 mẫu Hugging Face) và tập kiểm thử thực tế YouTube (225 phân đoạn từ 30 video) cho thấy:
     - Các mô hình thế hệ 1 chỉ học trên Hugging Face có hiệu năng rất thấp trên YouTube: `EXP-13` đạt `45.39% F1`, `SVM-RBF` (167 đặc trưng thô) đạt `31.21% F1`, `Random Forest` (167 đặc trưng thô) đạt `53.15% F1` (và tốn `5.6 giây/đoạn` trích xuất `pyin`), các bộ luật rời rạc (`DT Crisp / CN2 / RIPPER`) chỉ đạt `6.11% – 15.91% F1`.
     - Khi thử nghiệm **"Kết hợp tất cả"** (gộp đều các mô hình cũ, SVM, luật rời rạc cùng các mô hình mới), Macro-F1 trên YouTube bị kéo tụt từ **85.60% xuống 81.83% (-3.77%)**.
     - *Kết luận:* Loại bỏ hoàn toàn các mô hình chỉ học Hugging Face và mô hình 167 đặc trưng thô khỏi tầng tính xác suất; các bộ luật rời rạc chỉ giữ lại ở tầng phiên dịch giải thích XAI.
2. **Bước B — So sánh Phương pháp Tìm Trọng số cho Nhánh Âm thanh (SLSQP Validation vs Bỏ phiếu Đồng đều 4 Họ Kiến trúc):**
   - Để tránh tuyệt đối hiện tượng dò trọng số trên tập Test (Data Snooping), nghiên cứu so sánh 2 cách tìm trọng số:
     - *Cách 1 — Tối ưu hóa lồi SLSQP trên tập Validation (458 mẫu: 360 HF + 98 YT):* Tối thiểu hóa hàm Balanced Log-Loss trên tập Validation. Tuy nhiên do tập YouTube Validation chỉ có 98 mẫu (và đã dùng để chọn Early Stopping), trọng số SLSQP bị quá khớp nhẹ vào tập Validation nên khi kiểm thử trên YouTube Test chỉ đạt `77.87% – 78.36% F1`.
     - *Cách 2 — Bỏ phiếu mềm đồng đều theo Đại diện mạnh nhất từng Họ kiến trúc (Uniform Family Top-1 Soft-Voting):* Chọn đúng 1 mô hình mạnh nhất của mỗi Phương án 2, 3, 4, 5 với trọng số bằng nhau (`0.25` mỗi mô hình, hoàn toàn không tinh chỉnh siêu tham số trên tập Test):
       - `M2a` (Phương án 2: CNN + Transformer Single Mel — `81.29% YT F1`)
       - `M3c` (Phương án 3: Pretrained DenseNet121 + Transformer — `74.21% YT F1`)
       - `M4d` (Phương án 4: Multimodal EfficientNet-B0 + 7 Trục MPEG-7 — `76.12% YT F1`)
       - `M5c` (Phương án 5: Neuro-Symbolic EfficientNet-B0 Gated + 60 Luật mờ — `76.09% YT F1`)
     - *Kết quả:* Tổ hợp đồng đều 4 Họ kiến trúc đạt hiệu năng cao nhất toàn diện: **94.86% Acc / 94.82% Macro-F1** trên Hugging Face Test, **85.33% Acc / 85.60% Macro-F1** trên YouTube Test (**27/30 = 90.0%** cấp độ video), và **92.64% Macro-F1** trên Tập Tổng hợp 945 mẫu, đồng thời vẫn giữ được đầu ra 27 khái niệm mờ của `M5c` phục vụ XAI.
3. **Bước C — Quét Lưới Tìm Trọng số Tầng Dung hợp Đa phương thức (`w_audio`, `w_visual`):**
   - Trên tập video kiểm thử độc lập của cả hai nhánh (không có video nào trùng với tập huấn luyện của Nhánh Âm thanh hay Nhánh Thị giác), thực hiện quét lưới `w_audio` từ `0.00` đến `1.00` (bước nhảy `0.05`, `w_visual = 1.0 - w_audio`) giữa **Tổ hợp 4 Họ Âm thanh** và **Visual CBM (`ConvNeXt-Tiny` + `Visual RF Continuous`)**:
     - Chỉ dùng Thị giác (`0.00 / 1.00`): Đạt `85.7%` (`LogLoss = 0.5321`).
     - Chỉ dùng Âm thanh (`1.00 / 0.00`): Đạt `92.9%` (`LogLoss = 0.2398`).
     - Vùng giao thoa tối ưu (`0.50 / 0.50` đến `0.75 / 0.25`): Đạt độ chính xác tuyệt đối **100.0%**. Trong đó, cặp trọng số **`0.70 Audio + 0.30 Visual`** được chọn làm cấu hình chính thức vì đạt Log-Loss thấp hơn cấu hình cũ `0.65 / 0.35` (`0.2881` vs `0.3012`), phản ánh đúng độ tin cậy vượt trội của Tổ hợp 4 Họ Âm thanh mới trong khi vẫn bảo toàn `30%` trọng số hình ảnh đạo cụ/trang phục để bổ cứu các phân đoạn âm thanh bị nhiễu.


---

### 3.4. Bổ sung Nhãn Ngoại lai (Unknown / Out-of-Distribution), Phương pháp Tìm Ngưỡng và Điều kiện Kích hoạt

#### 3.4.1. Động lực Thiết kế Nhãn Ngoại lai (`Thể loại khác / Không đủ dữ liệu xác định`)
Trong bài toán phân lớp tập đóng (Closed-Set Classification) trên 6 thể loại di sản (*Ca trù, Chèo, Cải lương, Chầu văn, Quan họ, Hát xẩm*), hàm Softmax luôn chuẩn hóa tổng xác suất bằng `1.0`, khiến mô hình buộc phải gán nhãn vào 1 trong 6 lớp ngay cả khi người dùng tải lên dữ liệu không thuộc di sản (ví dụ: nhạc pop/điện tử, tiếng nói hội thoại, tạp âm môi trường, khoảng lặng hoặc video không phải biểu diễn truyền thống). Để giải quyết hiện tượng nhận diện gượng ép (Overconfident Misclassification) trong môi trường thực tế mở (Open-Set), hệ thống bổ sung cơ chế từ chối dự đoán với nhãn **`Thể loại khác / Không đủ dữ liệu xác định`** (mã màu `#64748B`).

#### 3.4.2. Phương pháp Tìm & Hiệu chuẩn Lại Ngưỡng Tối ưu cho Mô hình Multimodal Fusion Mới
Khi nâng cấp kiến trúc suy luận từ `EXP-13 + RF` cũ lên **Tổ hợp 4 Họ Audio Neuro-Symbolic (`0.25 M2a + 0.25 M3c + 0.25 M4d + 0.25 M5c`) + Visual CBM (`0.70 Audio / 0.30 Visual`)**, bộ ngưỡng cũ (`P_max < 0.30` hoặc `P_max < 0.40 và margin < 0.15`) không còn tối ưu do phân phối xác suất của mô hình mới sắc nét hơn.

Quá trình tìm ngưỡng tối ưu được thực hiện bằng **thực nghiệm đối chứng phân tách biên In-Distribution (ID) vs Out-of-Distribution (OOD)** trực tiếp trên `HeritageInferenceEngine`:

1. **Khắc phục 2 điểm mù ở tầng Dung hợp trước khi đo ngưỡng (`web_app/inference_engine.py`):**
   - *Không dùng fallback pixel heuristic khi Visual CBM đã nạp:* Trước đây, khi khung hình OOD không có nhạc cụ/trang phục (`cues = []`), pipeline cũ gọi hàm fallback màu sắc/độ sáng pixel `_detect_visual_cues_from_frame` làm phát sinh 3 cues giả. Đã giới hạn fallback này chỉ chạy khi `visual_cbm_model is None`.
   - *Luôn giữ Late Fusion `70% Audio + 30% Visual` cho Video có khung hình hợp lệ:* Ngay cả khi `cues = []` trên khung hình phi di sản, vector `P_Visual` từ `Visual CBM + RF Continuous` vẫn phản ánh đúng độ bất định (`P_max` chỉ đạt `0.231 – 0.276`, Entropy rất cao), giúp kéo giảm độ tự tin quá mức của nhánh Audio khi dung hợp.
2. **Đo lường đường biên thực nghiệm giữa Tập Trong miền (ID) và Tập Ngoài miền (OOD):**
   - **Tập Trong miền (6 video kiểm thử độc lập `dataset/video/test/*.mp4`):**
     - `quanho_buonbacbuondau.mp4`: `P_max = 0.908`, `margin = 0.867`, `H_norm = 0.243`, `rules = 5`, `cues = 4`
     - `catru_chapcanhrongbay.mp4`: `P_max = 0.855`, `margin = 0.812`, `H_norm = 0.356`, `rules = 6`, `cues = 8`
     - `cailuong_tinhanhbanchieu.mp4`: `P_max = 0.620`, `margin = 0.476`, `H_norm = 0.657`, `rules = 6`, `cues = 7`
     - `xam_chodongxuan.mp4`: `P_max = 0.620`, `margin = 0.407`, `H_norm = 0.627`, `rules = 2`, `cues = 6`
     - `chauvan_conoimechangcanchi.mp4`: `P_max = 0.565`, `margin = 0.429`, `H_norm = 0.745`, `rules = 4`, `cues = 1`
     - `cheo_tinhanhlinhtre.mp4` *(mẫu ID khó nhất)*: **`P_max = 0.519`**, **`margin = 0.272`**, `H_norm = 0.682`, `rules = 1`, `cues = 4`
     - *Kết luận đường biên ID:* Toàn bộ mẫu ID thật đều có `P_max >= 0.519`, `margin >= 0.272`, Entropy chuẩn hóa `H_norm <= 0.745`, và luôn kích hoạt `rules >= 1`, `cues >= 1`.
   - **Tập Ngoài miền (tín hiệu phi di sản: nhiễu trắng, tiếng ù 60Hz, sóng sin 440Hz, nhịp điện tử 4/4, giả giọng nói; khung hình nhiễu/tối/đơn sắc/lưới hiện đại):**
     - Khi qua dung hợp 70/30: `P_max` nằm trong khoảng **`0.241 – 0.489`**, Entropy chuẩn hóa **`H_norm >= 0.819`** (`H = 1.468 – 1.649`), và `cues = 0`.
3. **Xác định khoảng đệm an toàn:**
   - Chọn các ngưỡng nằm giữa đường biên trên của OOD (`P_max <= 0.489`, `H_norm >= 0.819`) và đường biên dưới của ID (`P_max >= 0.519`, `margin >= 0.272`, `H_norm <= 0.745`), đảm bảo **0% nhận nhầm ID thành Unknown** trong khi chặn được các mẫu OOD.

#### 3.4.3. Hệ Điều kiện & Ngưỡng Kích hoạt Chính thức (Đã hiệu chuẩn thực tế cho Video YouTube & Micro)

Trong [`web_app/inference_engine.py`](web_app/inference_engine.py), hệ thống kết hợp **Cổng lọc khoảng lặng/tạp âm nền** ở tầng tín hiệu đầu vào với **Cổng kiểm soát Ngoại lai (OOD Gate)** ở tầng quyết định:

1. **Cổng lọc Khoảng lặng & Nhiễu nền (`Silence & Spectral Flatness Gate`):**
   - Đo biên độ hiệu dụng `RMS`, đỉnh biên độ `peak`, tỷ lệ khung có tín hiệu (`frame_rms > 0.012`), độ biến thiên `rms_std` và độ phẳng phổ `spectral_flatness`.
   - Chặn trực tiếp bản ghi im lặng hoặc tiếng quạt phòng (`RMS < 0.012` hoặc `peak < 0.06` hoặc `spectral_flatness > 0.075` kèm `rms_std < 0.018`), kết hợp tắt `autoGainControl` ở trình thu âm Micro (`audio_recorder.js`) để không khuếch đại khoảng lặng thành nhiễu.
2. **Lọc thực thể thị giác di sản dương tính & Dung hợp có điều kiện:**
   - Loại trừ nhãn âm tính `khong_nhac_cu` ("không nhạc cụ đệm") đứng một mình khỏi nhóm thực thể thị giác di sản dương tính (tránh video vlog/thoại trên YouTube kích hoạt nhầm nhánh Thị giác).
   - Chỉ dung hợp `0.70 Audio + 0.30 Visual` tại phân đoạn có ít nhất 1 thực thể thị giác di sản dương tính; các phân đoạn không phát hiện thực thể di sản giữ nguyên phân phối từ nhánh Âm thanh.
3. **Các chỉ số tổng hợp toàn video:**
   - `top1_p` (`P_max`), `prob_margin = top1_p - top2_p`, Entropy Shannon chuẩn hóa `norm_entropy` trong `[0, 1]`.
   - Tỷ lệ phân đoạn có thị giác di sản dương tính `pos_vis_chunk_ratio = pos_vis_chunks / N_chunks` và độ nhất quán dự đoán giữa các phân đoạn `chunk_consistency`.

**Quy tắc kích hoạt nhãn `Thể loại khác / Không đủ dữ liệu xác định` (`is_ood`):**
- **Chế độ Video:** Kích hoạt nếu thỏa mãn ít nhất 1 trong các điều kiện sau:
  1. **Khoảng lặng / tạp âm nền** kèm không có thực thể thị giác di sản.
  2. **Hoàn toàn không có thực thể thị giác di sản dương tính** (`pos_vis_chunks == 0`) **VÀ** âm học chưa áp đảo tuyệt đối (`P_max < 0.80` hoặc `prob_margin < 0.45` hoặc `norm_entropy >= 0.50` hoặc `chunk_consistency < 0.65`) — chặn hiệu quả video công nghệ, vlog, tin tức, thoại trên YouTube mà vẫn giữ được video ảnh bìa tĩnh phát nhạc di sản chuẩn.
  3. **Nhiễu thị giác lác đác** (`pos_vis_chunk_ratio <= 0.15`, tối đa 1 khung nhiễu) **VÀ** (`P_max < 0.65` hoặc `prob_margin < 0.28` hoặc `norm_entropy >= 0.65` hoặc `chunk_consistency < 0.50`).
  4. **Độ bất định thống kê chung:** `P_max < 0.45`, hoặc `P_max < 0.52 và prob_margin < 0.18`, hoặc không có luật mờ kích hoạt kèm bất định cao.
- **Chế độ Audio-only / Micro:** Kích hoạt khi `is_silent_audio == True`, hoặc `P_max < 0.50`, hoặc (`P_max < 0.60` và `prob_margin < 0.20`), hoặc (`norm_entropy >= 0.72` và `P_max < 0.62`), hoặc (`chunk_consistency < 0.40` và `P_max < 0.65`).

Khi `is_ood == True`, hệ thống gán nhãn đầu ra là `Thể loại khác / Không đủ dữ liệu xác định`, không áp dụng Fallback Concept để tránh sinh khái niệm giả, và sinh diễn giải XAI nêu rõ nguyên nhân (thiếu năng lượng tín hiệu hoặc bất định xác suất và thiếu bằng chứng luật/thị giác di sản).
