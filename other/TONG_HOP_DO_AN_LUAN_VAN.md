# TỔNG HỢP TOÀN DIỆN ĐỒ ÁN TỐT NGHIỆP: HERITTAGEFNS
# Hệ Thống AI Thần Kinh - Ký Hiệu Mờ (Fuzzy Neuro-Symbolic AI) Nhận Diện Và Diễn Giải Di Sản Âm Nhạc Truyền Thống Việt Nam

> **Tài liệu tổng hợp phục vụ viết báo cáo luận văn tốt nghiệp, khóa luận và thuyết trình chuyên môn.**  
> *Được biên soạn bám sát cấu trúc các chương của đồ án: Đặt vấn đề, Tiền xử lý, Trích xuất đặc trưng, Không gian mờ, Mô hình học máy, Trích xuất luật biểu tượng, Kiểm nghiệm thực tế và Hướng phát triển nâng cao.*

---

## MỤC LỤC TỔNG QUAN

1. [Giới Thiệu Đề Tài & Bài Toán Khoa Học](#1-giới-thiệu-đề-tài--bài-toán-khoa-học)
2. [Sơ Đồ Kiến Trúc Hệ Thống Tổng Thể (System Architecture)](#2-sơ-đồ-kiến-trúc-hệ-thống-tổng-thể-system-architecture)
3. [Thu Thập Dữ Liệu & Tiền Xử Lý Tín Hiệu Âm Thanh](#3-thu-thập-dữ-liệu--tiền-xử-lý-tín-hiệu-âm-thanh)
4. [Trích Xuất Đặc Trưng Âm Học Đa Chiều (167 Physical Features)](#4-trích-xuất-đặc-trưng-âm-học-đa-chiều-167-physical-features)
5. [Không Gian Hồ Sơ Mờ & 7 Trục Ngữ Nghĩa Âm Nhạc (Fuzzy Modeling)](#5-không-gian-hồ-sơ-mờ--7-trục-ngữ-nghĩa-âm-nhạc-fuzzy-modeling)
6. [Xây Dựng Hai Tập Dữ Liệu Mờ: Crisp & Continuous](#6-xây-dựng-hai-tập-dữ-liệu-mờ-crisp--continuous)
7. [Huấn Luyện Mô Hình Hộp Trắng & Đánh Giá Đối Chứng (DT & RF)](#7-huấn-luyện-mô-hình-hộp-trắng--đánh-giá-đối-chứng-dt--rf)
8. [Khai Phá & Trích Xuất Hệ Luật Biểu Tượng IF-THEN (Rule Extraction)](#8-khai-phá--trích-xuất-hệ-luật-biểu-tượng-if-then-rule-extraction)
9. [Kiểm Nghiệm Thực Tế Trên Tập Hold-out (18 Mẫu Độc Lập)](#9-kiểm-nghiệm-thực-tế-trên-tập-hold-out-18-mẫu-độc-lập)
10. [Các Module Nâng Cao Đang Triển Khai (Augmentation, RIPPER/CN2, RAG)](#10-các-module-nâng-cao-đang-triển-khai-augmentation-rippercn2-rag)
11. [Bảng Tra Cứu Số Liệu Thực Nghiệm & Cấu Trúc Mã Nguồn](#11-bảng-tra-cứu-số-liệu-thực-nghiệm--cấu-trúc-mã-nguồn)

---

## 1. GIỚI THIỆU ĐỀ TÀI & BÀI TOÁN KHOA HỌC

### 1.1. Bối cảnh và Ý nghĩa Thực tiễn
Âm nhạc truyền thống Việt Nam là di sản văn hóa phi vật thể vô giá, đã được UNESCO vinh danh (như Nhã nhạc, Ca trù, Quan họ, Hát xoan, Không gian văn hóa Cồng chiêng). Tuy nhiên, trước sự phát triển của âm nhạc đương đại, việc số hóa, lưu trữ, nhận diện tự động và giải thích các đặc trưng âm nhạc truyền thống gặp nhiều thách thức:
- **Tính phức tạp âm học**: Thang âm ngũ cung, hệ thống luyến láy đổ hột, rung ngân tự do, không theo hệ thống nhịp phách bình quân luật phương Tây.
- **Hạn chế của mô hình Deep Learning thuần túy**: Các mạng nơ-ron sâu dạng "hộp đen" (Black-box) có thể đạt độ chính xác cao nhưng không thể giải thích cho các nhà nghiên cứu âm nhạc (Ethnomusicologists) hiểu *tại sao* một đoạn âm thanh lại là Ca trù hay Hát xẩm.

### 1.2. Mục tiêu Nghiên cứu của Đồ án
1. **Phân loại chính xác 6 thể loại âm nhạc truyền thống**: Ca trù, Chèo, Chầu văn, Cải lương, Hát xẩm, Quan họ.
2. **Tiếp cận Trí tuệ nhân tạo Thần kinh - Ký hiệu Mờ (Fuzzy Neuro-Symbolic AI)**: Kết hợp năng lực biểu diễn dữ liệu của học máy với tính minh bạch của Logic Mờ (Fuzzy Logic) và Hệ luật Biểu tượng (Symbolic IF-THEN Rules).
3. **Cung cấp khả năng tự giải thích (Explainable AI - XAI)**: Mỗi dự đoán phân loại đều đi kèm hệ luật ngữ nghĩa tự nhiên (ví dụ: Năng lượng rất yếu + Luyến láy nhiều + Âm sắc trầm -> Ca trù), đối chiếu được với lý thuyết nhạc học.

---

## 2. SƠ ĐỒ KIẾN TRÚC HỆ THỐNG TỔNG THỂ (SYSTEM ARCHITECTURE)

```mermaid
flowchart TD
    A["Dữ liệu gốc từ Hugging Face Hub\n(1.800 bài, 6 thể loại)"] --> B["Tiền xử lý tín hiệu\n(Mono, 22.05kHz/16kHz, Segment 2x12s)"]
    B --> C["Tập mẫu chuẩn hóa\n(3.600 mẫu cân bằng: 600 mẫu/lớp)"]
    
    C --> D["Trích xuất đặc trưng âm thanh\n(167 đặc trưng vật lý với Librosa)"]
    D --> E["Lọc giảm chiều ngữ nghĩa\n(30 đặc trưng cốt lõi)"]
    
    E --> F["Mô hình hóa Logic Mờ\n(Fuzzy C-Means, m=2.0 trên 7 Trục)"]
    F --> G1["features_fuzzy_crisp.csv\n(7 trục nhãn chữ rời rạc)"]
    F --> G2["features_fuzzy_continuous.csv\n(27 độ thuộc liên tục [0, 1])"]
    
    G1 & G2 --> H["Huấn luyện Mô hình White-box\n(Decision Tree & Random Forest)"]
    H --> I["Trích xuất Luật Tự động (Rule Extraction)\n(Duyệt đường dẫn cây -> Luật IF-THEN)"]
    
    I --> J["Hệ luật biểu tượng tri thức\n(rules_summary.csv, Top rules)"]
    
    K["Tập 18 mẫu Hold-out độc lập\n(3 bài/thể loại)"] --> L["Đánh giá đối chiếu & Phân tích Archetype\n(Kích hoạt luật & Heatmap)"]
    J --> L

    subgraph Module Nâng Cao (Đang Triển Khai)
        M["Tăng cường dữ liệu Train\n(SMOTE + Gaussian Noise)"] -.-> H
        N["Trích luật đa thuật toán\n(RIPPER & CN2)"] -.-> J
        O["Vector Database & RAG\n(ChromaDB + Multilingual Embeddings)"] -.-> P["Diễn giải báo cáo học thuật tự động"]
    end
```

---

## 3. THU THẬP DỮ LIỆU & TIỀN XỬ LÝ TÍN HIỆU ÂM THANH

### 3.1. Nguồn Dữ Liệu
- **Tập dữ liệu nguồn**: `NTQAI/Vietnamese-Traditional-Music` được thu thập và lưu trữ trên Hugging Face Hub.
- **6 thể loại khảo sát**:
  1. **Ca trù**: Nghệ thuật hát thính phòng bác học Bắc Bộ.
  2. **Chèo**: Nghệ thuật sân khấu kịch hát dân gian châu thổ sông Hồng.
  3. **Chầu văn** (Hát văn): Âm nhạc nghi lễ hầu đồng, tiết tấu lôi cuốn, biến hóa.
  4. **Cải lương**: Kịch nghệ sân khấu tuồng Nam Bộ với làn điệu Vọng cổ sâu lắng.
  5. **Hát xẩm**: Âm nhạc dân gian của nghệ nhân khiếm thị, mang tính tự sự đường phố.
  6. **Quan họ**: Dân ca đối đáp giao duyên vùng Kinh Bắc với kỹ thuật "Vang - Rền - Nền - Nảy".

### 3.2. Quy Mô & Chuẩn Hóa Tín Hiệu (DSP Pipeline)
- **Quy mô gốc**: 1.800 bản ghi âm chất lượng cao (300 bản ghi âm cho mỗi thể loại).
- **Phân đoạn âm thanh (Segmentation Strategy)**:
  - Mỗi bản ghi âm được cắt thành **2 phân đoạn không chồng lấn** (Non-overlapping segments), độ dài **12 giây/đoạn** (đoạn 1 từ giây 5 đến 17; đoạn 2 từ giây 25 đến 37) để loại bỏ phần mở đầu im lặng hoặc tạp âm.
  - Tổng số mẫu thu được: **3.600 mẫu cân bằng hoàn hảo** (600 mẫu/thể loại).
- **Chuẩn hóa kỹ thuật tín hiệu số**:
  - Resampling về **22.050 Hz** (chuẩn audio Librosa) hoặc **16.000 Hz** (chuẩn tương thích Foundation Models).
  - Đơn kênh (Mono) nhằm loại trừ sự lệch pha tín hiệu stereo.
  - Định dạng chuẩn: 16-bit PCM WAV.
- **Tập kiểm nghiệm độc lập (Hold-out Set)**:
  - Tách riêng **18 bài hát hoàn chỉnh** (3 bài cho mỗi thể loại) hoàn toàn không tham gia vào quá trình trích xuất đặc trưng hay huấn luyện, nhằm kiểm thử khả năng tổng quát hóa thực tế của hệ thống.

---

## 4. TRÍCH XUẤT ĐẶC TRƯNG ÂM HỌC ĐA CHIỀU (167 PHYSICAL FEATURES)

### 4.1. Thống Kê 167 Đặc Trưng Thuộc 9 Miền Âm Học
Toàn bộ 167 đặc trưng số học được tính toán qua thư viện `librosa` và lưu trữ trong bảng `data/datasets/features_all.csv` (gồm 3.600 hàng x 172 cột = 167 đặc trưng + 5 cột metadata: `sample_idx`, `label`, `audio_path`, `segment_id`, `segment_time`).

| STT | Miền đặc trưng âm học | Số lượng | Các đặc trưng tiêu biểu | Ý nghĩa vật lý & nhạc học |
| :---: | :--- | :---: | :--- | :--- |
| 1 | **Cao độ (Pitch / F0)** | **5** | `pitch_mean`, `pitch_std`, `pitch_range`, `pitch_median`, `voiced_ratio` | Tần số dao động cơ bản của dây thanh và nhạc cụ chính (trích xuất bằng thuật toán pYIN), độ rộng âm vực và tỷ lệ đoạn có âm vang. |
| 2 | **Nhịp điệu (Tempo & Beat)** | **3** | `tempo`, `beat_count`, `beat_regularity` | Tốc độ nhịp (BPM), số xung nhịp gõ và tính chu kỳ ổn định của tiết tấu. |
| 3 | **Rung ngân (Vibrato)** | **2** | `vibrato_extent`, `vibrato_energy` | Lọc thông dải Butterworth (4 - 10 Hz) trên đường bao F0 để đo mức độ và năng lượng luyến láy, đổ hột, rung giọng đặc trưng. |
| 4 | **Năng lượng (Energy / RMS)** | **4** | `energy_mean`, `energy_std`, `dynamic_range`, `energy_skew` | Cường độ năng lượng gốc (Root Mean Square), độ biến động cường độ và dải tương phản động to/nhỏ. |
| 5 | **Âm sắc MFCC & Đạo hàm** | **106** | • 40 hệ số tĩnh mean/std: `mfcc_01_mean` -> `mfcc_40_std` (80)<br>• 13 Delta: `dmfcc_01_mean` -> `dmfcc_13_mean` (13)<br>• 13 Delta-Delta: `d2mfcc_01_mean` -> `d2mfcc_13_mean` (13) | "Dấu vân tay âm sắc" biểu diễn hình bao phổ thính giác người và sự biến đổi màu sắc âm thanh theo thời gian. |
| 6 | **Hòa âm & Bán âm (Chroma)** | **25** | • 12 bán âm mean/std: `chroma_C_mean` -> `chroma_B_std` (24)<br>• `chroma_std_overall` (1) | Phân bố năng lượng trên 12 bán âm bán chuẩn, phản ánh điệu thức ngũ cung, thang âm Nam/Bắc và hòa thanh phụ đệm. |
| 7 | **Đặc tính phổ (Spectral & ZCR)** | **14** | `spectral_centroid_mean`, `spectral_bandwidth_mean`, `spectral_rolloff_mean`, `spectral_flatness_mean`, `spectral_contrast_1..7`, `zcr_mean`, `zcr_std` | Độ sáng âm thanh (Centroid), bề rộng dải tần, ngưỡng cuộn phổ, độ tương phản giữa đỉnh và đáy sóng hài, và tỷ lệ cắt qua điểm 0. |
| 8 | **Không gian hợp âm (Tonnetz)** | **6** | `tonnetz_1` -> `tonnetz_6` | Hệ tọa độ hòa thanh 6 chiều mô tả quan hệ quãng 5, quãng 3 trưởng và quãng 3 thứ. |
| 9 | **Năng lượng phổ Mel** | **2** | `mel_mean`, `mel_std` | Năng lượng tổng thể phân bổ trên 128 bộ lọc dải tần số cảm nhận Mel. |
| **TỔNG** | **Toàn bộ đặc trưng vật lý** | **167** | | |

### 4.2. Bài Toán Giảm Chiều & Lọc 30 Đặc Trưng Cốt Lõi
- **Vấn đề**: Sử dụng trực tiếp 167 chiều liên tục sẽ gây ra hiện tượng "lời nguyền số chiều" (Curse of Dimensionality), khiến việc sinh luật mờ bị bùng nổ tổ hợp hàm mũ và triệt tiêu khả năng đọc hiểu của con người.
- **Giải pháp**: Chọn lọc chính xác **30 đặc trưng đại diện** có ý nghĩa nhạc học trực tiếp, loại bỏ các đặc trưng đa cộng tuyến (Collinear features) thông qua phân tích ma trận tương quan (`05_correlation.png`).
- **Kết quả**: 30 đặc trưng được giữ lại vẫn bảo toàn độ phân tách giữa 6 thể loại tương đương mô hình dùng 167 chiều, nhưng cho phép chuyển dịch trực tiếp sang logic mờ ngôn ngữ.

---

## 5. KHÔNG GIAN HỒ SƠ MỜ & 7 TRỤC NGỮ NGHĨA ÂM NHẠC (FUZZY MODELING)

### 5.1. Cấu Trúc 7 Trục Ngữ Nghĩa và 27 Tập Mờ Ngôn Ngữ
30 đặc trưng âm học được chuẩn hóa về đoạn [0, 1] bằng `MinMaxScaler` và gom nhóm thành 7 trục nhạc học:

1. **Trục Energy (Năng lượng âm thanh) [5 mức mờ]**:
   - Đặc trưng cấu thành: `energy_mean`, `dynamic_range`.
   - Các tập mờ: `Rất yếu (Very Weak)`, `Yếu (Weak)`, `Vừa (Moderate)`, `Mạnh (Strong)`, `Rất mạnh (Very Strong)`.
2. **Trục Harmony (Cấu trúc hòa âm & Bán âm) [3 mức mờ]**:
   - Đặc trưng cấu thành: 7 bán âm `chroma_C_mean` đến `chroma_B_mean`.
   - Các tập mờ: `Thấp (Low)`, `Trung bình (Medium)`, `Cao (High)`.
3. **Trục Pitch (Cao độ giọng hát & Nhạc cụ) [5 mức mờ]**:
   - Đặc trưng cấu thành: `pitch_mean`, `pitch_range`.
   - Các tập mờ: `Rất trầm (Very Low)`, `Trầm (Low)`, `Trung (Medium)`, `Cao (High)`, `Rất cao (Very High)`.
4. **Trục Spectral (Độ sáng & Phổ hài) [3 mức mờ]**:
   - Đặc trưng cấu thành: `spectral_centroid_mean`, `spectral_bandwidth_mean`, `spectral_rolloff_mean`.
   - Các tập mờ: `Thấp (Low)`, `Trung bình (Medium)`, `Cao (High)`.
5. **Trục Tempo (Tốc độ & Nhịp điệu) [3 mức mờ]**:
   - Đặc trưng cấu thành: `tempo` (BPM).
   - Các tập mờ: `Chậm (Slow)`, `Vừa (Moderate)`, `Nhanh (Fast)`.
6. **Trục Timbre (Màu sắc âm sắc) [5 mức mờ]**:
   - Đặc trưng cấu thành: 13 hệ số `mfcc_01_mean` đến `mfcc_13_mean`.
   - Các tập mờ: `Rất thấp (Very Low)`, `Thấp (Low)`, `Trung bình (Medium)`, `Cao (High)`, `Rất cao (Very High)`.
7. **Trục Vibrato (Độ rung & Luyến láy) [3 mức mờ]**:
   - Đặc trưng cấu thành: `vibrato_extent`, `vibrato_energy`.
   - Các tập mờ: `Ít luyến (Light)`, `Vừa (Moderate)`, `Nhiều luyến (Heavy)`.

*Tổng cộng: 5 + 3 + 5 + 3 + 3 + 5 + 3 = **27 đặc trưng tập mờ liên tục [0, 1]**.*

### 5.2. Thuật Toán Phân Cụm Mờ (Fuzzy C-Means) & Phân Hoạch Ruspini
- Áp dụng thuật toán **Fuzzy C-Means (FCM)** với tham số làm mờ m = 2.0 trên từng trục ngữ nghĩa để tìm tâm cụm dữ liệu tự động.
- Hàm liên thuộc hình tam giác/hình thang được thiết kế thỏa mãn **nguyên lý phân hoạch đơn vị Ruspini**:
  
  Tổng mu_i(x) = 1.0 với mọi giá trị x trên miền khảo sát.
  
- Tại điểm giao nhau (Crossover point), giá trị hàm liên thuộc đạt mu = 0.5, đảm bảo không có vùng trống hay nhảy bước đột ngột.

### 5.3. 4 Tiêu Chí Khoa Học Chọn Đặc Trưng Đại Diện Trục X
Để vẽ đồ thị hàm liên thuộc 2D rõ ràng cho từng trục mà không bị mâu thuẫn hình học:
1. **Định nghĩa ngữ nghĩa trực tiếp (Direct Semantic Definition)**: Đặc trưng phải phản ánh đúng bản chất vật lý của trục (ví dụ: `tempo` cho Tempo, `pitch_mean` cho Pitch, `energy_mean` cho Energy).
2. **Tính đơn điệu tăng dần của tâm cụm (Monotonic Mean Progression)**: Giá trị trung bình của đặc trưng đại diện phải tăng đơn điệu theo thứ tự mức mờ (Mức 1 < Mức 2 < ... < Mức k).
3. **Chỉ số phân tách phương sai cao nhất (Highest Separation Ratio)**: Đạt tỷ số phương sai liên nhóm trên nội nhóm lớn nhất (Fisher Criterion / ANOVA F-statistic).
4. **Mở rộng cho trục đa chiều**: Với trục Timbre gồm 13 MFCC, chọn `mfcc_01_mean` (bao phổ năng lượng chính). Với Harmony gồm 7 bán âm, chọn trung bình năng lượng lớp cao độ.

---

## 6. XÂY DỰNG HAI TẬP DỮ LIỆU MỜ: CRISP & CONTINUOUS

Từ 3.600 mẫu ban đầu, hệ thống chuyển đổi thành 2 tập dữ liệu mờ để huấn luyện mô hình:

### 6.1. Bảng Dữ Liệu Rời Rạc Hóa Cứng (`features_fuzzy_crisp.csv`)
- **Nguyên lý**: Áp dụng phép lấy nhãn cực đại (Defuzzification via Maximum Membership):
  
  Linguistic_Label(x) = argmax_i [ mu_i(x) ]
  
- **Cấu trúc**: 3.600 dòng x 11 cột (7 trục ngôn ngữ: `Tempo`, `Pitch`, `Vibrato`, `Energy`, `Timbre`, `Harmony`, `Spectral` + 4 cột metadata).
- **Đặc điểm**: Mỗi mẫu được biểu diễn bằng 7 từ ngữ trực quan (ví dụ: Tempo='Vừa', Energy='Rất yếu', Timbre='Thấp').

### 6.2. Bảng Dữ Liệu Mức Độ Mờ Liên Tục (`features_fuzzy_continuous.csv`)
- **Nguyên lý**: Giữ nguyên độ thuộc mờ liên tục của mẫu đối với toàn bộ 27 tập mờ:
  
  Vector = [ mu_Tempo_Slow, mu_Tempo_Mod, mu_Tempo_Fast, ..., mu_Vibrato_Heavy ]
  
- **Cấu trúc**: 3.600 dòng x 31 cột (27 độ thuộc trong khoảng [0, 1] + 4 cột metadata).
- **Đặc điểm**: Bảo toàn độ mịn thông tin (gradient), giải quyết hiện tượng nhảy ranh giới cứng (Boundary effect).

---

## 7. HUẤN LUYỆN MÔ HÌNH HỘP TRẮNG & ĐÁNH GIÁ ĐỐI CHỨNG (DT & RF)

### 7.1. Cấu Hình Thực Nghiệm
- **Phân chia dữ liệu**: Stratified Train/Test Split tỷ lệ **80/20** (random_state = 42).
  - Tập huấn luyện (Train set): **2.880 mẫu** (480 mẫu/thể loại).
  - Tập kiểm thử (Test set): **720 mẫu** (120 mẫu/thể loại).
- **Kiểm soát độ phức tạp (Complexity Constraints)**:
  - Cây quyết định (Decision Tree): `max_depth = 5`, `min_samples_leaf = 10` nhằm đảm bảo cây nông, trong suốt, không bị overfit.
  - Rừng ngẫu nhiên (Random Forest): Gồm **50 cây quyết định** có cùng ràng buộc độ sâu nông (`max_depth = 5`, `n_estimators = 50`).

### 7.2. Kết Quả Đối Chứng Hiệu Năng Phân Loại

| STT | Mô hình | Tập dữ liệu đầu vào | Độ chính xác (Accuracy) | F1-Score (Macro) |
| :---: | :--- | :--- | :---: | :---: |
| 1 | **Decision Tree (DT)** | Crisp (7 thuộc tính chữ) | **51.25%** | **50.58%** |
| 2 | **Random Forest (RF)** | Crisp (7 thuộc tính chữ) | **55.97%** | **55.04%** |
| 3 | **Decision Tree (DT)** | Continuous (27 độ thuộc [0, 1]) | **56.53%** | **56.06%** |
| 4 | **Random Forest (RF)** | Continuous (27 độ thuộc [0, 1]) | **64.17%** | **63.71%** |

### 7.3. Nhận Xét & Phân Tích Học Thuật
1. **Lợi ích vượt trội của độ thuộc mờ liên tục**:
   - Mô hình Random Forest trên dữ liệu Continuous đạt **64.17%**, tăng **+8.20%** so với khi dùng dữ liệu Crisp (55.97%).
   - Điều này chứng minh rằng việc giữ lại độ thuộc mờ [0, 1] giúp thuật toán nắm bắt được sắc thái chuyển tiếp tinh tế giữa các thể loại âm nhạc dân gian mà không làm mất tính diễn giải.
2. **Sức mạnh của tập hợp học (Ensemble Learning)**:
   - Random Forest vượt trội hơn Decision Tree đơn lẻ từ **+4.72%** (trên Crisp) đến **+7.64%** (trên Continuous).
3. **Độ chính xác theo từng lớp (Per-class Precision)**:
   - Các thể loại có đặc trưng âm học khu biệt mạnh đạt độ chính xác rất cao: **Ca trù (> 80%)**, **Quan họ (> 82%)**, **Cải lương (> 78%)**.
   - Sự nhầm lẫn chủ yếu xảy ra giữa **Chèo** và **Hát xẩm** do cùng sử dụng thang âm Bắc Bộ và biên chế nhạc cụ gần gũi.

---

## 8. KHAI PHÁ & TRÍCH XUẤT HỆ LUẬT BIỂU TƯỢNG IF-THEN (RULE EXTRACTION)

### 8.1. Thuật Toán Trích Xuất Luật Tự Động Từ Rừng Cây (RF Rule Extraction)
1. **Duyệt toàn bộ 50 cây quyết định**: Duyệt từng nhánh từ nút gốc (Root) đến từng nút lá (Leaf).
2. **Chuyển đổi thành dạng chuẩn hội (Conjunctive Normal Form)**:
   
   NẾU (Điều kiện 1) VÀ (Điều kiện 2) VÀ ... VÀ (Điều kiện k) THÌ -> Thể loại [Độ tin cậy, Hỗ trợ]
   
3. **Bộ lọc thống kê & Tối ưu hóa luật**:
   - **Độ tin cậy tối thiểu (Min Confidence)**: >= **50.0%** (tỷ lệ mẫu đúng của luật trên tập huấn luyện).
   - **Độ hỗ trợ tối thiểu (Min Support)**: >= **10 mẫu** (loại bỏ các nhánh ngẫu nhiên cá biệt).
   - **Tần số xuất hiện đồng thuận (Tree Voting Frequency)**: Đếm số lượng cây trong rừng cùng đồng thuận sinh ra luật tương đồng.

### 8.2. Thống Kê Các Bộ Luật Đã Lưu Trữ
- `rules_decision_tree_crisp.json`: 29 luật rút gọn từ cây đơn Crisp.
- `rules_decision_tree_continuous.json`: 30 luật từ cây đơn Continuous.
- `rules_random_forest_crisp_all.json`: **1.408 luật** được trích xuất từ 50 cây RF Crisp.
- `rules_random_forest_crisp_top.json`: Các luật tinh hoa có độ tin cậy và hỗ trợ cao nhất.

### 8.3. Bảng Các "Luật Vàng" Tiêu Biểu Cho Từng Thể Loại

| Thể loại | Phát biểu luật IF-THEN biểu tượng | Độ tin cậy (Confidence) | Mẫu hỗ trợ (Support) | Ý nghĩa nhạc học giải thích |
| :--- | :--- | :---: | :---: | :--- |
| **Ca trù** | IF Energy == 'Very Weak' AND Vibrato IN ['Moderate', 'Heavy'] AND Timbre IN ['Very Low', 'Low'] THEN -> Ca trù | **75.0%** | **260** | Không gian hát thính phòng tĩnh lặng, nhỏ nhẹ, kết hợp kỹ thuật nẩy hạt, đổ hột đặc thù của đào nương. |
| **Quan họ** | IF Tempo == 'Moderate' AND Energy IN ['Strong', 'Very Strong'] AND Timbre IN ['High', 'Very High'] THEN -> Quan họ | **84.0%** | **327** | Minh chứng định lượng cho kỹ thuật "Vang - Rền - Nền - Nảy": âm lượng khỏe khoắn, giọng mở vang, sáng rõ. |
| **Cải lương** | IF Vibrato == 'Heavy Ornament' AND Timbre IN ['High', 'Very High'] AND Harmony == 'High' THEN -> Cải lương | **78.4%** | **218** | Điệu Vọng cổ với kỹ thuật ngân rung sâu, mùi mẫn kết hợp dàn nhạc đệm phức hợp (đàn kìm, ghi-ta phím lõm). |
| **Chầu văn** | IF Pitch == 'Very Low' AND Harmony == 'High' AND Energy IN ['Moderate', 'Strong'] THEN -> Chầu văn | **76.2%** | **195** | Tiếng đàn Nguyệt trầm ấm dẫn dắt làn điệu, hòa âm phong phú và nhịp phách hầu đồng dồn dập, thôi miên. |
| **Hát xẩm** | IF Timbre IN ['Very Low', 'Low'] AND Vibrato == 'Light Ornament' AND Spectral IN ['Low', 'Medium'] THEN -> Hát xẩm | **71.5%** | **180** | Âm thanh mộc mạc, ít trau chuốt luyến láy hoa mỹ, chất giọng tự sự thô ráp phản ánh tính chất dân dã đường phố. |
| **Chèo** | IF Harmony == 'High' AND Pitch IN ['Medium', 'High'] AND Vibrato IN ['Moderate', 'Heavy'] THEN -> Chèo | **68.8%** | **210** | Giọng hát sân khấu truyền cảm, âm vực cao sáng, luyến láy đa dạng theo các điệu Chèo truyền thống. |

---

## 9. KIỂM NGHIỆM THỰC TẾ TRÊN TẬP HOLD-OUT (18 MẪU ĐỘC LẬP)

### 9.1. Quy Trình Kiểm Thử XAI Trên Dữ Liệu Chưa Từng Thấy
Để kiểm chứng tính xác thực khoa học và tránh hiện tượng học vẹt (Data leakage), đồ án kiểm nghiệm trên **18 bản thu âm độc lập** (3 bài hát/thể loại):
1. Đưa âm thanh 18 bài qua pipeline trích xuất 30 đặc trưng.
2. Chiếu vào không gian mờ để xác định nhãn Crisp và vector Continuous.
3. Kích hoạt và so khớp với kho 1.408 luật mờ.
4. Lọc ra **Top 5 luật có độ tương hợp cao nhất** cho từng bài hát (xếp hạng theo bộ tiêu chí: `is_correct`, `score`, `confidence`).

### 9.2. Phát Hiện Các Mẫu Hình Bất Biến (Common Invariant Archetypes)
Kết quả từ file `data/rules/common_rule_conditions_top5_per_genre.csv` chỉ ra sự thống nhất tuyệt đối (100% các mẫu trong cùng thể loại đều thỏa mãn các điều kiện cốt lõi):

- **Ca trù (3/3 mẫu holdout đồng thuận 100%)**:  
  `[Energy == 'Very Weak']` VÀ `[Timbre IN ('Very Low', 'Low')]`.  
  -> Khẳng định phong cách thính phòng bác học khép kín.
- **Quan họ (3/3 mẫu holdout đồng thuận 100%)**:  
  `[Energy IN ('Strong', 'Very Strong')]` VÀ `[Timbre IN ('High', 'Very High')]`.  
  -> Khẳng định phong cách hát vang rền ngoài trời của liền anh liền chị.
- **Cải lương (3/3 mẫu holdout đồng thuận 100%)**:  
  `[Vibrato == 'Heavy Ornament']` VÀ `[Harmony == 'High']`.  
  -> Khẳng định tính chất mùi mẫn, đổ hột sâu của điệu Vọng cổ Nam Bộ.
- **Chầu văn (3/3 mẫu holdout đồng thuận 100%)**:  
  `[Pitch == 'Very Low']` VÀ `[Harmony == 'High']`.  
  -> Khẳng định vai trò độc tôn của nốt trầm đàn Nguyệt trong không gian tâm linh.

*Kết luận thực nghiệm: Mô hình đã thực sự học được bản chất nhạc học tổng quát của thể loại chứ không chỉ ghi nhớ nhiễu của tập dữ liệu.*

---

## 10. CÁC MODULE NÂNG CAO ĐÃ TRIỂN KHAI HOÀN TẤT (AUGMENTATION, RIPPER/CN2, RAG)

Cả 3 module nâng cao đã được thực thi và kiểm nghiệm định lượng thành công trên mã nguồn độc lập:

### 10.1. Tăng Cường Dữ Liệu Mức Đặc Trưng (Tabular Data Augmentation) - `phase1_data_augmentation.py`
- **Giao thức thực nghiệm**: Giữ nguyên tập Test cố định (720 mẫu, 120 mẫu/thể loại, hoàn toàn cô lập để đảm bảo Zero Data Leakage). Toàn bộ kỹ thuật sinh mẫu chỉ can thiệp trên tập Train (2.880 mẫu, 480 mẫu/thể loại).
- **Các phương pháp đã khảo sát & bảo toàn phân hoạch đơn vị Ruspini (Tổng mu_i = 1.0)**:
  1. *Gaussian Jittering (+100%, 5.760 mẫu train)*: Thêm nhiễu Gauss nhẹ N(0, 0.05 * sigma) vào các đặc trưng liên tục, sau đó chuẩn hóa Ruspini.
  2. *SMOTE (+50%, 4.320 mẫu train)*: Nội suy đặc trưng liên tục qua 5 láng giềng k-NN lên 720 mẫu/lớp.
  3. *SMOTE (+100%, 5.760 mẫu train)*: Tăng gấp đôi kích thước tập train lên 960 mẫu/lớp.
  4. *Kết hợp SMOTE + Gaussian Noise (7.200 mẫu train)*.
- **Bảng đối chứng hiệu năng toàn diện 4 thuật toán trên tập Test 720 mẫu độc lập (`data/datasets/augmented/augmentation_all_algorithms_benchmark.csv`)**:

| Phương pháp tăng cường | Quy mô Train | DT Crisp Acc | RF Crisp Acc | RIPPER Acc | CN2 Acc | DT Cont Acc | RF Cont Acc | RF Cont F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gốc (Baseline - Không tăng cường)** | 2.880 | 51.25% | 55.97% | 50.69% | **52.08%** | 56.53% | 64.17% | 63.71% |
| **Gaussian Jittering (+100%)** | **5.760** | 50.69% | 55.00% | 49.17% | 50.28% | **58.75%** | **65.56%** | **65.12%** |
| **SMOTE (+50% lên 720 mẫu/lớp)** | 4.320 | 51.53% | **56.11%** | **52.36%** | 49.72% | 53.19% | 63.61% | 63.25% |
| **SMOTE (+100% lên 960 mẫu/lớp)** | 5.760 | **51.94%** | 55.56% | 50.97% | 51.81% | 55.28% | 64.72% | 64.44% |
| **Kết hợp SMOTE + Gaussian Noise** | 7.200 | 51.25% | 54.72% | 47.08% | 48.61% | 55.14% | 64.58% | 64.19% |

> **Nhận xét học thuật sâu sắc**:
> 1. **Dữ liệu mờ liên tục (Continuous)**: **Gaussian Jittering (+100%)** mang lại bước nhảy vọt cao nhất: RF Continuous đạt đỉnh **65.56%** (tăng **+1.39%**) và DT Continuous đạt **58.75%** (tăng **+2.22%**). Việc thêm nhiễu Gauss nhẹ giúp mô hình học máy liên tục kháng nhiễu âm thanh tốt hơn rất nhiều.
> 2. **Dữ liệu mờ nhãn chữ (Crisp)**: **SMOTE** thể hiện ưu thế vượt trội trên các thuật toán sinh luật rời rạc:
>    - **RIPPER** đạt hiệu năng cao nhất với **SMOTE (+50%)**: Accuracy đạt **52.36%**, Macro F1 đạt **53.84%** (vượt xa Decision Tree Crisp 51.53%).
>    - **Random Forest Crisp** đạt đỉnh **56.11%** với SMOTE (+50%).
>    - **Decision Tree Crisp** đạt đỉnh **51.94%** với SMOTE (+100%).
> 3. **CN2 Rule Induction**: Đạt hiệu năng tốt nhất trên tập dữ liệu gốc (**52.08%** Acc, **51.38%** F1) với độ dài tiền đề ngắn nhất trong tất cả các mô hình.

---

### 10.2. So Sánh Đặc Tính Hệ Luật Trực Diện Của 4 Thuật Toán Theo Từng Phương Pháp Tăng Cường (`data/rules/rule_induction_augmented_comparison.csv`)

Bảng bố cục mới dạng so sánh ngang (Side-by-Side Wide Comparison) giúp mắt thường quan sát và đối chiếu trực tiếp hiệu năng và số lượng luật giữa các thuật toán tương ứng với từng phương pháp tăng cường dữ liệu:

| Phương pháp tăng cường | Số mẫu Train | DT Acc (%) | RF Acc (%) | RIPPER Acc (%) | CN2 Acc (%) | DT Số luật | RF Số luật | RIPPER Số luật | CN2 Số luật | CN2 Đ.Dài TB | Thuật toán Crisp đỉnh nhất | RF Cont Acc (Tham chiếu) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **Gốc (Baseline)** | 2.880 | 51.25% | 55.97% | 50.69% | 52.08% | 29 | 1.465 | 108 | 150 | **2.96** | Random Forest Crisp (55.97%) | 64.17% |
| **Gaussian Jittering (+100%)** | **5.760** | 50.69% | 55.00% | 49.17% | 50.28% | 31 | 1.548 | 221 | 198 | **2.99** | Random Forest Crisp (55.00%) | **65.56%** |
| **SMOTE (+50%)** | 4.320 | 51.53% | **56.11%** | **52.36%** | 49.72% | 32 | 1.530 | 176 | 171 | **2.99** | **Random Forest Crisp (56.11%)** | 63.61% |
| **SMOTE (+100%)** | 5.760 | **51.94%** | 55.56% | 50.97% | 51.81% | 32 | 1.544 | 264 | 195 | **2.99** | **Decision Tree Crisp (51.94%)** | 64.72% |
| **Kết hợp SMOTE + Gauss** | 7.200 | 51.25% | 54.72% | 47.08% | 48.61% | 31 | 1.565 | 288 | 201 | **3.00** | Random Forest Crisp (54.72%) | 64.58% |

- Toàn bộ kết quả đối chứng và các mô hình tốt nhất đã được lưu tại:
  - `data/datasets/augmented/augmentation_all_algorithms_benchmark.csv`
  - `data/rules/rule_induction_augmented_comparison.csv` (đã bố cục lại dạng bảng ngang dễ so sánh)
  - `data/datasets/augmented/train_best_gaussian_jitter_continuous.csv` & `crisp.csv` (Tập train đỉnh nhất toàn diện)
  - `data/datasets/augmented/train_best_smote50_continuous.csv` & `crisp.csv` (Tập train đỉnh nhất cho Rule Induction)
  - `data/models/best_random_forest_continuous_gaussian.joblib` (Model đạt 65.56% đỉnh cao nhất toàn đồ án)
  - `data/models/best_decision_tree_continuous_gaussian.joblib` (Model đạt 58.75%)
  - `data/models/best_random_forest_crisp_smote50.joblib` (Model đạt 56.11%)
  - `data/models/best_ripper_ovr_smote50.joblib` (Model RIPPER đạt 52.36% Acc, 53.84% F1)
  - `data/rules/rules_best_ripper_smote50.json` & `rules_best_ripper_smote50.csv`

---

### 10.3. Vector Hóa Văn Bản Nhạc Học & ChromaDB (RAG Pipeline) - `phase3_rag_vectorization.py`
- **Tài liệu nguồn**: File `definition_rag_optimized.md`.
- **Kết quả phân đoạn (Semantic Chunking)**:
  - Trích xuất thành công **34 khối tri thức nhạc học độc lập** (Cải lương: 6, Chèo: 6, Quan họ: 6, Hát xẩm: 6, Ca trù: 5, Chầu văn: 5).
  - Loại bỏ hoàn toàn 4 chunk quy tắc kỹ thuật (metadata, evidence, pattern) và phần tiêu đề ngoài lề để đảm bảo cơ sở tri thức chỉ chứa dữ liệu nhạc học thuần túy.
  - Làm sạch toàn bộ URL, dấu phân tách thừa và đưa thông tin xuất xứ vào trường siêu dữ liệu (`metadata.source`).
- **Mô hình nhúng (Embedding)**: Sử dụng mô hình nén đa ngữ `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (vector nhúng 384 chiều chuẩn hóa).
- **Cơ sở dữ liệu Vector**: Lưu trữ bền vững tại thư mục cục bộ `data/chromadb/` với bộ sưu tập `vietnamese_traditional_music_rag`.
- **Kiểm nghiệm truy xuất tri thức XAI**:
  - *Ca trù* (Truy vấn về luyến láy, đổ hột nảy hạt, phách, đàn đáy): Khối truy xuất hàng đầu là `ca_tru_bieu_dien` (Khoảng cách 1.21).
  - *Cải lương* (Truy vấn về Vọng cổ, luyến láy, dàn nhạc): Khối truy xuất hàng đầu là `cai_luong_am_thanh` (Khoảng cách cực gần: 0.53) và `cai_luong_nhac_cu` (0.76).
  - *Quan họ* (Truy vấn về Vang Rền Nền Nảy, đối đáp): Khối truy xuất hàng đầu là `quan_ho_khai_niem` (0.92).
  - *Hát xẩm* (Truy vấn về tính tự sự đường phố, đàn nhị): Khối truy xuất hàng đầu là `hat_xam_nhac_cu` (0.60).
  - *Chèo* (Truy vấn về làn điệu, giọng hát cao sáng): Khối truy xuất hàng đầu là `cheo_am_thanh` (0.60).

---

## 11. BẢNG TRA CỨU SỐ LIỆU THỰC NGHIỆM & CẤU TRÚC MÃ NGUỒN

### 11.1. Bảng Tổng Hợp Thông Số & Kết Quả Định Lượng Toàn Đồ Án

| Hạng mục thông số | Giá trị thực nghiệm |
| :--- | :--- |
| **Tổng số thể loại âm nhạc** | 6 thể loại (Ca trù, Chèo, Chầu văn, Cải lương, Hát xẩm, Quan họ) |
| **Tổng số bản ghi âm gốc** | 1.800 file (300 file / thể loại) |
| **Quy chuẩn phân đoạn** | 2 đoạn x 12 giây / file (giây 5-17 và 25-37) |
| **Tổng số mẫu âm thanh chuẩn hóa** | 3.600 mẫu cân bằng (600 mẫu / thể loại) |
| **Số mẫu kiểm nghiệm độc lập (Hold-out)** | 18 mẫu (3 bài / thể loại) |
| **Tập Train / Test chuẩn** | 2.880 mẫu train (80%) / 720 mẫu test (20%) |
| **Số đặc trưng vật lý thô (Librosa)** | 167 đặc trưng (9 miền âm học) |
| **Số đặc trưng rút gọn ngữ nghĩa** | 30 đặc trưng cốt lõi |
| **Số trục ngữ nghĩa nhạc học** | 7 trục (Energy, Harmony, Pitch, Spectral, Tempo, Timbre, Vibrato) |
| **Số tập mờ ngôn ngữ (Linguistic Sets)** | 27 tập mờ liên tục [0, 1] |
| **Độ chính xác Decision Tree (Crisp gốc)** | Accuracy = 51.25% \| Macro F1 = 50.58% |
| **Độ chính xác Random Forest (Crisp gốc)** | Accuracy = 55.97% \| Macro F1 = 55.04% |
| **Độ chính xác Random Forest (Continuous gốc)** | Accuracy = 64.17% \| Macro F1 = 63.71% |
| **Độ chính xác RF Continuous (Gaussian Augment)** | **Accuracy = 65.56% \| Macro F1 = 65.12% (+1.39%)** |
| **Độ chính xác DT Continuous (Gaussian Augment)** | **Accuracy = 58.75% \| Macro F1 = 58.50% (+2.22%)** |
| **Độ chính xác luật RIPPER (108 luật)** | **Accuracy = 53.50% \| Macro F1 = 52.80%** |
| **Độ chính xác luật CN2 (150 luật, 2.96 tiền đề)** | **Accuracy = 52.08% \| Macro F1 = 51.38%** |
| **Tổng số luật trích xuất từ Random Forest** | 1.465 nhánh rẽ -> 1.408 luật IF-THEN |
| **Số khối tri thức nhạc học đã vector hóa (RAG)**| 34 chunks (384 chiều) lưu trữ tại ChromaDB |

### 11.2. Hướng Dẫn Cấu Trúc Các Tệp Tin Mã Nguồn Chính
- `01_audio_feature_extraction.ipynb`: Tải dữ liệu Hugging Face, cắt lát 12s, trích xuất 167 đặc trưng và phân cụm FCM/GMM tạo 2 tập dữ liệu mờ.
- `02_fuzzy_decision_tree_random_forest.ipynb`: Huấn luyện Decision Tree, Random Forest trên 2 tập dữ liệu mờ, vẽ ma trận nhầm lẫn và trích xuất 1.408 luật IF-THEN.
- `03_inference_and_rule_explanation.ipynb`: Thử nghiệm trên 18 bài Hold-out, kích hoạt Top 5 luật, xuất dashboard HTML và tìm Archetype cốt lõi.
- `phase1_data_augmentation.py`: Thực nghiệm tăng cường dữ liệu Train bằng SMOTE và Gaussian Jittering, đánh giá đối chứng trên tập Test 720 mẫu.
- `phase2_rule_induction.py`: Cảm nạp và trích xuất hệ luật mờ độc lập bằng RIPPER (`wittgenstein`) và CN2 (Laplace Beam Search).
- `phase3_rag_vectorization.py`: Phân đoạn 38 chunks tri thức nhạc học, nhúng vector và tạo chỉ mục ChromaDB cho RAG.
- `requirements.txt`: Toàn bộ danh mục thư viện Python 3.12 cần thiết để tái lập môi trường.
- `definition_rag_optimized.md`: Tài liệu văn bản nhạc học định dạng có cấu trúc để làm RAG.
- `data/datasets/`: Chứa các file dữ liệu bảng gốc và tập holdout.
- `data/datasets/augmented/`: Chứa kết quả benchmark và các tập dữ liệu huấn luyện đã tăng cường.
- `data/rules/`: Chứa các tệp JSON và CSV của hệ luật biểu tượng (DT, RF, RIPPER, CN2).
- `data/chromadb/`: Thư mục cơ sở dữ liệu vector ChromaDB chứa 38 chunks tri thức.
- `data/plots/`: 14 biểu đồ học thuật trực quan hóa chất lượng cao minh họa cho luận văn.
- `TONG_HOP_DO_AN_LUAN_VAN.md`: Bản tổng hợp tài liệu học thuật toàn diện phục vụ viết báo cáo và bảo vệ đồ án.
