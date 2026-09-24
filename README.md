# HerittageFNS — Bảo Tồn & Nhận Diện Di Sản Âm Nhạc Dân Tộc Việt Nam
> **Bước 1: Trích Xuất Đặc Trưng Âm Thanh Đa Chiều & Xây Dựng Không Gian Hồ Sơ Mờ (Audio Feature Extraction & Fuzzy Clustering Pipeline)**

---

## 1. Tổng Quan Dữ Liệu & Tiền Xử Lý (Audio Preprocessing)

* **Bộ dữ liệu nguồn**: `NTQAI/Vietnamese-Traditional-Music` (Hugging Face Hub).
* **6 thể loại âm nhạc truyền thống**: Cải lương, Quan họ, Chầu văn, Chèo, Ca trù, Hát xẩm.
* **Quy mô thực nghiệm**: Lấy mẫu cân bằng (Balanced sampling) 300 mẫu/thể loại, tổng cộng 1.800 mẫu âm thanh.
* **Chuẩn hóa kỹ thuật DSP**:
  * **Resample**: Chuẩn hóa toàn bộ tần số lấy mẫu về **16.000 Hz** (16 kHz), đồng bộ tương thích với các mô hình Audio Foundation Models ở các bước sau (MERT, AudioMAE).
  * **Channel**: Chuyển đổi về đơn kênh (Mono) nhằm triệt tiêu hiện tượng lệch pha tín hiệu stereo.
  * **Audio Segmentation**: Áp dụng cửa sổ trung tâm **10 giây** tiêu biểu sau khi lọc bỏ khoảng lặng (trim silence), tối ưu hóa việc nắm bắt đặc trưng cục bộ (luyến láy, rung âm) và tăng tốc độ xử lý.

---

## 2. Thống Kê 167 Đặc Trưng Âm Thanh Được Trích Xuất

Toàn bộ 167 đặc trưng số học sau khi trích xuất được lưu trữ trong file `data/features_all.csv` (1.800 dòng × 169 cột, gồm 2 cột nhãn metadata và 167 cột đặc trưng), phân bổ theo 9 nhóm âm học chuyên sâu:

| STT | Nhóm đặc trưng | Số lượng | Danh sách đặc trưng chi tiết | Ý nghĩa âm học |
| :---: | :--- | :---: | :--- | :--- |
| 1 | **Cao độ (Pitch / F0)** | **5** | `pitch_mean`, `pitch_std`, `pitch_range`, `pitch_median`, `voiced_ratio` | Tần số dao động thanh đới cơ bản trích từ thuật toán pYIN, độ biến thiên cao độ và tỷ lệ đoạn có tiếng. |

| 2 | **Nhịp điệu (Tempo & Beat)** | **3** | `tempo`, `beat_count`, `beat_regularity` | Tốc độ nhịp (BPM), số lượng xung nhịp và độ ổn định của bước nhịp. |

| 3 | **Rung âm (Vibrato)** | **2** | `vibrato_extent`, `vibrato_energy` | Đặc trưng luyến láy, đổ hột qua bộ lọc thông dải Butterworth (dải tần 4 - 10 Hz) trên đường bao F0. |

| 4 | **Năng lượng (Energy / RMS)** | **4** | `energy_mean`, `energy_std`, `dynamic_range`, `energy_skew` | Cường độ năng lượng gốc (RMS), độ phân tán, dải động lực và độ lệch phân phối biên độ. |

| 5 | **Âm sắc MFCC & Đạo hàm** | **106** | • 40 hệ số tĩnh (mean & std): `mfcc_01_mean` -> `mfcc_40_std` (80 đặc trưng)<br>• 13 hệ số vận tốc Delta (mean): `dmfcc_01_mean` -> `dmfcc_13_mean` (13 đặc trưng)<br>• 13 hệ số gia tốc Delta-Delta (mean): `d2mfcc_01_mean` -> `d2mfcc_13_mean` (13 đặc trưng) | Dấu vân tay âm sắc (Timbre) mô tả hình dáng bao phổ cảm nhận bởi thính giác con người và sự biến thiên âm sắc theo thời gian. |

| 6 | **Hòa âm (Chroma STFT)** | **25** | • 12 bán âm (mean & std): `chroma_C_mean` -> `chroma_B_std` (24 đặc trưng)<br>• Độ lệch chuẩn toàn cục: `chroma_std_overall` (1 đặc trưng) | Phân bố năng lượng trên 12 nửa cung (C, C#, D, D#, E, F, F#, G, G#, A, A#, B), phản ánh thang âm và điệu thức dân gian. |

| 7 | **Đặc tính phổ & Tần số (Spectral & ZCR)** | **14** | `spectral_centroid_mean`, `spectral_centroid_std`, `spectral_bandwidth_mean`, `spectral_rolloff_mean`, `spectral_flatness_mean`, `spectral_contrast_1` -> `spectral_contrast_7` (7 dải tần), `zcr_mean`, `zcr_std` | Trọng tâm phổ (độ sáng âm thanh), độ rộng dải phổ, điểm suy giảm phổ, độ phẳng phổ, độ tương phản âm sắc và tỷ lệ đổi dấu tín hiệu. |

| 8 | **Không gian hợp âm (Tonnetz)** | **6** | `tonnetz_1`, `tonnetz_2`, `tonnetz_3`, `tonnetz_4`, `tonnetz_5`, `tonnetz_6` | Hệ tọa độ hòa thanh 6 chiều mô tả quan hệ quãng năm, quãng ba trưởng/thứ trong cấu trúc điệu thức. |

| 9 | **Quang phổ Mel (Mel-Spectrogram)** | **2** | `mel_mean`, `mel_std` | Giá trị trung bình và độ lệch chuẩn năng lượng trên 128 dải lọc tần số Mel. |
| **TỔNG** | **Toàn bộ đặc trưng âm thanh** | **167** | | |

---

## 3. Các Trục Ngữ Nghĩa (Semantic Axes) & 27 Mức Độ Mờ (Fuzzy Levels)

Để chuyển đổi từ các con số kỹ thuật vô hướng sang **Hồ sơ mờ có thể diễn giải được (Interpretable Fuzzy Profiles)**, 167 đặc trưng được chuẩn hóa theo thang đo [0, 1] qua `MinMaxScaler` và gom nhóm thành **7 Trục ngữ nghĩa cốt lõi** (`FUZZY_AXES`).

Dựa trên tổng quan y văn nghiên cứu chuyên ngành (Music Information Retrieval & Fuzzy Audio Analysis), số lượng mức độ mờ cho từng trục được thiết lập chuẩn xác như sau:
* **Energy**: 5 mức (`Rất yếu`, `Yếu`, `Vừa`, `Mạnh`, `Rất mạnh`)
* **Harmony**: 3 mức (`Thấp`, `Trung bình`, `Cao`)
* **Pitch**: 5 mức (`Rất trầm`, `Trầm`, `Trung`, `Cao`, `Rất cao`)
* **Spectral**: 3 mức (`Thấp`, `Trung bình`, `Cao`)
* **Tempo**: 3 mức (`Chậm`, `Vừa`, `Nhanh`)
* **Timbre**: 5 mức (`Rất thấp`, `Thấp`, `Trung bình`, `Cao`, `Rất cao`)
* **Vibrato**: 3 mức (`Ít luyến`, `Vừa`, `Nhiều luyến`)
* **Tổng cộng**: 5 + 3 + 5 + 3 + 3 + 5 + 3 = **27 đặc trưng mức độ mờ liên tục [0, 1]**.

```
167 Đặc trưng số học thô (features_all.csv)
       │
       ▼ (Gom nhóm theo tính chất vật lý & nhạc học)
 7 Trục ngữ nghĩa (Semantic Axes)
       │
       ▼ (Phân cụm mờ Fuzzy C-Means & GMM)
 Hồ sơ mờ ngữ nghĩa (fuzzy_profiles.json)
```

### Chi tiết 7 Trục ngữ nghĩa và các đặc trưng thành phần:

1. **Trục `Tempo` (Nhịp điệu)**
   * **Các đặc trưng thành phần**: `tempo` (BPM), `beat_regularity` (độ ổn định nhịp).
   * **Các tập mờ ngôn ngữ (Linguistic terms)**: `Chậm`, `Vừa`, `Nhanh`.
   * **Ý nghĩa**: Phản ánh tốc độ diễn tấu của tiết tấu bài nhạc.

2. **Trục `Pitch` (Cao độ)**
   * **Các đặc trưng thành phần**: `pitch_mean` (tần số F0 trung bình), `pitch_range` (độ rộng âm vực).
   * **Các tập mờ ngôn ngữ**: `Trầm`, `Trung`, `Cao`.
   * **Ý nghĩa**: Định hình cao độ giọng hát chính của nghệ nhân và nhạc cụ dẫn dắt.

3. **Trục `Vibrato` (Độ rung & Luyến láy)**
   * **Các đặc trưng thành phần**: `vibrato_extent` (độ sâu rung ngân), `vibrato_energy` (năng lượng dải luyến 4 - 10 Hz).
   * **Các tập mờ ngôn ngữ**: `Ít luyến`, `Vừa`, `Nhiều luyến`.
   * **Ý nghĩa**: Bắt trọn kỹ thuật hát đặc thù của nhạc cổ truyền (đổ hột trong Ca trù, nảy hạt trong Quan họ, ngân rung trong Chầu văn).

4. **Trục `Energy` (Năng lượng âm thanh)**
   * **Các đặc trưng thành phần**: `energy_mean` (RMS trung bình), `dynamic_range` (khoảng chênh lệch to/nhỏ).
   * **Các tập mờ ngôn ngữ**: `Yếu`, `Vừa`, `Mạnh`.
   * **Ý nghĩa**: Biểu thị cường độ âm thanh và tính chất trữ tình êm dịu hay hào hùng, sôi động.

5. **Trục `Timbre` (Màu sắc âm sắc)**
   * **Các đặc trưng thành phần**: 13 hệ số MFCC bậc đầu (`mfcc_01_mean` -> `mfcc_13_mean`).
   * **Các tập mờ ngôn ngữ**: `Rất thấp`, `Thấp`, `Trung bình`, `Cao`.
   * **Ý nghĩa**: Mô tả cấu trúc chất giọng và sự pha trộn âm sắc giữa các nhạc cụ dân tộc trong dàn nhạc.

6. **Trục `Harmony` (Cấu trúc hòa âm & Bán âm)**
   * **Các đặc trưng thành phần**: 7 bán âm chủ đạo (`chroma_C_mean`, `chroma_D_mean`, `chroma_E_mean`, `chroma_F_mean`, `chroma_G_mean`, `chroma_A_mean`, `chroma_B_mean`).
   * **Các tập mờ ngôn ngữ**: `Rất thấp`, `Thấp`, `Trung bình`, `Cao`.
   * **Ý nghĩa**: Thể hiện các nốt trụ và cấu trúc thang âm ngũ cung đặc trưng của âm nhạc phương Đông.

7. **Trục `Spectral` (Độ sáng & Phân bố dải tần)**
   * **Các đặc trưng thành phần**: `spectral_centroid_mean` (độ sáng phổ), `spectral_bandwidth_mean` (độ rộng dải tần), `spectral_rolloff_mean` (ngưỡng suy giảm tần số).
   * **Các tập mờ ngôn ngữ**: `Thấp`, `Trung bình`, `Cao`.
   * **Ý nghĩa**: Đánh giá sự phong phú của họa âm tần số cao do các nhạc cụ gõ (thanh la, não bạt, mõ, trống) tạo ra.

---

## 4. Cấu Trúc Thư Mục Kết Quả (`data/`)

```
data/
├── features_all.csv              # Bảng chứa đầy đủ 1.800 mẫu × 167 đặc trưng âm học
├── features_fuzzy_crisp.csv      # Bảng 1.800 mẫu × 7 cột nhãn chữ (Rời rạc hóa cứng)
├── features_fuzzy_continuous.csv # Bảng 1.800 mẫu × 27 cột mức độ mờ liên tục [0, 1]
├── features_cache.pkl            # Cache nhị phân chứa spectrogram, F0 contour, waveform thô
├── fuzzy_profiles.json           # Hồ sơ mờ định lượng dạng JSON cho 6 thể loại
├── fcm_results.pkl               # Kết quả huấn luyện ma trận độ thuộc Fuzzy C-Means
├── gmm_results.pkl               # Kết quả xác suất hậu nghiệm Gaussian Mixture Model
│
├── 01_genre_distribution.png     # Biểu đồ phân phối số lượng mẫu gốc và cân bằng
├── 02_violin_comparison.png      # Violin plot so sánh mật độ phân phối đặc trưng giữa 6 thể loại
├── 03_radar_chart.png            # Biểu đồ mạng nhện "dấu vân tay âm thanh" 8 chiều
├── 04_pca_tsne.png               # Không gian đặc trưng giảm chiều 2D (PCA & t-SNE)
├── 05_correlation.png            # Ma trận tương quan Pearson giữa các đặc trưng
├── 06_optimal_k.png              # Xác định số cụm tối ưu qua Elbow, Silhouette và BIC
├── 07_clustering_comparison.png  # So sánh 4 thuật toán (K-Means, GMM, FCM, DBSCAN vs Ground Truth)
├── 08_fuzzy_profile_heatmap.png  # Bản đồ nhiệt hồ sơ mờ tổng hợp của 6 thể loại
│
├── membership_*.png              # 7 biểu đồ hàm liên thuộc mờ trên 7 trục ngữ nghĩa
└── viz_*.png                     # 6 biểu đồ phân tích âm phổ mẫu tiêu biểu cho từng thể loại
```

---

## 5. Huấn Luyện Decision Tree, Random Forest & Trích Xuất Hệ Luật Mờ (`02_fuzzy_decision_tree_random_forest.ipynb`)

Pipeline đã thực hiện huấn luyện đối chứng 4 mô hình trên tập Train/Test (80/20 Stratified) từ 2 bảng dữ liệu mờ:

### Kết quả đối chứng hiệu năng phân loại:
* **1. Decision Tree (Crisp 7 trục)**: Accuracy = **43.89%** | F1-Score = **43.23%**
* **2. Random Forest (Crisp 7 trục)**: Accuracy = **50.56%** | F1-Score = **50.37%**
* **3. Decision Tree (Continuous 27 mức)**: Accuracy = **50.00%** | F1-Score = **48.70%**
* **4. Random Forest (Continuous 27 mức)**: Accuracy = **58.61%** | F1-Score = **58.44%**

> **Nhận xét học thuật**: Giữ lại các giá trị mức độ mờ liên tục [0, 1] giúp mô hình Random Forest tăng thêm hơn **8.0%** độ chính xác so với việc ép cứng thành 7 nhãn chữ, đồng thời vẫn bảo toàn 100% tính diễn giải ngôn ngữ tự nhiên.

### Hệ thống file kết quả luật được tạo:
* `data/rules_decision_tree_crisp.json`: 28 luật từ cây quyết định rời rạc.
* `data/rules_decision_tree_continuous.json`: 29 luật từ cây quyết định mờ liên tục.
* `data/rules_random_forest_all.json`: Toàn bộ 1.391 nhánh rẽ luật từ 50 cây con của Random Forest.
* `data/rules_random_forest_top.json`: Top 5 luật vàng có độ tin cậy và độ phủ cao nhất cho từng thể loại.
* `data/rules_summary.csv`: Bảng tổng kết các luật vàng dạng văn bản dễ đọc.
* `data/09_decision_tree_crisp_graph.png`: Sơ đồ trực quan Cây quyết định Crisp.
* `data/10_decision_tree_continuous_graph.png`: Sơ đồ trực quan Cây quyết định Mờ Continuous.
* `data/11_rf_feature_importance.png`: Đồ thị tỷ trọng quan trọng của 27 mức độ mờ.
* `data/12_model_performance_comparison.png`: Biểu đồ so sánh Accuracy 4 mô hình và Confusion Matrix.
