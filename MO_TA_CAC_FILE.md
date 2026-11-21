# BẢN MÔ TẢ CẤU TRÚC THƯ MỤC & CHỨC NĂNG CÁC FILE CỐT LÕI TRONG DỰ ÁN

*(Tài liệu bàn giao cấu trúc mã nguồn, dữ liệu, trọng số mô hình và quy trình khởi chạy hệ thống)*

---

## 1. Sơ đồ Bố cục Thư mục Dự án

```text
LuanVan/
├── BAO_CAO_TONG_HOP_TOAN_DU_AN.md       # Báo cáo tổng hợp quá trình nghiên cứu, phát hiện khoa học & Pipeline E2E
├── MO_TA_CAC_FILE.md                    # Tài liệu này: Mô tả bố cục thư mục & chức năng từng file cốt lõi
├── MO_TA_GIAO_DIEN.md                   # Đặc tả chi tiết toàn bộ giao diện, luồng tương tác & 6 khu vực XAI của Web HERA
├── overall.txt                          # Tóm tắt nhanh các chỉ số cốt lõi của toàn bộ dự án
├── requirements.txt                     # Danh sách thư viện Python cần thiết
├── howtorun.txt                         # Hướng dẫn câu lệnh khởi chạy nhanh ứng dụng Web
│
├── web_app/                             # Hệ thống Ứng dụng Web Phân lớp & Giải thích Đa phương thức (FastAPI + Canvas 2D + GSAP)
│   ├── app.py                           # Máy chủ Backend FastAPI (REST API)
│   ├── inference_engine.py              # Bộ máy suy luận Đa phương thức (Audio 4-Family Ensemble + Visual CBM + Silence/OOD Gate + XAI)
│   ├── test_backend.py                  # Script kiểm thử tự động các API và luồng can thiệp khái niệm CBM
│   ├── templates/index.html             # Giao diện người dùng chính (Editorial Academic Light Mode + Slide Drawer 6 khu vực)
│   └── static/                          # Tài nguyên Frontend (CSS, JS, ảnh tư liệu 6 di sản, biểu tượng, Canvas 2D)
│
├── phase7_audio_visual_concept_mapper.py # Cầu nối ngữ nghĩa 2 tầng dịch luật mờ & ánh xạ khái niệm UNESCO/ICHCAP
├── concept/                             # Từ điển khái niệm di sản chuẩn hóa (43 Audio Concepts, 38 Visual Concepts)
│
├── experiments/                         # Kết quả huấn luyện Thế hệ 2: 21 mô hình Đa phương thức & Neuro-Symbolic (M1a–M5e)
│   ├── checkpoints/                     # Đúng 21 file trọng số (_best.pth) và 21 file lịch sử huấn luyện (_history.json)
│   ├── results/                         # Bảng chỉ số 21 mô hình & Bảng đối chứng tổ hợp Multimodal (CSV)
│   └── plots/                           # Hệ thống biểu đồ 300 DPI gom nhóm theo 5 Phương án kiến trúc
│
├── data/                                # Kho dữ liệu đã xử lý, mô hình thị giác, bộ luật mờ và báo cáo đối chứng
│   ├── datasets/                        # Tập đặc trưng 167 chiều, 27 độ thuộc mờ và bảng chia chống rò rỉ (split_group_v2.csv)
│   ├── annotations/                     # Tập dữ liệu gán nhãn 25 khái niệm thị giác (visual_concepts_dataset.csv)
│   ├── models/                          # Trọng số Visual CBM (ConvNeXt-Tiny), Visual RF, GMM 7 trục và hồ sơ mờ
│   ├── rules/                           # Các bộ luật IF-THEN (60 luật mờ liên tục DT/RF, Crisp, RIPPER, CN2, Visual RIPPER)
│   ├── results/                         # Các báo cáo đối chứng (Multimodal Ablation, In-The-Wild YouTube, 167 Raw Features)
│   └── plots/                           # Biểu đồ phân tích dữ liệu âm học, hàm thuộc mờ Ruspini và phổ mẫu
│
├── dataset/video/                       # Kho 131 video biểu diễn thực tế thu thập trên YouTube (6 thể loại + thư mục test/)
│
└── [Các Notebook & Script Huấn luyện / Đánh giá chính ở thư mục gốc]
    ├── 01_audio_feature_extraction.ipynb
    ├── 02_fuzzy_decision_tree_random_forest.ipynb
    ├── 03_raw_features_benchmark_colab.ipynb
    ├── Visual_Data_Prep_and_SemiAuto_Labeling_Colab.ipynb
    ├── Visual_CBM_and_Rule_Extraction_Colab.ipynb
    ├── build_multimodal_audio_cnn_notebook.py
    ├── train_multimodal_audio_cnn_experiments.ipynb
    ├── evaluate_wild_videos.py
    └── evaluate_audio_youtube_benchmark.py
```

---

## 2. Nhóm File Vận hành Hệ thống Ứng dụng Web & Suy luận Đa phương thức (`web_app/`)

1. **[web_app/app.py](web_app/app.py)**
   - Máy chủ Backend FastAPI chính của hệ thống.
   - Cung cấp các REST API: phân tích file Audio/Video tải lên (`/api/analyze`), phân tích trực tiếp link YouTube bằng `yt-dlp` hỗ trợ Node JS runtime và tự động cắt 180s nếu video dài quá 10 phút (`/api/analyze_url`), cắt ngắn video dài trên 10 phút kèm âm thanh bằng `ffmpeg` (`/api/trim_video`), nạp 6 mẫu biểu diễn chuẩn đã hiệu chuẩn (`/api/sample_files`, `/api/analyze_sample`) và xử lý can thiệp khái niệm phản sự thực CBM (`/api/intervene`).
2. **[web_app/inference_engine.py](web_app/inference_engine.py)**
   - Bộ máy suy luận trung tâm (`HeritageInferenceEngine`):
     - Vận hành **Tổ hợp 4 Họ Kiến trúc Neuro-Symbolic Audio (`0.25 M2a + 0.25 M3c + 0.25 M4d + 0.25 M5c`)** trên cửa sổ ngữ cảnh 12 giây, trích xuất đồng thời 7 trục MPEG-7 và 27 khái niệm mờ từ Concept Head của `M5c`.
     - Vận hành **Nhánh Thị giác (`best_visual_cbm.pth` ConvNeXt-Tiny + `best_visual_rf_continuous.joblib`)** trên các khung hình keyframe sắc nét nhất của mỗi phân đoạn.
     - Thực hiện dung hợp muộn có điều kiện (`0.70 Audio + 0.30 Visual` chỉ khi phát hiện ít nhất 1 thực thể thị giác di sản dương tính, loại trừ nhãn âm tính `khong_nhac_cu`; nếu không có dấu hiệu thị giác di sản thì giữ nguyên nhánh Âm thanh), gộp phân đoạn thông minh (Smart Grouping), **Cổng kiểm tra im lặng / độ phẳng phổ (`Silence & Spectral Flatness Gate`)** kết hợp **Cổng nhận diện ngoại lai (OOD Gate)**, đối chiếu 60 luật mờ liên tục + luật Crisp/RIPPER/Visual, sinh Bản đồ Bằng chứng Đa phương thức (`Multimodal Evidence Map`), Giải thích Đối chứng (`Contrastive Explanation`) và Can thiệp Khái niệm (`Counterfactual CBM Intervention`).
3. **[phase7_audio_visual_concept_mapper.py](phase7_audio_visual_concept_mapper.py)**
   - Cầu nối ngữ nghĩa 2 tầng (`AudioRuleToConceptBridge` và `VisualConceptBridge`): Phiên dịch tự động các mệnh đề luật toán học (như `Pitch_Trầm > 0.300`) sang câu giải thích âm học tiếng Việt và ánh xạ sang từ điển khái niệm chuẩn UNESCO/ICHCAP.
4. **[web_app/templates/index.html](web_app/templates/index.html)**, **[web_app/static/css/index.css](web_app/static/css/index.css)**, **[web_app/static/js/app.js](web_app/static/js/app.js)**, **[web_app/static/js/audio_wave.js](web_app/static/js/audio_wave.js)**, **[web_app/static/js/audio_recorder.js](web_app/static/js/audio_recorder.js)**, **[web_app/static/img/heritage/](web_app/static/img/heritage/)**
   - Toàn bộ mã nguồn giao diện Frontend *Editorial Academic Light Mode* (xem đặc tả chi tiết trong [MO_TA_GIAO_DIEN.md](MO_TA_GIAO_DIEN.md)): trạng thái khởi đầu dạng dàn trang mở gồm khu vực nạp tư liệu, **Canvas 60fps Hình thái Âm học 6 thể loại** đồng bộ cùng **Slideshow triển lãm 6 di sản** (ảnh tư liệu độ nét cao + liên kết bài báo chính thống), trình phát media kèm dải phổ FFT 2D (`audio_wave.js`), dải phân đoạn hiển thị phần trăm từng phương thức, ngăn kéo XAI 4 khu vực (Đồ thị Bằng chứng SVG, Khám phá Luật, Đối chiếu & Can thiệp CBM, Báo cáo Tổng kết), trình cắt video trực tuyến và thu âm micro không khuếch đại khoảng lặng.
5. **[web_app/test_backend.py](web_app/test_backend.py)**
   - Bộ kiểm thử tự động xác nhận trạng thái nạp mô hình, các API phân tích mẫu và tính đối xứng của chức năng can thiệp khái niệm CBM.

---

## 3. Nhóm Trọng số Mô hình, Kho Khái niệm & Bộ Luật Suy diễn

1. **[experiments/checkpoints/](experiments/checkpoints/)**
   - Chứa đúng **21 checkpoint `.pth` và 21 file `_history.json`** của 21 mô hình Thế hệ 2 thuộc 5 Phương án (`M1a–M1c`, `M2a–M2e`, `M3a–M3c`, `M4a–M4e`, `M5a–M5e`). Trong đó 4 đại diện mạnh nhất của Phương án 2, 3, 4, 5 (`M2a`, `M3c`, `M4d`, `M5c`) được nạp trực tiếp vào `web_app/inference_engine.py`.
2. **[data/models/visual/](data/models/visual/)**
   - `best_visual_cbm.pth`: Mạng `ConvNeXt-Tiny` dự đoán đa nhãn 25 khái niệm thị giác.
   - `visual_concept_thresholds.json`: Bộ ngưỡng tối ưu F1 riêng cho từng khái niệm thị giác.
   - `best_visual_rf_continuous.joblib`: Bộ phân loại Random Forest đặt trên 25 xác suất liên tục của Visual CBM (đạt `90.0%` trên tập video kiểm thử độc lập).
3. **[data/models/gmm_results.pkl](data/models/gmm_results.pkl)** & **[data/models/fuzzy_profiles.json](data/models/fuzzy_profiles.json)**
   - Bộ phân cụm GMM 7 trục âm học và hồ sơ đặc trưng thể loại phục vụ chiếu mờ.
4. **[concept/audio_concepts.md](concept/audio_concepts.md)** & **[concept/visual_concepts.md](concept/visual_concepts.md)**
   - Kho tri thức gồm **43 khái niệm âm thanh** và **38 khái niệm thị giác** trích dẫn chuẩn theo UNESCO, ICHCAP và Viện Âm nhạc.
5. **[data/rules/](data/rules/)**
   - `rules_decision_tree_continuous.json` & `rules_random_forest_top.json`: **60 luật mờ liên tục** biên dịch trực tiếp thành hàm mất mát khả vi (`DifferentiableFuzzyRuleLoss`) ở Phương án 5 và dùng cho hậu kiểm XAI.
   - `rules_decision_tree_crisp.json`, `rules_best_ripper_smote50.json`, `rules_cn2.json`, `rules_visual_ripper.json`: Các bộ luật âm học và thị giác dùng để sinh diễn giải mệnh đề tiếng Việt trên giao diện.

---

## 4. Nhóm Notebook & Script Huấn luyện, Đánh giá Thực nghiệm

1. **[train_multimodal_audio_cnn_experiments.ipynb](train_multimodal_audio_cnn_experiments.ipynb)** (sinh tự động từ **[build_multimodal_audio_cnn_notebook.py](build_multimodal_audio_cnn_notebook.py)**)
   - Pipeline huấn luyện và đánh giá chính thức của toàn bộ **21 mô hình Đa phương thức & Neuro-Symbolic (`M1a–M5e`)** trên tập hợp nhất Hugging Face + YouTube (chạy trên Google Colab GPU T4).
2. **[Visual_Data_Prep_and_SemiAuto_Labeling_Colab.ipynb](Visual_Data_Prep_and_SemiAuto_Labeling_Colab.ipynb)** & **[Visual_CBM_and_Rule_Extraction_Colab.ipynb](Visual_CBM_and_Rule_Extraction_Colab.ipynb)**
   - Chuẩn bị khung hình từ video YouTube, gán nhãn bán tự động 25 khái niệm thị giác, huấn luyện `ConvNeXt-Tiny Visual CBM` với Asymmetric Loss và cảm nạp luật thị giác.
3. **[01_audio_feature_extraction.ipynb](01_audio_feature_extraction.ipynb)** & **[02_fuzzy_decision_tree_random_forest.ipynb](02_fuzzy_decision_tree_random_forest.ipynb)**
   - Trích xuất 167 đặc trưng âm học bằng Librosa, thiết kế 7 trục ngữ nghĩa MPEG-7, phân cụm mờ FCM/GMM và cảm nạp luật mờ Decision Tree / Random Forest.
4. **[03_raw_features_benchmark_colab.ipynb](03_raw_features_benchmark_colab.ipynb)** (sinh từ **[build_raw_features_notebook.py](build_raw_features_notebook.py)**)
   - Huấn luyện và tối ưu 10 mô hình Học máy truyền thống trên 167 đặc trưng thô bằng Optuna kèm giải thích SHAP TreeExplainer.
5. **[create_group_split.py](create_group_split.py)**
   - Script chia tập dữ liệu 3,600 mẫu Hugging Face theo cặp bài hát gốc (`sample_idx // 2`) để đảm bảo không rò rỉ dữ liệu giữa Train/Val/Test.
6. **[evaluate_wild_videos.py](evaluate_wild_videos.py)** & **[evaluate_audio_youtube_benchmark.py](evaluate_audio_youtube_benchmark.py)**
   - Hai script kiểm thử độ lệch miền (Out-of-Distribution Stress Test) trên tập video YouTube thực tế.

---

## 5. Nhóm Tập Dữ liệu Chuẩn, Bảng Số liệu & Biểu đồ Báo cáo

1. **Tập Dữ liệu Chuẩn:**
   - [data/datasets/features_all.csv](data/datasets/features_all.csv): Bảng 167 đặc trưng âm học thô của 3,600 mẫu Hugging Face.
   - [data/datasets/features_fuzzy_continuous.csv](data/datasets/features_fuzzy_continuous.csv) & [data/datasets/features_fuzzy_crisp.csv](data/datasets/features_fuzzy_crisp.csv): Tập 27 độ thuộc mờ liên tục và 7 nhãn định tính ngôn ngữ.
   - [data/datasets/split_group_v2.csv](data/datasets/split_group_v2.csv): Bảng phân chia Train/Val/Test chống rò rỉ cấp bài hát.
   - [data/annotations/visual_concepts_dataset.csv](data/annotations/visual_concepts_dataset.csv): Tập dữ liệu gán nhãn 25 khái niệm thị giác của các khung hình video và ảnh tĩnh.
2. **Bảng Số liệu & Báo cáo Kết quả:**
   - [experiments/results/model_comparison_metrics.csv](experiments/results/model_comparison_metrics.csv): Bảng kết quả chi tiết của cả 21 mô hình (`M1a–M5e`) trên 3 kịch bản kiểm thử.
   - [experiments/results/multimodal_fusion_comparison.csv](experiments/results/multimodal_fusion_comparison.csv): Bảng đối chứng gọn gồm 14 phương án (Mô hình cũ, Học máy 167 đặc trưng, Luật mờ, Đại diện 5 phương án mới và Tổ hợp tối ưu).
   - [experiments/results/youtube_video_duration_split_stats.csv](experiments/results/youtube_video_duration_split_stats.csv): Thống kê phân bổ thời lượng và hạn ngạch lấy mẫu của 131 video YouTube theo tập Train/Val/Test.
   - [data/results/multimodal_fusion_ablation_report.md](data/results/multimodal_fusion_ablation_report.md): Báo cáo giải thích chi tiết các bảng đối chứng và lựa chọn trọng số Đa phương thức.
   - [data/results/audio_youtube_131_evaluation_report.md](data/results/audio_youtube_131_evaluation_report.md) & [data/results/wild_video_evaluation_report.md](data/results/wild_video_evaluation_report.md): Báo cáo thí nghiệm phát hiện lệch miền của các mô hình Thế hệ 1 trên video YouTube thực tế.
3. **Biểu đồ Trực quan chất lượng cao (300 DPI):**
   - [experiments/plots/](experiments/plots/): Chứa 5 biểu đồ tổng hợp của 21 mô hình (`benchmark_family_comparison.png`, `benchmark_macro_f1_vs_rule_violation.png`, `confusion_matrices_in_domain_test.png`, `confusion_matrices_youtube_zero_leakage.png`, `training_loss_f1_rule_curves.png`).
   - [data/plots/](data/plots/): Chứa biểu đồ phân tích dữ liệu âm học (`01` đến `14`), đồ thị 7 hàm thuộc mờ chuẩn Ruspini (`fuzzy_logic_graphs_all_axes.png`) và ảnh phổ mẫu (`viz_*.png`, `spectrogram_previews/preview_*.png`).

---

## 6. Hướng dẫn Khởi chạy Nhanh Hệ thống

1. **Cài đặt thư viện cần thiết:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Kiểm tra nhanh toàn bộ Backend & Tích hợp Mô hình:**
   ```bash
   python web_app/test_backend.py
   ```
3. **Khởi chạy Ứng dụng Web Phân lớp & Giải thích Đa phương thức:**
   ```bash
   python -m uvicorn web_app.app:app --host 127.0.0.1 --port 8000 --reload
   ```
   Sau đó truy cập trình duyệt tại địa chỉ: `http://127.0.0.1:8000`
