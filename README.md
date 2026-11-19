# HeritageFNS — Hệ thống Phân lớp & Giải thích Đa phương thức Di sản Âm nhạc Cổ truyền Việt Nam

> **Interpretable Multimodal Neuro-Symbolic XAI for Vietnamese Intangible Cultural Heritage**  
> **6 Thể loại nghiên cứu:** Ca trù, Chèo, Chầu văn, Cải lương, Hát xẩm, Quan họ

---

## 1. Tổng quan Đề tài

Dự án xây dựng một hệ thống Trí tuệ Nhân tạo Đa phương thức có khả năng giải thích (**Multimodal Neuro-Symbolic XAI**) nhằm phân loại và giải thích trực quan 6 loại hình nghệ thuật diễn xướng di sản văn hóa phi vật thể Việt Nam từ tệp âm thanh, video biểu diễn thực tế, đường dẫn YouTube hoặc thu âm micro trực tiếp.

Hệ thống giải quyết đồng thời hai thách thức cốt lõi của bài toán nhận diện di sản ngoài đời thực:
1. **Khắc phục hiện tượng lệch miền phân phối (Domain Shift):** Các mô hình học sâu thế hệ đầu chỉ huấn luyện trên dữ liệu thu âm phòng thu (Hugging Face) bị sụt giảm mạnh khi gặp video YouTube có tạp âm sân khấu. Dự án hợp nhất dữ liệu đa nguồn (**3,600 mẫu Hugging Face + 902 phân đoạn 12 giây từ 131 video YouTube** theo giao thức chống rò rỉ cấp độ bài hát và cấp độ video) và phát triển **21 mô hình Đa phương thức & Neuro-Symbolic (`M1a–M5e`)** nâng Macro-F1 trên tập kiểm thử YouTube thực tế từ **45.39% lên 85.60%**.
2. **Minh bạch hóa quyết định (Explainable AI):** Kết hợp **7 trục ngữ nghĩa âm học MPEG-7**, **27 khái niệm mờ liên tục**, **60 luật mờ khả vi (`IF-THEN`)** và **25 khái niệm thị giác (Visual CBM — nhạc cụ, trang phục, bối cảnh)** chuẩn hóa theo kho tri thức UNESCO/ICHCAP.

### Tài liệu Chi tiết đi kèm
- **Báo cáo tổng hợp toàn dự án, các phát hiện khoa học & kiến trúc Pipeline:** [BAO_CAO_TONG_HOP_TOAN_DU_AN.md](BAO_CAO_TONG_HOP_TOAN_DU_AN.md)
- **Sơ đồ bố cục thư mục & mô tả chức năng từng file cốt lõi:** [MO_TA_CAC_FILE.md](MO_TA_CAC_FILE.md)
- **Báo cáo thực nghiệm đối chứng & lựa chọn bộ trọng số Multimodal:** [data/results/multimodal_fusion_ablation_report.md](data/results/multimodal_fusion_ablation_report.md)
- **Bảng số liệu benchmark 21 mô hình & tổ hợp đa phương thức:** [experiments/results/model_comparison_metrics.csv](experiments/results/model_comparison_metrics.csv) | [experiments/results/multimodal_fusion_comparison.csv](experiments/results/multimodal_fusion_comparison.csv)

---

## 2. Kiến trúc Hệ thống Đầu cuối (End-to-End Pipeline)

```text
[ĐẦU VÀO: Video MP4 / Audio WAV-MP3 / Link YouTube / Thu âm Micro]
                                │
                                ▼
   ┌────────────────────────────────────────────────────────────┐
   │ TIỀN XỬ LÝ & PHÂN ĐOẠN THÍCH ỨNG (<= 10 PHÚT, 6–14 CHUNKS) │
   │ • Cắt video trực tuyến (ffmpeg) nếu thời lượng > 600 giây  │
   │ • Mỗi phân đoạn: Cửa sổ âm thanh 12.0s + 1 Keyframe nét    │
   └──────────────┬─────────────────────────────┬───────────────┘
                  │                             │
                  ▼                             ▼
┌───────────────────────────────────┐ ┌───────────────────────────────────┐
│ NHÁNH ÂM THANH (NEURO-SYMBOLIC)   │ │ NHÁNH THỊ GIÁC (VISUAL CBM)       │
│ • Ảnh phổ Mel & CQT (128x128)     │ │ • Khung hình RGB (224x224)        │
│ • 7 trục MPEG-7 + 27 độ thuộc mờ  │ │ • ConvNeXt-Tiny (Asymmetric Loss) │
│ • Tổ hợp 4 Họ Kiến trúc (0.25/mô):│ │   -> 25 xác suất khái niệm Sigmoid│
│   M2a + M3c + M4d + M5c           │ │ • Ngưỡng F1 riêng từng khái niệm  │
│ • Xuất P_Audio & 27 concept mờ    │ │ • Visual Random Forest Continuous │
└─────────────────┬─────────────────┘ └─────────────────┬─────────────────┘
                  │                                     │
                  └──────────────────┬──────────────────┘
                                     ▼
   ┌────────────────────────────────────────────────────────────┐
   │ DUNG HỢP MUỘN ĐA PHƯƠNG THỨC: 0.70 * P_Audio + 0.30 * P_Vis│
   │ • Tổng hợp chuỗi phân đoạn (Temporal Average Pooling)      │
   │ • Bộ lọc 3 điều kiện nhận diện video ngoại lai (OOD)       │
   │ • Đối chiếu 60 Luật mờ + Phiên dịch khái niệm UNESCO/ICHCAP│
   └─────────────────────────────────┬──────────────────────────┘
                                     ▼
[ĐẦU RA XAI: Phán quyết | Evidence Map 4 tầng | Contrastive XAI | Can thiệp CBM | Đồ thị 3D]
```

---

## 3. Kết quả Thực nghiệm Nổi bật

Xuyên suốt đồ án đã huấn luyện và đối chứng tổng cộng **57 mô hình phân loại / học sâu có giám sát** (cùng 4 thuật toán phân cụm mờ không giám sát `K-Means`, `DBSCAN`, `FCM`, `GMM`):
- **6 mô hình phân loại & cảm nạp luật mờ âm thanh** (`DT Crisp/Cont`, `RF Crisp/Cont`, `RIPPER`, `CN2`)
- **10 mô hình Học máy trên 167 đặc trưng thô** (tối ưu bằng Optuna + giải thích SHAP)
- **13 mô hình CNN Transfer Learning Thế hệ 1** (`EXP-01` đến `EXP-13`)
- **7 mô hình nhánh Thị giác** (`ConvNeXt-Tiny Visual CBM` 25 khái niệm + 6 bộ phân loại/luật tầng trên)
- **21 mô hình Đa phương thức & Neuro-Symbolic Thế hệ 2** (`M1a–M5e` thuộc 5 Phương án kiến trúc)

### Bảng So sánh Hiệu năng Cốt lõi

| Nhóm Phương pháp / Mô hình | HF Test Acc (%) *(720 mẫu)* | HF Test Macro-F1 (%) | YouTube Test Acc (%) *(225 mẫu / 30 video)* | YouTube Test Macro-F1 (%) | Unified Test Macro-F1 (%) *(945 mẫu)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `EXP-13` (CNN Thế hệ 1 — chỉ học Hugging Face) | 92.50 | 92.50 | 49.78 | 45.39 | 81.28 |
| `SVM-RBF` (167 đặc trưng thô — chỉ học Hugging Face) | 93.75 | 93.88 | 39.11 | 31.21 | 78.96 |
| `M1c`: Baseline CNN (Mel+CQT) *[PA 1]* | 81.67 | 81.65 | 64.44 | 62.57 | 77.49 |
| `M2a`: CNN + Transformer (Single Mel) *[PA 2]* | 93.33 | 93.29 | 80.44 | 81.29 | 90.34 |
| `M3c`: Pretrained DenseNet121 + Transformer *[PA 3]* | 92.22 | 92.11 | 74.67 | 74.21 | 88.12 |
| `M4d`: Multimodal EfficientNet-B0 + 7 Trục Concat *[PA 4]* | 91.53 | 91.46 | 76.00 | 76.12 | 87.98 |
| `M5c`: Neuro-Symbolic EffNet-B0 Gated + 60 Luật mờ *[PA 5]* | 93.06 | 93.02 | 75.11 | 76.09 | 88.85 |
| **Tổ hợp 4 Họ Audio (`0.25 M2a + 0.25 M3c + 0.25 M4d + 0.25 M5c`)** | **94.86** | **94.82** | **85.33** | **85.60** | **92.64** |

- **Hiệu năng cấp độ Video:** Tổ hợp 4 Họ Audio đạt **27/30 (90.0%)** trên 30 video YouTube Test độc lập; khi kết hợp cùng **Visual CBM (`ConvNeXt-Tiny` mAP `86.25%` + `Visual RF Continuous` `90.0%`)** theo tỷ lệ tối ưu **`0.70 Audio + 0.30 Visual`**, hệ thống đạt **100.0%** trên tập video kiểm thử độc lập của cả hai nhánh (`LogLoss = 0.2881`).

---

## 4. Các Tính năng Chính trên Ứng dụng Web XAI (`web_app/`)

- **Phân tích đa nguồn đầu vào:** Hỗ trợ tải file Audio/Video, dán link YouTube trực tiếp, thu âm micro và 6 mẫu biểu diễn chuẩn.
- **Trình cắt Video trực tuyến (Interactive Trimmer):** Tự động phát hiện video dài quá 10 phút và cho phép người dùng kéo thanh trượt chọn đoạn cần phân tích (có hình ảnh xem trước và âm thanh).
- **Dòng thời gian đa phương thức:** Đồng bộ trình phát video với dải phân đoạn thông minh (Smart Grouping), biểu đồ diễn biến xác suất theo thời gian, ảnh phổ và biểu đồ Radar 7 trục MPEG-7.
- **Bản đồ Bằng chứng Đa phương thức 4 tầng (`Multimodal Evidence Map`):** Đồ thị tương tác truy vết từ Tín hiệu đầu vào -> Đặc trưng -> Khái niệm CBM -> Luật suy luận mờ -> Phán quyết; bấm vào mốc thời gian để tua video tới đúng bằng chứng.
- **Giải thích Đối chứng (`Contrastive Explanation`) & Can thiệp Phản sự thực (`Counterfactual CBM Intervention`):** Lý giải tại sao chọn thể loại A thay vì thể loại B, đồng thời cho phép bật/tắt từng khái niệm âm thanh và thị giác để kiểm chứng sự thay đổi xác suất theo thời gian thực.
- **Nhận diện ngoại lai (Out-of-Distribution):** Tự động gắn nhãn *"Thể loại khác / Không đủ dữ liệu xác định"* khi đầu vào không thuộc 6 thể loại di sản.

---

## 5. Cấu trúc Thư mục Chính

```text
LuanVan/
├── web_app/                                      # Ứng dụng Web FastAPI + Frontend Editorial Dashboard + Three.js
├── experiments/                                  # 21 checkpoint chuẩn (M1a–M5e), bảng kết quả CSV & biểu đồ 300 DPI
├── data/                                         # Dữ liệu đặc trưng, mô hình Visual CBM, bộ luật IF-THEN & báo cáo
├── concept/                                      # Từ điển 43 Audio Concepts & 38 Visual Concepts (UNESCO/ICHCAP)
├── dataset/video/                                # Kho 131 video biểu diễn thực tế thu thập từ YouTube
├── phase7_audio_visual_concept_mapper.py         # Cầu nối ngữ nghĩa dịch luật mờ & ánh xạ khái niệm
├── train_multimodal_audio_cnn_experiments.ipynb  # Notebook huấn luyện 21 mô hình Thế hệ 2 (M1a–M5e)
├── Visual_CBM_and_Rule_Extraction_Colab.ipynb    # Notebook huấn luyện Visual CBM & cảm nạp luật thị giác
├── BAO_CAO_TONG_HOP_TOAN_DU_AN.md                # Báo cáo tổng hợp chi tiết toàn dự án
└── MO_TA_CAC_FILE.md                             # Mô tả đầy đủ cấu trúc và chức năng từng file
```

---

## 6. Hướng dẫn Cài đặt & Khởi chạy

### Yêu cầu môi trường
- Python 3.10+ và `ffmpeg` đã được cài đặt trong biến môi trường hệ thống (`PATH`).

### Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Kiểm thử hệ thống Backend
```bash
python web_app/test_backend.py
```

### Khởi chạy Ứng dụng Web
```bash
python -m uvicorn web_app.app:app --host 127.0.0.1 --port 8000 --reload
```
Mở trình duyệt tại địa chỉ: `http://127.0.0.1:8000`
