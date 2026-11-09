# BÁO CÁO KẾT QUẢ THỰC NGHIỆM ĐẶC TRƯNG THÔ & SO SÁNH TOÀN DIỆN

**Thời gian sinh báo cáo:** 2026-10-27 04:40:50

**Quy mô tập dữ liệu:** 3,600 phân đoạn âm thanh (1,800 bài hát x 2 đoạn 12 giây), 6 thể loại nhạc cổ truyền.


## 1. TỔNG HỢP HIỆU NĂNG CÁC MÔ HÌNH TRÊN ĐẶC TRƯNG THÔ (PROTOCOL B - SONG-LEVEL)

| Model                    |   Test_Acc |   Test_Macro_F1 |   Train_Time_s |
|:-------------------------|-----------:|----------------:|---------------:|
| LogisticRegression       |      86.53 |           86.53 |              0 |
| KNN                      |      91.81 |           91.77 |              0 |
| SVM_RBF                  |      93.89 |           93.88 |              0 |
| DecisionTree             |      67.08 |           67.02 |              0 |
| RandomForest             |      89.86 |           89.84 |              0 |
| ExtraTrees               |      90.14 |           90.12 |              0 |
| XGBoost                  |      92.22 |           92.21 |              0 |
| LightGBM                 |      92.5  |           92.49 |              0 |
| CatBoost                 |      90.69 |           90.68 |              0 |
| PyTorch_MLP              |      93.47 |           93.45 |              0 |
| Ensemble_SoftVoting_Top3 |      95.14 |           95.13 |            nan |


## 2. HIỆU NĂNG CÁC PHƯƠNG ÁN LATE FUSION MỞ RỘNG (F1 - F6)

| Method                    | Best_Weight_CNN                      |   Test_Acc |   Test_Macro_F1 |
|:--------------------------|:-------------------------------------|-----------:|----------------:|
| F1_CNN_FuzzyRF            | 0.66                                 |      94.03 |           93.96 |
| F2_CNN_SVM_RBF            | 0.62                                 |      97.36 |           97.36 |
| F3_TriModal_CNN_Raw_Fuzzy | w_cnn=0.4, w_raw=0.27, w_fuzzy=0.33  |      97.22 |           97.22 |
| F6_Song_Level_Pooling_F2  | Pooled Across Segments               |      97.78 |           97.78 |


## 3. SO SÁNH ĐỐI ĐẦU 3 NHÁNH NGHIÊN CỨU

| Nhánh                            | Mô hình                                 | Đầu vào                        | Protocol                   |   Test Acc (%) |   Macro F1 (%) | Khả năng giải thích (XAI)       | Thời gian train   |
|:---------------------------------|:----------------------------------------|:-------------------------------|:---------------------------|---------------:|---------------:|:--------------------------------|:------------------|
| Luật mờ (Fuzzy / Rule)           | Random Forest Continuous + Gaussian     | 31 Đặc trưng mờ liên tục       | Protocol A (Segment-level) |        65.56   |        65.12   | Rất cao (Luật IF-THEN rõ ràng)  | ~2 phút (CPU)     |
| Luật mờ (Fuzzy / Rule)           | Random Forest Continuous (Re-evaluated) | 31 Đặc trưng mờ liên tục       | Protocol B (Song-level)    |        70.28   |        70.22   | Rất cao (Luật IF-THEN rõ ràng)  | ~2 phút (CPU)     |
| Đặc trưng thô (Raw Features)     | Ensemble_SoftVoting_Top3 (Optuna 50t)   | 167 Đặc trưng số âm học thô    | Protocol B (Song-level)    |        95.14   |        95.13   | Cao (SHAP TreeExplainer)        | < 1s (Ensemble)   |
| Thị giác CNN (Transfer Learning) | EXP-13_efficientnet_b0_mel+cqt_partial  | Spectrogram 2 kênh (Mel + CQT) | Protocol A (Spectrograms)  |        89.3056 |        89.1899 | Thấp (Hộp đen, Grad-CAM cục bộ) | 13.28 phút (GPU)  |


## 4. TOP 10 ĐẶC TRƯNG ÂM HỌC CỐT LÕI (SHAP)

| Feature             |   Mean_Absolute_SHAP |
|:--------------------|---------------------:|
| energy_mean         |             0.454455 |
| mfcc_07_mean        |             0.427605 |
| spectral_contrast_1 |             0.399597 |
| spectral_contrast_6 |             0.298305 |
| mfcc_01_mean        |             0.254182 |
| mfcc_04_mean        |             0.241246 |
| tonnetz_1           |             0.236352 |
| mfcc_01_std         |             0.232341 |
| chroma_std_overall  |             0.224056 |
| mfcc_16_std         |             0.220061 |


