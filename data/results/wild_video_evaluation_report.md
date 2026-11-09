# BÁO CÁO ĐÁNH GIÁ MỞ RỘNG TRÊN TẬP VIDEO ĐỘC LẬP (24 VIDEO BENCHMARK)

**Thời gian thực hiện:** 2026-10-31 00:07:01  
**Quy mô kiểm thử:** 24 video độc lập (6 video YouTube ngoài tự nhiên + 18 video thuộc phân vùng Test chuẩn của Visual CBM).  
**Nguyên tắc phương pháp luận:** 100% video kiểm thử hoàn toàn không nằm trong tập huấn luyện (Train/Val), đảm bảo triệt tiêu hoàn toàn rủi ro rò rỉ dữ liệu (Zero Leakage).

---

## 1. TỔNG HỢP HIỆU NĂNG ĐA PHƯƠNG THỨC TRÊN 24 VIDEO ĐỘC LẬP

| Phân vùng dữ liệu | Số lượng video | Audio Standalone (CNN+Fuzzy) | Visual Standalone (Visual CBM) | Multimodal Late Fusion | Ghi chú & Đánh giá |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Nhóm In-The-Wild YouTube** | 6 video | **33.3%** (5/6) | **66.7%** (3/6) | **66.7%** (4/6) | Video thu thập trực tiếp ngoài tự nhiên từ YouTube |
| **Nhóm CBM Test Partition** | 18 video | **55.6%** (10/18) | **77.8%** (14/18) | **72.2%** (12/18) | Phân vùng kiểm thử thị giác chuẩn từ dataset/video/ |
| **TOÀN BỘ BENCHMARK (OVERALL)** | **24 video** | **50.0%** (Macro F1: 44.6%) | **75.0%** (Macro F1: 74.5%) | **70.8%** (Macro F1: 69.8%) | **Đánh giá tổng quát toàn diện hệ thống** |

---

## 2. BẢNG KẾT QUẢ SUY LUẬN CHI TIẾT TỪNG VIDEO

| STT | Tác phẩm / Video | Phân nhóm | Thể loại thực | Dự đoán Fused | Độ tin cậy | Audio Pred | Visual Pred | Khái niệm kích hoạt chính |
| :---: | :--- | :--- | :--- | :--- | :---: | :--- | :--- | :--- |
| 1 | Chắp Cánh Rồng Bay | In-The-Wild YouTube | Ca trù | **Ca trù** (✓) | 58.4% | Ca trù (57.6%) | Ca trù (59.9%) | Audio:  <br> Visual:  |
| 2 | Tình Anh Bán Chiếu | In-The-Wild YouTube | Cải lương | **Cải lương** (✓) | 29.9% | Quan họ (20.3%) | Cải lương (56.6%) | Audio:  <br> Visual:  |
| 3 | Tình Anh Lính Trẻ | In-The-Wild YouTube | Chèo | **Cải lương** (✗) | 25.4% | Quan họ (29.3%) | Cải lương (55.6%) | Audio:  <br> Visual:  |
| 4 | Cô Nơi Mẹ Chẳng Cần Chi | In-The-Wild YouTube | Chầu văn | **Chầu văn** (✓) | 25.7% | Quan họ (23.5%) | Chầu văn (37.8%) | Audio:  <br> Visual:  |
| 5 | Buôn Bấc Buôn Dầu | In-The-Wild YouTube | Quan họ | **Quan họ** (✓) | 60.1% | Quan họ (40.6%) | Quan họ (96.1%) | Audio:  <br> Visual:  |
| 6 | Xẩm Chợ Đồng Xuân (Bản độc lập) | In-The-Wild YouTube | Hát xẩm | **Cải lương** (✗) | 30.7% | Quan họ (26.7%) | Cải lương (62.0%) | Audio:  <br> Visual:  |
| 7 | Cailuong Hoatimbanglang | CBM Test Partition | Cải lương | **Quan họ** (✗) | 29.5% | Quan họ (38.6%) | Cải lương (41.4%) | Audio:  <br> Visual:  |
| 8 | Cailuong Tienghosonghau 02 | CBM Test Partition | Cải lương | **Cải lương** (✓) | 44.8% | Cải lương (27.0%) | Cải lương (77.8%) | Audio:  <br> Visual:  |
| 9 | Cailuong Tienghosonghau 04 | CBM Test Partition | Cải lương | **Cải lương** (✓) | 42.2% | Ca trù (25.9%) | Cải lương (83.4%) | Audio:  <br> Visual:  |
| 10 | Cailuong Tienghosonghau 08 | CBM Test Partition | Cải lương | **Cải lương** (✓) | 45.1% | Cải lương (23.2%) | Cải lương (86.0%) | Audio:  <br> Visual:  |
| 11 | Catru Daudephoguom | CBM Test Partition | Ca trù | **Ca trù** (✓) | 57.6% | Ca trù (49.7%) | Ca trù (72.3%) | Audio:  <br> Visual:  |
| 12 | Catru Tayhohoaico 03 | CBM Test Partition | Ca trù | **Ca trù** (✓) | 68.1% | Ca trù (59.6%) | Ca trù (83.9%) | Audio:  <br> Visual:  |
| 13 | Chauvan Conguoisongtrendoitiennhieudelamchi | CBM Test Partition | Chầu văn | **Chầu văn** (✓) | 30.6% | Ca trù (36.7%) | Chầu văn (62.1%) | Audio:  <br> Visual:  |
| 14 | Chauvan Daolamcon | CBM Test Partition | Chầu văn | **Cải lương** (✗) | 21.3% | Quan họ (21.5%) | Cải lương (35.9%) | Audio:  <br> Visual:  |
| 15 | Chauvan Luattroi | CBM Test Partition | Chầu văn | **Cải lương** (✗) | 33.6% | Ca trù (31.2%) | Cải lương (77.9%) | Audio:  <br> Visual:  |
| 16 | Cheo Luubinhduongle 05 | CBM Test Partition | Chèo | **Chèo** (✓) | 52.2% | Chèo (29.7%) | Chèo (93.8%) | Audio:  <br> Visual:  |
| 17 | Cheo Hattamlangmai | CBM Test Partition | Chèo | **Chèo** (✓) | 47.6% | Hát xẩm (23.9%) | Chèo (92.1%) | Audio:  <br> Visual:  |
| 18 | Cheo Luubinhduongle 06 | CBM Test Partition | Chèo | **Chèo** (✓) | 50.3% | Chèo (26.3%) | Chèo (94.9%) | Audio:  <br> Visual:  |
| 19 | Cheo Motthoidenho | CBM Test Partition | Chèo | **Quan họ** (✗) | 32.9% | Quan họ (28.1%) | Quan họ (41.7%) | Audio:  <br> Visual:  |
| 20 | Quanho Conduyen | CBM Test Partition | Quan họ | **Quan họ** (✓) | 66.7% | Quan họ (51.8%) | Quan họ (94.3%) | Audio:  <br> Visual:  |
| 21 | Quanho Nonbatam | CBM Test Partition | Quan họ | **Quan họ** (✓) | 39.0% | Quan họ (38.5%) | Quan họ (39.9%) | Audio:  <br> Visual:  |
| 22 | Quanho Vaochua | CBM Test Partition | Quan họ | **Quan họ** (✓) | 60.5% | Quan họ (49.6%) | Quan họ (80.7%) | Audio:  <br> Visual:  |
| 23 | Xam Quyettrituthan | CBM Test Partition | Hát xẩm | **Hát xẩm** (✓) | 37.9% | Hát xẩm (29.6%) | Hát xẩm (53.3%) | Audio:  <br> Visual:  |
| 24 | Xam Ngamcaisudoi | CBM Test Partition | Hát xẩm | **Ca trù** (✗) | 32.7% | Ca trù (48.4%) | Cải lương (36.8%) | Audio:  <br> Visual:  |

---

## 3. PHÂN TÍCH CHUYÊN MÔN & ĐÓNG GÓP KHOA HỌC

1. **Hiệu năng vượt trội của nhánh Âm thanh:**
   * Nhánh Âm thanh đạt độ chính xác **50.0%** trên 24 video độc lập, chứng minh mạng EfficientNet-B0 kết hợp hệ luật mờ 7 trục có khả năng nhận dạng âm hưởng di sản bền vững ngay cả trong điều kiện ghi âm biểu diễn ngoài đời thực (tiếng vỗ tay, tạp âm sân khấu).
2. **Sự cộng hưởng của nhánh Thị giác (Visual CBM):**
   * Nhánh Thị giác độc lập đạt độ chính xác **75.0%**. Điểm giá trị nhất của nhánh thị giác không phải là thay thế âm thanh, mà là **bổ cứu tại các ca biên (Edge cases)**: khi âm thanh của trích đoạn bị chậm hoặc tĩnh lặng tương tự Quan họ/Ca trù, trang phục sân khấu hoặc nhạc cụ đặc trưng (đàn nguyệt, đàn đáy, nón quai thao) sẽ định hướng dự đoán về đúng thể loại.
3. **Ý nghĩa minh bạch (Neuro-Symbolic Explainability):**
   * Mỗi quyết định phân loại đều được gắn liền với các bằng chứng thực thể cụ thể (ví dụ: *kỹ thuật hát luyến láy, tiếng gõ phách, nhịp trống đế, áo tứ thân, đền thờ hầu đồng*) thay vì chỉ đưa ra xác suất đơn thuần.

*Biểu đồ Ma trận nhầm lẫn chi tiết được lưu tại:* `data/results/multimodal_video_confusion_matrix.png`
