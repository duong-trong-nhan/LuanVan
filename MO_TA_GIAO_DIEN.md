# MÔ TẢ CHI TIẾT GIAO DIỆN & CHỨC NĂNG HỆ THỐNG WEB HERA

*(Tài liệu đặc tả giao diện người dùng, luồng tương tác và các khu vực giải thích XAI của ứng dụng Web HERA — Hệ thống Phân tích & Giải thích Di sản Âm nhạc Đa phương thức)*

---

## 1. Triết lý Thiết kế & Ngôn ngữ Thị giác

- **Phong cách tổng thể:** *Editorial Academic / Museum Archival Light Mode* — dàn trang báo chí học thuật mở trực tiếp trên nền giấy ấm (`#FAF9F5`), tiết chế tối đa việc lồng nhiều khung card nổi, ưu tiên đường kẻ mảnh kiến trúc `1px` (`#E2DCD0`) và khoảng trắng thoáng.
- **Hệ thống kiểu chữ (Typography):**
  - **Newsreader (Serif):** Dùng có chọn lọc cho thương hiệu, tiêu đề thể loại di sản, tiêu đề kết luận và báo cáo tổng kết.
  - **Plus Jakarta Sans (Sans-serif):** Văn bản diễn giải XAI, điều khiển tương tác và bảng đối chiếu.
  - **JetBrains Mono (Monospace):** Các chỉ số xác suất, mốc thời gian (`00:10 - 00:21`), tên trục MPEG-7 và thông số kỹ thuật.
- **Bảng màu nhận diện 6 thể loại di sản:**
  - **Ca trù:** Đỏ son (`#C84B31`)
  - **Cải lương:** Hổ phách (`#C97A1B`)
  - **Chèo:** Xanh chàm (`#2E5A88`)
  - **Chầu văn:** Tím trầm (`#8C3B88`)
  - **Quan họ:** Xanh ngọc bích (`#1E6B5B`)
  - **Hát xẩm:** Nâu đất (`#7A6248`)
  - **Thể loại khác / Không đủ dữ liệu xác định (OOD):** Xám đá (`#64748B`)

---

## 2. Thanh Điều hướng Trên cùng (Editorial Header)

- **Bên trái:** Biểu trưng **HERA** và phụ đề *"Phân tích & Giải thích Di sản Âm nhạc Đa phương thức"*.
- **Bên phải:**
  - **Thanh Thử nhanh (6 mẫu mặt tiền tiêu biểu):** 6 nút bấm gọn (chấm màu thể loại + tên thể loại; thời lượng và dung lượng hiển thị khi rê chuột qua tooltip) cho phép chạy phân tích tức thì 1 trong 6 video biểu diễn chuẩn được tuyển chọn sẵn (`Ca trù`, `Cải lương`, `Chèo`, `Chầu văn`, `Quan họ`, `Hát xẩm`).
  - **Nút Thông tin hệ thống (`i`):** Mở hộp thoại giới thiệu kiến trúc mô hình, quy mô dữ liệu và trích dẫn học thuật.

---

## 3. Màn hình Khởi đầu (Empty State — Bố cục 2 Cột Mở)

Khi chưa nạp tư liệu, màn hình chia 2 cột trực tiếp trên nền giấy ấm, ngăn cách bởi 1 đường kẻ dọc `1px` và tự động căn bằng mép dưới:

### 3.1. Cột Trái — Khu vực Nạp Dữ liệu & Hình thái Âm học Trực quan
1. **3 Chế độ nạp đầu vào linh hoạt (3 Tab gạch chân):**
   - **Tệp Video & Âm thanh:** Khu vực kéo thả hoặc chọn tệp cục bộ (`MP4, MOV, AVI, WAV, MP3, FLAC`, tối đa 200 MB, dưới 10 phút). Nếu video tải lên vượt quá 10 phút, hệ thống tự động mở **Trình cắt video trực tuyến (Video Trimmer Modal)** có phát trước hình ảnh và âm thanh để người dùng chọn đoạn cắt hợp lệ rồi phân tích trực tiếp bằng `ffmpeg`.
   - **Ghi âm trực tiếp (Micro):** Thu âm trực tiếp từ microphone trình duyệt (`Web Audio API`), đồng hồ đếm thời gian (tối thiểu 5s, khuyến nghị 6–15s), dải sóng phổ FFT 2D thời gian thực (`audio_wave.js`). Tự động tắt `autoGainControl` và không khuếch đại khoảng lặng để tránh nhận nhầm tạp âm nền thành thể loại âm nhạc.
   - **Link YouTube / URL:** Dán liên kết YouTube (`watch`, `shorts`, `live`, `youtu.be`) hoặc URL video trực tiếp, kèm 3 nút thử nhanh. Backend dùng `yt-dlp` kèm Node JS runtime để tải và tự động cắt 180s đầu nếu video dài quá 10 phút.
2. **Biểu đồ Hình thái Âm học Di sản (`#empty-left-artwork`, Canvas 60fps):**
   - Đồng bộ theo thời gian thực với thể loại đang hiển thị ở Slideshow bên phải.
   - Minh họa trực quan 6 hình thái sóng âm riêng biệt theo đặc trưng nhạc lý từng thể loại (xung phách đôi sắc gọn của **Ca trù**, vòng ngân sâu trải dài của **Cải lương**, cụm sóng đảo phách rộn ràng của **Chèo**, hòa âm dày dồn dập của **Chầu văn**, hai bè trên – dưới đối đáp của **Quan họ**, dạng sóng kéo dây đàn nhị bất đối xứng của **Hát xẩm**).
   - Khi chuyển thể loại, thuật toán nội suy toán học `easeInOutCubic` trong `1.85 giây` giữa hàm sóng cũ và mới giúp sóng tan chảy mượt mà, không giật cục. Kèm 3 chỉ mục gọn: **Biên độ**, **Mật độ phổ**, **Cấu trúc nhịp**.

### 3.2. Cột Phải — Slideshow Triển lãm 6 Thể loại Di sản (`#empty-heritage-guide`)
- **Thanh chỉ mục 6 thể loại phía trên:** Tự động chuyển slide mỗi **8 giây** bằng một đồng hồ `requestAnimationFrame` duy nhất điều khiển đồng thời thanh tiến trình `scaleX` và thời điểm chuyển slide (tạm dừng khi rê chuột vào và chạy tiếp đúng mili-giây đã dừng khi rời chuột, không bị kẹt ở 100%).
- **Khung ảnh tư liệu thực tế (16:9):** 6 ảnh biểu diễn thực tế độ nét cao, không dính watermark (`web_app/static/img/heritage/`), hiệu ứng phóng nhẹ Ken Burns 8s, kèm nhãn vùng miền & danh hiệu UNESCO và nút mũi tên điều hướng trái/phải.
- **Nội dung biên tập & Liên kết bài viết:** Tiêu đề Newsreader serif, 2 đoạn giới thiệu ngắn gọn giàu thông tin (nguồn gốc, biên chế nhạc cụ, kỹ thuật thanh nhạc, không gian diễn xướng), 3 cụm đặc trưng nhận diện và **nút liên kết bài báo/tư liệu chính thống** mở trong thẻ mới (Viện Âm nhạc, Lý luận VHNT, Bản sắc Việt, Cục Di sản, UNESCO ICH, Cổng TTĐT Hà Nội).

---

## 4. Màn hình Chờ Xử lý (Frameless Editorial Loading)

- Thiết kế toàn màn hình không đóng khung hộp vuông trên nền giấy mờ ấm.
- Hiển thị tiêu đề trạng thái, đồng hồ đếm thời gian xử lý thực (`00:xx`), hiệu ứng dải sóng dây đàn mảnh chuyển động ngang và thanh tiến trình 3 mốc giai đoạn: **Tách phân đoạn & trích xuất phổ** → **Suy luận 4 họ Audio + Visual CBM** → **Hợp nhất luật mờ & sinh XAI**.

---

## 5. Không gian Phân tích Chính (Analyzed Workspace — 2 Cột Đồng nhất)

Khi có kết quả phân tích (chuyển cảnh mờ êm bằng GSAP; bấm **Thay đổi tệp** cũng mờ dần rồi quay lại trang khởi đầu mượt mà):

### 5.1. Cột Trái — Trình phát Media & Phân tích Đa phương thức (`#sec-multimodal`)
1. **Trình phát Video / Âm thanh:**
   - Phát đồng bộ thời gian thực với toàn bộ phân đoạn và biểu đồ; hiển thị tên tệp, tỷ lệ khung hình (`16:9 HD`, `4:3 Classic`...) và nút **Thay đổi tệp**.
   - Khi phát tệp Âm thanh (Audio/Mic), hiển thị kèm dải phổ FFT 2D tối giản phản ứng theo tần số thực tế.
2. **Khu vực Phân đoạn Dự đoán Thời gian (Menu chọn 3 chế độ xem):**
   - **Dải Keyframes:** Danh sách thẻ phân đoạn đã gộp thông minh (Smart Grouping) kèm khung hình đại diện sắc nét nhất, mốc thời gian và **3 dòng phương thức hiển thị rõ phần trăm độ tin cậy**:
     - Âm thanh (`Audio %`),
     - Thị giác (`Visual %`, hoặc hiển thị rõ `Không rõ thị giác · —` nếu khung hình không có nhạc cụ/trang phục di sản),
     - Hợp nhất (`Fused %`).
     - Bấm vào bất kỳ thẻ phân đoạn nào sẽ tự động tua trình phát đến đúng mốc thời gian đó, làm nổi bật các khái niệm tương ứng, cập nhật 7 trục MPEG-7 và hiển thị diễn giải XAI chi tiết của riêng phân đoạn đó.
   - **Ảnh Phổ Mel & 7 Trục MPEG-7:** Trực quan hóa giá trị chuẩn hóa `[0, 100%]` của 7 trục ngữ nghĩa (`Energy`, `Harmony`, `Pitch`, `Spectral`, `Tempo`, `Timbre`, `Vibrato`) đồng bộ với Đồ thị Bằng chứng, đặt cạnh lưới ảnh phổ Mel-Spectrogram của từng phân đoạn.
   - **Biểu đồ Diễn biến & CBM:** Đồ thị đường SVG theo trục thời gian thể hiện sự biến thiên xác suất 6 thể loại kèm thanh tua tương tác và dải màu phân bố thể loại.

### 5.2. Cột Phải — Kết luận & Diễn giải Ngôn ngữ Tự nhiên XAI
1. **Thẻ Kết luận (Verdict Banner — `#verdict-main-card`):**
   - Hiển thị thể loại dự đoán hàng đầu (hoặc `Thể loại khác / Không đủ dữ liệu xác định` nếu bị Cổng OOD / Silence Gate chặn), phần trăm độ tin cậy và tóm tắt kết quả từng nhánh (`Âm thanh` · `Thị giác` · `Hợp nhất`).
   - **Mở rộng xác suất 5 thể loại còn lại:** Nút mũi tên xoay mượt (`rotate(180deg)`) trượt êm bằng CSS transition hiển thị đầy đủ thanh xác suất của cả 6 thể loại.
2. **Thẻ Diễn giải XAI (`#xai-card`):**
   - Trình bày bài diễn giải tiếng Việt mạch lạc, tự động điều chỉnh sắc thái theo mức độ tin cậy (`>= 80%`, `60–79.9%`, `< 60%`) và độ đồng thuận Âm thanh – Thị giác thực tế; bám sát các dấu hiệu thị giác thực sự phát hiện được trên khung hình.
   - **4 Lối tắt mở Ngăn kéo Kiểm chứng Chuyên sâu (Deep-Dive XAI):** Đặt gọn cuối thẻ diễn giải để mở trực tiếp Khu vực 03, 04, 05 hoặc 06.

---

## 6. Ngăn kéo Kiểm chứng Chuyên sâu (Slide Drawer — Khu vực 03 đến 06)

Có thể mở từ 4 lối tắt dưới thẻ XAI hoặc từ **Thanh cạnh phải (Edge Sidebar)** có hiệu ứng mở nhẹ khi rê chuột. Chuyển giữa 4 khu vực có hiệu ứng mờ êm `220ms`:

1. **Khu vực 03 — Bản đồ Bằng chứng Đa phương thức (`#sec-concept-graph`):**
   - Đồ thị SVG tương tác (hỗ trợ kéo thả pan, phóng to/thu nhỏ zoom, tự căn khung) kết nối 5 tầng: **Đầu vào Media → 7 Trục MPEG-7 & 25 Đặc trưng Thị giác → Khái niệm Di sản (UNESCO/ICHCAP) → Luật mờ kích hoạt → Phán quyết Thể loại**.
   - Bấm vào bất kỳ nút nào trên đồ thị sẽ mở bảng thanh tra chi tiết bên phải (Inspector) và cho phép làm nổi bật toàn bộ đường đi suy luận của luật đó.
2. **Khu vực 04 — Khám phá Luật Suy luận (`#sec-rule-explorer`):**
   - Bố cục Master–Detail: danh sách các luật mờ kích hoạt ở cột trái; cột phải hiển thị dải chỉ số 3 cột (Độ tin cậy, Độ phủ, Điểm trọng số), mệnh đề IF–THEN nguyên bản và bản dịch tiếng Việt từng điều kiện kèm nút định vị ngược lại trên đồ thị Khu vực 03.
3. **Khu vực 05 — Diễn giải Đối chiếu & Thử nghiệm Giả định (`#sec-explanation-explorer`):**
   - Cho phép chọn bất kỳ **Thể loại đối chứng** nào (mặc định là Hạng 2) để so sánh tức thì với Thể loại dự đoán mà không cần chạy lại mô hình. Gồm 3 tab:
     - **1. Khía cạnh phân biệt (Aspect Explorer):** Bảng đối chiếu 2 cột theo từng tiêu chí (nhạc cụ, thanh nhạc, tiết tấu, trang phục, không gian...) kèm bộ lọc nhanh và trạng thái kiểm chứng (có bằng chứng / giao thoa / không phát hiện).
     - **2. Ba nhóm đặc trưng:** Phân tách rõ nhóm *Ủng hộ dự đoán*, *Giao thoa chung* và *Đặc trưng chỉ có ở thể loại đối chứng*.
     - **3. Thử nghiệm Giả định (What-If & Can thiệp CBM):**
       - Thanh trượt mô phỏng dịch chuyển xác suất giữa Top 1 và thể loại đối chứng.
       - **Can thiệp Khái niệm CBM thời gian thực (`/api/intervene`):** Cho phép bật/tắt từng khái niệm âm thanh và thị giác; bảng theo dõi tức thì cập nhật xác suất mới, độ lệch Delta, cảnh báo lật ngược phán quyết (*Decision Flip*) và đồng bộ trực tiếp thanh so sánh thứ 2 theo đúng **thể loại đối chứng đang chọn** ở hộp phía trên, kèm nút **Khôi phục ban đầu**.
4. **Khu vực 06 — Báo cáo Tổng kết Chuyên khảo (`#sec-summary`):**
   - Bố cục 1 cột biên tập gọn gàng: dải chỉ số 4 cột tóm lược ở trên cùng, cột văn bản đọc toàn chiều rộng ưu tiên phần diễn giải tổng hợp ở giữa, và mục **Tài liệu nguồn kiểm chứng** dạng chú thích cuối trang (mặc định thu gọn, bấm mở khi cần tra cứu).
