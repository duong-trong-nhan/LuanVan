# BỘ CONCEPT HÌNH ẢNH CHUẨN HÓA (ATOMIC VISUAL CONCEPTS)

**Dự án:** Phân lớp đa phương thức và Diễn giải (XAI) 6 thể loại Âm nhạc Truyền thống Việt Nam

**Phạm vi:** Cải lương (`cai_luong`), Ca trù (`ca_tru`), Chèo (`cheo`), Chầu văn (`chau_van`), Quan họ (`quan_ho`), Hát Xẩm (`hat_xam`)

## 1. Quy ước Cấu trúc Dữ liệu (Schema)

Mỗi concept hình ảnh được viết dưới dạng một **mệnh đề khẳng định ngắn gọn (câu đơn)** trích xuất từ tài liệu chuyên ngành và quản lý qua 6 trường thông tin:

* **`concept_id`**: Mã định danh duy nhất (dùng làm nhãn thuộc tính cho Visual Concept Bottleneck Model hoặc biến đầu vào cho bộ luật DT/RF/CN2/RIPPER).
* **`genre_code`**: Mã thể loại (`cai_luong`, `ca_tru`, `cheo`, `chau_van`, `quan_ho`, `hat_xam`).
* **`aspect`**: Khía cạnh thị giác (`khong_gian_doi_hinh`, `nhac_cu_dao_cu`, `trang_phuc_hoa_trang`, `dong_tac_dien_xuat`).
* **`evidence_level`**:
  * `single_frame`: Có thể nhận diện trực tiếp từ một bức ảnh tĩnh hoặc một khung hình đơn lẻ (nhạc cụ, đạo cụ, trang phục, bố cục đội hình).
  * `video_sequence`: Cần quan sát chuỗi khung hình liên tiếp hoặc kết hợp ngữ cảnh sân khấu (động tác múa, diễn xuất hình thể, sự thay đổi trang phục theo giá hầu).
* **`concept_sentence`**: Câu đơn tiếng Việt dùng để ghép trực tiếp vào phần lý giải đầu ra của mô hình.
* **`source`**: Nguồn tài liệu học thuật đối chiếu.

## 2. Chi tiết Bộ Concept Hình ảnh theo Từng Thể loại

### 2.1. Cải lương (`cai_luong`)

| Mã Concept | Khía cạnh | Cấp độ bằng chứng | Câu Concept Lý giải (Atomic Sentence) | Nguồn tham khảo |
| ----- | ----- | ----- | ----- | ----- |
| `CL_VIS_01` | `khong_gian_doi_hinh` | `single_frame` | Biểu diễn trên sân khấu kịch hát với bối cảnh phục vụ cốt truyện và tình huống kịch. | Viện Âm nhạc Việt Nam; Nguyễn Thị Trúc Bạch (2022) |
| `CL_VIS_02` | `dong_tac_dien_xuat` | `video_sequence` | Nghệ sĩ kết hợp ca hát với diễn xuất biểu cảm và các động tác hình thể sân khấu (ca ra bộ). | Viện Âm nhạc Việt Nam; Nguyễn Thị Trúc Bạch (2022) |
| `CL_VIS_03` | `nhac_cu_dao_cu` | `single_frame` | Xuất hiện đàn guitar phím lõm trong dàn nhạc đệm. | Viện Âm nhạc Việt Nam |
| `CL_VIS_04` | `nhac_cu_dao_cu` | `single_frame` | Xuất hiện các nhạc cụ thuộc dàn nhạc đờn ca tài tử như đàn kìm, đàn cò/violon, đàn tranh và đàn sến. | Viện Âm nhạc Việt Nam |
| `CL_VIS_05` | `trang_phuc_hoa_trang` | `single_frame` | Nghệ sĩ mặc phục trang sân khấu cổ trang làm từ chất liệu lụa, gấm, nhung với màu sắc rực rỡ (trong các vở lịch sử hoặc tuồng cổ). | Hội đồng Lý luận, Phê bình VHNT Trung ương (lyluanphebinh.vn) |
| `CL_VIS_06` | `trang_phuc_hoa_trang` | `single_frame` | Trang phục nhân vật cổ trang có mũ/mão, khăn đóng, áo dài, áo ngắn, váy xòe, trang sức sân khấu và họa tiết thêu rồng, phượng biểu thị địa vị. | Hội đồng Lý luận, Phê bình VHNT Trung ương (lyluanphebinh.vn) |
| `CL_VIS_07` | `trang_phuc_hoa_trang` | `single_frame` | Nghệ sĩ mặc trang phục mô phỏng đời thường, áo dài, áo bà ba hoặc thời trang đương thời theo hoàn cảnh nhân vật (trong các vở đề tài xã hội hiện đại). | Hội đồng Lý luận, Phê bình VHNT Trung ương (lyluanphebinh.vn) |

### 2.2. Ca trù (`ca_tru`)

| Mã Concept | Khía cạnh | Cấp độ bằng chứng | Câu Concept Lý giải (Atomic Sentence) | Nguồn tham khảo |
| ----- | ----- | ----- | ----- | ----- |
| `CT_VIS_01` | `khong_gian_doi_hinh` | `single_frame` | Biểu diễn trong không gian thính phòng truyền thống hoặc không gian hát cửa đình. | UNESCO ("Ca trù singing"); ICHCAP ("Ca Trù Singing") |
| `CT_VIS_02` | `khong_gian_doi_hinh` | `single_frame` | Đội hình biểu diễn điển hình gồm ba thành phần: đào nương (người nữ hát), kép đàn (người nam chơi đàn đáy) và quan viên (người đánh trống chầu). | UNESCO ("Ca trù singing"); ICHCAP |
| `CT_VIS_03` | `nhac_cu_dao_cu` | `single_frame` | Đào nương ngồi hát đồng thời hai tay cầm dùi tự gõ vào bàn phách đặt phía trước để giữ nhịp. | UNESCO ("Ca trù singing"); ICHCAP |
| `CT_VIS_04` | `nhac_cu_dao_cu` | `single_frame` | Kép đàn ngồi đệm đàn đáy có cần đàn dài đặc trưng. | UNESCO ("Ca trù singing"); ICHCAP |
| `CT_VIS_05` | `nhac_cu_dao_cu` | `single_frame` | Quan viên ngồi cầm roi chầu gõ vào trống chầu để tương tác với người hát. | UNESCO ("Ca trù singing"); ICHCAP |
| `CT_VIS_06` | `trang_phuc_hoa_trang` | `single_frame` | Đào nương mặc trang phục truyền thống trang trọng, kín đáo và thanh lịch như áo thao (áo dài lụa tơ tằm màu nhạt hoặc sáng), quần đen và quấn khăn nhiễu trên đầu. | Viện Âm nhạc Việt Nam ("Vài nét về Ca trù Quảng Bình") |
| `CT_VIS_07` | `trang_phuc_hoa_trang` | `single_frame` | Đào nương mặc quần lĩnh, áo đoạn hoa, quấn khăn nhung và vấn tóc đuôi gà đoan trang. | Viện Âm nhạc Việt Nam ("Vài nét về Ca trù Quảng Bình") |
| `CT_VIS_08` | `trang_phuc_hoa_trang` | `single_frame` | Kép đàn mặc áo the truyền thống và đội khăn xếp trang trọng. | Viện Âm nhạc Việt Nam ("Vài nét về Ca trù Quảng Bình") |

### 2.3. Chèo (`cheo`)

| Mã Concept | Khía cạnh | Cấp độ bằng chứng | Câu Concept Lý giải (Atomic Sentence) | Nguồn tham khảo |
| ----- | ----- | ----- | ----- | ----- |
| `CH_VIS_01` | `khong_gian_doi_hinh` | `single_frame` | Không gian biểu diễn là chiếu Chèo truyền thống ở sân đình, sân làng hoặc sân khấu kịch hát dân gian. | Ủy ban Nhà nước về người Việt Nam ở nước ngoài (scov.gov.vn) |
| `CH_VIS_02` | `khong_gian_doi_hinh` | `single_frame` | Bố cục sân khấu có dàn nhạc công ngồi ở hai bên chiếu Chèo và khán giả đứng hoặc ngồi bao quanh. | Ủy ban Nhà nước về người Việt Nam ở nước ngoài (scov.gov.vn) |
| `CH_VIS_03` | `dong_tac_dien_xuat` | `video_sequence` | Diễn viên thực hiện đồng thời các động tác múa dân gian, diễn xuất hình thể kết hợp với hát và nói thoại để thể hiện nhân vật. | Ủy ban Nhà nước về người Việt Nam ở nước ngoài (scov.gov.vn) |
| `CH_VIS_04` | `nhac_cu_dao_cu` | `single_frame` | Xuất hiện bộ gõ gồm trống Chèo và mõ cùng các nhạc cụ dân tộc (như đàn bầu) trong dàn nhạc đệm. | Ủy ban Nhà nước về người Việt Nam ở nước ngoài; Học viện Âm nhạc Quốc gia Việt Nam |
| `CH_VIS_05` | `trang_phuc_hoa_trang` | `single_frame` | Trang phục và hóa trang mang tính ước lệ cao nhằm nhận diện tuyến nhân vật: vai vua, quan hoặc tướng lĩnh mặc áo bào, đội mũ mang tính quyền uy. | Ủy ban Nhà nước về người Việt Nam ở nước ngoài (scov.gov.vn) |
| `CH_VIS_06` | `trang_phuc_hoa_trang` | `single_frame` | Nhân vật thuộc tuyến thư sinh hoặc nho sĩ mặc áo dài hoặc áo the và đội khăn xếp. | Ủy ban Nhà nước về người Việt Nam ở nước ngoài (scov.gov.vn) |
| `CH_VIS_07` | `trang_phuc_hoa_trang` | `single_frame` | Nhân vật nữ mặc trang phục dân gian Bắc Bộ gồm áo tứ thân, yếm, váy và vấn khăn. | Ủy ban Nhà nước về người Việt Nam ở nước ngoài (scov.gov.vn) |
| `CH_VIS_08` | `trang_phuc_hoa_trang` | `single_frame` | Nhân vật hề Chèo sử dụng trang phục phóng đại hoặc có màu sắc nổi bật để hỗ trợ tính hài hước. | Ủy ban Nhà nước về người Việt Nam ở nước ngoài (scov.gov.vn) |

### 2.4. Chầu văn / Hát văn (`chau_van`)

| Mã Concept | Khía cạnh | Cấp độ bằng chứng | Câu Concept Lý giải (Atomic Sentence) | Nguồn tham khảo |
| ----- | ----- | ----- | ----- | ----- |
| `CV_VIS_01` | `khong_gian_doi_hinh` | `single_frame` | Không gian diễn xướng gắn với thực hành nghi lễ tín ngưỡng thờ Mẫu (đền, phủ, điện thờ) kết hợp giữa hầu đồng và hát văn. | Cục Di sản văn hóa (dsvh.gov.vn); UNESCO |
| `CV_VIS_02` | `dong_tac_dien_xuat` | `video_sequence` | Người thực hành nghi lễ (thanh đồng) thực hiện các điệu múa thiêng và nghi thức hầu đồng ở trung tâm, trong khi cung văn ngồi bên cạnh đàn, hát và quan sát để ứng tác. | Cục Di sản văn hóa (dsvh.gov.vn) |
| `CV_VIS_03` | `nhac_cu_dao_cu` | `single_frame` | Cung văn sử dụng đàn nguyệt kết hợp với bộ gõ gồm trống ban (trống con), phách, cảnh, thanh la (có thể thêm đàn nhị, sáo trúc, đàn tranh, trống cơm). | Cục Di sản văn hóa (dsvh.gov.vn); Hồ Thị Hồng Dung |
| `CV_VIS_04` | `nhac_cu_dao_cu` | `single_frame` | Người hầu đồng cầm các đạo cụ nghi lễ đặc trưng theo từng giá hầu như kiếm, cung, mái chèo, quang gánh, lẵng hoa hoặc nón quai thao. | Cục Di sản văn hóa (Tạp chí Di sản văn hóa số 49) |
| `CV_VIS_05` | `trang_phuc_hoa_trang` | `video_sequence` | Trang phục của người hầu đồng rực rỡ và thay đổi theo từng giá hầu (gồm áo dài, áo mớ ba, áo mớ bảy, khăn/mũ, yếm, thắt lưng và trang sức) tương ứng với hình tượng vị Thánh. | Cục Di sản văn hóa (Tạp chí Di sản văn hóa số 49) |
| `CV_VIS_06` | `trang_phuc_hoa_trang` | `single_frame` | Người hầu đồng mặc áo mớ ba màu trắng, đội khăn ba múi và đi hài cánh phượng (đặc trưng thị giác của giá Cô Bơ Thoải phủ). | Cục Di sản văn hóa (Tạp chí Di sản văn hóa số 49) |
| `CV_VIS_07` | `trang_phuc_hoa_trang` | `single_frame` | Người hầu đồng mặc áo màu vàng, đội khăn hoa và múa kiếm hoặc cung (đặc trưng thị giác của giá Chầu Mười Đồng Mỏ). | Cục Di sản văn hóa (Tạp chí Di sản văn hóa số 49) |

### 2.5. Quan họ (`quan_ho`)

| Mã Concept | Khía cạnh | Cấp độ bằng chứng | Câu Concept Lý giải (Atomic Sentence) | Nguồn tham khảo |
| ----- | ----- | ----- | ----- | ----- |
| `QH_VIS_01` | `khong_gian_doi_hinh` | `single_frame` | Biểu diễn theo cặp hoặc theo nhóm, đứng hoặc ngồi đối diện nhau để hát đối đáp giao duyên giữa liền chị (nữ) và liền anh (nam). | Viện Âm nhạc Việt Nam; UNESCO ("Quan Họ Bắc Ninh folk songs"); ICHCAP |
| `QH_VIS_02` | `khong_gian_doi_hinh` | `single_frame` | Không gian biểu diễn gắn với lễ hội truyền thống, nghi lễ, hội thi, các cuộc gặp gỡ cộng đồng và giao lưu giữa các làng vùng Kinh Bắc. | Viện Âm nhạc Việt Nam; UNESCO ("Quan Họ Bắc Ninh folk songs") |
| `QH_VIS_03` | `nhac_cu_dao_cu` | `single_frame` | Khung hình tập trung vào các cặp hoặc nhóm người hát đối đáp truyền thống mà không phụ thuộc vào dàn nhạc cụ đệm. | Viện Âm nhạc Việt Nam; ICHCAP ("Quan Họ Bắc Ninh in the North Vietnam") |
| `QH_VIS_04` | `trang_phuc_hoa_trang` | `single_frame` | Người nữ (liền chị) mặc áo mớ ba hoặc áo mớ bảy, áo năm thân truyền thống kết hợp với yếm và áo cánh bên trong. | Viện Âm nhạc Việt Nam ("Dân ca Quan họ") |
| `QH_VIS_05` | `trang_phuc_hoa_trang` | `single_frame` | Liền chị đầu vấn khăn mỏ quạ, tay cầm hoặc đội nón quai thao, đeo thắt lưng và xà tích. | Viện Âm nhạc Việt Nam ("Dân ca Quan họ") |
| `QH_VIS_06` | `trang_phuc_hoa_trang` | `single_frame` | Người nam (liền anh) mặc áo dài đen (áo the), quần dài trắng và đầu đội khăn xếp hoặc khăn nhiễu. | Viện Âm nhạc Việt Nam ("Dân ca Quan họ") |

### 2.6. Hát Xẩm (`hat_xam`)

| Mã Concept | Khía cạnh | Cấp độ bằng chứng | Câu Concept Lý giải (Atomic Sentence) | Nguồn tham khảo |
| ----- | ----- | ----- | ----- | ----- |
| `HX_VIS_01` | `khong_gian_doi_hinh` | `single_frame` | Không gian biểu diễn dân dã, bình dị như ngồi trên chiếu Xẩm tại chợ, đường phố, bến đò, lễ hội hoặc gắn với bối cảnh tàu điện đô thị. | Viện Âm nhạc Việt Nam (Tư liệu CD "Dạt nước cánh bèo"); Cổng GTĐT Hà Nội (hanoi.gov.vn) |
| `HX_VIS_02` | `dong_tac_dien_xuat` | `single_frame` | Nghệ nhân ngồi biểu diễn với tư thế mộc mạc, vừa trực tiếp chơi nhạc cụ vừa hát kể chuyện. | Viện Âm nhạc Việt Nam; Cổng GTĐT Hà Nội (hanoi.gov.vn) |
| `HX_VIS_03` | `nhac_cu_dao_cu` | `single_frame` | Xuất hiện bộ nhạc cụ Xẩm gồm đàn nhị, đàn bầu kết hợp với sênh (sênh sứa/sênh tiền), phách, trống mảnh, trống cơm hoặc thanh la. | Viện Âm nhạc Việt Nam (Tư liệu CD "Dạt nước cánh bèo"); Đặng Hoành Loan |
| `HX_VIS_04` | `trang_phuc_hoa_trang` | `single_frame` | Nghệ nhân mặc trang phục mộc mạc, bình dân với quần áo vải nâu giản dị (trong bối cảnh Xẩm chợ truyền thống). | Cổng Giao tiếp điện tử Hà Nội (hanoi.gov.vn) |
| `HX_VIS_05` | `trang_phuc_hoa_trang` | `single_frame` | Nghệ nhân nữ mặc áo nâu, yếm, váy, đội khăn mỏ quạ hoặc khăn vắt vai và đi chân đất (hình ảnh đặc trưng của Xẩm cổ và nghệ nhân Hà Thị Cầu). | Cổng Giao tiếp điện tử Hà Nội (hanoi.gov.vn) |
| `HX_VIS_06` | `trang_phuc_hoa_trang` | `single_frame` | Nghệ nhân nam mặc quần áo nâu, đội mũ cát và đeo kính đen; nghệ nhân nữ mặc áo tối màu, yếm sáng màu và váy dài đến khoảng đầu gối (trong bối cảnh Xẩm tàu điện). | Cổng Giao tiếp điện tử Hà Nội (hanoi.gov.vn) |

## 3. Dữ liệu Định dạng CSV (Dùng trực tiếp cho Code Python / Pandas)

Bạn có thể copy khối dưới đây lưu thành `visual_concepts.csv` để nạp thẳng vào pipeline:

```csv
concept_id,genre_code,genre_name,aspect,evidence_level,concept_sentence,source
CL_VIS_01,cai_luong,Cải lương,khong_gian_doi_hinh,single_frame,"Biểu diễn trên sân khấu kịch hát với bối cảnh phục vụ cốt truyện và tình huống kịch.","Viện Âm nhạc Việt Nam; Nguyễn Thị Trúc Bạch (2022)"
CL_VIS_02,cai_luong,Cải lương,dong_tac_dien_xuat,video_sequence,"Nghệ sĩ kết hợp ca hát với diễn xuất biểu cảm và các động tác hình thể sân khấu (ca ra bộ).","Viện Âm nhạc Việt Nam; Nguyễn Thị Trúc Bạch (2022)"
CL_VIS_03,cai_luong,Cải lương,nhac_cu_dao_cu,single_frame,"Xuất hiện đàn guitar phím lõm trong dàn nhạc đệm.","Viện Âm nhạc Việt Nam"
CL_VIS_04,cai_luong,Cải lương,nhac_cu_dao_cu,single_frame,"Xuất hiện các nhạc cụ thuộc dàn nhạc đờn ca tài tử như đàn kìm, đàn cò/violon, đàn tranh và đàn sến.","Viện Âm nhạc Việt Nam"
CL_VIS_05,cai_luong,Cải lương,trang_phuc_hoa_trang,single_frame,"Nghệ sĩ mặc phục trang sân khấu cổ trang làm từ chất liệu lụa, gấm, nhung với màu sắc rực rỡ (trong các vở lịch sử hoặc tuồng cổ).","Hội đồng Lý luận, Phê bình VHNT Trung ương (lyluanphebinh.vn)"
CL_VIS_06,cai_luong,Cải lương,trang_phuc_hoa_trang,single_frame,"Trang phục nhân vật cổ trang có mũ/mão, khăn đóng, áo dài, áo ngắn, váy xòe, trang sức sân khấu và họa tiết thêu rồng, phượng biểu thị địa vị.","Hội đồng Lý luận, Phê bình VHNT Trung ương (lyluanphebinh.vn)"
CL_VIS_07,cai_luong,Cải lương,trang_phuc_hoa_trang,single_frame,"Nghệ sĩ mặc trang phục mô phỏng đời thường, áo dài, áo bà ba hoặc thời trang đương thời theo hoàn cảnh nhân vật (trong các vở đề tài xã hội hiện đại).","Hội đồng Lý luận, Phê bình VHNT Trung ương (lyluanphebinh.vn)"
CT_VIS_01,ca_tru,Ca trù,khong_gian_doi_hinh,single_frame,"Biểu diễn trong không gian thính phòng truyền thống hoặc không gian hát cửa đình.","UNESCO (Ca trù singing); ICHCAP (Ca Trù Singing)"
CT_VIS_02,ca_tru,Ca trù,khong_gian_doi_hinh,single_frame,"Đội hình biểu diễn điển hình gồm ba thành phần: đào nương (người nữ hát), kép đàn (người nam chơi đàn đáy) và quan viên (người đánh trống chầu).","UNESCO (Ca trù singing); ICHCAP"
CT_VIS_03,ca_tru,Ca trù,nhac_cu_dao_cu,single_frame,"Đào nương ngồi hát đồng thời hai tay cầm dùi tự gõ vào bàn phách đặt phía trước để giữ nhịp.","UNESCO (Ca trù singing); ICHCAP"
CT_VIS_04,ca_tru,Ca trù,nhac_cu_dao_cu,single_frame,"Kép đàn ngồi đệm đàn đáy có cần đàn dài đặc trưng.","UNESCO (Ca trù singing); ICHCAP"
CT_VIS_05,ca_tru,Ca trù,nhac_cu_dao_cu,single_frame,"Quan viên ngồi cầm roi chầu gõ vào trống chầu để tương tác với người hát.","UNESCO (Ca trù singing); ICHCAP"
CT_VIS_06,ca_tru,Ca trù,trang_phuc_hoa_trang,single_frame,"Đào nương mặc trang phục truyền thống trang trọng, kín đáo và thanh lịch như áo thao (áo dài lụa tơ tằm màu nhạt hoặc sáng), quần đen và quấn khăn nhiễu trên đầu.","Viện Âm nhạc Việt Nam (Vài nét về Ca trù Quảng Bình)"
CT_VIS_07,ca_tru,Ca trù,trang_phuc_hoa_trang,single_frame,"Đào nương mặc quần lĩnh, áo đoạn hoa, quấn khăn nhung và vấn tóc đuôi gà đoan trang.","Viện Âm nhạc Việt Nam (Vài nét về Ca trù Quảng Bình)"
CT_VIS_08,ca_tru,Ca trù,trang_phuc_hoa_trang,single_frame,"Kép đàn mặc áo the truyền thống và đội khăn xếp trang trọng.","Viện Âm nhạc Việt Nam (Vài nét về Ca trù Quảng Bình)"
CH_VIS_01,cheo,Chèo,khong_gian_doi_hinh,single_frame,"Không gian biểu diễn là chiếu Chèo truyền thống ở sân đình, sân làng hoặc sân khấu kịch hát dân gian.","Ủy ban Nhà nước về người Việt Nam ở nước ngoài (scov.gov.vn)"
CH_VIS_02,cheo,Chèo,khong_gian_doi_hinh,single_frame,"Bố cục sân khấu có dàn nhạc công ngồi ở hai bên chiếu Chèo và khán giả đứng hoặc ngồi bao quanh.","Ủy ban Nhà nước về người Việt Nam ở nước ngoài (scov.gov.vn)"
CH_VIS_03,cheo,Chèo,dong_tac_dien_xuat,video_sequence,"Diễn viên thực hiện đồng thời các động tác múa dân gian, diễn xuất hình thể kết hợp với hát và nói thoại để thể hiện nhân vật.","Ủy ban Nhà nước về người Việt Nam ở nước ngoài (scov.gov.vn)"
CH_VIS_04,cheo,Chèo,nhac_cu_dao_cu,single_frame,"Xuất hiện bộ gõ gồm trống Chèo và mõ cùng các nhạc cụ dân tộc (như đàn bầu) trong dàn nhạc đệm.","Ủy ban Nhà nước về người Việt Nam ở nước ngoài; Học viện Âm nhạc Quốc gia Việt Nam"
CH_VIS_05,cheo,Chèo,trang_phuc_hoa_trang,single_frame,"Trang phục và hóa trang mang tính ước lệ cao nhằm nhận diện tuyến nhân vật: vai vua, quan hoặc tướng lĩnh mặc áo bào, đội mũ mang tính quyền uy.","Ủy ban Nhà nước về người Việt Nam ở nước ngoài (scov.gov.vn)"
CH_VIS_06,cheo,Chèo,trang_phuc_hoa_trang,single_frame,"Nhân vật thuộc tuyến thư sinh hoặc nho sĩ mặc áo dài hoặc áo the và đội khăn xếp.","Ủy ban Nhà nước về người Việt Nam ở nước ngoài (scov.gov.vn)"
CH_VIS_07,cheo,Chèo,trang_phuc_hoa_trang,single_frame,"Nhân vật nữ mặc trang phục dân gian Bắc Bộ gồm áo tứ thân, yếm, váy và vấn khăn.","Ủy ban Nhà nước về người Việt Nam ở nước ngoài (scov.gov.vn)"
CH_VIS_08,cheo,Chèo,trang_phuc_hoa_trang,single_frame,"Nhân vật hề Chèo sử dụng trang phục phóng đại hoặc có màu sắc nổi bật để hỗ trợ tính hài hước.","Ủy ban Nhà nước về người Việt Nam ở nước ngoài (scov.gov.vn)"
CV_VIS_01,chau_van,Chầu văn,khong_gian_doi_hinh,single_frame,"Không gian diễn xướng gắn với thực hành nghi lễ tín ngưỡng thờ Mẫu (đền, phủ, điện thờ) kết hợp giữa hầu đồng và hát văn.","Cục Di sản văn hóa (dsvh.gov.vn); UNESCO"
CV_VIS_02,chau_van,Chầu văn,dong_tac_dien_xuat,video_sequence,"Người thực hành nghi lễ (thanh đồng) thực hiện các điệu múa thiêng và nghi thức hầu đồng ở trung tâm, trong khi cung văn ngồi bên cạnh đàn, hát và quan sát để ứng tác.","Cục Di sản văn hóa (dsvh.gov.vn)"
CV_VIS_03,chau_van,Chầu văn,nhac_cu_dao_cu,single_frame,"Cung văn sử dụng đàn nguyệt kết hợp với bộ gõ gồm trống ban (trống con), phách, cảnh, thanh la (có thể thêm đàn nhị, sáo trúc, đàn tranh, trống cơm).","Cục Di sản văn hóa (dsvh.gov.vn); Hồ Thị Hồng Dung"
CV_VIS_04,chau_van,Chầu văn,nhac_cu_dao_cu,single_frame,"Người hầu đồng cầm các đạo cụ nghi lễ đặc trưng theo từng giá hầu như kiếm, cung, mái chèo, quang gánh, lẵng hoa hoặc nón quai thao.","Cục Di sản văn hóa (Tạp chí Di sản văn hóa số 49)"
CV_VIS_05,chau_van,Chầu văn,trang_phuc_hoa_trang,video_sequence,"Trang phục của người hầu đồng rực rỡ và thay đổi theo từng giá hầu (gồm áo dài, áo mớ ba, áo mớ bảy, khăn/mũ, yếm, thắt lưng và trang sức) tương ứng với hình tượng vị Thánh.","Cục Di sản văn hóa (Tạp chí Di sản văn hóa số 49)"
CV_VIS_06,chau_van,Chầu văn,trang_phuc_hoa_trang,single_frame,"Người hầu đồng mặc áo mớ ba màu trắng, đội khăn ba múi và đi hài cánh phượng (đặc trưng thị giác của giá Cô Bơ Thoải phủ).","Cục Di sản văn hóa (Tạp chí Di sản văn hóa số 49)"
CV_VIS_07,chau_van,Chầu văn,trang_phuc_hoa_trang,single_frame,"Người hầu đồng mặc áo màu vàng, đội khăn hoa và múa kiếm hoặc cung (đặc trưng thị giác của giá Chầu Mười Đồng Mỏ).","Cục Di sản văn hóa (Tạp chí Di sản văn hóa số 49)"
QH_VIS_01,quan_ho,Quan họ,khong_gian_doi_hinh,single_frame,"Biểu diễn theo cặp hoặc theo nhóm, đứng hoặc ngồi đối diện nhau để hát đối đáp giao duyên giữa liền chị (nữ) và liền anh (nam).","Viện Âm nhạc Việt Nam; UNESCO (Quan Họ Bắc Ninh folk songs); ICHCAP"
QH_VIS_02,quan_ho,Quan họ,khong_gian_doi_hinh,single_frame,"Không gian biểu diễn gắn với lễ hội truyền thống, nghi lễ, hội thi, các cuộc gặp gỡ cộng đồng và giao lưu giữa các làng vùng Kinh Bắc.","Viện Âm nhạc Việt Nam; UNESCO (Quan Họ Bắc Ninh folk songs)"
QH_VIS_03,quan_ho,Quan họ,nhac_cu_dao_cu,single_frame,"Khung hình tập trung vào các cặp hoặc nhóm người hát đối đáp truyền thống mà không phụ thuộc vào dàn nhạc cụ đệm.","Viện Âm nhạc Việt Nam; ICHCAP (Quan Họ Bắc Ninh in the North Vietnam)"
QH_VIS_04,quan_ho,Quan họ,trang_phuc_hoa_trang,single_frame,"Người nữ (liền chị) mặc áo mớ ba hoặc áo mớ bảy, áo năm thân truyền thống kết hợp với yếm và áo cánh bên trong.","Viện Âm nhạc Việt Nam (Dân ca Quan họ)"
QH_VIS_05,quan_ho,Quan họ,trang_phuc_hoa_trang,single_frame,"Liền chị đầu vấn khăn mỏ quạ, tay cầm hoặc đội nón quai thao, đeo thắt lưng và xà tích.","Viện Âm nhạc Việt Nam (Dân ca Quan họ)"
QH_VIS_06,quan_ho,Quan họ,trang_phuc_hoa_trang,single_frame,"Người nam (liền anh) mặc áo dài đen (áo the), quần dài trắng và đầu đội khăn xếp hoặc khăn nhiễu.","Viện Âm nhạc Việt Nam (Dân ca Quan họ)"
HX_VIS_01,hat_xam,Hát Xẩm,khong_gian_doi_hinh,single_frame,"Không gian biểu diễn dân dã, bình dị như ngồi trên chiếu Xẩm tại chợ, đường phố, bến đò, lễ hội hoặc gắn với bối cảnh tàu điện đô thị.","Viện Âm nhạc Việt Nam (Tư liệu CD Dạt nước cánh bèo); Cổng GTĐT Hà Nội (hanoi.gov.vn)"
HX_VIS_02,hat_xam,Hát Xẩm,dong_tac_dien_xuat,single_frame,"Nghệ nhân ngồi biểu diễn với tư thế mộc mạc, vừa trực tiếp chơi nhạc cụ vừa hát kể chuyện.","Viện Âm nhạc Việt Nam; Cổng GTĐT Hà Nội (hanoi.gov.vn)"
HX_VIS_03,hat_xam,Hát Xẩm,nhac_cu_dao_cu,single_frame,"Xuất hiện bộ nhạc cụ Xẩm gồm đàn nhị, đàn bầu kết hợp với sênh (sênh sứa/sênh tiền), phách, trống mảnh, trống cơm hoặc thanh la.","Viện Âm nhạc Việt Nam (Tư liệu CD Dạt nước cánh bèo); Đặng Hoành Loan"
HX_VIS_04,hat_xam,Hát Xẩm,trang_phuc_hoa_trang,single_frame,"Nghệ nhân mặc trang phục mộc mạc, bình dân với quần áo vải nâu giản dị (trong bối cảnh Xẩm chợ truyền thống).","Cổng Giao tiếp điện tử Hà Nội (hanoi.gov.vn)"
HX_VIS_05,hat_xam,Hát Xẩm,trang_phuc_hoa_trang,single_frame,"Nghệ nhân nữ mặc áo nâu, yếm, váy, đội khăn mỏ quạ hoặc khăn vắt vai và đi chân đất (hình ảnh đặc trưng của Xẩm cổ và nghệ nhân Hà Thị Cầu).","Cổng Giao tiếp điện tử Hà Nội (hanoi.gov.vn)"
HX_VIS_06,hat_xam,Hát Xẩm,trang_phuc_hoa_trang,single_frame,"Nghệ nhân nam mặc quần áo nâu, đội mũ cát và đeo kính đen; nghệ nhân nữ mặc áo tối màu, yếm sáng màu và váy dài đến khoảng đầu gối (trong bối cảnh Xẩm tàu điện).","Cổng Giao tiếp điện tử Hà Nội (hanoi.gov.vn)"
```