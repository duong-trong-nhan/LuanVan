# BỘ CONCEPT ÂM THANH CHUẨN HÓA (ATOMIC AUDIO CONCEPTS)
**Dự án:** Phân lớp đa phương thức và Diễn giải (XAI) 6 thể loại Âm nhạc Truyền thống Việt Nam  
**Phạm vi:** Cải lương (`cai_luong`), Ca trù (`ca_tru`), Chèo (`cheo`), Chầu văn (`chau_van`), Quan họ (`quan_ho`), Hát Xẩm (`hat_xam`)

---

## 1. Quy ước Cấu trúc Dữ liệu (Schema)
Mỗi concept âm thanh được tách thành một **mệnh đề khẳng định ngắn gọn (câu đơn)** và quản lý qua 6 trường thông tin:
* **`concept_id`**: Mã định danh duy nhất (dùng làm tên biến trong Concept Bottleneck Model hoặc Luật mờ DT/RF/CN2/RIPPER).
* **`genre_code`**: Mã thể loại (`cai_luong`, `ca_tru`, `cheo`, `chau_van`, `quan_ho`, `hat_xam`).
* **`aspect`**: Khía cạnh âm nhạc (`nhac_cu`, `ky_thuat_thanh_nhac`, `giai_dieu_hoi_dieu`, `tiet_tau_nhip`, `cau_truc_bieu_dien`).
* **`evidence_level`**:
  * `direct_audio`: Nhận diện trực tiếp từ tín hiệu âm thanh (Spectral, Timbre/MFCC, Pitch $F_0$, Vibrato, Tempo, RMS Energy).
  * `audio_plus_context`: Cần kết hợp tín hiệu âm thanh với lời ca (ASR transcript) hoặc metadata để khẳng định chắc chắn.
* **`concept_sentence`**: Câu đơn tiếng Việt dùng để ghép trực tiếp vào phần lý giải đầu ra của mô hình.
* **`source`**: Nguồn tài liệu học thuật đối chiếu.

---

## 2. Chi tiết Bộ Concept Âm thanh theo Từng Thể loại

### 2.1. Cải lương (`cai_luong`)
| Mã Concept | Khía cạnh | Cấp độ bằng chứng | Câu Concept Lý giải (Atomic Sentence) | Nguồn tham khảo |
| :--- | :--- | :--- | :--- | :--- |
| `CL_AUD_01` | `nhac_cu` | `direct_audio` | Sử dụng dàn nhạc tài tử Nam Bộ như đàn kìm, đàn cò/violon, đàn tranh và đàn sến. | Viện Âm nhạc Việt Nam; Nguyễn Thị Trúc Bạch (2022) |
| `CL_AUD_02` | `nhac_cu` | `direct_audio` | Có sự xuất hiện của đàn guitar phím lõm tạo ra màu âm đặc trưng trong phần nhạc đệm. | Viện Âm nhạc Việt Nam |
| `CL_AUD_03` | `ky_thuat_thanh_nhac` | `direct_audio` | Câu hát hoặc câu đàn có sử dụng các kỹ thuật rung, nhấn, vuốt, mổ và luyến láy. | Viện Âm nhạc Việt Nam; Trần Quang Hải |
| `CL_AUD_04` | `cau_truc_bieu_dien` | `direct_audio` | Âm nhạc kết hợp giữa giọng hát giàu tính biểu cảm, lời thoại kịch, ngâm thơ và nhạc đệm sân khấu. | Nguyễn Thị Trúc Bạch (2022); Viện Âm nhạc Việt Nam |
| `CL_AUD_05` | `giai_dieu_hoi_dieu` | `direct_audio` | Sử dụng hệ thống hơi và điệu đặc trưng của nhạc tài tử – cải lương như hơi Bắc, hơi Nam, hơi Oán, Lý, Bình, Ngâm. | Viện Âm nhạc Việt Nam |
| `CL_AUD_06` | `giai_dieu_hoi_dieu` | `direct_audio` | Giai điệu thể hiện các sắc thái biểu cảm đặc trưng như hơi Ai (buồn thương), hơi Xuân (tươi sáng) hoặc hơi Oán (trữ tình, ai oán). | Viện Âm nhạc Việt Nam |
| `CL_AUD_07` | `cau_truc_bieu_dien` | `audio_plus_context` | Sử dụng bài bản Vọng cổ (phổ biến với cấu trúc 6 câu, nhịp 32 phát triển từ Dạ Cổ Hoài Lang) hoặc hình thức tân cổ giao duyên. | Viện Âm nhạc Việt Nam; Trần Quang Hải |
| `CL_AUD_08` | `tiet_tau_nhip` | `direct_audio` | Tiết tấu, độ dài câu hát, nhịp nhấn và thời điểm ngân co giãn linh hoạt theo diễn xuất và tình huống sân khấu. | Nguyễn Thị Trúc Bạch (2022); Viện Âm nhạc Việt Nam |

---

### 2.2. Ca trù (`ca_tru`)
| Mã Concept | Khía cạnh | Cấp độ bằng chứng | Câu Concept Lý giải (Atomic Sentence) | Nguồn tham khảo |
| :--- | :--- | :--- | :--- | :--- |
| `CT_AUD_01` | `nhac_cu` | `direct_audio` | Kết hợp tổ hợp âm thanh đặc trưng gồm giọng hát của đào nương, tiếng phách, tiếng đàn đáy và tiếng trống chầu. | UNESCO ("Ca trù singing"); ICHCAP ("Ca Trù Singing") |
| `CT_AUD_02` | `ky_thuat_thanh_nhac` | `direct_audio` | Giọng hát nữ nổi bật ở âm vực tương đối cao, sử dụng kỹ thuật điều khiển hơi thở, rung giọng, luyến, nhấn và kỹ thuật nảy hạt. | UNESCO ("Ca trù singing"); ICHCAP |
| `CT_AUD_03` | `cau_truc_bieu_dien` | `audio_plus_context` | Lời ca gắn với thơ truyền thống qua lối hát thơ, ngâm vọng, ngắt câu, nhả chữ và ngân tiếng giàu chi tiết. | UNESCO ("Ca trù singing"); ICHCAP |
| `CT_AUD_04` | `tiet_tau_nhip` | `direct_audio` | Tiếng phách rõ nét do người hát tự gõ để giữ nhịp và tạo các điểm nhấn tiết tấu định hình dòng diễn xướng. | UNESCO ("Ca trù singing"); ICHCAP |
| `CT_AUD_05` | `nhac_cu` | `direct_audio` | Tiếng đàn đáy đóng vai trò nhạc cụ dây đệm với màu âm trầm đặc trưng. | UNESCO ("Ca trù singing"); ICHCAP |
| `CT_AUD_06` | `nhac_cu` | `direct_audio` | Tiếng trống chầu xuất hiện tạo các điểm nhấn nổi bật và phản hồi tương tác trực tiếp với câu hát của đào nương. | UNESCO ("Ca trù singing"); ICHCAP |
| `CT_AUD_07` | `giai_dieu_hoi_dieu` | `audio_plus_context` | Tiết tấu, độ dài câu và cách phối hợp nhạc cụ biến đổi đa dạng theo hệ thống 56 thể cách/giai điệu (như hát lót cửa đình, Bắc phản, Đò đưa, ngâm vọng). | UNESCO ("Ca trù singing"); ICHCAP |

---

### 2.3. Chèo (`cheo`)
| Mã Concept | Khía cạnh | Cấp độ bằng chứng | Câu Concept Lý giải (Atomic Sentence) | Nguồn tham khảo |
| :--- | :--- | :--- | :--- | :--- |
| `CH_AUD_01` | `cau_truc_bieu_dien` | `direct_audio` | Giọng hát mang đậm màu sắc dân gian Bắc Bộ, kết hợp linh hoạt giữa phần hát và phần nói thoại sân khấu. | Ủy ban Nhà nước về người Việt Nam ở nước ngoài (scov.gov.vn) |
| `CH_AUD_02` | `nhac_cu` | `direct_audio` | Tiếng trống Chèo và tiếng mõ đóng vai trò quan trọng, giữ nhịp gần như xuyên suốt và gắn chặt với động tác sân khấu. | Ủy ban Nhà nước về người Việt Nam ở nước ngoài (scov.gov.vn) |
| `CH_AUD_03` | `cau_truc_bieu_dien` | `audio_plus_context` | Làn điệu được cấu trúc thành các trổ (trổ mở đầu, trổ thân bài, trổ nhắc lại và trổ kết) có thể lặp lại hoặc điều chỉnh theo lời ca. | Tạp chí Giáo dục Nghệ thuật |
| `CH_AUD_04` | `cau_truc_bieu_dien` | `direct_audio` | Xuất hiện các câu nhạc lưu không (nối giữa các trổ hoặc kết bài) và câu nhạc xuyên tâm (xen ngắn giữa các câu hát để phân chia dòng nhạc). | Tạp chí Giáo dục Nghệ thuật |
| `CH_AUD_05` | `ky_thuat_thanh_nhac` | `direct_audio` | Cuối câu hát hoặc cuối trổ có phần ngân đuôi kéo dài những âm tiết như "i", "ơi" hoặc "hỡi" trước khi vào lưu không. | Tạp chí Giáo dục Nghệ thuật |
| `CH_AUD_06` | `ky_thuat_thanh_nhac` | `audio_plus_context` | Nghệ nhân sử dụng kỹ thuật bẻ làn, nắn điệu để điều chỉnh mô hình giai điệu linh hoạt theo lời thơ, thanh điệu và lối diễn. | Tạp chí Giáo dục Nghệ thuật |
| `CH_AUD_07` | `nhac_cu` | `direct_audio` | Phần nhạc đệm (như đàn bầu với âm sắc trong trẻo gần giọng người) tham gia bắt hơi, dẫn giai điệu và ứng tác đáp lại người hát. | Phạm Thị Huyền Trang (2016); Học viện Âm nhạc Quốc gia Việt Nam |
| `CH_AUD_08` | `cau_truc_bieu_dien` | `direct_audio` | Hát Chèo được thể hiện đa dạng dưới hình thức đơn ca, song ca hoặc đồng ca gắn với sắc thái từng vai diễn. | Ủy ban Nhà nước về người Việt Nam ở nước ngoài (scov.gov.vn) |

---

### 2.4. Chầu văn / Hát văn (`chau_van`)
| Mã Concept | Khía cạnh | Cấp độ bằng chứng | Câu Concept Lý giải (Atomic Sentence) | Nguồn tham khảo |
| :--- | :--- | :--- | :--- | :--- |
| `CV_AUD_01` | `nhac_cu` | `direct_audio` | Kết hợp chặt chẽ giữa giọng hát của cung văn với tiếng đàn nguyệt và bộ gõ nòng cốt gồm trống ban (trống con), phách, cảnh, thanh la. | Cục Di sản văn hóa (dsvh.gov.vn); Hồ Thị Hồng Dung; ICHCAP |
| `CV_AUD_02` | `nhac_cu` | `direct_audio` | Dàn nhạc Chầu văn có sự tham gia bổ sung của các nhạc cụ như đàn nhị, đàn tranh (đàn thập lục), sáo trúc hoặc trống cơm. | Cục Di sản văn hóa (dsvh.gov.vn); Hồ Thị Hồng Dung |
| `CV_AUD_03` | `ky_thuat_thanh_nhac` | `direct_audio` | Cung văn hát bằng giọng thật (giọng ngực) với âm thanh mềm mại, sáng và giàu tính biểu cảm. | Cục Di sản văn hóa (dsvh.gov.vn) |
| `CV_AUD_04` | `ky_thuat_thanh_nhac` | `direct_audio` | Câu hát và câu đàn sử dụng nhiều luyến láy, nốt hoa mỹ, kỹ thuật nảy hạt, rung (rõ ở các âm ngân dài), nhấn vuốt và thêu. | Cục Di sản văn hóa (dsvh.gov.vn); Hồ Thị Hồng Dung |
| `CV_AUD_05` | `giai_dieu_hoi_dieu` | `direct_audio` | Đường nét giai điệu vận động đa dạng (đi xuống dần, chữ V, làn sóng hoặc đi ngang) thuộc các nhóm làn điệu đặc trưng như Dọc, Phú, Cờn và Xá. | Hồ Thị Hồng Dung |
| `CV_AUD_06` | `tiet_tau_nhip` | `direct_audio` | Tổ chức nhịp đa dạng (nhịp một, nhịp đôi, nhịp ba, nhịp tự do) kết hợp lối nhấn lệch làm trọng âm lời ca rơi vào phách yếu hoặc kéo dài qua ô nhịp. | Hồ Thị Hồng Dung |
| `CV_AUD_07` | `tiet_tau_nhip` | `direct_audio` | Cường độ năng lượng, mật độ nhạc cụ và tốc độ tiết tấu thay đổi linh hoạt theo diễn tiến âm nhạc và từng giá hầu đồng. | UNESCO; Cục Di sản văn hóa; Hồ Thị Hồng Dung |

---

### 2.5. Quan họ (`quan_ho`)
| Mã Concept | Khía cạnh | Cấp độ bằng chứng | Câu Concept Lý giải (Atomic Sentence) | Nguồn tham khảo |
| :--- | :--- | :--- | :--- | :--- |
| `QH_AUD_01` | `cau_truc_bieu_dien` | `direct_audio` | Trình diễn bằng giọng hát thuần túy theo lối hát đôi hoặc hát đối đáp giao duyên giữa liền chị và liền anh, về cơ bản không có nhạc cụ đệm. | UNESCO ("Quan Họ Bắc Ninh folk songs"); ICHCAP; Viện Âm nhạc Việt Nam |
| `QH_AUD_02` | `cau_truc_bieu_dien` | `direct_audio` | Có sự phối hợp hòa giọng giữa một người dẫn giọng (hát giai điệu chính) và một người luồn giọng (hát phần phụ) để hai giọng hòa quyện về âm sắc. | ICHCAP ("Quan Họ Bắc Ninh in the North Vietnam") |
| `QH_AUD_03` | `ky_thuat_thanh_nhac` | `direct_audio` | Kỹ thuật thanh nhạc thể hiện bốn phẩm chất "vang, rền, nền, nảy" (âm thanh cộng hưởng lan tỏa, rung giọng nối liền, tuyến giai điệu gắn kết và âm tiết bật rõ gọn). | Học viện Âm nhạc Quốc gia Việt Nam; Bách khoa Toàn thư Việt Nam |
| `QH_AUD_04` | `giai_dieu_hoi_dieu` | `direct_audio` | Giai điệu kết hợp lời thơ/ca dao với nhiều tiếng đệm, tiếng đưa hơi như "i", "ư", "a", "tính tình tang" để nối câu và tạo độ ngân. | UNESCO ("Quan Họ Bắc Ninh folk songs"); Bách khoa Toàn thư Việt Nam |
| `QH_AUD_05` | `giai_dieu_hoi_dieu` | `audio_plus_context` | Bài hát thuộc hệ thống các nhóm giọng trong canh hát Quan họ gồm giọng lề lối (chậm, nhiều tiếng đệm/luyến láy), giọng sổng, giọng vặt và giọng giã bạn. | Bách khoa Toàn thư Việt Nam; Viện Âm nhạc Việt Nam |

---

### 2.6. Hát Xẩm (`hat_xam`)
| Mã Concept | Khía cạnh | Cấp độ bằng chứng | Câu Concept Lý giải (Atomic Sentence) | Nguồn tham khảo |
| :--- | :--- | :--- | :--- | :--- |
| `HX_AUD_01` | `nhac_cu` | `direct_audio` | Kết hợp giữa giọng hát kể chuyện với bộ nhạc cụ gồm đàn nhị, đàn bầu và bộ gõ tiết tấu (sênh/sênh sứa/sênh tiền, phách, trống mảnh, trống cơm, thanh la). | Viện Âm nhạc Việt Nam (Tư liệu CD "Dạt nước cánh bèo"); Đặng Hoành Loan |
| `HX_AUD_02` | `nhac_cu` | `direct_audio` | Tiếng đàn nhị hoặc đàn bầu giữ vai trò dẫn dắt giai điệu nổi bật, trong khi tiếng sênh và phách đảm nhiệm phần giữ nhịp tiết tấu. | Viện Âm nhạc Việt Nam; Đặng Hoành Loan |
| `HX_AUD_03` | `ky_thuat_thanh_nhac` | `direct_audio` | Cách nhả chữ, nhấn và ngắt lời mộc mạc, tạo cảm giác gần với lối kể chuyện (tự sự) hoặc nói có nhịp điệu. | Viện Âm nhạc Việt Nam; Tôi Yêu Hát Xẩm |
| `HX_AUD_04` | `giai_dieu_hoi_dieu` | `audio_plus_context` | Sử dụng hệ thống làn điệu đặc trưng của Xẩm như Xẩm Chợ, Xẩm Tàu Điện, Xẩm Hà Liễu, Xẩm Sai, Xẩm Thập Ân, Xẩm Ba Bậc, Xẩm Xoan và Xẩm Trống Quân. | Viện Âm nhạc Việt Nam; Đặng Hoành Loan |
| `HX_AUD_05` | `tiet_tau_nhip` | `direct_audio` | Tiết tấu nhanh, ngắn gọn, mô phỏng nhịp chuyển động chao đảo, giật và phanh của tàu điện đô thị (đặc trưng của biến thể Xẩm Tàu Điện). | VietnamPlus ("Bảo tồn nghệ thuật hát Xẩm") |
| `HX_AUD_06` | `tiet_tau_nhip` | `direct_audio` | Tiết tấu biến thiên linh hoạt, lúc khoan thai, da diết, ai oán (như Xẩm Sai, Hà Liễu), lúc mạnh, dồn dập, hóm hỉnh và châm biếm (như Xẩm Chợ). | Viện Âm nhạc Việt Nam; Đặng Hoành Loan |
| `HX_AUD_07` | `cau_truc_bieu_dien` | `audio_plus_context` | Lời ca mang tính tự sự, kể truyện dân gian hoặc sử dụng thơ ca/ca dao để phản ánh cuộc sống, thân phận, khuyên răn và phê phán xã hội. | Viện Âm nhạc Việt Nam; VietnamPlus |

---

## 3. Dữ liệu Định dạng CSV (Dùng trực tiếp cho Code Python / Pandas)
Bạn có thể copy khối dưới đây lưu thành `audio_concepts.csv` để nạp thẳng vào pipeline:

```csv
concept_id,genre_code,genre_name,aspect,evidence_level,concept_sentence,source
CL_AUD_01,cai_luong,Cải lương,nhac_cu,direct_audio,"Sử dụng dàn nhạc tài tử Nam Bộ như đàn kìm, đàn cò/violon, đàn tranh và đàn sến.","Viện Âm nhạc Việt Nam; Nguyễn Thị Trúc Bạch (2022)"
CL_AUD_02,cai_luong,Cải lương,nhac_cu,direct_audio,"Có sự xuất hiện của đàn guitar phím lõm tạo ra màu âm đặc trưng trong phần nhạc đệm.","Viện Âm nhạc Việt Nam"
CL_AUD_03,cai_luong,Cải lương,ky_thuat_thanh_nhac,direct_audio,"Câu hát hoặc câu đàn có sử dụng các kỹ thuật rung, nhấn, vuốt, mổ và luyến láy.","Viện Âm nhạc Việt Nam; Trần Quang Hải"
CL_AUD_04,cai_luong,Cải lương,cau_truc_bieu_dien,direct_audio,"Âm nhạc kết hợp giữa giọng hát giàu tính biểu cảm, lời thoại kịch, ngâm thơ và nhạc đệm sân khấu.","Nguyễn Thị Trúc Bạch (2022); Viện Âm nhạc Việt Nam"
CL_AUD_05,cai_luong,Cải lương,giai_dieu_hoi_dieu,direct_audio,"Sử dụng hệ thống hơi và điệu đặc trưng của nhạc tài tử – cải lương như hơi Bắc, hơi Nam, hơi Oán, Lý, Bình, Ngâm.","Viện Âm nhạc Việt Nam"
CL_AUD_06,cai_luong,Cải lương,giai_dieu_hoi_dieu,direct_audio,"Giai điệu thể hiện các sắc thái biểu cảm đặc trưng như hơi Ai (buồn thương), hơi Xuân (tươi sáng) hoặc hơi Oán (trữ tình, ai oán).","Viện Âm nhạc Việt Nam"
CL_AUD_07,cai_luong,Cải lương,cau_truc_bieu_dien,audio_plus_context,"Sử dụng bài bản Vọng cổ (phổ biến với cấu trúc 6 câu, nhịp 32 phát triển từ Dạ Cổ Hoài Lang) hoặc hình thức tân cổ giao duyên.","Viện Âm nhạc Việt Nam; Trần Quang Hải"
CL_AUD_08,cai_luong,Cải lương,tiet_tau_nhip,direct_audio,"Tiết tấu, độ dài câu hát, nhịp nhấn và thời điểm ngân co giãn linh hoạt theo diễn xuất và tình huống sân khấu.","Nguyễn Thị Trúc Bạch (2022); Viện Âm nhạc Việt Nam"
CT_AUD_01,ca_tru,Ca trù,nhac_cu,direct_audio,"Kết hợp tổ hợp âm thanh đặc trưng gồm giọng hát của đào nương, tiếng phách, tiếng đàn đáy và tiếng trống chầu.","UNESCO (Ca trù singing); ICHCAP (Ca Trù Singing)"
CT_AUD_02,ca_tru,Ca trù,ky_thuat_thanh_nhac,direct_audio,"Giọng hát nữ nổi bật ở âm vực tương đối cao, sử dụng kỹ thuật điều khiển hơi thở, rung giọng, luyến, nhấn và kỹ thuật nảy hạt.","UNESCO (Ca trù singing); ICHCAP"
CT_AUD_03,ca_tru,Ca trù,cau_truc_bieu_dien,audio_plus_context,"Lời ca gắn với thơ truyền thống qua lối hát thơ, ngâm vọng, ngắt câu, nhả chữ và ngân tiếng giàu chi tiết.","UNESCO (Ca trù singing); ICHCAP"
CT_AUD_04,ca_tru,Ca trù,tiet_tau_nhip,direct_audio,"Tiếng phách rõ nét do người hát tự gõ để giữ nhịp và tạo các điểm nhấn tiết tấu định hình dòng diễn xướng.","UNESCO (Ca trù singing); ICHCAP"
CT_AUD_05,ca_tru,Ca trù,nhac_cu,direct_audio,"Tiếng đàn đáy đóng vai trò nhạc cụ dây đệm với màu âm trầm đặc trưng.","UNESCO (Ca trù singing); ICHCAP"
CT_AUD_06,ca_tru,Ca trù,nhac_cu,direct_audio,"Tiếng trống chầu xuất hiện tạo các điểm nhấn nổi bật và phản hồi tương tác trực tiếp với câu hát của đào nương.","UNESCO (Ca trù singing); ICHCAP"
CT_AUD_07,ca_tru,Ca trù,giai_dieu_hoi_dieu,audio_plus_context,"Tiết tấu, độ dài câu và cách phối hợp nhạc cụ biến đổi đa dạng theo hệ thống 56 thể cách/giai điệu (như hát lót cửa đình, Bắc phản, Đò đưa, ngâm vọng).","UNESCO (Ca trù singing); ICHCAP"
CH_AUD_01,cheo,Chèo,cau_truc_bieu_dien,direct_audio,"Giọng hát mang đậm màu sắc dân gian Bắc Bộ, kết hợp linh hoạt giữa phần hát và phần nói thoại sân khấu.","Ủy ban Nhà nước về người Việt Nam ở nước ngoài (scov.gov.vn)"
CH_AUD_02,cheo,Chèo,nhac_cu,direct_audio,"Tiếng trống Chèo và tiếng mõ đóng vai trò quan trọng, giữ nhịp gần như xuyên suốt và gắn chặt với động tác sân khấu.","Ủy ban Nhà nước về người Việt Nam ở nước ngoài (scov.gov.vn)"
CH_AUD_03,cheo,Chèo,cau_truc_bieu_dien,audio_plus_context,"Làn điệu được cấu trúc thành các trổ (trổ mở đầu, trổ thân bài, trổ nhắc lại và trổ kết) có thể lặp lại hoặc điều chỉnh theo lời ca.","Tạp chí Giáo dục Nghệ thuật"
CH_AUD_04,cheo,Chèo,cau_truc_bieu_dien,direct_audio,"Xuất hiện các câu nhạc lưu không (nối giữa các trổ hoặc kết bài) và câu nhạc xuyên tâm (xen ngắn giữa các câu hát để phân chia dòng nhạc).","Tạp chí Giáo dục Nghệ thuật"
CH_AUD_05,cheo,Chèo,ky_thuat_thanh_nhac,direct_audio,"Cuối câu hát hoặc cuối trổ có phần ngân đuôi kéo dài những âm tiết như 'i', 'ơi' hoặc 'hỡi' trước khi vào lưu không.","Tạp chí Giáo dục Nghệ thuật"
CH_AUD_06,cheo,Chèo,ky_thuat_thanh_nhac,audio_plus_context,"Nghệ nhân sử dụng kỹ thuật bẻ làn, nắn điệu để điều chỉnh mô hình giai điệu linh hoạt theo lời thơ, thanh điệu và lối diễn.","Tạp chí Giáo dục Nghệ thuật"
CH_AUD_07,cheo,Chèo,nhac_cu,direct_audio,"Phần nhạc đệm (như đàn bầu với âm sắc trong trẻo gần giọng người) tham gia bắt hơi, dẫn giai điệu và ứng tác đáp lại người hát.","Phạm Thị Huyền Trang (2016); Học viện Âm nhạc Quốc gia Việt Nam"
CH_AUD_08,cheo,Chèo,cau_truc_bieu_dien,direct_audio,"Hát Chèo được thể hiện đa dạng dưới hình thức đơn ca, song ca hoặc đồng ca gắn với sắc thái từng vai diễn.","Ủy ban Nhà nước về người Việt Nam ở nước ngoài (scov.gov.vn)"
CV_AUD_01,chau_van,Chầu văn,nhac_cu,direct_audio,"Kết hợp chặt chẽ giữa giọng hát của cung văn với tiếng đàn nguyệt và bộ gõ nòng cốt gồm trống ban (trống con), phách, cảnh, thanh la.","Cục Di sản văn hóa (dsvh.gov.vn); Hồ Thị Hồng Dung; ICHCAP"
CV_AUD_02,chau_van,Chầu văn,nhac_cu,direct_audio,"Dàn nhạc Chầu văn có sự tham gia bổ sung của các nhạc cụ như đàn nhị, đàn tranh (đàn thập lục), sáo trúc hoặc trống cơm.","Cục Di sản văn hóa (dsvh.gov.vn); Hồ Thị Hồng Dung"
CV_AUD_03,chau_van,Chầu văn,ky_thuat_thanh_nhac,direct_audio,"Cung văn hát bằng giọng thật (giọng ngực) với âm thanh mềm mại, sáng và giàu tính biểu cảm.","Cục Di sản văn hóa (dsvh.gov.vn)"
CV_AUD_04,chau_van,Chầu văn,ky_thuat_thanh_nhac,direct_audio,"Câu hát và câu đàn sử dụng nhiều luyến láy, nốt hoa mỹ, kỹ thuật nảy hạt, rung (rõ ở các âm ngân dài), nhấn vuốt và thêu.","Cục Di sản văn hóa (dsvh.gov.vn); Hồ Thị Hồng Dung"
CV_AUD_05,chau_van,Chầu văn,giai_dieu_hoi_dieu,direct_audio,"Đường nét giai điệu vận động đa dạng (đi xuống dần, chữ V, làn sóng hoặc đi ngang) thuộc các nhóm làn điệu đặc trưng như Dọc, Phú, Cờn và Xá.","Hồ Thị Hồng Dung"
CV_AUD_06,chau_van,Chầu văn,tiet_tau_nhip,direct_audio,"Tổ chức nhịp đa dạng (nhịp một, nhịp đôi, nhịp ba, nhịp tự do) kết hợp lối nhấn lệch làm trọng âm lời ca rơi vào phách yếu hoặc kéo dài qua ô nhịp.","Hồ Thị Hồng Dung"
CV_AUD_07,chau_van,Chầu văn,tiet_tau_nhip,direct_audio,"Cường độ năng lượng, mật độ nhạc cụ và tốc độ tiết tấu thay đổi linh hoạt theo diễn tiến âm nhạc và từng giá hầu đồng.","UNESCO; Cục Di sản văn hóa; Hồ Thị Hồng Dung"
QH_AUD_01,quan_ho,Quan họ,cau_truc_bieu_dien,direct_audio,"Trình diễn bằng giọng hát thuần túy theo lối hát đôi hoặc hát đối đáp giao duyên giữa liền chị và liền anh, về cơ bản không có nhạc cụ đệm.","UNESCO (Quan Họ Bắc Ninh folk songs); ICHCAP; Viện Âm nhạc Việt Nam"
QH_AUD_02,quan_ho,Quan họ,cau_truc_bieu_dien,direct_audio,"Có sự phối hợp hòa giọng giữa một người dẫn giọng (hát giai điệu chính) và một người luồn giọng (hát phần phụ) để hai giọng hòa quyện về âm sắc.","ICHCAP (Quan Họ Bắc Ninh in the North Vietnam)"
QH_AUD_03,quan_ho,Quan họ,ky_thuat_thanh_nhac,direct_audio,"Kỹ thuật thanh nhạc thể hiện bốn phẩm chất 'vang, rền, nền, nảy' (âm thanh cộng hưởng lan tỏa, rung giọng nối liền, tuyến giai điệu gắn kết và âm tiết bật rõ gọn).","Học viện Âm nhạc Quốc gia Việt Nam; Bách khoa Toàn thư Việt Nam"
QH_AUD_04,quan_ho,Quan họ,giai_dieu_hoi_dieu,direct_audio,"Giai điệu kết hợp lời thơ/ca dao với nhiều tiếng đệm, tiếng đưa hơi như 'i', 'ư', 'a', 'tính tình tang' để nối câu và tạo độ ngân.","UNESCO (Quan Họ Bắc Ninh folk songs); Bách khoa Toàn thư Việt Nam"
QH_AUD_05,quan_ho,Quan họ,giai_dieu_hoi_dieu,audio_plus_context,"Bài hát thuộc hệ thống các nhóm giọng trong canh hát Quan họ gồm giọng lề lối (chậm, nhiều tiếng đệm/luyến láy), giọng sổng, giọng vặt và giọng giã bạn.","Bách khoa Toàn thư Việt Nam; Viện Âm nhạc Việt Nam"
HX_AUD_01,hat_xam,Hát Xẩm,nhac_cu,direct_audio,"Kết hợp giữa giọng hát kể chuyện với bộ nhạc cụ gồm đàn nhị, đàn bầu và bộ gõ tiết tấu (sênh/sênh sứa/sênh tiền, phách, trống mảnh, trống cơm, thanh la).","Viện Âm nhạc Việt Nam (Tư liệu CD Dạt nước cánh bèo); Đặng Hoành Loan"
HX_AUD_02,hat_xam,Hát Xẩm,nhac_cu,direct_audio,"Tiếng đàn nhị hoặc đàn bầu giữ vai trò dẫn dắt giai điệu nổi bật, trong khi tiếng sênh và phách đảm nhiệm phần giữ nhịp tiết tấu.","Viện Âm nhạc Việt Nam; Đặng Hoành Loan"
HX_AUD_03,hat_xam,Hát Xẩm,ky_thuat_thanh_nhac,direct_audio,"Cách nhả chữ, nhấn và ngắt lời mộc mạc, tạo cảm giác gần với lối kể chuyện (tự sự) hoặc nói có nhịp điệu.","Viện Âm nhạc Việt Nam; Tôi Yêu Hát Xẩm"
HX_AUD_04,hat_xam,Hát Xẩm,giai_dieu_hoi_dieu,audio_plus_context,"Sử dụng hệ thống làn điệu đặc trưng của Xẩm như Xẩm Chợ, Xẩm Tàu Điện, Xẩm Hà Liễu, Xẩm Sai, Xẩm Thập Ân, Xẩm Ba Bậc, Xẩm Xoan và Xẩm Trống Quân.","Viện Âm nhạc Việt Nam; Đặng Hoành Loan"
HX_AUD_05,hat_xam,Hát Xẩm,tiet_tau_nhip,direct_audio,"Tiết tấu nhanh, ngắn gọn, mô phỏng nhịp chuyển động chao đảo, giật và phanh của tàu điện đô thị (đặc trưng của biến thể Xẩm Tàu Điện).","VietnamPlus (Bảo tồn nghệ thuật hát Xẩm)"
HX_AUD_06,hat_xam,Hát Xẩm,tiet_tau_nhip,direct_audio,"Tiết tấu biến thiên linh hoạt, lúc khoan thai, da diết, ai oán (như Xẩm Sai, Hà Liễu), lúc mạnh, dồn dập, hóm hỉnh và châm biếm (như Xẩm Chợ).","Viện Âm nhạc Việt Nam; Đặng Hoành Loan"
HX_AUD_07,hat_xam,Hát Xẩm,cau_truc_bieu_dien,audio_plus_context,"Lời ca mang tính tự sự, kể truyện dân gian hoặc sử dụng thơ ca/ca dao để phản ánh cuộc sống, thân phận, khuyên răn và phê phán xã hội.","Viện Âm nhạc Việt Nam; VietnamPlus"
```