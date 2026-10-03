# Khái niệm di sản văn hóa phi vật thể --- phiên bản tối ưu cho Chunking và Vector hóa

Tài liệu này được **tái cấu trúc từ file khái niệm gốc** để phục vụ
semantic chunking, embedding/vector hóa và truy hồi cho phần diễn giải
của mô hình Neural-Symbolic AI đa phương thức. Nội dung không nhằm biến
các đặc trưng thành luật nhận dạng tuyệt đối.

**Quy tắc:** mỗi `CHUNK` là một đơn vị ngữ nghĩa tương đối độc lập; tên
di sản được lặp lại trong chính chunk; các khía cạnh được tách thành
khái niệm, biểu diễn, âm thanh, nhạc cụ, trang phục, bối cảnh và dấu
hiệu diễn giải. Khi embedding, có thể lưu `heritage`, `aspect`,
`chunk_id`, `keywords` và `source` làm metadata.

------------------------------------------------------------------------

# Ca trù

## CHUNK: ca_tru_khai_niem

**Ca trù --- khái niệm.** Ca trù là một hình thức ca hát thính phòng
truyền thống của người Việt, kết hợp thơ ca, giọng hát và nhạc cụ. Một
hình thức biểu diễn điển hình có ba thành phần: đào nương là người hát,
kép đàn là người chơi đàn đáy và quan viên là người đánh trống chầu.

## CHUNK: ca_tru_bieu_dien

**Ca trù --- cách biểu diễn.** Đào nương không chỉ hát mà còn tự gõ
phách để giữ nhịp. Giọng hát thường có âm vực tương đối cao và sử dụng
luyến, nhấn, đặc biệt là kỹ thuật nảy hạt. Kép đàn chơi đàn đáy và quan
viên sử dụng trống chầu để tương tác với người hát.

## CHUNK: ca_tru_am_thanh

**Ca trù --- đặc trưng âm thanh.** Ca trù nổi bật bởi giọng nữ, âm vực
cao, nhiều luyến láy và nảy hạt, tiết tấu đặc thù, tiếng phách rõ, màu
âm đặc trưng của đàn đáy và tiếng trống chầu tương tác với người hát. Tổ
hợp giọng hát + phách + đàn đáy + trống chầu là nhóm đặc trưng âm thanh
quan trọng.

## CHUNK: ca_tru_trang_phuc

**Ca trù --- trang phục.** Đào nương có phong cách trang trọng, kín đáo
và thanh lịch. Tư liệu Ca trù Quảng Bình ghi nhận áo thao bằng lụa tơ
tằm màu nhạt hoặc sáng, quần đen và khăn nhiễu; một tư liệu khác ghi
nhận quần lĩnh, áo đoạn hoa, khăn nhung và tóc đuôi gà. Kép đàn thường
gắn với áo the và khăn xếp. Trang phục có thể thay đổi theo địa phương,
thời kỳ và không gian diễn xướng.

## CHUNK: ca_tru_dien_giai

**Ca trù --- dấu hiệu diễn giải.** Khi giải thích một mẫu là Ca trù, nên
kết hợp đào nương nữ, cách hát có âm vực cao và luyến láy/nảy hạt, việc
người hát tự gõ phách, đàn đáy và trống chầu. Trang phục chỉ là bằng
chứng hỗ trợ vì có biến đổi theo địa phương và thời kỳ.

**Nguồn tham khảo:** -
https://www.unesco-bg.org/file_store/188329e.pdf -
https://vienamnhac.vn/di-san/ca-tru/bai-viet/vai-net-ve-ca-tru-quang-binh

------------------------------------------------------------------------

# Cải lương

## CHUNK: cai_luong_khai_niem

**Cải lương --- khái niệm.** Cải lương là một loại hình sân khấu kịch
hát. Quá trình phát triển được mô tả từ ca nhạc thính phòng đến biểu
diễn sân khấu, ca ra bộ, hát chập, hát lớp và hình thành sân khấu Cải
lương.

## CHUNK: cai_luong_bieu_dien

**Cải lương --- cách biểu diễn.** Cải lương kết hợp hát, thoại, diễn
xuất và âm nhạc sân khấu. Nghệ sĩ có thể thoại, hát các bài bản, ngâm
thơ, sử dụng vọng cổ và kết hợp động tác sân khấu. Tiết tấu và nhịp điệu
có thể thay đổi theo nhân vật, nhạc và tình huống sân khấu.

## CHUNK: cai_luong_nhac_cu

**Cải lương --- nhạc cụ.** Dàn nhạc Cải lương có nguồn gốc từ dàn nhạc
tài tử, với các nhạc cụ như đàn kìm, đàn cò hoặc violon, đàn tranh và
đàn sến. Đàn guitar phím lõm là một nhạc cụ đặc biệt quan trọng và tạo
màu âm đặc trưng.

## CHUNK: cai_luong_am_thanh

**Cải lương --- đặc trưng âm thanh.** Cải lương có giọng hát giàu tính
biểu cảm, nhiều luyến láy và tiết tấu linh hoạt. Có thể sử dụng các hơi
Bắc, Nam, Oán. Vọng cổ thường có cấu trúc 6 câu và có thể xuất hiện tân
cổ giao duyên. Vọng cổ, giọng ca Nam Bộ và guitar phím lõm là nhóm đặc
trưng âm thanh đáng chú ý.

## CHUNK: cai_luong_trang_phuc

**Cải lương --- trang phục.** Cải lương không có một bộ trang phục cố
định. Trang phục thay đổi theo bối cảnh, thời kỳ, thể loại vở diễn và
nhân vật. Vở xã hội hiện đại có thể dùng trang phục đời thường; vở lịch
sử, tuồng cổ hoặc tuồng Tàu có thể dùng lụa, gấm, nhung, áo dài, áo
ngắn, váy xòe, mũ/mão, khăn đóng, trang sức và phụ kiện sân khấu. Trang
phục giúp nhận diện nhân vật, địa vị, tính cách và hoàn cảnh.

## CHUNK: cai_luong_dien_giai

**Cải lương --- dấu hiệu diễn giải.** Nên kết hợp sân khấu kịch hát, hát
và thoại, diễn xuất, vọng cổ, giọng ca giàu tính biểu cảm, guitar phím
lõm và các nhạc cụ của dàn nhạc tài tử. Không nên dùng riêng áo bà ba
hoặc áo dài làm quy tắc nhận dạng Cải lương.

**Nguồn tham khảo:** -
https://vienamnhac.vn/bai-viet/nhac-co/nhac-tai-tu-nhac-cai-luong-nhung-net-tuong-%25C4%2591ong-va-di-biet -
https://lyluanphebinh.vn/tin-tuc/nghien-cuu-trao-doi/3162/phuc-trang-cai-luong-mien-nam-voi-no-luc-giu-gin-va-phat-trien

------------------------------------------------------------------------

# Chầu văn / Hát văn

## CHUNK: chau_van_khai_niem

**Chầu văn / Hát văn --- khái niệm.** Chầu văn của người Việt là một hệ
thống diễn xướng âm nhạc gắn với nghi lễ, đặc biệt là hầu đồng. Hình
thức này kết hợp hầu đồng, hát văn, âm nhạc, múa, trang phục, đạo cụ và
nghi lễ. Hát chầu văn là phần âm nhạc và ca hát phục vụ nghi lễ.

## CHUNK: chau_van_cung_van

**Chầu văn / Hát văn --- cung văn và cách biểu diễn.** Người hát được
gọi là cung văn. Cung văn đồng thời phải hát, đàn, giữ nhịp, quan sát
người hầu đồng và ứng tác theo từng giá hầu. Hoạt động biểu diễn có tính
tương tác giữa cung văn và người hầu đồng; tiết tấu có thể thay đổi tùy
giá hầu.

## CHUNK: chau_van_am_thanh

**Chầu văn / Hát văn --- đặc trưng âm thanh và nhạc cụ.** Giọng hát
thường sử dụng giọng thật hoặc giọng ngực, nhiều luyến láy, nốt hoa mỹ
và kỹ thuật nảy hạt; âm thanh có thể mềm mại, sáng và giàu tính biểu
cảm. Nhạc cụ gồm đàn nguyệt, trống ban hoặc trống con, phách, cảnh và
thanh la; ngoài ra có thể có nhị, sáo, đàn thập lục và trống cơm.

## CHUNK: chau_van_trang_phuc

**Chầu văn / Hát văn --- trang phục và đạo cụ.** Trang phục là thành
phần trực tiếp của nghi lễ, đặc biệt trong hầu đồng. Người thực hành
thay đổi trang phục và đạo cụ theo từng giá hầu, tương ứng với sự tích,
công trạng, tính cách và hình tượng của vị Thánh. Trang phục có thể gồm
áo dài, áo mớ ba, áo mớ bảy, khăn/mũ, yếm, thắt lưng và trang sức. Đạo
cụ có thể gồm kiếm, cung, mái chèo, quang gánh, lẵng hoa hoặc nón quai
thao.

## CHUNK: chau_van_dien_giai

**Chầu văn / Hát văn --- dấu hiệu diễn giải.** Các bằng chứng quan trọng
gồm bối cảnh hầu đồng, cung văn, sự thay đổi trang phục theo giá hầu,
đạo cụ nghi lễ, đàn nguyệt, phách, trống, cảnh hoặc thanh la, cùng phong
cách hát nhiều luyến láy và nảy hạt. Trang phục, đạo cụ, động tác nghi
lễ và không gian hầu đồng có thể kết hợp thành bằng chứng thị giác.

**Nguồn tham khảo:** -
https://dsvh.gov.vn/nghi-le-chau-van-cua-nguoi-viet-3150 -
https://dsvh.gov.vn/Upload/files/Tap%20chi%20DSVH/So%2049/4909_Suu%20tam_hien%20vat_trang_phuc_dao_cu_bao_ton_va_phat_huy_gia_tri_nghi_le_Chau_Van_cua_nguoi_Viet_o_Nam_Dinh.pdf

------------------------------------------------------------------------

# Chèo

## CHUNK: cheo_khai_niem

**Chèo --- khái niệm.** Chèo là một loại hình sân khấu kịch hát dân gian
truyền thống. Chèo kết hợp hát, nói, múa, diễn xuất, âm nhạc và cốt
truyện sân khấu. Hát Chèo có thể là đơn ca, song ca hoặc đồng ca.

## CHUNK: cheo_khong_gian

**Chèo --- không gian biểu diễn.** Không gian biểu diễn truyền thống có
thể là chiếu Chèo ở sân đình hoặc sân làng. Nhạc công thường ngồi hai
bên và khán giả có thể đứng hoặc ngồi xung quanh. Không gian, vị trí
nhạc công và tổ chức sân khấu có thể hỗ trợ nhận diện Chèo.

## CHUNK: cheo_bieu_dien

**Chèo --- cách biểu diễn.** Diễn viên Chèo đồng thời hát, múa, diễn
xuất, thoại và thể hiện nhân vật. Nhịp điệu âm nhạc có quan hệ với động
tác sân khấu. Vì Chèo kết hợp hát với nói và diễn xuất, dữ liệu âm thanh
có thể chứa cả phần hát và phần nói, còn hình ảnh có thể thể hiện động
tác múa và diễn xuất.

## CHUNK: cheo_am_thanh

**Chèo --- đặc trưng âm thanh.** Âm thanh Chèo mang màu sắc dân gian Bắc
Bộ và có nhiều làn điệu Chèo. Đặc điểm đáng chú ý là sự kết hợp giữa hát
và nói và mối liên hệ giữa nhịp điệu với động tác sân khấu. Tiếng trống
và mõ đóng vai trò quan trọng.

## CHUNK: cheo_trang_phuc

**Chèo --- trang phục.** Trang phục Chèo mang tính ước lệ và có chức
năng nhận diện nhân vật. Trang phục thay đổi theo tuyến nhân vật, địa vị
và tính cách. Vai vua, quan hoặc tướng lĩnh có thể có áo bào, mũ hoặc
phục trang mang tính quyền uy; vai thư sinh hoặc nho sĩ thường gắn với
áo dài hoặc áo the và khăn xếp; vai nữ có thể sử dụng áo tứ thân, yếm,
váy và khăn. Nhân vật hề có thể dùng trang phục phóng đại hoặc màu sắc
nổi bật.

## CHUNK: cheo_dien_giai

**Chèo --- dấu hiệu diễn giải.** Có thể kết hợp sân khấu dân gian, không
gian chiếu Chèo, hát và nói, múa và diễn xuất, màu sắc âm nhạc Bắc Bộ,
trống và mõ, cùng hệ thống trang phục ước lệ theo tuyến nhân vật. Một bộ
trang phục riêng lẻ không đủ để kết luận Chèo.

**Nguồn tham khảo:** -
https://scov.gov.vn/ban-sac-van-hoa/gioi-thieu-ban-sac-van-hoa/cheo-nghe-thuat-san-khau-truyen-thong-tieu-bieu-cua-nguoi-viet.html

------------------------------------------------------------------------

# Hát Xẩm

## CHUNK: hat_xam_khai_niem

**Hát Xẩm --- khái niệm.** Hát Xẩm là một loại hình ca nhạc cổ truyền
gắn với những người hát rong; trong lịch sử, phần lớn người thực hành
được mô tả là người khiếm thị. Xẩm thường được biểu diễn tại chợ, đường
phố, bến đò và lễ hội, từng là phương tiện mưu sinh.

## CHUNK: hat_xam_noi_dung

**Hát Xẩm --- nội dung và cách biểu diễn.** Xẩm có tính tự sự và kể
chuyện. Người hát dùng giọng ca để kể chuyện, nói về cuộc sống và thân
phận, khuyên răn, châm biếm và phản ánh các vấn đề xã hội. Phong cách
thể hiện có thể mộc mạc, trữ tình, hài hước hoặc châm biếm.

## CHUNK: hat_xam_nhac_cu

**Hát Xẩm --- nhạc cụ.** Bộ nhạc cụ Xẩm có thể gồm đàn nhị, đàn bầu,
sênh hoặc sênh sứa hoặc sênh tiền, phách và trống mảnh. Đàn nhị đóng vai
trò quan trọng trong việc dẫn dắt giai điệu; sênh và phách đảm nhiệm
phần tiết tấu. Tổ hợp giọng kể chuyện + đàn nhị/đàn bầu +
sênh/phách/trống mảnh là đặc điểm âm nhạc đáng chú ý.

## CHUNK: hat_xam_am_thanh

**Hát Xẩm --- làn điệu và đặc trưng âm thanh.** Các làn điệu được nhắc
đến gồm Xẩm Hà Liễu, Xẩm Thập ân, Xẩm Ba bậc, Xẩm Xoan và Xẩm Trống
quân. Âm thanh có thể mộc mạc, kể chuyện, trữ tình, hài hước và châm
biếm; tiết tấu có thể khoan thai hoặc mạnh và dồn hơn tùy cách thể hiện.

## CHUNK: hat_xam_trang_phuc

**Hát Xẩm --- trang phục.** Trang phục Xẩm không hoàn toàn cố định vì
Xẩm từng xuất hiện ở chợ, bến đò, đường phố và tàu điện, đồng thời thay
đổi theo thời kỳ. Xẩm chợ thường được phục dựng với hình ảnh mộc mạc như
áo/quần nâu, áo nâu, yếm và váy. Tư liệu về Hà Thị Cầu ghi nhận áo nâu,
chân đất và khăn mỏ quạ. Xẩm tàu điện có hình ảnh nam mặc quần áo nâu,
mũ cát, kính đen; nữ mặc áo tối màu, yếm sáng màu và váy dài khoảng đầu
gối.

## CHUNK: hat_xam_dien_giai

**Hát Xẩm --- dấu hiệu diễn giải.** Có thể kết hợp giọng hát mang tính
kể chuyện, đàn nhị hoặc đàn bầu, sênh/phách/trống mảnh, không gian chợ
hoặc đường phố và phong cách mộc mạc. Trang phục chỉ là bằng chứng hỗ
trợ theo bối cảnh. Xẩm thiên về tự sự/kể chuyện, trong khi Quan họ thiên
về đối đáp giao duyên.

**Nguồn tham khảo:** -
https://vienamnhac.vn/Media/Default/Default/Tai%2520lieu/Dat%2520nuoc%2520canh%2520beo.pdf -
https://hanoi.gov.vn/tin-tuc-su-kien-noi-bat/ha-noi-tim-lai-cho-dung-cho-hat-xam-4000102581.htm

------------------------------------------------------------------------

# Quan họ

## CHUNK: quan_ho_khai_niem

**Quan họ --- khái niệm.** Quan họ là một hình thức dân ca giao duyên và
hát đối đáp của vùng Kinh Bắc. Người hát biểu diễn theo cặp hoặc nhóm và
đối đáp giữa liền chị và liền anh bằng các giai điệu tương ứng và lời ca
khác nhau. Quan họ truyền thống về cơ bản không phụ thuộc vào dàn nhạc
cụ.

## CHUNK: quan_ho_bieu_dien

**Quan họ --- cách biểu diễn.** Đặc trưng nổi bật là đối đáp giữa các
nhóm người hát. Một nhóm hát trước và nhóm còn lại đáp lại, tạo thành
chuỗi tương tác giữa liền chị và liền anh. Trọng tâm của Quan họ truyền
thống nằm ở giọng hát, quan hệ đối đáp và sự hòa giọng thay vì sự phụ
thuộc vào dàn nhạc cụ.

## CHUNK: quan_ho_am_thanh

**Quan họ --- đặc trưng âm thanh.** Trọng tâm âm thanh nằm ở hai nhóm
giọng hát đối đáp và hòa giọng. Các đặc trưng liên quan đến cao độ, hòa
âm và tương tác giọng hát có thể có ý nghĩa lớn khi phân tích âm thanh.
Quan họ truyền thống về cơ bản không phụ thuộc vào dàn nhạc cụ.

## CHUNK: quan_ho_boi_canh

**Quan họ --- không gian và bối cảnh.** Quan họ truyền thống gắn với lễ
hội, nghi lễ, hội thi, các cuộc gặp gỡ cộng đồng và giao lưu giữa các
làng. Bối cảnh cộng đồng và sự tương tác giữa các nhóm người hát là
thông tin quan trọng khi diễn giải dữ liệu Quan họ.

## CHUNK: quan_ho_trang_phuc

**Quan họ --- trang phục.** Liền chị thường mặc áo mớ ba hoặc mớ bảy,
yếm, áo cánh, áo năm thân, kết hợp khăn mỏ quạ, nón quai thao, thắt lưng
và xà tích. Liền anh thường mặc áo dài đen, quần trắng và khăn xếp hoặc
khăn nhiễu. Trang phục có thể kết hợp với hình thức hát đối đáp để hỗ
trợ diễn giải Quan họ.

## CHUNK: quan_ho_dien_giai

**Quan họ --- dấu hiệu diễn giải.** Các dấu hiệu quan trọng có thể gồm
liền chị và liền anh, hình thức hát đối đáp, hòa giọng giữa các nhóm,
bối cảnh giao lưu cộng đồng và trang phục truyền thống. Không nên dùng
riêng nón quai thao hoặc một loại áo làm quy tắc nhận dạng Quan họ; hình
thức đối đáp và tương tác giọng hát là bằng chứng ngữ nghĩa quan trọng.

**Nguồn tham khảo:** -
https://vienamnhac.vn/di-san/dan-ca-quan-ho/bai-viet/dan-ca-quan-ho

------------------------------------------------------------------------

# Quy tắc truy hồi và diễn giải

## CHUNK: rag_vector_metadata

Khi lưu các chunk vào vector database, nên giữ metadata gồm `heritage`,
`aspect`, `chunk_id`, `keywords` và `source`. Ví dụ:
`heritage = Quan họ`, `aspect = trang phục`,
`chunk_id = quan_ho_trang_phuc`. Metadata giúp lọc và kết hợp các chunk
liên quan sau khi semantic search.

## CHUNK: rag_multimodal_evidence

Phần diễn giải nên kết hợp bằng chứng từ nhiều phương thức: **âm thanh,
hình ảnh, cách biểu diễn, nhạc cụ, trang phục, không gian và bối cảnh**.
Một chunk không cần mô tả toàn bộ di sản; các chunk chuyên biệt có thể
được truy hồi riêng rồi kết hợp.

## CHUNK: rag_no_single_feature_rule

Không nên biến một đặc trưng đơn lẻ thành luật tuyệt đối. Ví dụ, nón
quai thao không đồng nghĩa tuyệt đối với Quan họ và áo dài không đồng
nghĩa tuyệt đối với Cải lương. Diễn giải nên nêu các bằng chứng quan sát
được và liên hệ chúng với các đặc trưng trong cơ sở tri thức.

## CHUNK: rag_explanation_pattern

Một phần diễn giải có thể theo cấu trúc: **quan sát từ mô hình → đặc
trưng tương ứng → chunk tri thức hỗ trợ → kết luận có mức độ phù hợp**.
Ví dụ, nếu quan sát được hình thức hát đối đáp giữa liền anh và liền chị
cùng trang phục Quan họ, hệ thống có thể truy hồi các chunk
`quan_ho_bieu_dien`, `quan_ho_trang_phuc` và `quan_ho_am_thanh` để tạo
lời giải thích đa bằng chứng.
