## A. Nội dung đồ án

Sinh viên lựa chọn tutorial "Generative AI Meets Reinforcement Learning" từ ICML 2025.

[Project page](https://generative-rl-tutorial.github.io/)

[Slides](https://docs.google.com/presentation/d/1e4Plc9wvbaZKxFZXsvlk0nP5wzSuAoqXQlw_1Ct9vaI/edit?slide=id.g35f41101590_0_0#slide=id.g35f41101590_0_0)

[Notes](https://generative-rl-tutorial.github.io/notes.pdf)

## B. Các hạng mục/tiêu chí thực hiện

### Tiêu chí 1: Đọc hiểu tài liệu (20 điểm)

Sinh viên nghiên cứu toàn bộ tài liệu của tutorial, bao gồm:
- Slide thuyết trình của tutorial.
- Video bài giảng (nếu có) trên trang web của hội nghị hoặc YouTube.
- Các bài báo gốc được đề cập hoặc trích dẫn trong tutorial (tối thiểu 3 bài báo chính).
- Tài liệu tham khảo bổ sung liên quan (nếu cần).

### Tiêu chí 2: Báo cáo bằng tiếng Việt (30 điểm)

Nhóm viết một báo cáo kỹ thuật bằng tiếng Việt, trình bày đầy đủ nội dung của tutorial đã chọn. Báo cáo phải đảm bảo các yêu cầu sau:

- Về nội dung:
  1. Tổng quan chủ đề: Giới thiệu bài toán, động lực nghiên cứu và tầm quan trọng của hướng nghiên cứu.
  2. Kiến thức nền: Trình bày các khái niệm, lý thuyết cơ sở cần thiết để hiểu nội dung tutorial.
  3. Nội dung chính của tutorial: Trình bày chi tiết phương pháp, mô hình hoặc kỹ thuật được đề xuất. Nếu tutorial đề cập nhiều công trình, cần trình bày từng công trình có hệ thống.
  4. Thực nghiệm & Kết quả: Tóm tắt các thí nghiệm và kết quả được báo cáo trong tutorial; so sánh với các baseline nếu có.
  5. Đánh giá & Nhận xét: Nêu ưu điểm, hạn chế của phương pháp; thảo luận về các hướng nghiên cứu tương lai được đề xuất.
  6. Mô tả Demo: Mô tả chi tiết demo mà nhóm thực hiện và kết quả thu được.

- Về hình thức:
  - Viết bằng LATEX sử dụng template tuỳ ý.
  - Độ dài: tối thiểu 15 trang, không tính trang bìa và mục lục.
  - Có đầy đủ hình ảnh, sơ đồ minh hoạ (tự vẽ hoặc trích dẫn từ bài báo gốc có ghi nguồn).
  - Trích dẫn tài liệu tham khảo đầy đủ và đúng định dạng (BibTeX).
  - Mỗi thành viên trong nhóm ghi rõ phần mà mình phụ trách.

### Tiêu chí 3: Demo thực nghiệm (25 điểm)

Mỗi nhóm xây dựng hoặc kế thừa một demo thực nghiệm minh hoạ nội dung của tutorial. Có hai cách tiếp cận được chấp nhận:

**Cách 1: Kế thừa & mở rộng source code có sẵn**
Nhóm tìm một source code mã nguồn mở (official implementation hoặc third-party)
của một công trình được đề cập trong tutorial, sau đó:
- Chạy lại thí nghiệm trên tập dữ liệu gốc hoặc tập dữ liệu mới.
- Thực hiện ít nhất một trong các mở rộng sau: thử nghiệm trên dataset khác, so sánh với baseline, điều chỉnh siêu tham số, visualize kết quả bổ sung, v.v.
- Giải thích rõ từng bước chạy code trong báo cáo.

**Cách 2: Tự xây dựng demo từ đầu**
Nhóm tự cài đặt (implement) một phần hoặc toàn bộ một phương pháp được nêu trong
tutorial:
- Áp dụng trên tập dữ liệu phù hợp (benchmark hoặc real-world).
- Có so sánh với ít nhất một phương pháp baseline.
- Trình bày kết quả có định lượng (metric) và định tính (visualize).

**Lưu ý**
- Ngôn ngữ lập trình: Tuỳ ý. Có thể là Python hoặc Julia.
- Nhóm nên chạy thực nghiệm trong môi trường được ghi lại rõ ràng (ví dụ: requirements.txt cho Python, Project.toml cho Julia).
- Nếu sử dụng GPU/cloud, cần ghi chú rõ cấu hình phần cứng trong báo cáo.
- Demo phải có thể chạy được độc lập bởi người khác khi đọc hướng dẫn trong README.md

### Tiêu chí 4: Slide thuyết trình (15 điểm)

Mỗi nhóm chuẩn bị một bộ slide thuyết trình để báo cáo trước lớp. Yêu cầu:

- Thời lượng trình bày: 30–45 phút.
- Phần hỏi đáp: 10–15 phút sau khi nhóm trình bày xong hoặc trong lúc nhóm đang trình bày.
- Ngôn ngữ: Tiếng Việt.
- Nội dung slide cần bao gồm:
  1. Giới thiệu chủ đề và tutorial.
  2. Kiến thức chuẩn bị.
  3. Nội dung chính của tutorial (phương pháp, mô hình, kết quả).
  4. Thực nghiệm và phân tích kết quả.
  5. Kết luận và các hướng nghiên cứu tương lai.
- Định dạng: PowerPoint (.pptx), Beamer (LATEX) hoặc Google Slides.
- Slide phải chuyên nghiệp, có hình ảnh/sơ đồ minh hoạ, không quá 40 slide.

## C. Sản phẩm nộp

Nhóm nộp đầy đủ các sản phẩm sau:
1. Báo cáo kỹ thuật (PDF): File PDF được compile từ source LATEX.
2. Source LATEX của báo cáo:
3. Source code demo (Python/Julia):
4. Slide thuyết trình:
5. (Tuỳ chọn) Video demo:
  - Video ngắn (3–5 phút) chạy demo thực nghiệm, nếu nhóm không demo live.
  - Định dạng: .mp4, độ phân giải tối thiểu 720p.
  - Nộp kèm link Google Drive hoặc YouTube (unlisted).

## D. Tiêu chí đánh giá

Được đánh giá qua phần hỏi đáp trong buổi thuyết trình.

### Tiêu chí 1: Đọc hiểu tài liệu

| Mức độ | Điểm | Mô tả |
|---|---|---|
| Xuất sắc | 18–20 | Tất cả thành viên trả lời trôi chảy, chính xác, có chiều sâu; liên hệ được với kiến thức bên ngoài tutorial. |
| Tốt | 14–17 | Phần lớn câu hỏi được trả lời đúng; một số câu còn lúng túng nhưng hướng đúng. |
| Đạt | 10–13 | Trả lời được các câu hỏi cơ bản; gặp khó khăn với câu hỏi chuyên sâu. |
| Chưa đạt | 0–9 | Không trả lời được nhiều câu hỏi; thể hiện chưa đọc hiểu kỹ tài liệu. |

### Tiêu chí 2: Báo cáo kỹ thuật

| Tiêu chí | Điểm | Mô tả |
|---|---|---|
| Tổng quan chủ đề | 4 | Giới thiệu rõ ràng, nêu bật được tầm quan trọng và tính thời sự của hướng nghiên cứu. |
| Kiến thức nền | 6 | Trình bày đầy đủ, chính xác các lý thuyết nền tảng; có ví dụ minh hoạ dễ hiểu. |
| Nội dung chính | 10 | Trình bày đúng và sâu các phương pháp; có hình ảnh/sơ đồ; không sao chép nguyên văn. |
| Thực nghiệm & Demo | 5 | Mô tả rõ ràng thực nghiệm, kết quả, và phân tích ý nghĩa của kết quả. |
| Đánh giá & Nhận xét | 3 | Có nhận xét sắc bén về ưu/nhược điểm; đề xuất hướng cải tiến hoặc ứng dụng thực tế. |
| Hình thức & Trình bày | 2 | Đúng template, không lỗi chính tả, trích dẫn đầy đủ, bố cục rõ ràng. |

### Tiêu chí 3: Demo thực nghiệm

| Tiêu chí | Điểm | Mô tả |
|---|---|---|
| Tính đúng đắn | 8 | Code chạy đúng, kết quả hợp lý và nhất quán với nội dung tutorial. |
| Mức độ mở rộng | 7 | Không chỉ chạy lại nguyên bản; có thêm thí nghiệm mới, dataset khác, hoặc phân tích bổ sung. |
| Tái tạo được | 5 | README rõ ràng; người khác có thể cài đặt và chạy lại mà không cần hỗ trợ. |
| Chất lượng code | 3 | Code sạch, có comment, cấu trúc hợp lý, không có code thừa. |
| Trực quan hoá | 2 | Có biểu đồ, hình ảnh kết quả rõ ràng, có nhãn đầy đủ. |

### Tiêu chí 4: Slide và thuyết trình

| Tiêu chí | Điểm | Mô tả |
|---|---|---|
| Nội dung slide | 5 | Slide đầy đủ nội dung, súc tích, không quá nhiều chữ; có hình ảnh minh hoạ phù hợp. |
| Kỹ năng trình bày | 5 | Diễn đạt rõ ràng, tự tin, đúng thời gian; tất cả thành viên tham gia trình bày. |
| Demo trực tiếp | 3 | Demo chạy thành công và được giải thích rõ ràng trong buổi thuyết trình. |
| Thiết kế slide | 2 | Slide chuyên nghiệp, nhất quán về phong cách, dễ đọc. |