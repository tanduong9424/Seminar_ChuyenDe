**TEMPLATE BÁO CÁO ĐỒ ÁN CUỐI KỲ**

**Xây dựng Transformer cho bài toán phân loại cảm xúc văn bản**



_Môn học: Seminar chuyên đề_

| **Họ tên sinh viên**  | **.................................................................** |
| --------------------- | --------------------------------------------------------------------- |
| **MSSV**              | .................................................................     |
| ---                   | ---                                                                   |
| **Lớp**               | .................................................................     |
| ---                   | ---                                                                   |
| **Giảng viên**        | .................................................................     |
| ---                   | ---                                                                   |
| **Ngày nộp**          | .................................................................     |
| ---                   | ---                                                                   |
| **Phiên bản báo cáo** | .................................................................     |
| ---                   | ---                                                                   |

_Mẫu này được thiết kế để bám sát đúng yêu cầu nộp báo cáo 6-10 trang (không tính phụ lục), nội dung thực nghiệm, attention visualization, error analysis và rubric chấm điểm của đồ án._

# **1\. Quy cách và nguyên tắc viết báo cáo**

- Báo cáo nộp dạng PDF; độ dài khuyến nghị 15-20 trang, không tính phụ lục.
- Sơ đồ kiến trúc phải do sinh viên tự vẽ; không lấy ảnh kiến trúc từ internet.
- Phần Self-Attention và FFN phải được giải thích bằng công thức toán học và lời diễn giải ngắn gọn, sát với code của chính sinh viên.
- Bảng kết quả phải bao gồm baseline MLP và ít nhất 3 cấu hình Transformer.
- Phải có ít nhất 3 heatmap attention với nhận xét, và một mục phân tích lỗi (error analysis) rõ ràng.
- Báo cáo phải phản ánh trung thực phần tự cài đặt; không sao chép code, không dùng AI sinh code cho phần bắt buộc tự cài.

# **2\. Cấu trúc bắt buộc của báo cáo**

- Mở đầu và phát biểu bài toán
- Mô tả kiến trúc mô hình
- Chi tiết cài đặt Self-Attention và FFN/EncoderBlock
- Thiết lập thực nghiệm
- Kết quả thực nghiệm và so sánh
- Phân tích Attention
- Error Analysis
- Kết luận
- Tài liệu tham khảo
- Phụ lục (nếu có)

# **3\. Template chi tiết để sinh viên điền**

## **3.1. Mở đầu và phát biểu bài toán (khoảng 0.5-1 trang)**

- Nêu bài toán phân loại cảm xúc văn bản: đầu vào là câu tiếng Anh ngắn, đầu ra là 3 lớp Positive / Negative / Neutral.
- Tóm tắt mục tiêu học thuật của đồ án: hiểu cơ chế Self-Attention, không chạy theo accuracy cao nhất.
- Nêu rõ phạm vi cài đặt: thành phần nào tự cài, thành phần nào được phép dùng thư viện.
- Mô tả ngắn về dữ liệu: 600 mẫu, train/val/test = 420/90/90, cân bằng 3 nhãn.

## **3.2. Mô tả kiến trúc mô hình (1-2 trang)**

Sinh viên nên điền theo khung sau:

- Vẽ sơ đồ kiến trúc tổng thể của mô hình do chính mình tự vẽ.
- Mô tả luồng dữ liệu từ token IDs → embedding → positional encoding → encoder block → mean pooling → classifier head.
- Giải thích vai trò của từng thành phần: Embedding, Positional Encoding, Self-Attention, Add & LayerNorm, FFN, classifier head.
- Giải thích vì sao đồ án chỉ dùng 1 Transformer Encoder block trên bộ dữ liệu nhỏ.
- Trả lời ngắn: tại sao cần residual connection và Layer Normalization sau mỗi sublayer?

## **3.3. Chi tiết cài đặt phần tự làm (1-1.5 trang)**

Phần này phải bám sát code model.py của sinh viên, không viết chung chung.

| **Thành phần**               | **Phải trình bày gì**                             | **Công thức / shape**                                | **Minh chứng trong code** |
| ---------------------------- | ------------------------------------------------- | ---------------------------------------------------- | ------------------------- |
| Scaled Dot-Product Attention | Cách tính scores, softmax, output; trả về weights | Q,K,V: (B,L,d_k); scores: (B,L,L); output: (B,L,d_k) | Tên hàm / dòng code       |
| ---                          | ---                                               | ---                                                  | ---                       |
| FeedForward Network          | 2 Linear layers + ReLU                            | Linear(d_model,d_ff) → ReLU → Linear(d_ff,d_model)   | Tên lớp / dòng code       |
| ---                          | ---                                               | ---                                                  | ---                       |

## **3.4. Thiết lập thực nghiệm (0.5-1 trang)**

- Mô tả quy trình chạy: data_utils.py → model.py → train.py → visualize.py.
- Ghi rõ các siêu tham số chính: d_model, d_ff, batch_size, learning rate, num_epochs, max_len.
- Nêu cách đảm bảo tái lập kết quả: random seed, đường dẫn tương đối, requirements.txt.
- Nêu baseline bắt buộc: MLP do giảng viên cung cấp.

## **3.5. Kết quả thực nghiệm và so sánh (2-3 trang)**

Bắt buộc có bảng so sánh tối thiểu 4 cấu hình: baseline MLP + 3 cấu hình Transformer.

| **Mô hình**    | **d_model** | **d_ff** | **Train Acc** | **Val Acc** | **Test Acc** | **Train Loss cuối** |
| -------------- | ----------- | -------- | ------------- | ----------- | ------------ | ------------------- |
| Baseline MLP   | \-          | \-       | ...           | ...         | ...          | ...                 |
| ---            | ---         | ---      | ---           | ---         | ---          | ---                 |
| Transformer #1 | 64          | 128      | ...           | ...         | ...          | ...                 |
| ---            | ---         | ---      | ---           | ---         | ---          | ---                 |
| Transformer #2 | 128         | 256      | ...           | ...         | ...          | ...                 |
| ---            | ---         | ---      | ---           | ---         | ---          | ---                 |
| Transformer #3 | 32          | 64       | ...           | ...         | ...          | ...                 |
| ---            | ---         | ---      | ---           | ---         | ---          | ---                 |

- Chọn cấu hình tốt nhất và giải thích vì sao.
- Đưa learning curve của cấu hình tốt nhất.
- Phân tích dấu hiệu overfitting: xuất hiện từ epoch nào, dựa trên train/val loss hoặc accuracy như thế nào.
- Nhận xét ngắn về tác động của việc tăng/giảm d_model và d_ff.

## **3.6. Phân tích Attention (1-2 trang)**

Phải có ít nhất 3 heatmap attention, ưu tiên chọn đúng theo đề:

| **Câu** | **Loại câu**                     | **Kết quả mô hình** | **Nhận xét 2-3 câu** |
| ------- | -------------------------------- | ------------------- | -------------------- |
| Câu 1   | Mô hình phân loại đúng           | Đúng                | ...                  |
| ---     | ---                              | ---                 | ---                  |
| Câu 2   | Mô hình phân loại sai            | Sai                 | ...                  |
| ---     | ---                              | ---                 | ---                  |
| Câu 3   | Có phủ định: not / never / don't | Đúng hoặc sai       | ...                  |
| ---     | ---                              | ---                 | ---                  |

- Mỗi heatmap phải ghi rõ câu đầu vào, nhãn thật, nhãn dự đoán.
- Nhận xét xem attention đang tập trung vào từ nào, có hợp lý không.
- Ở câu sai, chỉ ra attention bị phân tán hoặc tập trung sai chỗ như thế nào.

## **3.7. Error Analysis (khoảng 1 trang)**

Bắt buộc liệt kê 5-10 câu mô hình phân loại sai.

| **STT** | **Câu văn**                      | **Nhãn đúng** | **Nhãn dự đoán** | **Nhóm lỗi / giải thích** |
| ------- | -------------------------------- | ------------- | ---------------- | ------------------------- |
| 1       | ................................ | ...           | ...              | ...                       |
| ---     | ---                              | ---           | ---              | ---                       |
| 2       | ................................ | ...           | ...              | ...                       |
| ---     | ---                              | ---           | ---              | ---                       |
| 3       | ................................ | ...           | ...              | ...                       |
| ---     | ---                              | ---           | ---              | ---                       |
| 4       | ................................ | ...           | ...              | ...                       |
| ---     | ---                              | ---           | ---              | ---                       |
| 5       | ................................ | ...           | ...              | ...                       |
| ---     | ---                              | ---           | ---              | ---                       |

- Nên phân nhóm lỗi: phủ định, câu mơ hồ, từ lạ/OOV, câu ngắn nhưng đa nghĩa, tín hiệu cảm xúc yếu.
- Đề xuất 1-2 hướng cải thiện cụ thể, không nêu chung chung.

## **3.8. Kết luận (khoảng 0.5 trang)**

- Tóm tắt những gì đã làm được.
- Nêu bài học quan trọng nhất khi tự cài đặt Self-Attention.
- Nêu giới hạn của mô hình trên bộ dữ liệu nhỏ.

# **4\. Phụ lục khuyến nghị**

- Ảnh chụp các lệnh chạy chính hoặc thư mục kết quả.
- Bổ sung learning curves còn lại nếu không đủ chỗ trong phần chính.
- Đoạn code then chốt của hàm scaled_dot_product_attention (nếu cần).
- Thông tin về random seed, package versions, môi trường chạy.

# **5\. Bảng tự kiểm tra trước khi nộp**

| **Mục cần có**                                            | **Đã có?** |
| --------------------------------------------------------- | ---------- |
| Báo cáo dài trong khoảng 15-20 trang (không tính phụ lục) | Có / Chưa  |
| ---                                                       | ---        |
| Có sơ đồ kiến trúc tự vẽ                                  | Có / Chưa  |
| ---                                                       | ---        |
| Giải thích rõ Self-Attention và FFN bằng công thức + lời  | Có / Chưa  |
| ---                                                       | ---        |
| Có bảng so sánh baseline MLP + 3 cấu hình Transformer     | Có / Chưa  |
| ---                                                       | ---        |
| Có learning curve của cấu hình tốt nhất                   | Có / Chưa  |
| ---                                                       | ---        |
| Có ít nhất 3 heatmap attention                            | Có / Chưa  |
| ---                                                       | ---        |
| Có nhận xét cho từng heatmap                              | Có / Chưa  |
| ---                                                       | ---        |
| Có mục error analysis với 5-10 câu sai                    | Có / Chưa  |
| ---                                                       | ---        |
| Có kết luận và bài học rút ra                             | Có / Chưa  |
| ---                                                       | ---        |

# **6\. Gợi ý phân bố dung lượng trang**

| **Mục**                     | **Số trang gợi ý** | **Ghi chú**                       |
| --------------------------- | ------------------ | --------------------------------- |
| Mở đầu + phát biểu bài toán | 0.5-1              | Ngắn gọn, đi thẳng vào mục tiêu   |
| ---                         | ---                | ---                               |
| Mô tả kiến trúc             | 1-2                | Có sơ đồ tự vẽ                    |
| ---                         | ---                | ---                               |
| Kết quả thực nghiệm         | 2-3                | Bảng + learning curve + phân tích |
| ---                         | ---                | ---                               |
| Phân tích Attention         | 1-2                | 3 heatmaps + nhận xét             |
| ---                         | ---                | ---                               |
| Phân tích lỗi               | 1                  | 5-10 câu sai + phân nhóm lỗi      |
| ---                         | ---                | ---                               |

_Template báo cáo đồ án - Seminar chuyên đề_