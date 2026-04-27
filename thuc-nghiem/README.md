
# 1. Thiết lập thực nghiệm

## 1.1. Quy trình thực hiện
Quy trình được thực hiện theo thứ tự sau: data_utils.py → model.py → train.py → visualize.py

| Thứ tự | File | Lệnh chạy | Nội dung thực hiện |
| --- | --- | --- | --- |
| 1 | `data_utils.py` | `python data_utils.py --max_len 20 --show_stats` | Đọc dữ liệu gốc từ `data/sentiment_raw.csv`, tiền xử lý văn bản, xây dựng từ điển từ tập train, và lưu dữ liệu tensor vào `data/processed/`. |
| 2 | `model.py` | `python model.py` | Định nghĩa mô hình Transformer tự cài đặt, gồm `scaled_dot_product_attention`, `SelfAttention`, `FeedForwardNetwork`, `TransformerEncoderBlock` và lớp phân loại. |
| 3 | `train.py` | `python train.py --run_all` | Huấn luyện mô hình bằng Adam, chọn checkpoint tốt nhất theo validation accuracy, và ghi tổng hợp kết quả vào `results/summary.json`. |
| 4 | `visualize.py` | `python visualize.py` | Nạp mô hình đã huấn luyện, trích xuất attention weights và xuất heatmap để phục vụ phân tích. |

Kết quả tiền xử lý bao gồm các tệp `data/processed/train.pt`, `val.pt`, `test.pt`, cùng `vocab.json` và `meta.json`, giúp các bước sau sử dụng đúng cùng một cấu hình đầu vào.


## 1.2. Các siêu tham số chính
Các siêu tham số chính của thí nghiệm được chia thành hai nhóm: nhóm giá trị cố định trong toàn bộ quá trình chạy và nhóm giá trị được khảo sát theo nhiều cấu hình.

**Nhóm giá trị cố định**

| Siêu tham số | Giá trị | Vai trò |
| --- | --- | --- |
| `max_len` | 20 | Độ dài tối đa của câu sau khi cắt/padding |
| `batch_size` | 32 | Số mẫu trong mỗi mini-batch |
| `learning rate` | 1e-3 | Tốc độ cập nhật tham số của optimizer |
| `num_epochs` | 20 | Số vòng lặp huấn luyện toàn bộ tập train |

**Nhóm giá trị được thử nghiệm trong project**

| Cấu hình | `d_model` | `d_ff` | Ghi chú |
| --- | --- | --- | --- |
| Transformer #1 | 64 | 128 | Cấu hình trung bình, dùng làm mốc chính |
| Transformer #2 | 128 | 256 | Mô hình lớn hơn, năng lực biểu diễn mạnh hơn |
| Transformer #3 | 32 | 64 | Mô hình nhỏ hơn, kiểm tra khả năng nén tham số |
| Baseline MLP | 64 | - | MLP do giảng viên cung cấp để đối sánh |



Baseline bắt buộc của đồ án là mô hình **MLP do giảng viên cung cấp**; mô hình này được cài sẵn trong `train.py` và dùng làm mốc so sánh với các cấu hình Transformer.

**[Chèn Bảng 1: bảng siêu tham số và mô tả chi tiết]**

## 1.3. Đảm bảo khả năng tái lập kết quả

Để kết quả có thể tái lập, thí nghiệm tuân thủ các nguyên tắc sau:

- Cố định `random seed = 42`.
- Sử dụng đường dẫn tương đối trong toàn bộ script.
- Cố định môi trường thư viện theo `requirements.txt`.

Nhờ đó, khi chạy lại cùng dữ liệu và cùng cấu hình, các chỉ số huấn luyện/đánh giá có thể được đối chiếu trực tiếp.

## 1.4. Ghi chú nội dung sẽ bổ sung vào báo cáo

**[Chèn Bảng 2: so sánh baseline MLP và các cấu hình Transformer]**

**[Chèn Hình 2: learning curve của mô hình tốt nhất]**

**[Chèn Hình 3-5: các heatmap attention cho ví dụ đúng, sai và có phủ định]**

**[Chèn Bảng 3: các câu dự đoán sai để làm error analysis]**

