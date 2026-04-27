
# 1. Thiết lập thực nghiệm

## 1.1. Quy trình thực hiện
Quy trình được thực hiện theo thứ tự sau: data_utils.py → model.py → train.py → visualize.py

**Bảng 1. Quy trình thực nghiệm**

| Thứ tự | File | Lệnh chạy | Nội dung thực hiện |
| --- | --- | --- | --- |
| 1 | `data_utils.py` | `python data_utils.py --max_len 20 --show_stats` | Đọc dữ liệu gốc từ `data/sentiment_raw.csv`, tiền xử lý văn bản, xây dựng từ điển từ tập train, và lưu dữ liệu tensor vào `data/processed/`. |
| 2 | `model.py` | `python model.py` | Định nghĩa mô hình Transformer tự cài đặt, gồm `scaled_dot_product_attention`, `SelfAttention`, `FeedForwardNetwork`, `TransformerEncoderBlock` và lớp phân loại. |
| 3 | `train.py` | `python train.py --run_all` | Huấn luyện mô hình bằng Adam, chọn checkpoint tốt nhất theo validation accuracy, và ghi tổng hợp kết quả vào `results/summary.json`. |
| 4 | `visualize.py` | `python visualize.py` | Nạp mô hình đã huấn luyện, trích xuất attention weights và xuất heatmap để phục vụ phân tích. |

Kết quả tiền xử lý bao gồm các tệp `data/processed/train.pt`, `val.pt`, `test.pt`, cùng `vocab.json` và `meta.json`, giúp các bước sau sử dụng đúng cùng một cấu hình đầu vào.


## 1.2. Các siêu tham số chính
Các siêu tham số chính của thí nghiệm được chia thành hai nhóm: nhóm giá trị cố định trong toàn bộ quá trình chạy và nhóm giá trị được khảo sát theo nhiều cấu hình.


**Bảng 2. Nhóm siêu tham số cố định**

| Siêu tham số | Giá trị |
| --- | --- |
| `max_len` | 20 |
| `batch_size` | 32 |
| `learning rate` | 1e-3 |
| `num_epochs` | 20 |


**Bảng 3. Các cấu hình mô hình được thử nghiệm**

| Cấu hình | `d_model` | `d_ff` |
| --- | --- | --- |
| Transformer #1 | 64 | 128 |
| Transformer #2 | 128 | 256 |
| Transformer #3 | 32 | 64 |
| Baseline MLP | 64 | - |



Baseline bắt buộc của đồ án là mô hình **MLP do giảng viên cung cấp**; mô hình này được cài sẵn trong `train.py` và dùng làm mốc so sánh với các cấu hình Transformer.

## 1.3. Đảm bảo khả năng tái lập kết quả

Để kết quả có thể tái lập, thí nghiệm tuân thủ các nguyên tắc sau:

- Cố định `random seed = 42`.
- Sử dụng đường dẫn tương đối trong toàn bộ script.
- Cố định môi trường thư viện theo `requirements.txt`.

Nhờ đó, khi chạy lại cùng dữ liệu và cùng cấu hình, các chỉ số huấn luyện/đánh giá có thể được đối chiếu trực tiếp.

# 2. Kết quả thực nghiệm và so sánh
...
# 3. Phân tích Attention
...
# 4. Error Analysis
...
