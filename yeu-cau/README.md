# ĐỒ ÁN CUỐI KỲ 

**Xây dựng Transformer cho bài toán phân loại cảm xúc văn bản**
Môn học: Seminar chuyên đề

---

## Hình thức nộp bài

- 01 file PDF Báo cáo đồ án + 01 file .zip chứa các code Python

## Hình thức đánh giá

- Chấm quyển Báo cáo đồ án + Vấn đáp (nếu cần)

---

## 1. Tổng quan

Trong đồ án này, sinh viên tự cài đặt một Transformer Encoder đơn giản bằng Python/PyTorch để giải quyết bài toán phân loại cảm xúc văn bản. Sinh viên cần hiểu kiến thức và có khả năng tự xây dựng thành phần cốt lõi của Transformer là Self-Attention.

Mục tiêu của đồ án không phải là đạt chỉ số accuracy cao nhất, mà là thực sự hiểu cơ chế Attention hoạt động như thế nào, và có khả năng phân tích, giải thích kết quả của mô hình.

### Bài toán

- Đầu vào: Câu văn tiếng Anh ngắn (<= 20 từ sau khi cắt)
- Đầu ra: Nhãn cảm xúc — Positive / Negative / Neutral (3 lớp)
- Dữ liệu: Tập giả lập do giảng viên cung cấp (600 mẫu, chia sẵn train/val/test = 420/90/90, cân bằng 200 mẫu mỗi nhãn)

### Phạm vi cài đặt

> Quy tắc chung: Sinh viên được phép dùng thư viện (PyTorch, sklearn, matplotlib...) cho mọi thành phần trong Transformer, ngoại trừ các thành phần Self-Attention và FFN phải tự cài đặt bằng các phép tính ma trận cơ bản.

### Thành phần bắt buộc tự cài / được dùng thư viện

| Thành phần | Bắt buộc tự cài | Được dùng thư viện |
|---|---|---|
| ScaledDotProductAttention | CÓ — cấm nn.MultiheadAttention |  |
| FeedForwardNetwork (FFN) | CÓ — chỉ dùng nn.Linear + activation |  |
| Add & LayerNorm (residual) |  | CÓ — dùng nn.LayerNorm |
| PositionalEncoding |  | CÓ — dùng code mẫu giảng viên cấp |
| Embedding, Optimizer, Loss |  | CÓ — nn.Embedding, Adam, CrossEntropy |
| Tokenization, vẽ đồ thị |  | CÓ — bất kỳ thư viện nào |

---

## 2. Tài nguyên được cung cấp

Giảng viên cung cấp sẵn toàn bộ dữ liệu và phần lớn code hỗ trợ. Sinh viên chỉ cần điền vào các phần đánh dấu # TODO.

Cấu trúc thư mục sau khi giải nén:

```text
MSSV_HoTen_DoAn/
  data/
    sentiment_raw.csv          <- 600 câu có nhãn (có sẵn cột split)
    processed/
      train.pt                 <- tập train (420 mẫu)
      val.pt                   <- tập val   (90 mẫu)
      test.pt                  <- tập test  (90 mẫu)
      vocab.json               <- từ điển (từ: id)
      meta.json                <- max_len, vocab_size, label_names
    ...
  data_utils.py                <- tiền xử lý dữ liệu [cho sẵn]
  model.py                     <- kiến trúc Transformer [SINH VIÊN ĐIỀN]
  train.py                     <- huấn luyện và thực nghiệm [cho sẵn]
  visualize.py                 <- heatmap attention [cho sẵn]
  requirements.txt
  README.md
```

### Hướng dẫn bắt đầu — chạy theo thứ tự

### Bước 1 — Cài thư viện

```bash
pip install -r requirements.txt
```

### Bước 2 — Tiền xử lý dữ liệu (chỉ cần chạy 1 lần)

```bash
python data_utils.py --max_len 20 --show_stats

# Kết quả in ra:
# [TRAIN] 420 mau | negative: 140 neutral: 140 positive: 140
# [VAL]    90 mau | negative:  30 neutral:  30 positive:  30
# [TEST]   90 mau | negative:  30 neutral:  30 positive:  30
# Vocab size: 1110 tu
# Tao ra: data/processed/train.pt, val.pt, test.pt, vocab.json, meta.json
```

### Bước 3 — Kiểm tra model.py (sau khi điền TODO)

```bash
python model.py

# Kết quả mong đợi khi điền đúng:
# TEST: scaled_dot_product_attention ... PASSED
# TEST: SelfAttention ............... PASSED
# TEST: FeedForwardNetwork .......... PASSED
# TEST: TransformerEncoderBlock ..... PASSED
# TAT CA TESTS PASSED -- model.py san sang de huan luyen!
```

### Bước 4 — Huấn luyện

```bash
python train.py
python train.py --run_all
python train.py --d_model 128
```

### Bước 5 — Visualize attention

```bash
python visualize.py
python visualize.py --model results/model_Transformer_d128_ff256.pt
python visualize.py --sentence "this film is absolutely terrible"
```

> Phải chạy train.py (Bước 4) trước khi visualize. train.py lưu model tốt nhất vào results/model_*.pt — đây là file visualize.py sẽ load. Dùng --model để chọn model cụ thể, --sentence để visualize câu tùy chọn. Tất cả kết quả lưu tự động trong results/.

### Mô tả file sentiment_raw.csv

File CSV gồm 600 dòng, mỗi dòng là một câu đã gắn nhãn:

| Cột | Giá trị | Ví dụ |
|---|---|---|
| id | 1 - 600 | 1 |
| split | train / val / test | train |
| text | Câu tiếng Anh | this movie is absolutely wonderful |
| label | 0 / 1 / 2 | 2 |
| label_name | negative/neutral/positive | positive |
| num_tokens | Số từ trong câu | 6 |

> Phân phối nhãn cân bằng: 200 positive / 200 negative / 200 neutral. Mỗi split có tỉ lệ 3 nhãn bằng nhau (stratified split). Cột split đã được điền sẵn — data_utils.py đọc cột này để tách tập, sinh viên không cần tự chia.

---

## 3. Yêu cầu kỹ thuật chi tiết

### 3.1 Kiến trúc mô hình

Mô hình gồm đúng 1 khối Transformer Encoder, theo thứ tự sau:

1. Embedding layer: chuyển token ID thành vector d_model chiều (dùng nn.Embedding)
2. Positional Encoding: cộng thêm thông tin vị trí (dùng code mẫu)
3. **1 x TransformerEncoderBlock**, gồm:
- Self-Attention: tính Q, K, V rồi tính attention score và context vector (TỰ CÀI ĐẶT)
- Add & LayerNorm: cộng residual rồi normalize (dùng nn.LayerNorm)
- Feed-Forward Network: Linear(d_m, d_ff) -> ReLU -> Linear(d_ff, d_m) (TỰ CÀI ĐẶT)
- Add & LayerNorm lần hai (dùng nn.LayerNorm)
4. Classifier Head: mean pooling -> Linear(d_m, num_classes) (dùng code mẫu)

> Lý do chỉ dùng 1 khối: trên tập 600 mẫu, stack nhiều khối dễ gây overfitting và không cải thiện kết quả. Mục tiêu là hiểu 1 khối thật sâu.

### 3.2 Hiểu về Tensor và Batch — đọc kỹ trước khi cài đặt

Trong PyTorch, dữ liệu được xử lý theo batch (một nhóm nhiều câu cùng lúc). Mọi tensor đều có thêm chiều batch ở đầu.

Ví dụ shape của tensor trong Self-Attention:

```python
# batch_size=32, seq_len=10 tu, d_model=64

x.shape = (32, 10, 64)  # dau vao: 32 cau x 10 tu x 64 chieu
Q.shape = (32, 10, 64)  # Query
K.shape = (32, 10, 64)  # Key
V.shape = (32, 10, 64)  # Value

# scores = Q @ K.transpose(-2, -1) / sqrt(d_k)
# scores.shape = (32, 10, 10)   # moi tu "nhin" vao 10 tu con lai
# weights.shape = (32, 10, 10)  # sau softmax -- moi hang tong = 1.0
# output.shape = (32, 10, 64)   # context vector -- cung shape voi x
```

> Hướng dẫn debug: Sau mỗi bước tính toán, in ra tensor.shape để kiểm tra. Nếu shape không như mong đợi, dừng lại và sửa trước khi tiếp tục. Với 420 câu train và batch_size=32, mỗi epoch có (420/32)=14 bước cập nhật (gọi là steps hay iterations).

### 3.3 Yêu cầu về Self-Attention

Đây là phần bắt buộc tự cài đặt và chiếm nhiều điểm nhất. Hàm scaled_dot_product_attention phải:

- Nhận đầu vào Q, K, V mỗi cái shape (batch, seq_len, d_k)
- Tính attention scores: scores = QK^T / sqrt(d_k), shape (batch, seq_len, seq_len)
- Áp dụng softmax trên chiều cuối (dim=-1) -> attention weights
- Tính context vector = weights @ V
- Trả về cả context vector lẫn attention weights (để dùng cho visualization)

Skeleton tham khảo — điền vào phần TODO:

```python
import math, torch


def scaled_dot_product_attention(Q, K, V):
    # Q, K, V: shape (batch_size, seq_len, d_k)
    d_k = Q.size(-1)

    # TODO: tinh scores = Q @ K^T / sqrt(d_k)
    scores = ...   # shape: (batch_size, seq_len, seq_len)

    # TODO: ap dung softmax
    weights = ...  # shape: (batch_size, seq_len, seq_len)

    # TODO: tinh output = weights @ V
    output = ...   # shape: (batch_size, seq_len, d_k)

    return output, weights
```

> Kiểm tra: với Q=K=V = torch.randn(2, 10, 32), output phải có shape (2, 10, 32) và weights shape (2, 10, 10). Kiểm tra weights.sum(dim=-1) xấp xỉ toàn bộ 1.0.

### 3.4 Siêu tham số gợi ý

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| d_model | 64 | Chiều embedding |
| d_ff | 128 | Chiều ẩn FFN (= 2 x d_model) |
| max_len | 20 | Độ dài tối đa sau padding |
| batch_size | 32 | Số câu mỗi bước cập nhật |
| lr | 1e-3 | Learning rate (Adam) |
| num_epochs | 30-50 | Số vòng huấn luyện |

### 3.5 Skeleton code được cung cấp

Giảng viên cung cấp sẵn, sinh viên không cần tự viết:

- data_utils.py: đọc sentiment_raw.csv, build vocab, tokenize, padding, lưu tensors vào data/processed/
- model.py (phần framework): PositionalEncoding, ClassifierHead, TransformerClassifier — sinh viên chỉ điền # TODO
- train.py: train loop, val loop, learning curve, baseline MLP, bảng so sánh
- visualize.py: hàm visualize_attention vẽ heatmap cho câu bất kỳ

> Sinh viên điền vào các hàm đánh dấu # TODO trong model.py. Không thay đổi tên hàm và tham số để đảm bảo code chạy được với bộ test của giảng viên.

---

## 4. Lộ trình thực hiện

| Tuần | Giai đoạn | Nhiệm vụ chính | Nộp / Kiểm tra |
|---|---|---|---|
| 1 | Hiểu luồng & Self-Attention | Chạy skeleton, hiểu tensor shape; cài đặt attention scaled_dot_product_attention đúng shape | Checkpoint 1: attention unit test đúng shape |
| 2 | FFN & EncoderBlock hoàn chỉnh | Cài FFN; ghép Add & LayerNorm; forward pass end-to-end; train/val loop; learning curve | Checkpoint 2: model huấn luyện được |
| 3 | Thực nghiệm & Báo cáo | Thử >= 3 cấu hình; visualize attention; viết báo cáo; error analysis; đóng gói | Nộp toàn bộ |

### Tuần 1 — Hiểu luồng dữ liệu & Self-Attention (ngày 1-7)

- Ngày 1-2: Đọc kỹ đề bài, chạy data_utils.py rồi chạy skeleton train.py từ đầu đến cuối, in shape của tensor ở mỗi bước
- Ngày 3-5: Cài đặt scaled_dot_product_attention — nhiệm vụ cốt lõi tuần này
- Ngày 6: Kiểm tra shape đầu ra, kiểm tra mỗi hàng trong weights tổng bằng 1
- Ngày 7: Chạy baseline MLP có sẵn để có con số tham chiếu ban đầu

Checkpoint tuần 1:
scaled_dot_product_attention(Q,K,V) với Q=K=V=torch.randn(2,10,32) cho output.shape=(2,10,32), weights.shape=(2,10,10), weights.sum(dim=-1) ~ ones.

### Tuần 2 — FFN, ghép EncoderBlock, huấn luyện (ngày 8-14)

- Ngày 8-9: Cài đặt FeedForwardNetwork (2 lớp Linear, ReLU ở giữa)
- Ngày 10-11: Ghép thành TransformerEncoderBlock: attention -> Add&LN -> FFN -> Add&LN
- Ngày 12-13: Chạy train loop, theo dõi learning curve, debug nếu loss không giảm
- Ngày 14: Thử thay đổi d_model hoặc d_ff, ghi nhận để chuẩn bị tuần 3

Checkpoint tuần 2:
Mô hình huấn luyện được, val loss giảm qua các epoch, val accuracy > 70%.

### Tuần 3 — Thực nghiệm, Visualization & Báo cáo (ngày 15-21)

- Ngày 15-16: Chạy >= 3 cấu hình, lập bảng so sánh
- Ngày 17-18: Visualize attention cho >= 3 câu, viết nhận xét
- Ngày 19-20: Viết báo cáo: mô tả kiến trúc, phân tích kết quả, error analysis
- Ngày 21: Kiểm tra lại code, set random seed, dọn dẹp, đóng gói nộp bài

---

## 5. Yêu cầu thực nghiệm

### 5.1 Baseline bắt buộc

Sinh viên phải chạy và báo cáo kết quả của mô hình baseline MLP đã được cung cấp sẵn trong train.py. Đây là điểm tham chiếu để đánh giá xem Transformer có thực sự cải thiện được gì không.

Chạy bằng lệnh:

```bash
python train.py --run_all
```

### 5.2 Thực nghiệm tối thiểu (>= 3 cấu hình Transformer)

- Cấu hình 1 (mặc định): d_model=64, d_ff=128, batch_size=32
- Cấu hình 2: d_model=128, d_ff=256 — mô hình to hơn có giúp ích không?
- Cấu hình 3: d_model=32, d_ff=64 — mô hình nhỏ hơn thì sao?

Với mỗi cấu hình (bao gồm cả baseline MLP), báo cáo:

- train accuracy
- val accuracy
- test accuracy
- train loss cuối cùng

Sinh viên có thể đề xuất cấu hình thứ 4 theo ý mình.

### 5.3 Visualization attention weights

Dùng hàm visualize_attention để tạo heatmap. Heatmap là ma trận (seq_len x seq_len): ô (i,j) cho biết từ thứ i đang chú ý vào từ thứ j bao nhiêu.

- Chọn >= 3 câu từ tập test
- Câu 1: mô hình phân loại đúng — attention tập trung vào từ nào? Có hợp lý không?
- Câu 2: mô hình phân loại sai — attention bị phân tán hay tập trung vào từ không quan trọng?
- Câu 3: câu có từ phủ định (not, never, don't) — mô hình có nhận ra sự phủ định không?
- Với mỗi heatmap, viết 2-3 câu nhận xét

---

## 6. Yêu cầu báo cáo

Báo cáo nộp dạng PDF, dài 6-10 trang (không tính phụ lục). Cần có đủ các mục:

### Mục 1 — Mô tả kiến trúc (1-2 trang)

- Sơ đồ kiến trúc tổng thể — vẽ tay hoặc công cụ, không lấy ảnh từ internet
- Mô tả từng thành phần tự cài đặt: công thức toán học và giải thích bằng lời
- Giải thích tại sao cần Add & LayerNorm sau mỗi sublayer

### Mục 2 — Kết quả thực nghiệm (2-3 trang)

- Bảng so sánh tất cả các cấu hình, bao gồm baseline MLP
- Đồ thị learning curve của cấu hình tốt nhất
- Phân tích: cấu hình nào tốt nhất? Tại sao? Có overfitting không? Xảy ra từ epoch nào?

### Mục 3 — Phân tích Attention (1-2 trang)

- 3 heatmap với nhận xét tương ứng
- Nhận xét tổng quát: mô hình đang chú ý vào loại từ nào? Có vẻ hợp lý không?

### Mục 4 — Error Analysis (1 trang)

- 5-10 câu mô hình phân loại sai, ghi nhãn đúng và nhãn mô hình đưa ra
- Phân nhóm lỗi (câu có từ phủ định, câu mơ hồ, từ lạ không có trong train...)
- Đề xuất 1-2 hướng cải thiện cụ thể

### Mục 5 — Kết luận (0.5 trang)

- Tóm tắt những gì đã làm được
- Bài học quan trọng nhất rút ra từ quá trình tự cài đặt Self-Attention

---

## 7. Tiêu chí chấm điểm

| Hạng mục | Tiêu chí | Điểm |
|---|---|---|
| Self-Attention (Q, K, V) | Tự cài đặt đúng: shape, scaling 1/sqrt(d_k), softmax, gradient flow | 3.5 |
| FFN & ghép EncoderBlock | FFN 2 lớp tự viết; Add & Norm dùng thư viện được; forward pass hoạt động | 2.0 |
| Thực nghiệm & so sánh | Bảng kết quả >= 4 cấu hình (gồm baseline), learning curve, nhận xét | 2.0 |
| Visualization attention | Heatmap >= 3 câu, ma trận đúng chiều, có nhận xét ý nghĩa | 1.0 |
| Báo cáo & phân tích lỗi | Rõ ràng, có insight, error analysis, không cần dài | 1.5 |
|  | Tổng cộng | 10 |

### Rubric chi tiết

| Hạng mục | Xuất sắc (A) | Đạt (B-C) | Yếu (D) | Không đạt |
|---|---|---|---|---|
| Self-Attention | Tự cài đúng, shape (B,L,d_k), scaling 1/sqrt(d_k), softmax, gradient != NaN | Đúng nhưng thiếu scaling hoặc shape sai chiều | Có code nhưng kết quả sai | Không tự cài / copy nguyên |
| FFN & Block | FFN 2 lớp tự viết; Add&Norm đúng vị trí; end-to-end hoạt động | FFN đúng nhưng Add&Norm sai vị trí | Chỉ có Attention, thiếu FFN | Không hoàn chỉnh |
| Thực nghiệm | >= 4 cấu hình (gồm baseline), bảng rõ, learning curve, nhận xét | 2-3 cấu hình, có bảng, ít nhận xét | 1 cấu hình, không bảng | Chỉ chạy 1 lần |
| Visualization | Heatmap >= 3 câu, đúng chiều, nhận xét có ý nghĩa | Có heatmap nhưng thiếu nhận xét | Hình sai chiều ma trận | Không có |
| Báo cáo | Mạch lạc, có error analysis, giải thích kết quả, sạch | Đủ mục nhưng phân tích nông | Thiếu mục quan trọng | Sơ sài |

### Điểm cộng (tùy chọn)

- (+5đ) Cài đặt Multi-Head Attention (num_heads > 1): tách Q/K/V thành nhiều head, tính song song, ghép lại
- (+5đ) Cài đặt padding mask: bỏ qua các vị trí [PAD] khi tính attention
- (+3đ) Thêm Dropout sau attention weights và sau FFN — phân tích tác động lên overfitting

> Điểm cộng chỉ áp dụng khi điểm bắt buộc đạt từ 70/100 trở lên.

---

## 8. Quy định nộp bài

### 8.1 Cấu trúc thư mục nộp bài

Nộp một file .zip tên MSSV_HoTen_DoAn.zip:

```text
MSSV_HoTen_Seminar_Codes/
  data/
    sentiment_raw.csv          <- file dữ liệu gốc (giữ nguyên)
  model.py                     <- kiến trúc Transformer (phần tự cài đặt)
  train.py                     <- vòng lặp huấn luyện và thực nghiệm
  visualize.py                 <- code tạo heatmap attention
  data_utils.py                <- tiền xử lý dữ liệu
  requirements.txt             <- danh sách thư viện và phiên bản
  README.md                    <- hướng dẫn chạy code (<=1 trang)

MSSV_HoTen_Seminar_Report.pdf  <- báo cáo đồ án
```

> Lưu ý: không nộp thư mục data/processed/ (các file .pt nặng và có thể tái tạo). Giảng viên sẽ chạy lại data_utils.py để tạo.

### 8.2 Yêu cầu về code

- Chạy được trên máy giảng viên theo thứ tự: pip install -r requirements.txt -> python data_utils.py -> python train.py
- Không dùng đường dẫn tuyệt đối — dùng đường dẫn tương đối
- Kết quả tái tạo được: set random seed (torch.manual_seed(42))
- Comment giải thích cho mỗi hàm và mỗi bước tính toán quan trọng

### 8.3 Quy định làm bài

- Được phép tham khảo tài liệu, sách giáo khoa, bài báo gốc
- Được thảo luận ý tưởng với bạn, nhưng code và báo cáo phải độc lập
- Ghi rõ nguồn tham khảo trong báo cáo
- Nghiêm cấm: sao chép code của người khác; dùng AI sinh code phần tự cài đặt

> Trong buổi vấn đáp (nếu có), sinh viên cần giải thích được từng dòng code trong hàm scaled_dot_product_attention. Không giải thích được sẽ bị trừ điểm nặng.

---

## 9. Câu hỏi vấn đáp tham khảo

Giảng viên có thể hỏi bất kỳ câu nào dưới đây. Sinh viên nên chuẩn bị trả lời được và liên kết với code cụ thể.

### Về Self-Attention và Tensor

- Trong hàm của bạn, Q, K, V có shape như thế nào? Chiều nào là batch, chiều nào là seq_len?
- Tại sao phải chia cho sqrt(d_k)? Điều gì xảy ra với gradient nếu không chia khi d_k lớn?
- Attention weights phải thỏa mãn điều kiện gì? Làm thế nào để kiểm tra trong code?
- Nếu hai từ trong câu giống nhau hoàn toàn, attention weights giữa chúng trông như thế nào?

### Về kiến trúc

- Residual connection (Add) có tác dụng gì? Điều gì xảy ra khi bỏ nó đi?
- Tại sao Transformer dùng Layer Normalization thay vì Batch Normalization?
- Feed-Forward Network trong block của bạn có bao nhiêu tham số? Tính thế nào?

### Về kết quả và phân tích

- Chỉ ra trong learning curve: mô hình bắt đầu overfit từ epoch nào? Nhận ra bằng dấu hiệu gì?
- Nhìn vào heatmap của câu The movie was not good at all — mô hình chú ý vào từ nào nhiều nhất?
- Tại sao mô hình lớn hơn (d_model=128) không nhất thiết tốt hơn trên tập 600 mẫu?

---

## 10. Tài liệu tham khảo

### Bắt buộc đọc

- Vaswani et al. (2017). Attention Is All You Need. arXiv:1706.03762
- Nội dung bài giảng tại lớp về Self-Attention và Transformer

### Tham khảo thêm

- Illustrated Transformer — Jay Alammar (jalammar.github.io) — giải thích trực quan, rất nên đọc
- The Annotated Transformer — Harvard NLP Group (nlp.seas.harvard.edu) — có code Python đầy đủ
- PyTorch documentation: torch.matmul, torch.nn.functional.softmax

> The Annotated Transformer chỉ dùng để hiểu khái niệm. Không copy code từ đó cho phần tự cài đặt.

---

Chúc sinh viên làm bài tốt. Mọi thắc mắc vui lòng hỏi trong giờ thực hành hoặc qua diễn đàn môn học.
