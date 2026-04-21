# Danh sách công việc thực nghiệm

Tài liệu này dùng để theo dõi tiến độ phần thực nghiệm theo đúng template báo cáo. Mỗi mục có thứ tự làm và tiêu chí hoàn thành để dễ tick tiến độ.

## 1) Thiết lập thực nghiệm

### Thứ tự thực hiện

1. Chạy tiền xử lý dữ liệu.
2. Kiểm tra dữ liệu đầu ra đã sinh đúng.
3. Kiểm tra model skeleton hoạt động.
4. Chạy thử pipeline train.
5. Chốt cấu hình thực nghiệm và môi trường chạy.

### Checklist chi tiết

- [ ] Chạy `python data_utils.py --max_len 20 --show_stats`.
- [ ] Xác nhận thư mục `data/processed/` đã có đủ `train.pt`, `val.pt`, `test.pt`, `vocab.json`, `meta.json`.
- [ ] Mở `meta.json` và ghi lại các giá trị quan trọng: `max_len`, `vocab_size`, `pad_id`, `unk_id`, `num_classes`.
- [ ] Chạy `python model.py` để kiểm tra toàn bộ `TODO` trong mô hình.
- [ ] Đảm bảo các test nội bộ trong `model.py` đều qua.
- [ ] Chạy `python train.py` một lần để kiểm tra train/val loop và việc load dữ liệu.
- [ ] Ghi lại cấu hình sẽ dùng cho báo cáo: `max_len`, `d_model`, `d_ff`, `batch_size`, `lr`, `num_epochs`.
- [ ] Ghi lại phiên bản thư viện từ `requirements.txt` và môi trường Python đang dùng.
- [ ] Cố định random seed trước khi chạy các thí nghiệm chính.

### Tiêu chí hoàn thành

- [ ] Dữ liệu đã xử lý được tạo thành công và không báo lỗi.
- [ ] `model.py` chạy qua các kiểm tra cơ bản.
- [ ] `train.py` chạy được ít nhất một lần từ đầu đến cuối.
- [ ] Các thông số thí nghiệm được ghi lại rõ ràng để có thể tái lập.

## 2) Kết quả thực nghiệm và so sánh

### Thứ tự thực hiện

1. Chạy baseline MLP để lấy mốc tham chiếu.
2. Chạy Transformer mặc định.
3. Chạy thêm ít nhất 2 cấu hình Transformer khác nhau.
4. Thu thập chỉ số train/val/test cho từng mô hình.
5. Chọn mô hình tốt nhất theo validation.
6. Lưu learning curve và nhận xét overfitting.
7. Chốt bảng kết quả cuối cùng.

### Checklist chi tiết

- [ ] Chạy baseline bằng `python train.py --run_all`.
- [ ] Ghi lại kết quả của baseline MLP: train accuracy, val accuracy, test accuracy, train loss cuối.
- [ ] Chạy Transformer mặc định với cấu hình gốc của project.
- [ ] Chạy Transformer lớn hơn, ví dụ `d_model=128`, `d_ff=256`.
- [ ] Chạy Transformer nhỏ hơn, ví dụ `d_model=32`, `d_ff=64`.
- [ ] Nếu cần, chạy thêm 1 cấu hình phụ để so sánh rõ hơn.
- [ ] Lưu đầy đủ metrics của từng mô hình vào một bảng tổng hợp.
- [ ] Chọn mô hình tốt nhất theo validation accuracy hoặc validation loss.
- [ ] Lưu learning curve của mô hình tốt nhất để dùng trong báo cáo.
- [ ] Quan sát dấu hiệu overfitting: train loss giảm mạnh nhưng val loss tăng hoặc val accuracy dừng cải thiện.
- [ ] Ghi lại epoch bắt đầu có dấu hiệu overfitting nếu có.

### Tiêu chí hoàn thành

- [ ] Có ít nhất 4 dòng so sánh: baseline MLP + 3 cấu hình Transformer.
- [ ] Mỗi mô hình đều có đủ train/val/test accuracy và train loss cuối.
- [ ] Có mô hình tốt nhất được xác định rõ.
- [ ] Có learning curve hoặc dữ liệu đủ để vẽ learning curve.
- [ ] Có nhận xét ngắn về tác động của việc tăng/giảm `d_model` và `d_ff`.

## 3) Phân tích Attention

### Thứ tự thực hiện

1. Chọn 3 câu đại diện từ tập test.
2. Chạy visualize cho từng câu.
3. Kiểm tra heatmap và nhãn dự đoán.
4. Viết nhận xét ngắn cho từng ảnh.

### Checklist chi tiết

- [ ] Chọn 1 câu mô hình dự đoán đúng.
- [ ] Chọn 1 câu mô hình dự đoán sai.
- [ ] Chọn 1 câu có từ phủ định như `not`, `never`, hoặc `don't`.
- [ ] Chạy `python visualize.py` với từng câu bằng tham số `--sentence`.
- [ ] Nếu cần, chỉ định rõ model bằng tham số `--model`.
- [ ] Lưu từng heatmap attention vào thư mục `results/`.
- [ ] Kiểm tra heatmap có đúng kích thước `seq_len x seq_len`.
- [ ] Ghi rõ câu đầu vào, nhãn thật, nhãn dự đoán cho từng heatmap.
- [ ] Xác định từ nào được attention tập trung nhiều nhất.
- [ ] Viết 2-3 câu nhận xét cho mỗi heatmap về mức độ hợp lý của attention.

### Tiêu chí hoàn thành

- [ ] Có ít nhất 3 heatmap attention.
- [ ] Mỗi heatmap đều có mô tả câu, nhãn thật, nhãn dự đoán.
- [ ] Mỗi heatmap đều có nhận xét riêng.
- [ ] Ít nhất một ví dụ có từ phủ định được phân tích rõ.

## 4) Error Analysis

### Thứ tự thực hiện

1. Tìm các mẫu dự đoán sai.
2. Chọn các câu sai tiêu biểu.
3. Phân nhóm lỗi theo nguyên nhân.
4. Viết hướng cải thiện.

### Checklist chi tiết

- [ ] Lọc ra các câu dự đoán sai trên validation hoặc test.
- [ ] Chọn tối thiểu 5 câu sai để đưa vào phân tích.
- [ ] Ghi lại nguyên văn từng câu, nhãn đúng và nhãn dự đoán.
- [ ] Phân loại lỗi theo nhóm: phủ định, câu mơ hồ, từ lạ/OOV, tín hiệu cảm xúc yếu, câu quá ngắn.
- [ ] Tìm 1-2 ví dụ điển hình cho mỗi nhóm lỗi nếu có.
- [ ] Ghi chú vì sao mô hình đoán sai ở từng trường hợp.
- [ ] Đề xuất ít nhất 1-2 cách cải thiện cụ thể, ví dụ padding mask, thêm dropout, hoặc mở rộng dữ liệu.
- [ ] Tóm tắt pattern lỗi chung để đưa vào báo cáo.

### Tiêu chí hoàn thành

- [ ] Có ít nhất 5 câu sai được phân tích.
- [ ] Mỗi câu sai đều có nhãn đúng, nhãn dự đoán và giải thích.
- [ ] Có nhóm lỗi rõ ràng, không chỉ liệt kê rời rạc.
- [ ] Có hướng cải thiện cụ thể, không nêu chung chung.

## 5) Hoàn thiện báo cáo

### Thứ tự thực hiện

1. Chèn bảng kết quả thực nghiệm.
2. Chèn learning curve.
3. Chèn heatmap attention.
4. Chèn error analysis.
5. Đọc rà soát và hoàn thiện bố cục.

### Checklist chi tiết

- [ ] Chèn bảng so sánh kết quả vào báo cáo.
- [ ] Chèn learning curve của mô hình tốt nhất.
- [ ] Chèn đủ 3 heatmap attention và phần nhận xét.
- [ ] Chèn mục error analysis với các câu sai tiêu biểu.
- [ ] Kiểm tra báo cáo có bám đúng template và yêu cầu trong README gốc.
- [ ] Đảm bảo phần mô tả thực nghiệm khớp với các file code và kết quả đã chạy.

### Tiêu chí hoàn thành

- [ ] Báo cáo có đủ 4 phần: thiết lập, so sánh kết quả, attention, error analysis.
- [ ] Nội dung trong báo cáo khớp với thí nghiệm đã chạy.
- [ ] Không còn mục checklist quan trọng nào bị bỏ trống.