# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:**2A202603362
- **Họ và tên:**Dương Quang Bảo
- **SEED** (in ở ô đầu notebook):8098

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | v |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | v |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | v |

## 2. Kết quả chính

<img width="889" height="489" alt="image" src="https://github.com/user-attachments/assets/2cbc4b3e-4329-407b-9407-c9079753d3ed" />


## 3. Dự đoán và thực tế

1. dl_relu (bỏ ReLU)

Đoán: F1 giảm còn khoảng 0.8.
Ra: F1 test 0.933, bằng đúng MLP có ReLU.
Vì sao lệch: bỏ ReLU thì 2 lớp Linear chỉ còn là 1 phép tuyến tính, giống logistic regression. Bag-of-words gần tách tuyến tính được nên không cần phi tuyến. Mô hình này vẫn hơn LogReg tự code (0.880), nên phần chênh đến từ cách tối ưu (Adam, mini-batch), không phải từ “deep”.

2. ml_lr (lr = 0.001 và 50)

Đoán: lr = 50 thì không hội tụ.
Ra: loss tăng vọt lúc đầu rồi giảm nhanh về 0.006.
Vì sao lệch: loss logistic là hàm lồi, sigmoid bão hoà nên gradient nhỏ lại, không nổ. Loss 0.006 là trên tập train, có thể chỉ là học thuộc chứ chưa chắc tốt hơn.

3. hard_eval (accuracy 50 case khó)

Đoán: khoảng 0.5.
Ra: BoW+LogReg 0.50 (đúng), nhưng kNN 0.70, 3b 0.64, 3c 0.66.
Vì sao lệch: tôi nghĩ embedding cũng sẽ sập như BoW. Thực tế embedding hiểu ngữ cảnh nên hơn BoW 14–20 điểm, nhưng vẫn chưa qua 0.70.

## 4. Phân tích 50 case khó

BoW + LogReg tụt nhiều nhất: F1 từ 0.919 (SUB) xuống 0.286, accuracy chỉ còn 0.50. Nhóm khó nhất là spam giọng tự nhiên, vì cả 6 case mà mọi model đều sai (#12, 13, 14, 15, 16, 19) đều thuộc nhóm này.

Case #14 (“working from home, making £300 a day…”): tin không có từ khoá spam, các từ như “i”, “me”, “hi”, “home” kéo về ham. Embedding thì thấy câu giống người thật kể chuyện. Tập train năm 2011 cũng không có kiểu lừa việc làm này.
Case #19 (“Hey sweetie, missed you lately…”): spam người lớn trong train dùng từ rõ ràng, tin này nói ẩn ý nên model không có gì để bám.
Cơ chế chung: model học từ vựng và giọng văn của train chứ không hiểu ý đồ của người gửi.
## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

Chọn 3c (MiniLM + MLP head).

p95 = 38.6 ms, đạt SLA 50 ms. F1 test 0.950, cao nhất trong các cách đạt SLA.
Trên 50 case khó F1 = 0.653, hơn BoW+MLP (0.378).
Loại kNN vì p95 126.9 ms. Loại BoW vì nhanh nhưng sập ở case khó.
LLM: tính tay $10.5/ngày cho 1 triệu tin, nên 10 triệu tin là khoảng $105/ngày, tức ~$3.150/tháng. Độ trễ chưa đo được vì không có API key.
Đánh đổi: biên SLA mỏng (38.6 so với 50 ms, đo trên Colab). Mỗi instance chỉ ~45 tin/s, trong khi tải trung bình là ~116 tin/s nên cần vài instance. Vẫn sai khoảng 1/3 case khó.

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
