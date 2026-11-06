# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603895
- **Họ và tên:** Nguyễn Thảo Quyên
- **SEED** (in ở ô đầu notebook): 8250

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

Ô báo cáo cuối in 15/15 kiểm tra đạt, gồm 5 đáp án tính tay khớp mã kiểm.

## 2. Kết quả chính

| Cách tiếp cận | F1 test | F1 SUB 40 | p95 ms/tin | USD/1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.917 | 0.857 | 0.24 | 0 (chạy local) |
| 2. BoW + MLP | 0.947 | 0.857 | 0.33 | 0 (chạy local) |
| 3a. MiniLM + kNN | 0.913 | 0.974 | 67.91 | 0 (chạy local) |
| 3b. MiniLM + LogReg | 0.939 | 0.947 | 57.17 | 0 (chạy local) |
| 3c. MiniLM + MLP head | 0.964 | 0.947 | 37.34 | 0 (chạy local) |
| 4a. LLM nhỏ zero-shot (deepseek-flash) | — | 0.947 | 1585.78 | chưa đo |
| 4b. LLM lớn zero-shot (deepseek-v4-pro) | — | 0.974 | 8126.57 | chưa đo |
| 4c. LLM nhỏ few-shot (deepseek-flash) | — | 0.974 | 2070.26 | chưa đo |

LLM chỉ chạy trên SUB 40 tin nên chỉ so được bằng cột F1 SUB. Chênh lệch dưới khoảng 0.05 F1 trên SUB 40 coi như hoà.

## 3. Dự đoán và thực tế

**3.1. `ml_f1` — F1 trên tập test**
- **Dự đoán:** accuracy khoảng 0.8, recall khoảng 0.7, F1 khoảng 0.7; precision cao hơn recall.
- **Thực tế:** accuracy 0.981, precision 0.988, recall 0.856, F1 0.917.
- **Lệch:** accuracy lệch khoảng 0.18, F1 lệch khoảng 0.22. Hướng "precision cao hơn recall" là đúng.
- **Vì sao lệch:** Mình cho rằng dữ liệu ít, nhưng sau khi bỏ trùng vẫn còn 5.169 tin (3.619 tin để train). Mình cũng đánh giá thấp mức độ mạnh của các từ khoá: trong nhóm từ đẩy về spam có "call", "txt", "free", "now", "150p", và chỉ vài từ như vậy đã đủ để BoW phân loại khá tốt.

**3.2. `hard_eval` — accuracy trên 50 case khó**
- **Dự đoán:** accuracy chung khoảng 80%; BoW khó nhất ở nhóm spam giọng tự nhiên; LLM khó nhất ở nhóm prompt injection.
- **Thực tế:** accuracy của BoW + LogReg 0.52, BoW + MLP 0.54, kNN 0.70, Emb + LogReg 0.70, Emb + MLP 0.66, LLM nhỏ zero-shot 1.00.
- **Lệch:** Gần 80% là sai với cả hai nhóm model. LLM không sai case nào, nên prompt injection không đánh lừa được LLM zero-shot. Phần "BoW khó nhất ở nhóm nào" chưa xác nhận được vì notebook chưa có số liệu theo nhóm.
- **Vì sao lệch:** BoW và embedding học từ phân phối của tập train, còn 50 case này được viết để né đúng những gì model đã học. LLM không học từ tập train nên không bị lệch phân phối tương tự.

**3.3. `tradeoff` — cách nào đạt SLA p95 < 50 ms**
- **Dự đoán:** LLM nằm bên phải vạch 50 ms (đúng). BoW và MiniLM đều có lợi thế tốc độ lớn (đúng một phần).
- **Thực tế:** BoW (0.24 và 0.33 ms) và 3c (37.34 ms) đạt SLA. 3a (67.91 ms) và 3b (57.17 ms) vượt SLA. LLM mất từ 1.6 đến 8.1 giây.
- **Lệch:** Không phải mọi cách dùng MiniLM đều đạt SLA. 3b và 3c dùng cùng encoder nhưng p95 chênh khoảng 20 ms, nhiều khả năng là nhiễu khi đo trên Colab, nên không nên kết luận 3c nhanh hơn 3b.

## 4. Phân tích 50 case khó

- **Model tụt nhiều nhất:** BoW + LogReg, F1 từ 0.857 (SUB) xuống 0.333, tụt khoảng 0.52. Accuracy 0.52 gần như đoán ngẫu nhiên. Trong nhóm embedding, 3c tụt nhiều nhất (0.947 → 0.667). LLM nhỏ zero-shot không tụt (0.947 → 1.000); bản có prompt phòng thủ đạt 0.982.
- **Nhóm khó nhất:** Chưa điền được vì notebook chỉ in từng case của một model. Chạy `groupby("category")` trên bảng kết quả của ô 88 để có số theo nhóm.
- **Case 1 — #2 (thật = ham):** "FREE pizza in the office kitchen right now, come quick before it's gone". Cơ chế: BoW cộng trọng số của "free" và "now", hai từ nằm trong nhóm từ đẩy về spam mạnh nhất ở Phần 1, nên dễ bị đoán thành spam. Cần xác nhận bằng ô INSPECT.
- **Case 2 — #16 (thật = spam, prompt phòng thủ đoán ham):** "Long time no see! I've started a little side business with skincare, you'd love it…". Câu này không có từ khoá. Prompt phòng thủ định nghĩa ham là tin nhắn cá nhân nên câu giống tin của người quen bị kéo về ham. Đây là đánh đổi của prompt phòng thủ: giảm nhạy với spam giả dạng người quen.

## 5. Thử thách cá nhân

**Đề #1 — Đổi ngưỡng quyết định của mô hình Phần 1**

| Ngưỡng | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.2 | 90 | 12 | 7 | 0.882 | 0.928 | 0.905 |
| 0.5 | 83 | 1 | 14 | 0.988 | 0.856 | 0.917 |
| 0.8 | 63 | 0 | 34 | 1.000 | 0.650 | 0.788 |

**Kết luận:** Ngân hàng sợ chặn nhầm tin OTP, tức là sợ false positive, nên cần ưu tiên precision. Ở ngưỡng 0.8 không có tin ham nào bị chặn nhầm, nhưng bỏ lọt 34 spam (recall 0.65). Ở ngưỡng 0.5 đã có precision 0.988 với chỉ 1 tin ham bị chặn nhầm, nên đây là điểm cân bằng hợp lý. Nếu chi phí chặn nhầm OTP rất cao thì nâng dần lên 0.8 và chấp nhận lọt nhiều spam hơn. Lưu ý: dataset không có tin OTP, nên các con số này chỉ là ước lượng chung.

## 6. Phương án cho sản phẩm thật

- **Tải:** 10 triệu SMS/ngày ≈ 116 tin/giây trung bình (chưa tính đỉnh tải).
- **Loại LLM đồng bộ:** p95 từ 1.6 đến 8.1 giây, gấp 30 đến 160 lần SLA 50 ms. Chi phí ước tính khoảng 105 USD/ngày, tương đương 3.150 USD/tháng (10 triệu × 100 token input với giá $0.10/1M, cộng 10 triệu × 1 token output với giá $0.50/1M). Con số này khớp với phép tính của bạn.
- **Chọn:**
  - Tầng 1: BoW + MLP (p95 0.33 ms, khoảng 25.600 tin/giây, F1 test 0.947).
  - Tầng 2: 3c cho các tin có xác suất trong vùng bất định (giả định p ∈ [0.3, 0.7], chọn lại bằng tập validation).
  - LLM chỉ xử lý bất đồng bộ cho tin khó hoặc đưa người review, không nằm trong đường SLA.
- **Kiểm tra tải:** 3c xử lý khoảng 41 tin/giây trên một luồng Colab. Nếu 10% tin đi sang tầng 2 (giả định) thì trung bình khoảng 12 tin/giây, một worker đủ cho tải trung bình nhưng cần thêm worker dự phòng cho đỉnh.
- **Đánh đổi:**
  - Tầng 1 chỉ đạt F1 0.41 trên 50 case khó, nên cần tầng 2 và vòng phản hồi để thu thập tin lỗi.
  - 3c cần encoder MiniLM (khoảng 22,7 triệu tham số, khoảng 90 MB) và CPU hoặc GPU.
  - Số liệu lấy từ một lần chia dữ liệu trên Colab, cần đo lại trên hạ tầng thật.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| TODO 3 – Các hàm xử lý vector | Hướng dẫn viết và giải thích các hàm tính Cosine Similarity, Top-K Similarity. | Gợi ý cách tính độ tương đồng giữa các vector và tìm K phần tử gần nhất. | Kiểm tra shape của vector, kết quả tính toán và trường hợp vector toàn 0. |
| TODO 4 – LLM Classification | Hướng dẫn hoàn thiện các hàm gọi API LLM, xây dựng prompt phân loại và xử lý nhãn đầu ra. | Gợi ý cách viết hàm gọi API, thiết lập tham số và parse nhãn Spam/Ham. | Kiểm tra lỗi trong code, thử nhiều input và đối chiếu nhãn dự đoán với kết quả mong đợi. |
| Đánh giá mô hình | Giải thích Precision, Recall, F1 và các chỉ số tốc độ trong bảng kết quả. | Giải thích ý nghĩa từng chỉ số và so sánh các mô hình BoW, MiniLM và LLM. | Đối chiếu với bảng kết quả thực nghiệm, giữ lại các số liệu thực tế. |
| Parse label | Vì sao hàm parse_label_v2 đếm sai nhiều trường hợp? | Phát hiện lỗi do tìm chuỗi "spam" mà không xét ngữ cảnh, ví dụ "not spam". | Kiểm tra bằng các trường hợp đầu vào có chứa từ Spam nhưng nhãn thực tế là Ham. |
| Pareto và SLA | Giải thích đường Pareto, cách bị lấn át và lựa chọn mô hình đáp ứng SLA 50 ms. | Phân tích sự đánh đổi giữa F1 và độ trễ, xác định các mô hình nằm trên đường Pareto. | Đối chiếu với biểu đồ thực tế, sử dụng p95 latency và F1 trên SUB 40. |
| Adversarial Testing | Nhóm dữ liệu nào khó nhất đối với BoW và LLM? | BoW khó với Spam giọng tự nhiên; LLM khó với Prompt injection. | Ban đầu ghi nhận đây là nhận định dự đoán. Sau khi chạy 50 case khó, LLM nhỏ zero-shot đúng 50/50 (kể cả 6 case prompt injection), nên nhận định "LLM khó với Prompt injection" đã không được số liệu ủng hộ. |

