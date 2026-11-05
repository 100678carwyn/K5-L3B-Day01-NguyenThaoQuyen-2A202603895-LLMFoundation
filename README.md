# Bài thực hành Ngày 1 — Một bài toán, bốn thế hệ AI: Lọc tin nhắn rác (Spam / Ham)

<a href="https://colab.research.google.com/github/VinUni-AI20k/K5-L3-Day-01-AI-Foundation-Lab/blob/main/notebooks/day01_spam_ml_dl_embedding_llm.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

[Mở notebook `day01_spam_ml_dl_embedding_llm.ipynb` trên Colab](https://colab.research.google.com/github/VinUni-AI20k/K5-L3-Day-01-AI-Foundation-Lab/blob/main/notebooks/day01_spam_ml_dl_embedding_llm.ipynb)

> **Thời lượng:** 2h30 · **Hình thức:** cá nhân, trên Google Colab
> **Bài nộp:** repository GitHub có thư mục `report/`; chỉ nộp **link repo** trên LMS.

Bạn giải **cùng một bài toán** là phân loại SMS thành `spam` hay `ham` (tin bình thường) bằng **4 cách**, theo đúng
lịch sử AI trong slide Ngày 1. Cuối bài, bạn so sánh 4 cách trên cùng một tập test về chất lượng, tốc độ và chi phí.

```text
Machine Learning  →  Deep Learning  →  Embedding  →  LLM
người chọn feature   mạng tự học        mượn "bản đồ    không train,
máy học trọng số     đặc trưng          nghĩa" pre-train  chỉ cần prompt
```

| Phần | Thời gian | Cách tiếp cận | Bạn tự làm gì |
| --- | --- | --- | --- |
| 0 | 0:00 – 0:10 | Cài đặt, khám phá dữ liệu | Tìm tin trùng lặp, hiểu vì sao accuracy dễ đánh lừa |
| 1 | 0:10 – 0:40 | **Machine Learning** | Bag-of-words + logistic regression **từ đầu bằng numpy** + precision/recall/F1 |
| 2 | 0:40 – 1:05 | **Deep Learning** | Mạng MLP + vòng lặp huấn luyện PyTorch, nhận diện overfitting |
| 3 | 1:05 – 1:30 | **Embedding** | Cosine, tìm kiếm theo nghĩa, kNN trên `all-MiniLM-L6-v2` |
| — | 1:30 – 1:35 | Nghỉ | |
| 4 | 1:35 – 2:05 | **LLM** | `call_openai` (đúng chữ ký template LLM API), prompt zero/few-shot, đo chi phí |
| 5 | 2:05 – 2:30 | Tổng hợp | Bảng so sánh, thử thách cá nhân, phản ánh, nộp bài |

## Mục tiêu học tập

Sau bài thực hành, bạn có thể:

1. Giải thích máy "học" thế nào: feature → trọng số → loss → gradient descent, và tự viết được vòng lặp đó.
2. Chọn đúng thước đo khi dữ liệu lệch lớp (accuracy vs precision / recall / F1) và nhận ra data leakage.
3. Giải thích vì sao deep learning không phải lúc nào cũng thắng, và đọc được dấu hiệu overfitting.
4. Giải thích embedding là gì, vì sao nó tổng quát hoá tốt hơn bag-of-words với câu "ngụy trang".
5. Dùng LLM cho một downstream task: thiết kế prompt, đọc output không chắc chắn, ước tính chi phí theo token.
6. Chọn phương án cho sản phẩm thật dựa trên số đo (F1, ms/tin, USD/1k tin), không dựa trên cảm giác.

## Dữ liệu

[SMS Spam Collection — Kaggle `uciml/sms-spam-collection-dataset`](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset):
5.572 tin nhắn tiếng Anh (747 spam). Notebook tự tải qua `kagglehub` (không cần tài khoản Kaggle); nếu lỗi sẽ tự
chuyển sang bản gốc trên UCI. Sau khi bỏ 403 tin trùng lặp còn 5.169 tin.

## Luật chơi: học để hiểu, không vibe code

| Cơ chế | Cách hoạt động |
| --- | --- |
| 🚫 Vùng tự code (Phần 1–3) | Không dùng AI để *viết* code; được hỏi AI để *giải thích khái niệm*. Hàm của bạn không được dùng `sklearn`, vì bộ kiểm tra đọc mã nguồn. |
| ✅ Vùng được dùng AI (Phần 4) | Được dùng AI viết code, nhưng phải ghi **AI log** và tự review trước khi gọi API. |
| 🔮 Dự đoán trước khi chạy | Ô thí nghiệm **báo lỗi** nếu chưa gọi `predict("key", "...")`. Dự đoán sai không bị trừ điểm; không dự đoán mới bị trừ. |
| 🎲 Dữ liệu riêng | Cách chia train/test phụ thuộc MSSV, nên số liệu của mỗi người là duy nhất. |
| ✍️ Tính tay | 5 phép tính làm trên giấy, chụp ảnh nộp kèm. |
| 🐞 Bắt lỗi | 3 đoạn code "do AI viết" có lỗi phương pháp; bạn phải chỉ ra lỗi. |
| 🎯 Thử thách cá nhân | Mỗi người nhận một đề sửa code khác nhau, làm trong 15 phút, không dùng AI. |
| 🗣️ Vấn đáp | Lab Coach có thể hỏi bất kỳ dòng code nào bạn nộp. |

## Checklist trước khi bắt đầu

- [ ] Đăng nhập được tài khoản GitHub và tài khoản Google dùng cho Colab.
- [ ] Dùng Chrome hoặc Edge phiên bản mới, Internet ổn định.
- [ ] Có **OpenAI API key** cho Phần 4 (Colab → 🔑 *Secrets* → thêm `OPENAI_API_KEY`). Không có key vẫn làm được Phần 0–3.
- [ ] Biết mã khóa (`KHOA`, ví dụ `K5`), họ tên không dấu và MSSV để đặt tên repository.
- [ ] Giấy và bút cho phần tính tay.

## Hướng dẫn từng bước

### Bước 1: Tạo repository bài làm từ template

Mở repository này, nhấn **Use this template → Create a new repository**. Chọn tài khoản GitHub của bạn làm owner,
đặt tên theo mẫu `<KHOA>-DAY01-HoVaTen-MSSV` (ví dụ `K5-DAY01-NguyenVanAn-12345678`) và chọn **Public** để Lab
Coach mở được link nộp. Không chọn **Include all branches**.

### Bước 2: Mở notebook bằng Google Colab

Nhấn badge **Open in Colab** ở đầu trang, rồi nhấn **Copy to Drive** để có bản bạn sửa được. Chạy được bằng CPU;
nếu có GPU (`Runtime → Change runtime type → T4 GPU`) thì Phần 3 sẽ nhanh hơn.

### Bước 3: Làm bài từ trên xuống dưới

Điền `STUDENT_ID` = MSSV của bạn ở ô đầu tiên. Làm lần lượt các TODO; sau mỗi TODO có một ô kiểm tra (✅ / ❌)
kèm gợi ý khi sai. Nếu runtime bị ngắt, chọn **Runtime → Run all** để chạy lại từ đầu.

### Bước 4: Đưa bài vào repo

Chạy ô **Báo cáo cuối** (tự tải `lab1_report.json`), rồi tải notebook đã chạy (**File → Download → .ipynb**, đổi
tên thành `day01_lab.ipynb`). Trên GitHub, mở thư mục `report/` → **Add file → Upload files** → kéo 3 file vào →
**Commit changes**.

### Bước 5: Nộp link

Tải lại trang GitHub, kiểm tra `report/` đủ 3 file, rồi nộp **link repository** trên LMS.

## Cấu trúc bài nộp

```text
<KHOA>-DAY01-HoVaTen-MSSV/
└── report/
    ├── day01_lab.ipynb      # notebook đã chạy, đã điền câu trả lời
    ├── lab1_report.json     # do ô "Báo cáo cuối" sinh ra (checks, dự đoán, kết quả, SEED)
    └── hand_calc.jpg        # ảnh tờ giấy tính tay
```

Nếu đã quen Git trên máy:

```bash
git add report
git commit -m "submit day 1 lab"
git push
```

## Lỗi thường gặp

| Hiện tượng | Cách xử lý |
| --- | --- |
| `✋ Hãy điền STUDENT_ID` | Điền MSSV vào ô `STUDENT_ID`, chạy lại ô đó và ô bộ công cụ. |
| `✋ Chưa có dự đoán!` | Đây là thiết kế của bài, không phải lỗi: chạy ô `predict("...", "...")` ngay phía trên trước. |
| `⬜ ... chưa làm` / `NotImplementedError` | Bạn chưa hoàn thành TODO đó; các ô phía sau dùng hàm này sẽ lỗi theo. |
| Kaggle tải lỗi | Notebook tự chuyển sang nguồn UCI. Nếu cả hai lỗi, kiểm tra Internet rồi **Run all** lại. |
| Phần 3 encode chậm | Bình thường trên CPU (khoảng 1 phút). Có thể bật GPU T4. |
| Phần 4 báo không có key | Thêm `OPENAI_API_KEY` vào Colab Secrets và bật *Notebook access*, rồi chạy lại ô API key. |
| `RateLimitError` | `retry_with_backoff` sẽ tự thử lại; nếu vẫn lỗi, chờ 1 phút rồi chạy lại ô đó. |
| Ô sau lỗi vì thiếu biến | Chọn `Runtime → Restart session` rồi **Run all** từ đầu. |

## Quy tắc quan trọng

- **Không bao giờ** commit API key. Kiểm tra notebook trước khi upload.
- Không ghi họ tên, email, số điện thoại vào notebook hoặc report. MSSV chỉ dùng làm `STUDENT_ID` để tạo seed;
  `lab1_report.json` chỉ lưu `SEED`.
- Phần 4 tốn tiền thật nhưng rất ít (dưới $0.05 cho cả bài với `gpt-4o-mini` + 40 tin của `gpt-4o`). Không tự tăng
  số tin gửi lên LLM.
- Mô hình dự đoán sai không làm bài kém đi. Phân tích lỗi đúng cách là bằng chứng có giá trị.

## Tệp trong repository

| Tệp | Dùng để làm gì |
| --- | --- |
| `notebooks/day01_spam_ml_dl_embedding_llm.ipynb` | notebook bài lab (làm trên Colab) |
| `RUBRIC.md` | tiêu chí chấm điểm |
| `report/` | thư mục học viên upload bài làm |
| `requirements.txt` | thư viện nếu muốn chạy trên máy cá nhân |
| `tests/test_repository_contract.py` | kiểm tra notebook hợp lệ, sạch output, đủ TODO |
| `THIRD_PARTY_NOTICES.md` | nguồn và license của dữ liệu, model |

## Nguồn

- Almeida, T. A., Hidalgo, J. M. G., & Yamakami, A. (2011). *Contributions to the Study of SMS Spam Filtering.* — [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- [Sentence-Transformers `all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat)

## Bài phụ: LLM API (`template.py`)

Phần 4 của notebook dùng đúng chữ ký hàm của bài LLM API trong `template.py` (`call_openai`, `call_openai_mini`,
`compare_models`, `streaming_chatbot`, `retry_with_backoff`, ...). Nếu GV giao thêm bài này để làm trên máy:

```bash
pip install -r requirements.txt
mkdir -p solution && cp template.py solution/solution.py   # rồi hoàn thiện các TODO
cp exercises.md solution/exercises.md                      # điền phần phản ánh
pytest tests/test_solution.py -v                           # test dùng mock, không cần API key
```
