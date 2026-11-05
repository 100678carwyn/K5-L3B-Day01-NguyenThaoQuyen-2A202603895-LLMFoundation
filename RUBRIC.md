# Rubric — Lab Ngày 1: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> **Cách chấm:** mở **link repository** học viên nộp trên LMS, đọc thư mục `report/`
> (`day01_lab.ipynb`, `lab1_report.json`, `hand_calc.jpg`). Chạy lại notebook với MSSV trong tên repository
> sẽ cho đúng `SEED` và cùng cách chia dữ liệu như bài của học viên.

| Hạng mục | Điểm | Đủ bằng chứng | Chưa đạt |
| --- | ---: | --- | --- |
| Bộ kiểm tra tự động (TODO 1.1 → 4.4) | 35 | Các ô kiểm tra ✅ trong notebook đã chạy và trong `lab1_report.json` | ❌ / ⬜, hoặc hàm dùng `sklearn` trong vùng tự code |
| Dự đoán + câu hỏi + giải thích (Phần 0–4) | 25 | Đủ 8 dự đoán có dấu thời gian **trước** thí nghiệm; câu trả lời dẫn số liệu của chính mình và giải thích đúng cơ chế | Thiếu dự đoán, trả lời chung chung không gắn với kết quả chạy |
| Bắt lỗi 1–3 | 10 | Chỉ ra đúng lỗi (data leakage / đánh giá trên train, thiếu chuẩn hoá + sai chiều sắp xếp, `"not spam"` chứa `"spam"`) và hậu quả | Nêu lỗi cú pháp/đặt tên thay vì lỗi phương pháp |
| Tính tay | 10 | 5 giá trị khớp + ảnh viết tay có các bước tính | Chỉ có số, không có ảnh hoặc ảnh không có bước tính |
| Thử thách cá nhân + phản ánh + AI log | 20 | Đúng đề theo `SEED`, có code + kết quả + giải thích; phản ánh dùng bảng so sánh của mình; AI log trung thực | Làm sai đề, không có kết quả, phản ánh không dẫn số liệu |
| **Tổng** | **100** | | |

## Quy tắc vấn đáp

Lab Coach có thể chọn ngẫu nhiên một hàm học viên đã nộp và yêu cầu giải thích, hoặc sửa nhỏ tại chỗ (ví dụ:
"đổi dấu bước cập nhật thì loss thế nào?"). Nếu không giải thích được, điểm phần code tương ứng bị **chia đôi**.

## Evidence tối thiểu

- `report/day01_lab.ipynb` đã chạy (có output), không chứa API key.
- `report/lab1_report.json` có `seed`, `checks`, `predictions`, `results`.
- `report/hand_calc.jpg` là ảnh viết tay.
- Không có họ tên, email, số điện thoại trong notebook hoặc report.
