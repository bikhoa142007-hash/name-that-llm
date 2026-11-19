# Kaggle CPU baseline — đã chạy trên Kaggle

Ngày: **03/11/2026**, Asia/Saigon. Notebook riêng tư của tài khoản `khoabi142007`:

[Name That LLM - CPU TF-IDF Baseline, Version 1](https://www.kaggle.com/code/khoabi142007/name-that-llm-cpu-tf-idf-baseline?scriptVersionId=365287782).

## Kết quả đã kiểm chứng trên giao diện Kaggle

- Lượt interactive hoàn tất; tiếp theo **Save Version → Save & Run All** hoàn tất với trạng thái **Successful**.
- Version ID: `365287782`; tên phiên bản `CPU baseline offline - verified run`.
- Kaggle hiển thị thời gian phiên lưu: **61,3 giây** (1m 1s).
- Notebook **Private**, accelerator **None (CPU)**, **Internet off**, environment **Pin to original environment**.
- Dữ liệu cuộc thi được attach trực tiếp; không upload dữ liệu gốc, không dataset ngoài, không tải model weights.
- Sau khi lưu thành công, draft session tương tác đã được tắt để ngừng sử dụng CPU.
- **Không gửi submission lên cuộc thi; chưa có điểm leaderboard hoặc kết quả hidden test.**

## Thay đổi để chạy được trên Kaggle

Notebook cũ `notebooks/kaggle_baseline.ipynb` là entry point inference từ artifact local, cần attach source/model và khớp dependency. Notebook đó vẫn giữ nguyên.

Tạo thêm `notebooks/kaggle_cpu_standalone.ipynb`: nhúng nguyên logic từ `src/baseline.py` và cấu hình từ `configs/baseline.json`; không nhúng dữ liệu hoặc credential. Nó train/evaluate bằng CPU ngay trong môi trường Kaggle, refit trên toàn train và inference cùng môi trường, tránh mismatch artifact giữa Windows local và Linux Kaggle. Không cần cài thêm package hoặc bật mạng.

Script build: `scripts/build_kaggle_cpu_notebook.py`. Chạy từ gốc repo:

```powershell
& .\.venv\Scripts\python.exe scripts/build_kaggle_cpu_notebook.py
```

Script không ghi đè notebook đã tồn tại. Nếu cần tái tạo sau sửa code, chọn/lưu tên mới hoặc chủ động quản lý bản cũ trước; không chạy mù lệnh trên một lần nữa. Đã kiểm tra cú pháp các cell và đối chiếu code nhúng khớp source repository.

## Môi trường Kaggle thực tế

Python 3.13.18, Linux x86_64, numpy 2.1.3, scipy 1.16.3, pandas 2.3.3, scikit-learn 1.6.1, pyarrow 23.0.1, joblib 1.6.0, threadpoolctl 3.7.0. Đây là phiên bản runtime Kaggle được quan sát, không thay requirements đã pin cho môi trường local.

## Dữ liệu, validation và output

Đọc train/test từ `/kaggle/input/competitions/name-that-llm`. Cùng seed 42 và cấu hình TF-IDF 3–5 ký tự, tối đa 40.000 features, LR `C=1`, `tol=1e-6`. Holdout: **2.159 train / 540 validation**, group theo prompt và duplicate như baseline local. TF-IDF fit trong Pipeline trên phần train của holdout.

| Model | Balanced accuracy | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Dummy | 0,055556 | 0,055556 | 0,005848 |
| TF-IDF + LR | **0,290741** | **0,290741** | **0,269549** |

Đánh giá lại bằng validation artifact được lưu trong cùng run trả `matches_saved_metrics=true`. Sau đó pipeline toàn tập dự đoán **3 dòng test công khai**, đúng thứ tự ID; lưu, đọc lại và validate cả bản trong run và bản cuối:

- `/kaggle/working/submission.parquet`
- `/kaggle/working/submission.checks.json`
- `/kaggle/working/baseline-cpu-jnzxbsn2/`: config, artifacts, evaluation, reload_check và kết quả của Version 1.

Output notebook có **27 file**. Parquet cuối có đúng hai cột chuỗi `id,model`, 3 dòng, không null, ID duy nhất và nhãn hợp lệ. Xác suất nội bộ hữu hạn và tổng theo dòng gần 1. SHA-256 Parquet trên Kaggle:

`74496ed630d06e03042d75b16e07f954b4e9deca82d98c6b9218a800c8e8869a`

| ID | Dự đoán, không biết đúng/sai |
|---|---|
| 00353fb1c541 | gemini-3.6-flash |
| 005f5a9a6139 | gpt-5.6-terra |
| 0062a53d5487 | gemini-3.7-flash |

Hash test Kaggle `4545aa3a3459f4c0724f97ad5bd06c4caa3506bdfea5fab905e8203184fc0c57` khớp test local cũ. Hash Parquet khác file local do môi trường/serialization khác; các giá trị dự đoán quan sát được giống local. Không khẳng định đã đo checksum train Kaggle trong lần kiểm tra UI này.

## Trạng thái và cách mở lại

Mở liên kết Version 1 → **Notebook** để xem code/kết quả, **Output** để xem các file, **Logs** để xem trạng thái và thời gian. Lượt Save & Run All đã chạy từ phiên sạch; chưa thử hidden competition rerun. Không tự bấm Submit to Competition.

File local mới chỉ gồm script build, notebook standalone và báo cáo này; README-baseline được bổ sung liên kết. Không commit/push trong công việc chạy Kaggle này. Các báo cáo cũ ghi “chưa chạy Kaggle” phản ánh trạng thái lịch sử; bằng chứng mới nhất nằm trong báo cáo này.
