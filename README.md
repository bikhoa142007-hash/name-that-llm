# Name That LLM — Data Overview

Phân tích khám phá dữ liệu bằng CPU cho cuộc thi [Name That LLM](https://www.kaggle.com/competitions/name-that-llm).

- [Báo cáo tiếng Việt](reports/name_that_llm_data_overview.md)
- [Danh mục dữ liệu — lần chạy hoàn chỉnh](reports/name_that_llm_data_inventory_1.csv)
- [Thống kê chi tiết — lần chạy hoàn chỉnh](reports/name_that_llm_data_statistics_1.json)
- [Script kiểm tra dữ liệu](scripts/inspect_name_that_llm_data.py)

## Tái lập

Yêu cầu Python với `pandas`, `pyarrow`, `matplotlib`. Phiên bản đã kiểm chứng được ghi trong báo cáo.

Đặt dữ liệu cuộc thi đã được cấp quyền truy cập trong một thư mục local, sau đó chạy:

```powershell
python scripts/inspect_name_that_llm_data.py --data-dir <duong-dan-du-lieu> --output-dir reports
```

Script đọc toàn bộ các file `train.csv`, `test.csv`, `sample_submission.parquet` tìm được trong thư mục dữ liệu; đầu ra đã tồn tại được giữ lại và lần chạy mới có hậu tố số.

Dữ liệu gốc, credential, model weights và thư mục IDE không được đưa vào repository. Báo cáo/thống kê có trích mẫu văn bản ngắn để kiểm tra schema; repository được tạo riêng tư. Công việc hiện tại chưa huấn luyện mô hình hoặc tạo/gửi submission.
