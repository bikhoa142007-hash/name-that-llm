# Baseline CPU — Name That LLM

Đã chạy local trên dữ liệu thật: **TF-IDF ký tự + Logistic Regression**, so sánh với DummyClassifier, lưu pipeline và xuất/đọc lại Parquet. Kết quả và giới hạn: [báo cáo baseline](reports/baseline_report.md). Chưa chạy trên Kaggle và chưa gửi submission.

## Ý nghĩa ngắn gọn

- **TF-IDF:** đổi các cụm 3–5 ký tự thành đặc trưng số; cụm phổ biến ở mọi đoạn được giảm trọng số. Giữ chữ hoa/thường và dấu câu để giữ dấu hiệu phong cách.
- **Logistic Regression:** mô hình tuyến tính học trọng số cho từng lớp; tên có chữ “regression” nhưng ở đây dùng để phân loại 18 model.
- **Validation:** phần train có nhãn được giữ riêng để đo khả năng gặp prompt mới. Không phải ba dòng test công khai.
- **Balanced accuracy:** trung bình recall của 18 lớp. Recall là tỷ lệ tìm đúng trong từng lớp. Accuracy đếm đúng trên toàn bộ dòng; macro-F1 lấy trung bình F1 của từng lớp.

## Môi trường PowerShell

Chạy lệnh từ thư mục gốc repository. Nếu đã có `.venv` của phiên này, dùng trực tiếp `.\.venv\Scripts\python.exe`; không cần tạo/cài lại.

Với Python 3.12 thông thường, cài mới như sau:

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-baseline.txt
```

Máy hiện tại chưa có Python qua `py`; đã dùng runtime đi kèm Codex. Lệnh thay thế thực tế dùng cho môi trường riêng:

```powershell
$runtimePython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $runtimePython -m venv .venv --without-pip
& $runtimePython -m pip --python .venv/Scripts/python.exe install -r requirements-baseline.txt
```

Không chạy bước tạo lại nếu `.venv` đang dùng hoặc chứa môi trường của bạn. Runtime embedded không bootstrap được ensurepip, nhưng venv `--without-pip` dùng được; pip của runtime cài vào venv qua `--python`. Không thay Python hệ thống. Thư viện chỉ cần Internet lúc cài đặt; train/evaluate/predict không gọi mạng.

## Train và evaluate

Đặt `train.csv` local ở đường dẫn bạn được phép sử dụng. Không tải dataset ngoài. Lệnh dưới đọc train, tạo **một holdout theo group với seed=42**, so sánh Dummy với LR, rồi fit lại cùng cấu hình trên toàn bộ train để inference:

```powershell
& .\.venv\Scripts\python.exe -m src.baseline train --train .\train.csv --config .\configs\baseline.json --artifact-dir .\artifacts\baseline --output-dir .\outputs\baseline
```

**Lượt chạy này đã tạo hai thư mục trên.** Để chạy train lại, chọn thư mục mới, ví dụ `artifacts/baseline_run2` và `outputs/baseline_run2`. CLI không ghi đè kết quả đã tồn tại. Không cần chạy train lại để thử inference.

Đánh giá lại từ model validation đã lưu và manifest split cũ, không refit:

```powershell
& .\.venv\Scripts\python.exe -m src.baseline evaluate --train .\train.csv --artifact-dir .\artifacts\baseline --output-dir .\outputs\evaluation_again
```

Lệnh evaluate kiểm tra hash train/split, dùng `validation_pipeline.joblib`, không dùng mô hình đã fit toàn bộ train. Tất cả 18 lớp phải có trong train và cả hai phía holdout; nếu thiếu, chương trình dừng với thông báo. `prompt_id` nếu có phải đầy đủ; nếu không có, chỉ còn nhóm theo text trùng và đảm bảo yếu hơn.

## Predict và kiểm tra output

```powershell
& .\.venv\Scripts\python.exe -m src.baseline predict --test .\test.csv --artifact-dir .\artifacts\baseline --output .\outputs\baseline\submission_new.parquet
& .\.venv\Scripts\python.exe -m src.baseline validate --test .\test.csv --submission .\outputs\baseline\submission_new.parquet
```

File đã tạo và kiểm chứng trong phiên này: `outputs/baseline/submission.parquet`; lượt đọc artifact độc lập thứ hai: `outputs/baseline/submission_reloaded.parquet`. Hai bảng phải có cùng prediction/ID theo thứ tự input. Số dòng hoàn toàn lấy từ test thực tế, không hardcode 3 hay 4.000. Có thể đổi `--batch-size` (mặc định 256) để giảm bộ nhớ inference.

Đầu ra đúng hai cột **`id`, `model`** theo thứ tự đó; nhãn là chuỗi, không xuất 18 cột xác suất. Code ánh xạ argmax qua `pipeline.classes_`, kiểm tra xác suất hữu hạn, [0,1], tổng gần 1. Sau khi ghi, code đọc lại Parquet để kiểm tra số dòng, ID/thứ tự, kiểu giá trị chuỗi, nhãn hợp lệ và null. File `.checks.json` kế bên lưu bằng chứng kiểm tra.

Quy tắc lỗi dữ liệu: ID thiếu/trống/trùng hoặc cột thiếu → dừng; train text rỗng/null hoặc thiếu nhãn → dừng; test text null → chuỗi rỗng, test chỉ khoảng trắng được giữ nguyên, có cảnh báo, **không bỏ dòng**. Không tự đoán nhãn. CSV đọc kiểu chuỗi để giữ số 0 đầu ID; ô trống coi là null, chữ `NA` giữ nguyên. Test 0 dòng vẫn xuất bảng Parquet rỗng đúng schema. Không tự dùng `.env`/credential.

## File và artifact

| Đường dẫn | Vai trò |
|---|---|
| `src/baseline.py` | CLI đọc dữ liệu, gom nhóm duplicate/prompt, train/evaluate/predict/validate |
| `configs/baseline.json` | Một cấu hình CPU cố định, không tìm kiếm siêu tham số |
| `requirements-baseline.txt` | Phiên bản dependency đã dùng |
| `artifacts/baseline/pipeline.joblib` | Pipeline TF-IDF + LR fit toàn bộ train, dùng inference |
| `artifacts/baseline/validation_pipeline.joblib` | Pipeline chỉ fit phần train của holdout |
| `artifacts/baseline/validation_dummy.joblib` | Dummy của cùng holdout |
| `artifacts/baseline/split.csv` | ID, nhóm và vai trò train/validation |
| `artifacts/baseline/label_mapping.json` | Tên lớp, thứ tự lớp trong model và cột xác suất nội bộ |
| `artifacts/baseline/metadata.json` | Hash, cấu hình, phiên bản, thời gian, metrics và audit split |
| `outputs/baseline/` | Metrics, confusion matrix CSV, dự đoán validation, ví dụ lỗi và Parquet |
| `tests/test_baseline.py` | Fixture tổng hợp nhỏ chỉ để kiểm tra code, không dùng làm dữ liệu thật |

Chỉ load joblib do chính bạn tạo hoặc tin cậy: định dạng này dựa trên pickle và có thể chạy code lúc load. Hash kiểm tra tính toàn vẹn, không chứng minh an toàn của artifact bên thứ ba. Các artifact fitted và output local đã được `.gitignore`; báo cáo/source/config nhỏ có thể được quản lý phiên bản khi bạn yêu cầu commit sau.

## Kiểm tra code

```powershell
# Dùng thư mục tạm trong workspace để tránh lỗi quyền TEMP của sandbox Windows.
$env:TEMP = Join-Path (Get-Location) 'outputs\test-tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Chín test kiểm tra schema, ID, null, thiếu lớp, nối nhóm prompt/duplicate, vocabulary không học từ validation, optimizer chưa học và Parquet nhỏ/rỗng. Không dùng fixture để công bố điểm thi.

## Đưa tài nguyên vào Kaggle

Entry point: [kaggle_baseline.ipynb](notebooks/kaggle_baseline.ipynb). Notebook chỉ inference từ artifact, CPU, Internet OFF; chưa chạy trên hạ tầng Kaggle.

1. Khi bạn có quyền tham gia, tự tạo/mở Notebook của cuộc thi và attach dữ liệu Name That LLM. Đường dẫn chính thức: `/kaggle/input/competitions/name-that-llm/test.csv`; hidden rerun sẽ thay nội dung file đó. Không dùng test local cố định làm Kaggle input.
2. Tự attach một dataset **riêng tư của bạn** tên tùy chọn, ví dụ `name-that-llm-baseline-assets`, chứa cây thư mục `src/__init__.py`, `src/baseline.py` và `artifacts/baseline/` từ repo này. Không cần upload train/test, không upload `.venv`, `.git`, credential hay bản `baseline_initial_tolerance`. Sửa `ASSET_ROOT` trong cell cấu hình theo đường dẫn Input thực tế. Nếu UI đổi cây thư mục sau upload, điều chỉnh cho đúng.
3. Khớp các phiên bản trong `requirements-baseline.txt`/metadata. Nếu Kaggle khác phiên bản, chuẩn bị wheelhouse trên môi trường **Linux và Python tương thích Kaggle**, attach riêng rồi cài offline trước cell inference:

```python
# Chỉ chạy sau khi bạn đã chuẩn bị và attach đúng wheels cùng requirements.
import subprocess, sys
subprocess.check_call([
    sys.executable, '-m', 'pip', 'install', '--no-index',
    '--find-links', '/kaggle/input/your-baseline-wheels',
    '-r', '/kaggle/input/your-baseline-wheels/requirements-baseline.txt'
])
```

Không dùng wheels Windows của `.venv` cho Kaggle Linux. Chưa kiểm tra wheelhouse trên Kaggle; không có cell tự tải mạng. Notebook dừng rõ nếu thiếu asset hoặc khác phiên bản. Sau khi chỉnh/cài package trong kernel đang chạy, restart kernel trước inference.

Chạy Notebook sẽ ghi `/kaggle/working/submission.parquet` và đọc lại để kiểm tra. Không có lệnh/API gửi bài. Các cell đã smoke test local bằng cùng dữ liệu/artifact và thay đường dẫn; điều đó không chứng minh lượt hidden rerun thành công. Bạn cần chạy thử trên Kaggle trước khi tự quyết định nộp bài.

Quy định nguồn: [Data](https://www.kaggle.com/competitions/name-that-llm/data), [Evaluation/nhãn/submission](https://www.kaggle.com/competitions/name-that-llm/overview#evaluation), [Code Requirements](https://www.kaggle.com/competitions/name-that-llm/overview#code-requirements), [Rules §2.6](https://www.kaggle.com/competitions/name-that-llm/rules). Không dùng dữ liệu ngoài trong baseline này.
