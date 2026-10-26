# Báo cáo baseline CPU — Name That LLM

Ngày thực hiện: **26/10/2026**, Asia/Saigon. Đã hoàn thành **baseline phân loại có đánh giá local**, không chỉ baseline kỹ thuật. Không fine-tune LLM, dùng GPU, tải model weights/dataset ngoài, gọi API AI, sửa dữ liệu gốc, gửi Kaggle submission hoặc commit/push.

## 1. Kết quả chính và phạm vi đã chạy

| Mô hình, cùng holdout 540 dòng | Balanced accuracy | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| DummyClassifier, most_frequent | 0,055556 | 0,055556 | 0,005848 |
| TF-IDF ký tự + Logistic Regression | **0,290741** | **0,290741** | **0,269549** |

LR dự đoán đúng **157/540** đoạn; Dummy đúng 30/540. Balanced accuracy cao hơn Dummy **23,52 điểm phần trăm**. Đây là **một holdout theo prompt**, không phải trung bình CV hoặc điểm Kaggle. Chỉ có 30 mẫu/lớp trong validation, nên không suy ra độ chính xác hidden test từ kết quả này. Accuracy và balanced accuracy bằng nhau ở đây vì validation cân bằng hoàn toàn; không có nghĩa hai metric luôn giống nhau.

Đã thực hiện:

- Đọc, kiểm tra toàn bộ train và chia holdout trước khi fit TF-IDF.
- Huấn luyện/evaluate hai baseline trên cùng split; lưu metrics, kết quả từng lớp, confusion matrix và ví dụ sai.
- Fit lại cùng cấu hình trên toàn bộ 2.699 dòng để tạo pipeline inference riêng.
- Đọc artifact trong tiến trình khác, đánh giá lại validation: metrics trùng chính xác kết quả gốc.
- Inference test thật 3 dòng, xuất Parquet, đọc lại và kiểm tra; lặp inference trong tiến trình mới: cùng bảng và SHA-256.
- Chạy các cell Notebook local với đường dẫn thay thế: output giống CLI. **Chưa chạy trên Kaggle**, chưa thử hidden rerun.
- Chín unit test đạt; thêm kiểm tra inference từ artifact với input giả 0/1/2 dòng, có null/chỉ khoảng trắng, không bỏ dòng. Fixture chỉ dùng kiểm thử, không thêm vào train thật.

## 2. Hiện trạng, nguồn và schema dữ liệu

Git đầu lượt: `main...origin/main`, working tree sạch, HEAD `83a5702`. Không tìm thấy AGENTS.md ở repo hoặc các thư mục cha đã kiểm tra. Đã đọc README, `.gitignore`, báo cáo EDA, inventory và script EDA có sẵn. Bảo toàn các file đó; chỉ bổ sung mục liên quan trong `.gitignore`. Báo cáo EDA là ảnh chụp lịch sử trước khi repository được tạo và trước baseline này; các câu “chưa huấn luyện/chưa có Git” trong báo cáo cũ mô tả thời điểm cũ.

| File local | Byte | Dòng × cột | Schema, vai trò |
|---|---:|---:|---|
| train.csv | 2.139.418 | 2.699 × 4 | `id`, `prompt_id`, `text`, `model`; chỉ `model` ở train là ground truth |
| test.csv | 2.218 | 3 × 2 | `id`, `text`; không nhãn, chỉ test công khai |
| sample_submission.parquet | 899 | 3 × 2 | `id`, `model`; mẫu định dạng, không lấy làm nhãn |

Nguồn local phù hợp mô tả chính thức ở Data: 2.699 đoạn/150 prompt, có train gán nhãn và 18 model. **Chưa có bằng chứng checksum chính thức từ Kaggle hoặc manifest tải về**, nên không tuyên bố đã xác thực provenance tuyệt đối. Không cần dữ liệu ngoài để chạy baseline và không tải thêm dữ liệu. Hash của cả ba file khớp inventory EDA trước/sau; dữ liệu gốc không đổi. Chi tiết: [inventory EDA](name_that_llm_data_inventory_1.csv), [báo cáo EDA](name_that_llm_data_overview.md), [verification local](../outputs/baseline/verification.json).

Cột CSV đọc dạng chuỗi để giữ nguyên ID. Train không thiếu text, nhãn, ID hoặc prompt_id; không ID trùng. Có 150 prompt: 149 prompt đủ 18 model, `p002` có 17, thiếu `claude-fable-5`. Train toàn tập gồm 149 dòng claude-fable-5 và 150 dòng mỗi lớp còn lại. Không tự sinh dòng thiếu.

### Thông tin chính thức đã xác minh lại

Đã đọc nội dung hiển thị trên trình duyệt Kaggle trong lượt này; công cụ web thuần trả trang không có nội dung, không dùng sự vắng nội dung đó để đoán quy định.

| Nguồn và mục | Yêu cầu dùng trong triển khai |
|---|---|
| [Overview → Description](https://www.kaggle.com/competitions/name-that-llm/overview) | Input là đoạn text do LLM sinh; dự đoán một trong 18 model, không phải AI-vs-human |
| [Data → Files, Columns](https://www.kaggle.com/competitions/name-that-llm/data) | Train 2.699 đoạn/150 prompt; train có `id,prompt_id,text,model`, test có `id,text`; prompt test mới; test tương tác 3 dòng, hidden khoảng 4.000 |
| [Evaluation → Evaluation, Valid Labels, Submission File](https://www.kaggle.com/competitions/name-that-llm/overview#evaluation) | Balanced accuracy là trung bình recall từng lớp; 18 chuỗi model hợp lệ; output `submission.parquet`, cột `id,model`; mỗi test ID đúng một lần, không null/trùng/nhãn lạ |
| [Overview → Code Requirements](https://www.kaggle.com/competitions/name-that-llm/overview#code-requirements) | Nộp qua Notebook, CPU ≤9 giờ hoặc GPU ≤9 giờ; Internet OFF; cho phép dữ liệu ngoài công khai miễn phí, gồm model tiền huấn luyện |
| [Rules → Competition-Specific Rules §2.6](https://www.kaggle.com/competitions/name-that-llm/rules) | Dữ liệu ngoài phải công khai và tiếp cận bình đẳng miễn phí hoặc đáp ứng tiêu chuẩn Reasonableness; baseline này không dùng dữ liệu ngoài/API |

Không có thứ tự 18 cột xác suất trong submission: output là một chuỗi model. `LABELS` trong code giữ thứ tự trình bày chính thức; confusion matrix và báo cáo dùng thứ tự đó. Xác suất nội bộ luôn ánh xạ bằng `pipeline.classes_`; thứ tự alphabet của estimator khác thứ tự trình bày và được lưu riêng trong `label_mapping.json`.

Danh sách chính thức, cũng là 18 lớp thực tế:

```text
gpt-5.5, gpt-5.6-luna, gpt-5.6-sol, gpt-5.6-terra, gpt-6-astra
gemini-3.1-pro-preview, gemini-3.6-flash, gemini-3.7-flash, gemini-3.8-flash
claude-fable-5, claude-opus-4-6, claude-opus-4-8, claude-opus-5, claude-sonnet-5
qwen3.8-max, glm-5.3, glm-5.3-flash, kimi-k3
```

Mâu thuẫn nguồn vẫn tồn tại: Overview → Submission File nói 1 dòng test mẫu, Data nói 3; local đo 3. Code không dựa vào con số đó. Hidden khoảng 4.000 chỉ là công bố; chưa đọc hidden, chưa biết phân bố thật. Loại CPU/core/RAM cụ thể và điều kiện từng dịch vụ API ngoài chưa xác minh; không cần chúng cho lượt local này.

## 3. Validation và kiểm soát rò rỉ

Một `GroupShuffleSplit`, seed **42**, validation_fraction **0,2**:

| Phần | Dòng | Prompt/nhóm | Số mẫu mỗi lớp |
|---|---:|---:|---|
| Train holdout | 2.159 | 120 | 120; riêng claude-fable-5 có 119 |
| Validation | 540 | 30 | 30 cho cả 18 lớp |

Không lớp nào thiếu ở hai phía. Code kiểm tra thiếu lớp và dừng nếu split khác không đủ lớp, không thử nhiều seed để chọn điểm tốt.

Trước split, nối các dòng cùng `prompt_id` và các text trùng sau **NFC + gộp khoảng trắng** thành các thành phần liên thông. Nếu text trùng nối hai prompt, toàn bộ hai prompt được giữ chung. Trên dữ liệu thật có 150 nhóm, 0 duplicate nguyên văn, 0 duplicate chuẩn hóa, 0 nhóm text trùng khác nhãn. Sau split: giao prompt **0**, giao text nguyên văn **0**, giao text chuẩn hóa **0**. Manifest ID/split lưu ở [split.csv](../artifacts/baseline/split.csv); audit ở [split_audit.json](../outputs/baseline/split_audit.json).

TF-IDF nằm trong sklearn Pipeline, chỉ `.fit` trên 2.159 dòng train holdout. Validation chỉ `.predict`, không fit vocabulary/IDF. Không đọc test trong lệnh train, không dùng leaderboard để chọn cấu hình. Full-data refit được lưu riêng và không dùng đánh giá holdout. Chưa kiểm tra near-duplicate theo tương đồng ngữ nghĩa, nên chưa loại trừ text gần giống ở những prompt khác nhau; group prompt cũng chưa đảm bảo tách hoàn toàn chủ đề. Không có metadata source để group theo source.

## 4. Tiền xử lý và cấu hình

Giữ nguyên chữ hoa/thường, dấu câu, Unicode và nội dung; không stemming, bỏ stopword, lowercase hoặc chuẩn hóa text trước học. Chuẩn hóa NFC/khoảng trắng chỉ dùng nhận biết duplicate. Bộ phân tích char của sklearn vẫn có xử lý chuỗi khoảng trắng theo triển khai nội bộ; không tuyên bố giữ mọi chi tiết whitespace trong biểu diễn đặc trưng.

- TF-IDF: `analyzer='char'`, n-gram `(3,5)`, `lowercase=False`, `min_df=2`, `max_features=40000`, `sublinear_tf=True`, L2 mặc định, dtype float32 cho ma trận sparse. Học được 40.000 đặc trưng ở cả hai pipeline.
- Logistic Regression đa lớp: `C=1.0`, solver `lbfgs`, `max_iter=400`, `tol=1e-6`, `class_weight='balanced'`, random_state 42. C kiểm soát regularization, giúp hạn chế trọng số quá lớn. Class weight được tính từ phần train tương ứng.
- Dummy: `strategy='most_frequent'`, cùng split; dự đoán lớp phổ biến được sklearn chọn khi có nhiều lớp đồng hạng, không học text.
- Giới hạn luồng BLAS/OpenMP bằng `threadpool_limits(1)` lúc fit/predict. Không grid search, không GPU. Max_features giới hạn số cột ma trận, nhưng chưa đo peak RAM; không coi đây là giới hạn RAM tuyệt đối khi xây vocabulary.

`tol=1e-3` ở lượt khởi đầu khiến LBFGS dừng ngay tại iteration 0, hệ số toàn 0. Đã chẩn đoán qua `n_iter_` và `coef_`, sửa ngưỡng hội tụ thành `1e-6`, thêm guard và regression test. Không đổi C, n-gram, seed hoặc split theo điểm validation. Kết quả khởi đầu không được dùng làm baseline phân loại hợp lệ; artifact/output đó được giữ riêng ở `artifacts/baseline_initial_tolerance` và `outputs/baseline_initial_tolerance`, bị Git ignore. Cấu hình cuối hội tụ sau **40 iteration** ở holdout và **34 iteration** khi fit toàn tập; không ConvergenceWarning.

## 5. Kết quả theo lớp và lỗi

Mỗi lớp validation có support 30. Precision là tỷ lệ đúng trong các dự đoán mang tên lớp; recall là tỷ lệ nhận diện đúng trong các đoạn thực thuộc lớp đó. F1 kết hợp precision và recall.

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| gpt-5.5 | 0,2083 | 0,1667 | 0,1852 |
| gpt-5.6-luna | 0,2609 | 0,2000 | 0,2264 |
| gpt-5.6-sol | 0,3529 | 0,2000 | 0,2553 |
| gpt-5.6-terra | 0,1250 | 0,1000 | 0,1111 |
| gpt-6-astra | 0,3600 | 0,6000 | 0,4500 |
| gemini-3.1-pro-preview | 0,5500 | 0,7333 | 0,6286 |
| gemini-3.6-flash | 0,3429 | 0,4000 | 0,3692 |
| gemini-3.7-flash | 0,2857 | 0,4000 | 0,3333 |
| gemini-3.8-flash | 0,2813 | 0,3000 | 0,2903 |
| claude-fable-5 | 0,0625 | 0,0333 | 0,0435 |
| claude-opus-4-6 | 0,4286 | 0,4000 | 0,4138 |
| claude-opus-4-8 | 0,4615 | 0,2000 | 0,2791 |
| claude-opus-5 | 0,2923 | 0,6333 | 0,4000 |
| claude-sonnet-5 | 0,2308 | 0,2000 | 0,2143 |
| qwen3.8-max | 0,2778 | 0,3333 | 0,3030 |
| glm-5.3 | 0,0000 | 0,0000 | 0,0000 |
| glm-5.3-flash | 0,2500 | 0,1000 | 0,1429 |
| kimi-k3 | 0,1842 | 0,2333 | 0,2059 |

Không giấu lớp recall 0: GLM-5.3 có mặt trong train và validation nhưng mô hình không nhận đúng mẫu nào trong 30 mẫu giữ lại. Confusion matrix 18×18 dạng CSV: [LR](../outputs/baseline/tfidf_logistic_confusion_matrix.csv), [Dummy](../outputs/baseline/dummy_confusion_matrix.csv). Hàng là nhãn thật (`true_model`), cột là nhãn dự đoán. Ví dụ nhầm phổ biến: gemini-3.8-flash → gemini-3.7-flash **12 lần**, glm-5.3 → claude-opus-5 **9 lần**. Chưa thể kết luận nguyên nhân từ thống kê này.

Một số mẫu sai, chọn **ba lỗi đầu theo thứ tự file**, text rút ngắn; không dùng để sửa nhãn test:

| ID | Text rút gọn | Nhãn thật | Dự đoán |
|---|---|---|---|
| 016f17711735 | Lanterns smoked above the courtyard, where camel breath mingled with saffron… | qwen3.8-max | kimi-k3 |
| 03293c742476 | Constructing a Dyson sphere represents a civilization’s triumphant ascension… | gemini-3.8-flash | gemini-3.7-flash |
| 0408e914e870 | Encircling a star in gleaming lattices of light, a civilization would transcend… | kimi-k3 | glm-5.3 |

Toàn bộ metrics và các ví dụ: [metrics.json](../outputs/baseline/metrics.json), [kết quả từng lớp](../outputs/baseline/tfidf_logistic_per_class.json), [10 ví dụ lỗi](../outputs/baseline/error_examples.json). Không báo accuracy train hoặc accuracy test không có nhãn.

## 6. Inference và kiểm tra output

Mô hình cuối dùng toàn bộ train, được load từ `pipeline.joblib`. Dự đoán test local:

| id | model dự đoán, chưa biết đúng/sai |
|---|---|
| 00353fb1c541 | gemini-3.6-flash |
| 005f5a9a6139 | gpt-5.6-terra |
| 0062a53d5487 | gemini-3.7-flash |

File: [submission.parquet](../outputs/baseline/submission.parquet). Đã kiểm tra **3 dòng**, đúng tên/thứ tự cột `id,model`, giá trị chuỗi, ID duy nhất và đúng thứ tự input, không null, nhãn nằm trong 18 lớp chính thức. Số dòng/ID không hardcode. Xác suất nội bộ hữu hạn, trong [0,1], tổng mỗi dòng gần 1, argmax ánh xạ theo đúng tên lớp estimator. Không xuất xác suất vào submission.

Đọc lại bằng pyarrow/pandas đạt toàn bộ điều kiện. SHA-256 file: `896ca78577094d2f76600aad223e4854eadd6ac75439a6c716d2142b03613696`. Lượt predict độc lập thứ hai và smoke Notebook local có cùng SHA và bảng. Bằng chứng: [submission.checks.json](../outputs/baseline/submission.checks.json), [verification.json](../outputs/baseline/verification.json).

Input lỗi quan trọng đã kiểm tra: cột thiếu/trùng, ID null/rỗng/trùng, nhãn thiếu lớp; lỗi trả rõ và dừng. Train text rỗng bị từ chối; test null đổi thành chuỗi rỗng, cảnh báo nhưng giữ dòng, text chỉ khoảng trắng được giữ. Inference fixture 0/1/2 dòng đều xuất đúng số dòng và đọc lại được; không trộn fixture với dữ liệu thật. Artifact và output đã tồn tại không bị ghi đè.

## 7. Thời gian, môi trường và lệnh đã chạy

Môi trường riêng `.venv`: **Python 3.12.14, Windows 11**, numpy 2.3.5, scipy 1.16.3, pandas 3.0.1, scikit-learn 1.7.2, pyarrow 26.0.0, joblib 1.5.2, threadpoolctl 3.6.0. Dependency phụ đã pin: python-dateutil 3.0.0, six 1.17.0, tzdata 2026.5. Không thay Python hệ thống. Chưa đo peak RAM hoặc benchmark trên phần cứng Kaggle.

| Hoạt động | Giây đo trong chương trình |
|---|---:|
| Fit pipeline holdout | 29,87 |
| Fit pipeline toàn bộ train | 29,15 |
| Lệnh train tổng, gồm evaluate/lưu artifact | 67,59 |
| Inference 3 dòng gồm load/check/ghi/đọc lại | 0,285 |
| Lượt inference độc lập thứ hai | 0,253 |

Thời gian không bao gồm tải/cài dependency, khởi động Python và các công việc tài liệu. Artifact pipeline toàn tập **5.523.531 byte**, pipeline validation **5.484.321 byte**. Một cấu hình cuối, một holdout; không mở rộng mô hình hoặc dò tham số.

Các lệnh thực tế từ gốc repository (PowerShell):

```powershell
& .\.venv\Scripts\python.exe -m src.baseline train --train train.csv --config configs/baseline.json --artifact-dir artifacts/baseline --output-dir outputs/baseline
& .\.venv\Scripts\python.exe -m src.baseline evaluate --train train.csv --artifact-dir artifacts/baseline --output-dir outputs/baseline/reloaded_evaluation
& .\.venv\Scripts\python.exe -m src.baseline predict --test test.csv --artifact-dir artifacts/baseline --output outputs/baseline/submission.parquet
& .\.venv\Scripts\python.exe -m src.baseline predict --test test.csv --artifact-dir artifacts/baseline --output outputs/baseline/submission_reloaded.parquet
& .\.venv\Scripts\python.exe -m src.baseline validate --test test.csv --submission outputs/baseline/submission.parquet
$env:TEMP = Join-Path (Get-Location) 'outputs\test-tmp'
$env:TMP = $env:TEMP
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Các cell code Notebook đã được đọc từ JSON và thực thi local, thay `ASSET_ROOT` bằng gốc repo, `TEST_CSV` bằng test local, `OUTPUT` thành `outputs/baseline/notebook_smoke/submission.parquet`. So sánh bằng `pandas.testing.assert_frame_equal` và SHA thành công. Đây không phải chạy kernel/hạ tầng Kaggle.

Lỗi môi trường gặp và cách xử lý: Python launcher không có Python thông thường; runtime embedded tạo venv nhưng ensurepip thất bại. Venv vẫn chạy được, cài dependency vào venv bằng pip runtime `--python` (chi tiết ở README-baseline). Unit test ban đầu lỗi quyền TEMP sandbox; chuyển TEMP/TMP sang thư mục dưới workspace, chín test chạy lại đạt. Một lần gọi evaluate quá sớm khi full refit chưa ghi metadata đã báo FileNotFoundError; đợi train hoàn tất rồi chạy lại thành công, không cần đổi code/model. Lỗi tolerance và cách sửa được ghi ở mục 4.

## 8. File tạo/sửa và cách dùng lại

| File/thư mục | Trạng thái/vai trò |
|---|---|
| `src/__init__.py`, `src/baseline.py` | Mới; logic và CLI train/evaluate/predict/validate |
| `configs/baseline.json` | Mới; cấu hình cuối cố định |
| `requirements-baseline.txt` | Mới; dependency đã pin |
| `notebooks/kaggle_baseline.ipynb` | Mới; entry point CPU offline từ artifact, chưa chạy Kaggle |
| `tests/test_baseline.py` | Mới; 9 unit test có fixture tổng hợp riêng |
| `README-baseline.md` | Mới; lệnh PowerShell tái lập và cách attach assets/wheels offline |
| `reports/baseline_report.md` | Mới; báo cáo này |
| `.gitignore` | Chỉ thêm `/artifacts/`, `/outputs/`, `*.joblib`, `.ipynb_checkpoints/`; bảo toàn nội dung cũ |
| `artifacts/baseline/` | Local, ignore; hai fitted pipeline, dummy, split, config, mapping, metadata/hash |
| `outputs/baseline/` | Local, ignore; metrics, confusion CSV, dự đoán, Parquet/checks, smoke và fixture kiểm thử |
| `.venv/` | Local, đã ignore từ trước; môi trường riêng, không commit |

Đầu ra vòng khởi đầu được giữ riêng và ignore, không sử dụng inference. Git không được commit/push trong lượt này. Dữ liệu gốc và file EDA có sẵn không bị ghi đè. Không đọc credential hoặc tài liệu PDF/DOCX không liên quan.

Tái lập đầy đủ: [README-baseline.md](../README-baseline.md). Để evaluate/predict lại, dùng artifact sẵn có; để train lại, chọn thư mục output/artifact mới. Joblib chỉ được load từ nguồn tin cậy; hash không làm pickle của bên thứ ba trở nên an toàn.

## 9. Hạn chế và ba cải tiến tiếp theo

Chưa có điểm Kaggle hoặc nhãn hidden test. Test 3 dòng chỉ xác minh luồng kỹ thuật. Một holdout 30 prompt chưa đo biến thiên giữa fold; các lớp như GLM-5.3 và Claude Fable còn yếu. Dấu câu, tên model và chủ đề có thể là shortcut; chưa có thí nghiệm tách tác động. Provenance local chưa được đối chiếu checksum chính thức. Notebook còn cần bạn attach assets riêng tư và môi trường dependency phù hợp Linux/Python Kaggle; hướng dẫn wheelhouse đã có nhưng chưa kiểm thử trên Kaggle.

Ba cải tiến **đề xuất, chưa thực hiện**:

1. Chạy 5-fold theo prompt với cấu hình cố định để đo độ ổn định; kiểm tra near-duplicate giữa prompt trước tuning.
2. Đánh giá lần lượt thêm word TF-IDF hoặc các đặc trưng dấu câu/độ dài và masking tên model, trên cùng protocol; không đổi nhiều yếu tố cùng lúc.
3. Phân tích confusion theo nhóm model và prompt, rồi thử một thay đổi regularization nhỏ có kiểm soát; giữ một holdout cuối chưa dùng chọn cấu hình, không lấy leaderboard làm bộ tuning.

Baseline cơ bản đã chạy local từ đầu đến cuối; dừng ở đây, chưa triển khai các cải tiến hoặc mô hình phức tạp.
