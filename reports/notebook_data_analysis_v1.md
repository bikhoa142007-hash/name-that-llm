# Phân tích cách notebook xử lý dữ liệu và nhận diện mô hình LLM

Tài liệu phân tích `notebooks/kaggle_cpu_standalone.ipynb`, notebook đã chạy thành công trên Kaggle ngày 03/11/2026. Nội dung được đối chiếu với các cell thực tế, `src/baseline.py`, `configs/baseline.json`, báo cáo chạy Kaggle và audit validation local trong repository.

Notebook dùng văn bản đầu ra của một LLM để dự đoán tên mô hình đã tạo ra văn bản đó, trong 18 nhãn. Phương pháp là **TF-IDF ở mức ký tự kết hợp Logistic Regression**, chạy trên CPU. Nó học mối liên hệ thống kê giữa các cụm ký tự và nhãn mô hình từ dữ liệu có sẵn.

## 1. Notebook gồm những gì?

| Cell | Nội dung | Vai trò |
|---|---|---|
| 1 — Markdown | Giới thiệu baseline và yêu cầu môi trường | Nêu phạm vi CPU, Internet OFF và dữ liệu cuộc thi |
| 2 — Code | Các hàm được nhúng từ `src/baseline.py` | Định nghĩa cách đọc, kiểm tra, chia dữ liệu, huấn luyện, đánh giá và dự đoán |
| 3 — Code | Cấu hình và chuỗi lời gọi `main(...)` | Thực thi toàn bộ quy trình và tạo file kết quả |

Cell 2 định nghĩa các hàm; cell 3 mới gọi chúng để chạy. Mã nhúng đã được đối chiếu và khớp với source ở thời điểm viết tài liệu. Notebook độc lập nên không cần tải repository hoặc model đã huấn luyện từ bên ngoài vào phiên chạy.

Luồng dữ liệu có hai nhánh:

```text
train.csv → kiểm tra → chia theo nhóm prompt/văn bản trùng
                      ├─ train holdout → học TF-IDF + Logistic Regression
                      └─ validation → dự đoán → metrics và phân tích lỗi

train.csv toàn bộ → học lại pipeline cùng cấu hình → lưu model cuối
test.csv → kiểm tra → model cuối dự đoán → submission.parquet → đọc lại, kiểm tra
```

## 2. Đọc dữ liệu và xác định vai trò của từng cột

Notebook tìm dữ liệu trong `/kaggle/input/competitions/name-that-llm`. Nếu không có `train.csv` ở đó, nó thử `/kaggle/input/name-that-llm`. Cả train và test phải tồn tại trước khi chạy.

| Cột | Có trong | Được sử dụng thế nào? |
|---|---|---|
| `id` | Train, test | Giữ danh tính và thứ tự của mẫu; không đưa vào đặc trưng |
| `prompt_id` | Train hiện tại | Gom các câu trả lời cùng prompt vào một phía của phép chia |
| `text` | Train, test | Đầu vào duy nhất của bộ phân loại |
| `model` | Train | Nhãn đúng dùng để học và đánh giá |

Dữ liệu đã khảo sát có 2.699 dòng train, 150 prompt và 18 lớp. Có 17 lớp với 150 dòng mỗi lớp; `claude-fable-5` có 149 dòng. Test công khai trong lần chạy có 3 dòng và không có nhãn đúng. Notebook đọc số dòng thực tế, không cố định test phải có 3 dòng.

`sample_submission.parquet` không được đọc trong quy trình này. Các nhãn ví dụ trong file mẫu không phải đáp án của test.

### Kiểm tra bằng `read_input()`

CSV được đọc dạng chuỗi để tránh làm biến đổi ID, ví dụ mất số 0 ở đầu. Hàm kiểm tra tên cột không trùng và có đủ các cột bắt buộc. ID phải có giá trị, không chỉ gồm khoảng trắng và không trùng nhau.

Đối với train, chương trình dừng nếu dữ liệu rỗng, text rỗng, nhãn thiếu hoặc có nhãn ngoài danh sách 18 lớp. Nó cũng yêu cầu toàn bộ 18 lớp đều xuất hiện. Nếu có `prompt_id`, cột này phải đầy đủ. Code có thể chạy khi thiếu `prompt_id`, nhưng lúc đó chỉ gom nhóm theo văn bản trùng, nên khả năng ngăn rò rỉ yếu hơn.

Đối với test, text thiếu được điền bằng chuỗi rỗng và phát cảnh báo; dòng vẫn được giữ để không làm mất ID cần dự đoán. Điều này bảo đảm đủ đầu ra nhưng không bảo đảm dự đoán đáng tin cậy cho text rỗng.

Audit lưu số dòng, danh sách cột, số text thiếu/rỗng, kích thước file, SHA-256 và phân bố lớp của train. Hash giúp phát hiện file thay đổi; riêng hash không chứng minh dữ liệu có nguồn gốc chính thức.

## 3. Tiền xử lý và nhận diện văn bản trùng

Hàm `normalized()` chuẩn hóa Unicode về NFC, tách rồi nối lại bằng một dấu cách. Ví dụ, hai chuỗi khác nhau chỉ vì nhiều khoảng trắng có thể trở thành cùng một khóa.

**Khóa chuẩn hóa chỉ phục vụ gom nhóm văn bản trùng.** Notebook không thay toàn bộ text đầu vào bằng khóa này trước khi học. Nó không chủ động xóa dấu câu, bỏ stopword, stemming hoặc chuyển chữ thường. Cấu hình `lowercase=False` giữ khả năng phân biệt chữ hoa/thường. Tuy nhiên, bộ phân tích ký tự của TF-IDF vẫn có xử lý khoảng trắng nội bộ; không nên hiểu rằng biểu diễn giữ nguyên mọi chi tiết whitespace.

Lựa chọn này cho phép mô hình khai thác dấu câu, cách viết từ và các mẫu định dạng ngắn. Đây là những tín hiệu có thể liên quan đến phong cách sinh văn bản; notebook chưa chứng minh tín hiệu nào thực sự quyết định một dự đoán cụ thể.

## 4. Chia train/validation để hạn chế rò rỉ

Một prompt có thể được trả lời bởi nhiều mô hình. Nếu chia ngẫu nhiên từng dòng, các câu trả lời cùng prompt có thể xuất hiện ở cả train và validation. Khi đó, điểm số có thể hưởng lợi từ nội dung/chủ đề đã xuất hiện trong quá trình học.

Hàm `make_split()` xử lý bằng cách nối các dòng có cùng `prompt_id` hoặc cùng text chuẩn hóa thành các nhóm liên thông. Ví dụ, nếu một text trùng nối prompt A với prompt B, toàn bộ các dòng của A và B phải cùng phía. Cách này tránh trường hợp chỉ giữ cặp text trùng chung nhóm nhưng lại tách những dòng còn lại của hai prompt.

Sau đó, `GroupShuffleSplit` chia một lần với `seed=42`, `validation_fraction=0.2`. Tỷ lệ này áp dụng cho nhóm, không bảo đảm chính xác 20% số dòng khi kích thước nhóm khác nhau. Phép chia cũng không tự bảo đảm cân bằng lớp; code kiểm tra lại và dừng nếu một phía thiếu lớp.

Kết quả trên bộ dữ liệu hiện tại:

| Phần dữ liệu | Số dòng | Số nhóm/prompt | Phân bố lớp |
|---|---:|---:|---|
| Train holdout | 2.159 | 120 | 120 mỗi lớp, riêng `claude-fable-5` là 119 |
| Validation | 540 | 30 | 30 mẫu cho mỗi lớp |

Audit local ghi nhận 0 dòng trùng thêm, 0 nhóm text chuẩn hóa có nhãn xung đột; giao prompt, text nguyên văn và text chuẩn hóa giữa hai phía đều bằng 0. Mã kiểm tra các điều kiện giao nhau này trong mỗi lượt chạy.

Một lớp bảo vệ khác nằm trong `Pipeline`: vocabulary và IDF chỉ được học từ 2.159 dòng train holdout. Validation được biến đổi bằng bộ đặc trưng đã học, không tham gia xây dựng vocabulary hay thống kê IDF. Lệnh `train` không đọc test.

## 5. Biến văn bản thành đặc trưng TF-IDF

Máy học cần đầu vào số. `TfidfVectorizer` chuyển mỗi văn bản thành một vector thưa, trong đó mỗi cột tương ứng với một cụm ký tự đã học từ tập train.

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `analyzer` | `char` | Tách đặc trưng theo ký tự, không theo từ |
| `ngram_range` | `(3, 5)` | Lấy các chuỗi liên tiếp dài 3, 4 và 5 ký tự |
| `lowercase` | `False` | Không chuyển toàn bộ text thành chữ thường |
| `min_df` | `2` | Bỏ cụm ký tự chỉ xuất hiện trong một văn bản train |
| `max_features` | `40000` | Giữ tối đa 40.000 đặc trưng theo tần suất trong corpus |
| `sublinear_tf` | `True` | Dùng `1 + log(tần suất)` cho cụm có xuất hiện |
| `dtype` | `float32` | Lưu giá trị ma trận đặc trưng bằng số thực 32 bit |

Ví dụ, chuỗi `hello` tạo các cụm 3 ký tự `hel`, `ell`, `llo`; các cụm 4 ký tự `hell`, `ello`; và cụm 5 ký tự `hello`. Đây chỉ là ví dụ minh họa cơ chế, không phải kết quả giải thích mô hình đã học.

TF biểu thị mức xuất hiện của một cụm trong văn bản. IDF giảm trọng số tương đối của cụm xuất hiện ở quá nhiều văn bản. Với các giá trị mặc định đang dùng, IDF có dạng `log((1 + N) / (1 + df)) + 1`, trong đó `N` là số văn bản dùng để fit và `df` là số văn bản có cụm đó. Sau khi nhân TF và IDF, vector được chuẩn hóa L2 theo mặc định.

Vì vậy, mô hình có thể học từ các mẫu ký tự ngắn mà không cần tokenizer hoặc LLM tiền huấn luyện. Các cụm chưa có trong vocabulary sẽ không tạo cột mới khi dự đoán. Giới hạn 40.000 đặc trưng giúp giảm kích thước biểu diễn, nhưng không phải cam kết về mức RAM tối đa trong toàn bộ quá trình fit.

## 6. Huấn luyện và kiểm tra mô hình có thực sự học

Notebook huấn luyện hai mô hình trên cùng phép chia:

- `DummyClassifier(strategy='most_frequent')` luôn đoán một lớp phổ biến trong train, không học nội dung text. Đây là mốc so sánh tối thiểu.
- `LogisticRegression` học trọng số cho các đặc trưng TF-IDF để phân biệt 18 lớp. Khi dự đoán, mô hình tính xác suất cho các lớp rồi chọn lớp có xác suất lớn nhất.

Logistic Regression dùng `C=1.0`, solver `lbfgs`, `max_iter=400`, `tol=1e-6` và `class_weight='balanced'`. `C` điều khiển mức regularization; C nhỏ hơn tương ứng regularization mạnh hơn. Trọng số lớp được tính từ phần dữ liệu đang fit, nhằm bù chênh lệch số lượng mẫu giữa các lớp.

`fit_checked()` giới hạn luồng tính toán bằng `threadpool_limits(1)`, đo thời gian và kiểm tra cảnh báo hội tụ. Nếu có `ConvergenceWarning`, hoặc mô hình không thực hiện bước tối ưu/hệ số vẫn toàn 0, chương trình dừng. Các kiểm tra này giúp phát hiện việc chạy xong về mặt kỹ thuật nhưng mô hình chưa học được trọng số; chúng không thay thế đánh giá chất lượng trên validation.

## 7. Đánh giá kết quả và phân tích lỗi

`evaluate_models()` chỉ đánh giá phần validation bằng model học từ train holdout. Các chỉ số được tính cho cả Dummy và TF-IDF + Logistic Regression.

| Chỉ số | Cách đọc |
|---|---|
| Accuracy | Tỷ lệ tổng số mẫu dự đoán đúng |
| Balanced accuracy | Trung bình recall của các lớp, mỗi lớp có trọng lượng bằng nhau |
| Macro-F1 | Trung bình F1 của các lớp; F1 kết hợp precision và recall |

Với một lớp, recall là số mẫu đúng của lớp đó chia cho số mẫu thật thuộc lớp; precision là số mẫu đúng chia cho số mẫu được dự đoán mang nhãn lớp đó. Confusion matrix có hàng là nhãn thật, cột là nhãn dự đoán.

Kết quả được ghi nhận trong lần chạy Kaggle Version 1:

| Mô hình | Balanced accuracy | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Dummy | 5,56% | 5,56% | 0,005848 |
| TF-IDF + Logistic Regression | **29,07%** | **29,07%** | **0,269549** |

TF-IDF + Logistic Regression đúng 157/540 mẫu, hơn Dummy khoảng 23,52 điểm phần trăm balanced accuracy. Accuracy bằng balanced accuracy ở phép chia này vì validation có đúng 30 mẫu mỗi lớp; hai chỉ số không luôn bằng nhau.

Ngoài điểm tổng, notebook ghi các file để kiểm tra sâu hơn:

| File trong thư mục `evaluation/` | Nội dung |
|---|---|
| `metrics.json` | Các chỉ số tổng của hai mô hình |
| `tfidf_logistic_per_class.json` | Precision, recall, F1 và support theo lớp |
| `tfidf_logistic_confusion_matrix.csv` | Số lần dự đoán mỗi nhãn thật thành từng nhãn khác |
| `validation_predictions.csv` | ID, nhãn thật, dự đoán LR và Dummy cho từng mẫu |
| `error_examples.json` | 10 mẫu sai đầu tiên, text cắt còn tối đa 180 ký tự |
| `split_audit.json` | Quy mô phép chia, phân bố lớp và kiểm tra giao nhóm/text |

Các ví dụ lỗi là những lỗi đầu theo thứ tự dữ liệu, không phải mẫu ngẫu nhiên hay những lỗi có xác suất cao nhất. Do đó, chúng hữu ích để đọc nhanh nhưng không đại diện đầy đủ cho toàn bộ lỗi. Notebook xuất dữ liệu phân tích, chưa tự viết lời giải thích nguyên nhân nhầm lẫn hoặc vẽ heatmap.

## 8. Lưu model, kiểm tra tái lập và học lại toàn bộ train

Notebook lưu riêng model đánh giá (`validation_pipeline.joblib`) và Dummy tương ứng. Nó cũng lưu `split.csv`, cấu hình, môi trường thư viện, số đặc trưng, số vòng tối ưu, thời gian và hash artifact.

Sau khi đánh giá, một pipeline mới với cùng cấu hình được fit trên toàn bộ 2.699 dòng và lưu thành `pipeline.joblib`. Mục đích là dùng đầy đủ dữ liệu có nhãn cho dự đoán test. **Model toàn tập không được dùng để tính điểm holdout**, vì nó đã học cả những dòng trước đó thuộc validation.

Lệnh `evaluate` tải lại model validation và phép chia cũ, kiểm tra hash dữ liệu/artifact, thứ tự ID và phiên bản scikit-learn. Nó yêu cầu các metrics sau khi tải lại phải khớp metrics đã lưu. Lần chạy Kaggle trả `matches_saved_metrics=true`. Đây là kiểm tra khả năng khôi phục kết quả từ artifact trong môi trường đó, không phải một lần cross-validation độc lập.

## 9. Dự đoán test và kiểm tra file đầu ra

`predict()` tải model toàn tập và xử lý test theo batch, mặc định 256 dòng. Xác suất phải hữu hạn, nằm trong khoảng 0–1 và tổng mỗi dòng gần bằng 1. Nhãn được lấy bằng `pipeline.classes_[argmax]`, tránh nhầm thứ tự lớp với danh sách nhãn trình bày trong báo cáo.

Xác suất chỉ được dùng nội bộ. File kết quả có đúng hai cột chuỗi `id`, `model`, mỗi ID tương ứng một nhãn dự đoán. `validate_submission()` kiểm tra số dòng, thứ tự ID, ID duy nhất, không null và nhãn thuộc danh sách hợp lệ, cả trước khi ghi lẫn sau khi đọc lại Parquet.

Mỗi lần chạy tạo một thư mục mới `/kaggle/working/baseline-cpu-.../`. File đã kiểm tra được sao chép sang `/kaggle/working/submission.parquet` và kiểm tra thêm lần nữa. Thư mục riêng giữ kết quả từng lượt; file đích ở cấp `/kaggle/working` có thể được thay bằng kết quả mới khi chạy lại cell.

Kaggle Version 1 hoàn tất trong 61,3 giây và tạo đủ kết quả cho 3 dòng test công khai. Các kiểm tra đầu ra chứng minh file có cấu trúc phù hợp với các điều kiện trong code, không chứng minh nhãn dự đoán là đúng. Lần chạy được ghi nhận chưa gửi chấm điểm và chưa có điểm hidden test.

## 10. Phạm vi phân tích hiện có và những giới hạn

Notebook tập trung vào kiểm tra chất lượng dữ liệu phục vụ huấn luyện, đánh giá có kiểm soát và tạo output. EDA chi tiết về độ dài văn bản, biểu đồ và thống kê mô tả nằm trong báo cáo EDA riêng; notebook này không tự thực hiện lại toàn bộ EDA đó.

Các giới hạn cần giữ khi diễn giải kết quả:

- Chỉ có một holdout với 30 prompt và 30 mẫu/lớp trong validation; chưa đo độ dao động qua nhiều phép chia nhóm.
- Gom prompt và text trùng giúp hạn chế rò rỉ nhưng chưa phát hiện mọi near-duplicate hoặc sự trùng chủ đề giữa các prompt khác nhau.
- TF-IDF ký tự biểu diễn các mẫu cục bộ; không trực tiếp mô hình hóa toàn bộ ngữ cảnh dài hay cấu trúc lập luận.
- Điểm validation 29,07% không phải điểm Kaggle leaderboard và không bảo đảm hiệu quả trên prompt mới.
- Chưa phân tích trọng số đặc trưng, độ tin cậy đã hiệu chỉnh của xác suất, hoặc nguyên nhân lỗi theo độ dài/ngôn ngữ/chủ đề.

Những bước phân tích tiếp theo có thể gồm grouped cross-validation, xem confusion matrix theo lớp, nhóm lỗi theo độ dài và đọc các n-gram có trọng số lớn. Đây là đề xuất mở rộng, chưa được triển khai trong notebook hiện tại.

## 11. Các nguồn trong repository

- `notebooks/kaggle_cpu_standalone.ipynb`: notebook được phân tích.
- `src/baseline.py`: các hàm xử lý và mô hình được nhúng trong notebook.
- `configs/baseline.json`: cấu hình thực tế của baseline.
- `scripts/build_kaggle_cpu_notebook.py`: cách tạo notebook độc lập từ source.
- `reports/kaggle_cpu_run_2026-11-03.md`: bằng chứng và kết quả lần chạy Kaggle.
- `reports/baseline_report.md`: kết quả chi tiết của baseline local, gồm thống kê từng lớp.
- `outputs/baseline/split_audit.json`: audit phép chia local được dùng đối chiếu trong tài liệu.
- `reports/name_that_llm_data_overview.md`: EDA riêng trước bước huấn luyện.

Các con số trong tài liệu phản ánh dữ liệu và lần chạy đã ghi nhận. Khi đổi dữ liệu, cấu hình hoặc runtime, cần đọc lại audit và kết quả của lượt chạy mới.
