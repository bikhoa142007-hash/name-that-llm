# Debate: Hướng cải tiến tiếp theo cho baseline TF-IDF + LR (Name That LLM)

**Created:** 2026-11-19
**Agent 1:** Claude (Opus 5.5)
**Agent 2:** Astra (Codex, gpt-6-astra)
**Max Rounds:** 3
**Status:** CONVERGED

## Context

Baseline hiện tại (TF-IDF ký tự 3–5 + Logistic Regression, CPU) đạt balanced accuracy **0,2907** trên một holdout theo prompt (540 dòng, 30 prompt, 30 mẫu/lớp), so với Dummy 0,0556. Câu hỏi của debate: **bước tiếp theo nên làm gì để tăng điểm thật trên hidden test, với rủi ro thấp nhất?** Có nên thay mô hình, hay giữ mô hình và đổi cách đánh giá / hậu xử lý?

Phạm vi cho phép trong debate: được **chạy script đo đạc chỉ đọc** (không sửa `src/`, `configs/`, notebook). Code pipeline chỉ thay đổi sau khi debate hội tụ ("implement debate 1").

### Evidence

**E1 — Kết quả baseline (Kaggle Version 1, 03/11/2026, và local khớp):**

| Mô hình | Balanced acc | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Dummy | 0,055556 | 0,055556 | 0,005848 |
| TF-IDF + LR | 0,290741 | 0,290741 | 0,269549 |

Nguồn: `reports/kaggle_cpu_run_2026-11-03.md`, `outputs/baseline/metrics.json`.

**E2 — Lớp yếu và nhầm lẫn (local, `reports/baseline_report.md:93-114`):** `glm-5.3` recall 0/30 (F1 0), `claude-fable-5` F1 0,0435, `gpt-5.6-terra` F1 0,1111. Nhầm lớn nhất: `gemini-3.8-flash → gemini-3.7-flash` 12 lần, `glm-5.3 → claude-opus-5` 9 lần.

**E3 — Lỗi trong cùng họ model (đo 2026-11-19 bằng `debates/evidence/1_measure_prompt_constraint.py`):** 383 lỗi / 540; **177 lỗi (46%) là nhầm sang model cùng họ** (gpt/gemini/claude/glm…).

**E4 — Cấu trúc dữ liệu train (`reports/name_that_llm_data_overview.md:79`):** 149 prompt có đủ 18 model, p002 có 17 (thiếu `claude-fable-5`); không trùng cặp `(prompt_id, model)`. Test **không có** `prompt_id` (`reports/name_that_llm_data_overview.md:32`). Hidden test khoảng 4.000 đoạn (`reports/name_that_llm_data_overview.md:21`), xấp xỉ 222 × 18.

**E5 — Ràng buộc "mỗi model một lần trong mỗi prompt" trên holdout hiện có** (cùng script, dùng `artifacts/baseline/validation_pipeline.joblib` chỉ fit trên train holdout; Hungarian `scipy.optimize.linear_sum_assignment` trên `-log P`, mỗi nhóm có các cột lớp lặp `ceil(n/18)` lần):

| Cách gán nhãn trên 540 dòng validation | Accuracy |
|---|---:|
| argmax (hiện tại) | 0,2907 |
| Hungarian theo `prompt_id` thật (oracle) | **0,4722** |
| Hungarian theo cụm chủ đề tự gom (không dùng `prompt_id`) | **0,4481** |

Gom cụm: word TF-IDF (`stop_words='english'`, `min_df=2`, `sublinear_tf`) chỉ trên text validation → `AgglomerativeClustering(n_clusters=30, metric='cosine', linkage='average')`. ARI so với `prompt_id` = **0,9241**; kích thước cụm min/median/max = 1/18/36.

**E6 — Gom cụm ở quy mô gần hidden test (toàn bộ 2.699 dòng train, 150 prompt), `k = round(n/18) = 150`:** ARI = **0,9087**; kích thước cụm min/median/max = 1/18/54; 122/150 cụm có đúng 18 phần tử.

~~**E7 — Độ nhiễu của một holdout:** sai số chuẩn của accuracy ≈ √(0,29·0,71/540) ≈ **0,020**; mỗi lớp chỉ 30 mẫu nên recall từng lớp dao động khoảng ±0,08. Chênh lệch < ~0,04 trên một holdout không phân biệt được với nhiễu.~~
[A2-R1] Công thức E7 giả định các dòng độc lập và chỉ xét một tỷ lệ. Hai cách dự đoán dùng chung 540 dòng thuộc 30 prompt. Có 128 dòng chuyển từ sai sang đúng và 43 dòng chuyển từ đúng sang sai. Vì vậy không dùng 0,04 làm ngưỡng phân biệt cho chênh lệch ghép cặp. SC3 bên dưới dùng chênh lệch thực đo theo fold.
[A1-R2] Chấp nhận. Đã kiểm tra độc lập bằng `debates/evidence/1_r2_verify.py` (phần a): holdout argmax 157 đúng → topic 242 đúng, `wrong_to_right=128`, `right_to_wrong=43`, khớp từng số với [A2-R1]. E7 không còn được dùng làm cổng; cổng là SC3 theo fold.

**E8 [A1-R2] — Tái lập độc lập 5-fold (`debates/evidence/1_r2_verify.py`, phần b):** dùng `make_split()` để lấy `groups` (src/baseline.py:134), `StratifiedGroupKFold(5, shuffle=True, random_state=42)`, `build_pipeline` + `fit_checked` (src/baseline.py:165-183), gom cụm chỉ trên text validation với `k=round(n/18)`. Mọi fold có 18 lớp validation, giao prompt và giao text nguyên văn = 0 (assert). Kết quả trùng khớp đến 6 chữ số với bảng [A2-R1] ở P1: balanced accuracy trung bình 0,299655 → 0,418225, **tăng trung bình +0,118570, tăng nhỏ nhất +0,085185 (fold 2)**. Lưu ý: script E5 của [A1-R1] lấy `k` từ `prompt_id` thật (dòng 47) — [A2-R1] chỉ ra đúng; với `k=round(540/18)=30` kết quả holdout vẫn là 242/540 (phần a dùng k ước lượng).

### Relevant Files
- `src/baseline.py:109-147` — `make_split()`: gom thành phần liên thông theo `prompt_id` + text chuẩn hóa, `GroupShuffleSplit` một lần.
- `src/baseline.py:165-171` — `build_pipeline()`: TfidfVectorizer + LogisticRegression.
- `src/baseline.py:192-215` — `evaluate_models()`: dùng `model.predict` (argmax).
- `src/baseline.py:290-323` — `predict()`: dự đoán theo batch, lấy `classes_[argmax]` tại dòng 308.
- `configs/baseline.json` — cấu hình cố định.
- `notebooks/kaggle_cpu_standalone.ipynb`, `scripts/build_kaggle_cpu_notebook.py` — notebook nhúng `src/baseline.py`.
- `debates/evidence/1_measure_prompt_constraint.py` — script đo E3, E5.

### Constraints
- Kaggle code competition: notebook CPU, Internet OFF, không dataset ngoài (README-baseline, Rules §2.6).
- Không dùng test / leaderboard để chọn cấu hình.
- Đầu ra giữ đúng schema `id, model`.
- Không tự gửi submission; người dùng quyết định.

---

## Assumptions

- [A1-R1] Assumption: Hidden test được sinh giống train, tức các prompt mới, mỗi prompt có (gần) đủ 18 model. Basis: train 149/150 prompt đủ 18 (E4); hidden ≈ 4.000 ≈ 222 × 18. **Chưa xác minh**; nếu sai, ràng buộc cứng có thể làm hại.
[A2-R2] Kaggle cho biết số đoạn mỗi model trong test khác nhau và tỷ lệ model khác nhau giữa hai leaderboard. Điều này loại trừ giả thiết mọi prompt đều có đúng một đoạn cho mỗi model trên toàn test. Mức thiếu theo từng prompt vẫn chưa biết. [Mô tả chính thức](https://www.kaggle.com/competitions/name-that-llm/overview/abstract)
[A1-R3] Xác minh: đã mở trang Overview (2026-11-19), mục Evaluation ghi "The number of test passages per model is not equal". Chấp nhận — A1 dạng "đủ 18 model mỗi prompt" là **sai** với hidden test. Đo tác động ở **E9** bên dưới.

**E9 [A1-R3] — P2 khi số đoạn mỗi model không bằng nhau** (`debates/evidence/1_r3_imbalance.py`): trong mỗi fold validation của 5-fold (cùng split E8), giữ mỗi dòng của model m với xác suất r_m ~ Uniform(lo, 1), 3 seed/fold, `k=round(n/18)` như P2.

| Kịch bản | Số lượt | Số dòng/model (min–max) | Tăng bal. acc trung bình | Nhỏ nhất | Lớn nhất | Lượt âm |
|---|---:|---|---:|---:|---:|---:|
| Cân bằng (E8) | 5 | 29–30 | +0,1186 | +0,0852 | +0,1521 | 0 |
| lo=0,3 | 15 | 4–30 | +0,0358 | +0,0030 | +0,0683 | 0 |
| lo=0,1 | 15 | 1–30 | +0,0260 | −0,0195 | +0,1332 | 1 (fold 4 seed 0: 0,3207 → 0,3012) |

Kết luận: khi số lượng lệch, mức tăng giảm còn khoảng 1/3–1/4 và **không đạt ngưỡng trung bình ≥ 0,05 của SC3**; có lượt làm giảm điểm. Một phần do `k=round(n/18)` đánh giá thấp số prompt (vd. n=300 → k=17 trong khi có 30 prompt). Điều này **ủng hộ** quyết định [A2-R2] để mặc định argmax ở cả CLI và notebook.
- [A1-R1] Assumption: Có thể ước lượng số prompt trong test bằng `round(n_test/18)`. Basis: trên train, `round(2699/18)=150` đúng bằng số prompt thật (E6).
[A2-R1] Đã đếm trực tiếp train.csv: 149 nhóm 18 dòng, một nhóm 17 dòng, 0 cặp prompt/model trùng. Khi lấy ngẫu nhiên 16 hoặc 13 dòng từ mỗi prompt trên holdout (NumPy default_rng(42), hai lần lấy liên tiếp theo thứ tự này, nhóm theo prompt_id đã sắp), số prompt thật vẫn là 30 nhưng round(n/18) lần lượt là 27 và 22. Số dòng đúng argmax → topic là 143 → 202 trên 480 và 119 → 145 trên 390. Hai phép thử này chỉ kiểm trường hợp thiếu dòng ngẫu nhiên.
[A1-R2] Đã kiểm độc lập với mẫu ngẫu nhiên khác (`default_rng(7)`, `1_r2_verify.py` phần c): giữ 16/prompt → 480 dòng, k=27, argmax 141 → topic 204; giữ 13/prompt → 390 dòng, k=22, argmax 119 → topic 149. Cùng hướng với [A2-R1]: mức tăng giảm dần khi thiếu dòng (+63 → +30) nhưng vẫn dương trong hai trường hợp đã đo. Trường hợp thiếu hẳn một model trên toàn test chưa đo → giữ trong Parking Lot.
- [A1-R1] Assumption: Dùng thông tin chéo giữa các dòng test bên trong notebook (gom cụm + gán nhãn đồng thời) không vi phạm Rules. Basis: chỉ dùng text test có sẵn trong môi trường chấm, không dữ liệu ngoài, không gán nhãn thủ công. **Chưa đọc lại Rules để xác nhận** — phải xác nhận trước khi nộp.
[A2-R1] UNCLEAR: trang Rules chính thức không trả về nội dung điều khoản đọc được trong lượt kiểm tra. Giữ việc xác nhận trong Parking Lot trước khi dùng trên Kaggle. [Kaggle Rules](https://www.kaggle.com/competitions/name-that-llm/rules)
[A1-R3] Đã đọc được trang Rules qua trình duyệt (2026-11-19, ~35.600 ký tự). Điều khoản liên quan duy nhất tìm thấy cấm thông tin từ "hand labeling or human prediction" trên validation/test; không có điều khoản cấm xử lý thuật toán trên text test trong notebook. Đây **không** phải xác nhận rõ ràng từ BTC; người dùng vẫn nên tự đọc và quyết định trước khi bật `topic`. Giữ PARKED.
- [A1-R1] Assumption: Mức tăng ở E5 (một holdout, 30 prompt) phản ánh mức tăng trên prompt mới. Basis: holdout tách prompt hoàn toàn (`prompt_overlap=0`, `artifacts/baseline/metadata.json`). Sẽ kiểm bằng 5-fold (P1).
- [A1-R1] Assumption: Bộ test công khai 3 dòng quá nhỏ để gom cụm; với n < 18 phải giữ argmax. Basis: 3 < 18, không thể tạo nhóm prompt có ý nghĩa.

---

## Success Criteria

- [ ] SC1 — 5-fold CV theo nhóm prompt chạy được, không rò rỉ: mỗi fold có đủ 18 lớp, giao prompt/text giữa train và validation bằng 0 — `python -m src.baseline cv --train train.csv --config configs/baseline.json --output-dir outputs/cv_run1` → `cv_metrics.json` có 5 fold, `prompt_overlap=0` mọi fold.
[A2-R1] Đọc cả prompt_overlap, exact_text_overlap và normalized_text_overlap, đồng thời đếm lớp ở cả hai phía. src/baseline.py:143-155 đã định nghĩa ba phép kiểm. Trong phép đo R1, mỗi fold có 18/18 lớp train/validation và cả ba số overlap đều bằng 0.
[A1-R2] Chấp nhận nội dung. Sửa nhỏ trích dẫn: ba phép kiểm nằm ở src/baseline.py:143-147 (`require(... 'Text/prompt leakage detected.')` tại dòng 147); dòng 149-155 chỉ ghi các số đó vào `audit`. P1 nên tái dùng đoạn 140-147 cho mỗi fold.
- [ ] SC2 — Báo cáo argmax 5-fold: mean ± std balanced accuracy, recall từng lớp — có trong `cv_metrics.json`.
~~- [ ] SC3 — Hậu xử lý gán nhãn theo cụm được giữ lại **chỉ khi** trên 5-fold: accuracy (cụm + Hungarian) > argmax ở **cả 5 fold** và mức tăng trung bình ≥ 0,05 — đọc `cv_metrics.json`.~~
[A2-R1] Dùng balanced accuracy cho cổng chọn P2, với cùng hai điều kiện (> 0 ở cả 5 fold, trung bình ≥ 0,05). src/baseline.py:186-189 đã tính chỉ số này. Fold 1 có 539 dòng và 18 lớp, accuracy argmax 165/539 = 0,306122 trong khi balanced accuracy là 0,305683. Cả hai chỉ số đều vượt cổng trong phép đo dưới đây.
[A1-R2] Chấp nhận: balanced accuracy đúng là metric của báo cáo baseline và fold chứa p002 (539 dòng) làm hai chỉ số lệch nhau. Đã xác minh fold 1 = 539 dòng, bal argmax 0,305683 (E8). Theo đo đạc chỉ đọc E8, SC3 **đã đạt** (tăng nhỏ nhất +0,085 > 0, trung bình +0,119 ≥ 0,05); SC3 vẫn phải được xác nhận lại bằng chính lệnh `cv` sau khi triển khai.
- [ ] SC4 — Khi n_test < 18 hoặc tắt cờ, đầu ra giống hệt argmax — chạy `predict` trên `test.csv` (3 dòng) với cả hai chế độ → hai file Parquet có cùng cột `model`.
[A1-R2] Bổ sung theo fallback đã gộp (xem P2): unit test trên fixture tổng hợp ≥ 18 dòng — (i) một dòng chỉ stopword: dòng đó bằng nhãn argmax, các dòng còn lại không lỗi, thứ tự ID giữ nguyên; (ii) mọi dòng chỉ stopword: toàn bộ bằng argmax, không exception.
[A2-R2] Chấp nhận SC4. Trên 18 dòng “the and”, bộ tạo đặc trưng báo empty vocabulary và nhánh fallback trả đủ 18 nhãn argmax. Trên holdout có một dòng vector 0, đảo ngược đầu vào rồi đối chiếu theo ID cho cùng 540/540 nhãn.
[A1-R3] Đã kiểm độc lập (hàm `topic` trong `1_r3_imbalance.py`): 18 dòng “the and” → toàn bộ bằng argmax (True); holdout có dòng 0 = “the and”, đảo thứ tự đầu vào rồi đảo lại → 540/540 nhãn giống nhau.
- [ ] SC5 — Toàn bộ test hiện có vẫn pass — `python -m unittest discover -s tests -v` → OK, và có thêm test cho P1/P2 trên fixture tổng hợp.
- [ ] SC6 — Notebook Kaggle standalone được build lại và chạy Save & Run All thành công (CPU, Internet OFF); `submission.parquet` hợp lệ — người dùng xác nhận trên Kaggle.

---

## Proposal

STATUS: CONVERGED

~~**Tóm tắt [A1-R1]:** Chưa thay mô hình. Làm hai việc nhỏ theo thứ tự: (P1) đo bằng 5-fold theo prompt; (P2) thêm bước hậu xử lý "mỗi model một lần trong mỗi cụm chủ đề", chỉ bật nếu P1 xác nhận. Lý do: E5 cho thấy hậu xử lý tăng +0,16 accuracy trên holdout, lớn hơn nhiều so với mức thường thấy khi chỉnh đặc trưng tuyến tính, và nhắm đúng loại lỗi chiếm 46% (E3).~~
[A2-R1] Giữ baseline và đánh giá P2 trên cùng fold. Đối chiếu nhãn từng dòng từ debates/evidence/1_measure_prompt_constraint.py:26-53: topic sửa 61 lỗi cùng họ và 67 lỗi khác họ, đồng thời làm sai 43 dòng argmax vốn đúng. Mức tăng ròng là 85/540. Bằng chứng ủng hộ P2 cho cả hai loại lỗi.
[A1-R2] Chấp nhận việc bỏ lập luận "lỗi cùng họ" làm lý do chính. Đã xác minh tổng: 128 − 43 = 85, 157 + 85 = 242 (E8/phần a). Tách 61/67 theo họ chưa được kiểm lại riêng nhưng không ảnh hưởng kết luận vì tổng đã khớp. Lý do giữ P2 bây giờ là mức tăng balanced accuracy theo fold (E8), không phải phân loại lỗi.

### P1 — Thêm lệnh `cv` (5-fold theo nhóm prompt) [A1-R1]

- **Hiện tại:** `src/baseline.py:136-138` chỉ chia một lần bằng `GroupShuffleSplit(n_splits=1, ...)`; mọi so sánh dựa trên 540 dòng (độ nhiễu ±0,02, E7).
- **Đề xuất:** Tách phần gom nhóm trong `make_split()` (`src/baseline.py:115-134`, tính `groups`) thành hàm `connected_groups(frame)` dùng chung; thêm hàm `cross_validate(args)` dùng `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)` trên `groups`. Mỗi fold: `build_pipeline(config)` → `fit_checked` → tính `predict_proba` trên fold validation → ghi argmax metrics (dùng `score()` tại `src/baseline.py:186-189`) và các kiểm tra rò rỉ giống `src/baseline.py:139-147`. Xuất `cv_metrics.json` (từng fold + mean/std) và per-class recall. Không lưu model, không đọc test.
- **Không đổi:** lệnh `train`, `evaluate`, `predict`, cấu hình, holdout cũ.
[A1-R2] Lý do "độ nhiễu ±0,02, E7" ở dòng "Hiện tại" không còn hiệu lực (E7 đã bị gạch). Lý do giữ P1: một holdout không cho biết độ dao động giữa các phép chia; E8 cho thấy argmax dao động 0,283–0,311 và mức tăng P2 dao động +0,085 đến +0,152 giữa các fold.
[A2-R1] Đã chạy độc lập 5 lần fit bằng src/baseline.py:109-162,165-189 và configs/baseline.json, StratifiedGroupKFold(5, shuffle=True, random_state=42). Với mỗi validation, chỉ dùng text để fit bộ gom cụm và lấy k=round(n/18). Script E5 ở dòng 47 lấy k từ prompt_id thật. Cả hai cách cho k=30 ở cả 5 fold. Kết quả dưới đây là phép đo chỉ đọc, chưa tạo lệnh cv hay cv_metrics.json.

| Fold [A2-R1] | Dòng | Đúng argmax → topic | Balanced accuracy argmax → topic |
|---|---:|---:|---:|
| 1 [A2-R1] | 539 | 165 → 247 | 0,305683 → 0,457791 |
| 2 [A2-R1] | 540 | 168 → 214 | 0,311111 → 0,396296 |
| 3 [A2-R1] | 540 | 153 → 220 | 0,283333 → 0,407407 |
| 4 [A2-R1] | 540 | 160 → 212 | 0,296296 → 0,392593 |
| 5 [A2-R1] | 540 | 163 → 236 | 0,301852 → 0,437037 |
| Trung bình theo fold [A2-R1] | | | 0,299655 → 0,418225 (+0,118570) |

### P2 — Hậu xử lý gán nhãn theo cụm chủ đề (tùy chọn, tắt mặc định cho tới khi qua SC3) [A1-R1]

- **Hiện tại:** `src/baseline.py:308` gán `pipeline.classes_[probabilities.argmax(axis=1)]` độc lập cho từng dòng.
- **Đề xuất:**
  1. Trong `predict()`, gom `probabilities` của mọi batch thành một ma trận thay vì argmax ngay.
  2. Hàm `assign_by_topic(texts, probabilities, classes)`: nếu `n < 18` → trả argmax (fallback, SC4). Ngược lại: word TF-IDF trên text test (`stop_words='english'`, `min_df=2`, `sublinear_tf=True`) → `AgglomerativeClustering(n_clusters=round(n/18), metric='cosine', linkage='average')` → trong mỗi cụm, `linear_sum_assignment` trên `-log(clip(P, 1e-12))` với các cột lớp lặp `ceil(size/18)` lần.
  3. Cờ CLI `--assignment {argmax,topic}` cho `predict`, mặc định `argmax`. Lệnh `cv` báo cả hai chế độ trên mỗi fold (để kiểm SC3). Trong `cv`, gom cụm chỉ dùng text của fold validation (giống tình huống test).
  [A1-R2] Bật P2 ở đâu: vì E8 cho thấy SC3 đạt, đề xuất **CLI giữ mặc định `argmax`** (không đổi hành vi lệnh hiện có), còn ~~**notebook standalone dùng biến cấu hình `ASSIGNMENT = 'topic'`**~~ ở cell cấu hình, kèm ghi chú rằng người dùng phải tự xác nhận Rules (A3) trước khi nộp; đổi về `'argmax'` là cách quay lui một dòng. Lý do: notebook là thứ duy nhất chạy trên hidden test, nên nếu P2 chỉ có ở CLI thì mức tăng không bao giờ được dùng. Nếu Astra muốn notebook cũng mặc định `argmax` cho tới khi có xác nhận Rules, mình chấp nhận được — điểm cần thống nhất là chỗ đặt công tắc, không phải thuật toán.
[A2-R2] Chọn phương án argmax mặc định cho cả notebook và CLI mà Agent 1 đã chấp thuận ở câu cuối. Giữ biến ASSIGNMENT trong cell cấu hình và truyền nó vào lệnh predict tại scripts/build_kaggle_cpu_notebook.py:50-53. Người dùng có thể chọn topic sau khi xác nhận Rules. CV cải thiện trung bình 0,118570 trên các fold gần cân bằng. Theo Assumptions, tỷ lệ model trên test khác train, và lượt Kaggle đã kiểm chỉ có 3 dòng (reports/kaggle_cpu_run_2026-11-03.md:50), dưới ngưỡng kích hoạt 18. Trang Rules chưa cung cấp văn bản đọc được để xác nhận việc gom chung text test. [Kaggle Rules](https://www.kaggle.com/competitions/name-that-llm/rules)
[A1-R3] Chấp nhận, và rút lại đề xuất notebook mặc định `topic` của [A1-R2] vì bằng chứng chứ không phải để kết thúc debate: E9 cho thấy khi số đoạn mỗi model lệch, mức tăng trung bình chỉ +0,026–0,036 và có lượt âm. Đã kiểm trích dẫn: `scripts/build_kaggle_cpu_notebook.py:50-51` là lời gọi `main(['predict', ...])` — đây là chỗ thêm `'--assignment', ASSIGNMENT`; `reports/kaggle_cpu_run_2026-11-03.md:50` ghi Parquet cuối 3 dòng.
~~- **Fallback / độ bền (guardrail "no brittle assumptions"):** nếu A1 sai (test không đủ 18 model mỗi prompt), cụm sẽ lệch kích thước; việc lặp cột `ceil(size/18)` vẫn cho mỗi dòng một nhãn hợp lệ, không crash. Mức suy giảm khi A1 sai **chưa đo** — xem Parking Lot.~~
~~[A2-R1] Thêm fallback argmax cho toàn bộ tập dự đoán khi word TF-IDF không có từ vựng hoặc có hàng vector 0.~~ Đã thay một trong 18 text holdout bằng “the and”: bộ lọc ở debates/evidence/1_measure_prompt_constraint.py:46 loại cả hai từ, dòng 48 báo “Cosine affinity cannot be used when X contains zero vectors”. Nếu cả 18 dòng đều như vậy thì không còn từ để fit. Bổ sung hai fixture này vào SC4, yêu cầu trả đúng thứ tự và nhãn argmax từng dòng. Cụm lệch kích thước vẫn xử lý theo P2.
[A1-R2] Đồng ý lỗi là thật và cần fallback; không đồng ý fallback cho **toàn bộ** tập khi chỉ có vài hàng vector 0. Lý do: `read_input` điền text test null bằng chuỗi rỗng và giữ dòng (`reports/notebook_data_analysis_v1.md:49`), nên chỉ một dòng rỗng trong ~4.000 dòng hidden sẽ tắt P2 hoàn toàn. Đo (`1_r2_verify.py` phần d, thay 1 dòng holdout bằng “the and”): fallback toàn tập → 157/540 đúng (mất toàn bộ mức tăng); fallback từng dòng → 242/540. **Đề xuất gộp:** (1) `TfidfVectorizer` ném `ValueError` (từ vựng rỗng) hoặc n < 18 → argmax toàn tập; (2) hàng có vector 0 giữ argmax riêng từng dòng, các hàng còn lại gom cụm với `k=round(m/18)`, `m` = số hàng khác 0; nếu `k < 1` → argmax toàn tập. Giữ hai fixture của [A2-R1] trong SC4 (một dòng “the and” → dòng đó bằng argmax, các dòng khác vẫn được gán; toàn bộ stopword → bằng argmax toàn tập). Đã cài đúng logic này trong `debates/evidence/1_r2_verify.py`, hàm `topic(..., per_row_fallback=True)`.
[A2-R2] Chấp nhận fallback từng dòng. Phép đo của Agent 1 giữ P của text gốc khi thay text để gom cụm (debates/evidence/1_r2_verify.py:60,66-70). Tôi thay text đầu tiên bằng “the and” trước cả predict_proba và TF-IDF, dùng cùng validation_pipeline.joblib. Có 539/540 vector khác 0, k=30. Argmax toàn tập đúng 156/540, fallback từng dòng đúng 241/540. Dòng bị thay giữ nhãn argmax kimi-k3. Chênh lệch vẫn là 85 dòng đúng ở cả hai phép đo.
[A1-R3] Xác minh: [A2-R2] đúng là phép đo R2 của mình dùng P của text gốc. Tính lại P sau khi thay text: argmax 156/540, topic 241/540, dòng 0 = `kimi-k3` ở cả hai chế độ — khớp từng số.
- **Chi phí:** gom cụm agglomerative trên ma trận dense 4.000 × |vocab| có thể tốn RAM; trên 2.699 dòng train chạy được trên máy local (E6). Nếu cần, giảm chiều bằng `max_features` cho TF-IDF chủ đề. Chưa đo thời gian trên Kaggle.

### Không đề xuất ở vòng này [A1-R1]

Thay đổi đặc trưng/mô hình (word n-gram, tăng `max_features`, LinearSVC, transformer) chưa có bằng chứng đo → Parking Lot. Chúng nên được so sánh **sau** khi có P1, trên cùng 5 fold.

---

## Plan

PLAN_STATUS: CONVERGED

[A1-R4] Plan triển khai đúng phần đã hội tụ: P1 (lệnh `cv`) và P2 (chế độ `topic`, mặc định `argmax` ở CLI và notebook, fallback từng dòng). Không đổi `configs/baseline.json`, không đổi lệnh `train`/`evaluate`, không đổi mô hình. Không đụng notebook cũ đã chạy trên Kaggle; notebook mới được build ra tên mới.

### Changes

| # | File | What to change | Why |
|---|------|----------------|-----|
| 1 [A1-R4] | `src/baseline.py:22-28` | Thêm import `AgglomerativeClustering` (`sklearn.cluster`), `StratifiedGroupKFold` (`sklearn.model_selection`, cạnh `GroupShuffleSplit` dòng 26), `linear_sum_assignment` (`scipy.optimize`). Không thêm dependency: `scipy==1.16.3` đã có trong `requirements-baseline.txt`. | P1, P2 |
| 2 [A1-R4] | `src/baseline.py:115-134` → hàm mới `connected_groups(frame)` | Chuyển **nguyên văn** đoạn union-find (dòng 115-134, kết thúc bằng `groups = np.array(...)`) thành `connected_groups(frame)` trả `(groups, keys)`. `make_split` gọi `groups, keys = connected_groups(frame)`. Không đổi thuật toán, nên cột `component` trong `split.csv` phải giữ nguyên. | P1 cần cùng nhóm với holdout |
| 3 [A1-R4] | `src/baseline.py:139-147` → hàm mới `check_disjoint(frame, groups, train_idx, valid_idx)` | Chuyển nguyên văn kiểm tra lớp thiếu (140-142), giao nhóm (143), giao text/prompt (144-147) thành hàm trả `(absent, exact_overlap, normalized_overlap, prompt_overlap)`; `make_split` dùng giá trị trả về để dựng `audit` như hiện tại (149-161 không đổi). ~~Thông báo lỗi `Holdout lacks classes` đổi thành `Split lacks classes` để dùng chung cho fold.~~ (xem ↳ [A1-R5] bên dưới) | SC1 dùng đúng phép kiểm của holdout |
| 4 [A1-R4] | `src/baseline.py`, sau `score()` (dòng 186-189) | Thêm `assign_by_topic(texts, probabilities, classes)` và helper `_assign_within(log_p, classes, labels)`. Logic giống hàm `topic(...)` trong `debates/evidence/1_r3_imbalance.py` (đoạn code bên dưới). | P2 |
| 5 [A1-R4] | `src/baseline.py`, hàm mới `cross_validate(args)` trước `main()` | 5-fold `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config['seed'])` trên `connected_groups`. Mỗi fold: `check_disjoint` → `build_pipeline` → `fit_checked` → `predict_proba` (trong `threadpool_limits(1)`) → `score()` cho `argmax` và `topic` (topic chỉ dùng text của fold validation) + recall từng lớp argmax. Ghi `cv_metrics.json`. Không lưu model, không đọc test, thư mục output phải mới (giống `train`, dòng 234). | P1, SC1-SC3 |
| 6 [A1-R4] | `src/baseline.py:301-309` (`predict`) | Gom `probabilities` của mọi batch (vẫn kiểm từng batch như dòng 305-307) vào danh sách, `np.vstack` sau vòng lặp; nếu `args.assignment == 'topic'` gọi `assign_by_topic(test['text'].tolist(), P, pipeline.classes_)`, ngược lại `pipeline.classes_[P.argmax(axis=1)]`. Thêm `assignment=args.assignment` vào `checked.update(...)` (dòng 316). Test 0 dòng: `P` = mảng rỗng shape `(0, 18)`, trả danh sách rỗng như hiện tại. | P2, SC4 |
| 7 [A1-R4] | `src/baseline.py:342-367` (`main`) | Thêm subcommand `cv` (`--train`, `--config`, `--output-dir`); thêm `--assignment` vào `predict` với `choices=['argmax','topic']`, `default='argmax'`. | P1, P2 |
| 8 [A1-R4] | `tests/test_baseline.py` | Thêm 4 test (fixture tổng hợp, không dữ liệu thật): ~~(a) `connected_groups` trả cùng nhóm như `make_split` cũ trên fixture của `test_union_of_prompt_and_normalized_duplicates_cannot_leak`;~~ (a, d: xem ↳ [A1-R5] bên dưới) (b) `assign_by_topic` với n < 18 = argmax; (c) 18 dòng `"the and"` = argmax, không exception; (d) 36 dòng thuộc 2 chủ đề rõ rệt, một dòng `"the and"`: dòng đó = argmax, ~~mỗi chủ đề nhận 18 nhãn khác nhau, đảo thứ tự đầu vào cho cùng nhãn theo vị trí gốc~~. | SC4, SC5 |
| 9 [A1-R4] | `scripts/build_kaggle_cpu_notebook.py` | (i) Thêm `ASSIGNMENT = 'argmax'  # 'topic' chỉ sau khi tự xác nhận Rules; xem debates/1-...` vào chuỗi `run` sau `RUN_DIR` (dòng 39); (ii) dòng 50-51 thêm `'--assignment', ASSIGNMENT` vào `main(['predict', ...])`; (iii) thêm tham số tùy chọn `--output` (mặc định giữ `notebooks/kaggle_cpu_standalone.ipynb`) để build ra file mới, vì `open('x')` ở dòng 77 không ghi đè. | SC6, công tắc đã hội tụ |
| 10 [A1-R4] | `README-baseline.md` | Thêm mục ngắn: lệnh `cv`, cờ `--assignment` (mặc định argmax), lý do không bật topic mặc định (E9) và rằng việc bật là quyết định của người dùng sau khi đọc Rules. | Người dùng biết công tắc |
| ↳ [A2-R4] | 3 | Giữ a, b trong make_split hoặc dựng lại từ frame.iloc trước audit. Sau khi tách dòng 139, src/baseline.py:152,160-161 vẫn dùng len(a), len(b) và cả hai tập để đếm lớp. Giữ nguyên thông báo lỗi hiện có cho holdout khi tách hàm. | Plan review |
| ↳ [A2-R4] | 8 | Thay 8a bằng test gọi main([“cv”, …]) trên 8 prompt × 18 lớp của tests/test_baseline.py:54-66 (hai prompt nối qua text chuẩn hóa). Đọc JSON, kiểm 5 fold, đủ lớp hai phía, không giao nhóm/prompt/text, tổng validation_rows = 144 và mỗi fold đủ 18 giá trị recall. So component thật trong Verification 2 độc lập với helper. Hiện 8a chỉ kiểm helper, còn SC5 yêu cầu fixture cho P1. | Plan review |
| ↳ [A2-R4] | 8 | Sửa 8d thành 37 dòng: một dòng stopword riêng và hai cụm 18 dòng dùng từ vựng tách biệt. Với fixture cũ 36 dòng, P đều 1/18, chạy snippet #4 cho 17 và 18 nhãn khác nhau, đảo thứ tự chỉ 2/36 nhãn khớp do hòa điểm. Tôi chạy fixture 37 dòng với P có nghiệm tối ưu duy nhất: mỗi cụm đủ 18 nhãn, 1 nhãn khác argmax và đảo lại khớp 37/37. Có thể tái lập với nền 0,01 và đường chéo 0,83 cho từng cụm, sửa hai hàng đầu cụm 1 ở cột 0,1 lần lượt thành (0,55; 0,29) và (0,45; 0,39); dòng stopword ưu tiên lớp 0. Gắn kết quả với ID gốc. | Plan review |
| ↳ [A1-R5] | 3 | Chấp nhận [A2-R4]. Đã kiểm: trong `make_split`, `audit` dùng `len(a)`, `len(b)` (src/baseline.py:152) và `part` của `('train', a), ('validation', b)` (dòng 160-161). **#3 chốt:** `make_split` giữ nguyên dòng 139 `a, b = frame.iloc[train_idx], frame.iloc[valid_idx]`; `check_disjoint(frame, groups, train_idx, valid_idx)` tự dựng `a, b` bên trong và trả `(absent, exact_overlap, normalized_overlap, prompt_overlap)`; giữ nguyên thông báo lỗi hiện có (`Holdout lacks classes...`), kể cả khi gọi từ fold `cv`. Phần "đổi thành `Split lacks classes`" ở #3 bị bỏ. | Plan review |
| ↳ [A1-R5] | 8 | Chấp nhận cả hai sửa đổi [A2-R4]. Đã kiểm độc lập (chạy snippet #4 nguyên văn): **8a mới** — `StratifiedGroupKFold(5, shuffle=True, random_state=42)` trên fixture `tests/test_baseline.py:54-66` (8 prompt × 18, hai prompt nối qua “café duplicate”): fold validation 36/36/18/36/18 dòng, tổng **144**, mỗi fold 18/18 lớp hai phía, giao nhóm 0, `fit_checked` không lỗi hội tụ. **8d mới** — fixture 37 dòng đúng mô tả [A2-R4] (từ vựng riêng cho hai cụm, ví dụ “galaxy nebula orbit telescope …” và “recipe flour butter oven …”): dòng stopword = argmax, mỗi cụm 18 nhãn khác nhau, đúng 1 nhãn khác argmax, đảo thứ tự rồi đối chiếu theo ID khớp **37/37**. Fixture cũ 36 dòng P đều: đảo thứ tự chỉ khớp 2/36 → xác nhận lỗi hòa điểm. **#8 chốt:** (a) `main(['cv', ...])` trên fixture 8 prompt, đọc `cv_metrics.json` kiểm 5 fold, đủ lớp hai phía, overlap = 0, tổng `validation_rows` = 144, mỗi fold có 18 giá trị recall; (b) n < 18 = argmax; (c) 18 dòng “the and” = argmax; (d) fixture 37 dòng như trên. | Plan review |

**Code cho #4 [A1-R4]** (khớp logic đã đo ở E8/E9):

```python
def _assign_within(log_p, classes, labels):
    """Each cluster: Hungarian on -log P, class columns repeated ceil(size/K) times."""
    classes = np.asarray(classes)
    result = np.empty(len(log_p), dtype=object)
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        repeats = -(-len(idx) // len(classes))
        rows, cols = linear_sum_assignment(-np.tile(log_p[idx], (1, repeats)))
        result[idx[rows]] = classes[cols % len(classes)]
    return result


def assign_by_topic(texts, probabilities, classes):
    """Optional post-processing (debate 1). Falls back to argmax; zero-vector rows keep argmax."""
    classes = np.asarray(classes)
    result = classes[probabilities.argmax(axis=1)] if len(texts) else np.array([], dtype=object)
    if len(texts) < len(classes):
        return result
    try:
        x = TfidfVectorizer(analyzer='word', stop_words='english', min_df=2,
                            sublinear_tf=True).fit_transform(texts)
    except ValueError:  # empty vocabulary
        return result
    usable = np.asarray(x.getnnz(axis=1) > 0)
    clusters = round(int(usable.sum()) / len(classes))
    if clusters < 1:
        return result
    labels = AgglomerativeClustering(n_clusters=clusters, metric='cosine',
                                     linkage='average').fit_predict(x[usable].toarray())
    result = result.astype(object)
    result[usable] = _assign_within(np.log(np.clip(probabilities[usable], 1e-12, 1)), classes, labels)
    return result
```

Ghi chú: dùng `getnnz` thay `np.linalg.norm(X.toarray())` để không dựng ma trận dense toàn bộ hai lần; kết quả tương đương vì TF-IDF không âm. `round()` của Python làm tròn kiểu banker — giống script đo (E8/E9 dùng cùng hàm), nên số liệu tái lập được.

**Cấu trúc `cv_metrics.json` [A1-R4]:**

```json
{"config": {...}, "n_splits": 5, "seed": 42,
 "folds": [{"fold": 1, "train_rows": 2160, "validation_rows": 539,
            "validation_groups": 30, "absent_classes": {"train": [], "validation": []},
            "prompt_overlap": 0, "exact_text_overlap": 0, "normalized_text_overlap": 0,
            "argmax": {"balanced_accuracy": 0.305683, "accuracy": ..., "macro_f1": ...},
            "topic": {...}, "argmax_recall_per_class": {"gpt-5.5": ..., ...}}],
 "summary": {"argmax": {"balanced_accuracy_mean": ..., "balanced_accuracy_std": ...},
             "topic": {...}, "topic_gain_min": ..., "topic_gain_mean": ...,
             "sc3_balanced_folds_pass": true},
 "note": "Balanced folds only; unequal per-model counts reduce topic gain (debate 1, E9)."}
```

### Order of Operations
1. [A1-R4] #2, #3 (tách hàm, không đổi hành vi) → chạy test hiện có + kiểm `component` (Verification 1-2). Dừng nếu khác.
2. [A1-R4] #1, #4 + test 8b-8d.
3. [A1-R4] #5, #7 (`cv`) + test 8a → chạy `cv` trên `train.csv` thật (Verification 3).
4. [A1-R4] #6, #7 (`--assignment`) → Verification 4.
5. [A1-R4] #9 → build `notebooks/kaggle_cpu_standalone_v2.ipynb`; #10 README.
6. [A1-R4] Người dùng chạy notebook v2 trên Kaggle (Verification 6). Không tự submit.
7. [A1-R4] Ghi `## Implementation Results` vào file debate này.

### Verification

[A2-R4] SC1–SC3 có lệnh cv thật và cổng E8; SC4 có đối chiếu Parquet và fixture fallback; SC5 cần sửa #8; SC6 có bước Kaggle riêng. Thứ tự kiểm thử: sau #2–3 chạy 9 test cũ, sau khi thêm đủ các test mới mới chạy toàn bộ suite. Khi so E8, làm tròn 6 chữ số như debates/evidence/1_r2_verify.py:93-98 và tính cổng từ các điểm chưa làm tròn.
1. [A1-R4] `python -m unittest discover -s tests -v` → 9 test cũ + 4 test mới đều OK (SC5).
2. [A1-R4] So `connected_groups(read_input('train.csv', labeled=True)[0])[0]` với cột `component` của `artifacts/baseline/split.csv` → bằng nhau 2.699/2.699; và `python -m src.baseline evaluate --train train.csv --artifact-dir artifacts/baseline --output-dir outputs/evaluation_debate1` → `matches_saved_metrics: true` (refactor không làm đổi holdout).
[A2-R4] So thêm train_idx/valid_idx do make_split vừa sửa trả về với manifest theo ID. Trước sửa có 2.699/2.699 component khớp, 2.159 ID train và 540/540 ID validation khớp. evaluate chỉ nạp split.csv ở src/baseline.py:329-337 nên dù phép chia mới sai nó vẫn có thể báo true.
[A1-R5] Chấp nhận. Đã kiểm `evaluate` (src/baseline.py:324-340): chỉ đọc `split.csv` và model đã lưu, không gọi `make_split` → không phát hiện được lỗi refactor. Mốc trước sửa tự đo lại: component 2.699/2.699, ID train 2.159/2.159, ID validation 540/540. Verification 2 chốt: sau #2-#3, ba số này phải giữ nguyên **và** `evaluate` vẫn trả `matches_saved_metrics: true`.
3. [A1-R4] `python -m src.baseline cv --train train.csv --config configs/baseline.json --output-dir outputs/cv_run1` → 5 fold, mỗi fold 18/18 lớp hai phía, ba số overlap = 0 (SC1); có mean/std và recall từng lớp (SC2); balanced accuracy argmax trung bình **0,299655**, topic **0,418225**, gain min **0,085185** khớp E8; `sc3_balanced_folds_pass: true` (SC3).
4. [A1-R4] `predict` trên `test.csv` (3 dòng) với `--assignment argmax` và `--assignment topic` ra hai file mới → cột `model` giống hệt nhau, `validate` OK cả hai (SC4); `checks.json` ghi đúng `assignment`.
5. [A1-R4] Build notebook v2: script in ra đường dẫn mới, notebook cũ không đổi (~~`git diff --stat notebooks/kaggle_cpu_standalone.ipynb` rỗng~~); cell code parse được (`ast.parse`), có `ASSIGNMENT = 'argmax'`.
[A2-R4] Dùng SHA-256 notebook v1 trước và sau build v2. File hiện là untracked (git status --short trả ??), nên git diff không phát hiện thay đổi. Hash trước sửa f3ae493a4a85d066d0d200eb68b488807bec0ad6681092fc6bc9ca1644a2886f.
[A1-R5] Chấp nhận. Đã kiểm: `git status --short` liệt kê `?? notebooks/kaggle_cpu_standalone.ipynb`; `sha256sum` = `f3ae493a4a85d066d0d200eb68b488807bec0ad6681092fc6bc9ca1644a2886f`, khớp [A2-R4].
6. [A1-R4] Người dùng: Save & Run All notebook v2 trên Kaggle (CPU, Internet OFF) → Successful, `submission.parquet` hợp lệ, 3 dòng giống Version 1 vì vẫn argmax (SC6).
[A2-R4] SC6 kiểm số dòng và thứ tự ID theo test gắn vào notebook (src/baseline.py:278,283); so từng nhãn với Version 1 khi đầu vào khớp. Mẫu đã lưu có 3 dòng. Chỉ ghi đạt SC6 sau khi có kết quả Kaggle.
[A1-R5] Chấp nhận: câu "3 dòng giống Version 1" ở bước 6 chỉ áp dụng khi test Kaggle có hash khớp `4545aa3a…0c57` (reports/kaggle_cpu_run_2026-11-03.md); nếu khác thì chỉ kiểm số dòng/thứ tự ID qua `validate_submission`. Cũng chấp nhận ghi chú [A2-R4] ở đầu Verification: so E8 bằng giá trị làm tròn 6 chữ số, còn cổng `sc3_balanced_folds_pass` tính từ số chưa làm tròn.

[A1-R5] Đặt PLAN_STATUS: CONVERGED (đầu mục Plan) — mọi sửa đổi [A2-R4] đã được kiểm độc lập và gộp nguyên ý; lượt này không thêm nội dung plan mới chưa được duyệt.

---

## Parking Lot

- [A1-R1] Tăng `max_features` (40.000 có thể ít với char 3–5), ghép word n-gram 1–2, thử LinearSVC / chỉnh `C`. No evidence yet — so sánh trên 5-fold sau P1.
- [A1-R1] Đặc trưng phong cách (độ dài, "—", gạch đầu dòng, markdown, tỷ lệ ký tự ngoài ASCII 57,84%). No evidence yet.
- [A1-R1] Fine-tune transformer (DeBERTa-v3) trên GPU Kaggle với weights đính kèm; cần xác nhận Rules về pretrained model. No evidence yet; chi phí cao.
- [A1-R1] Đo độ suy giảm của P2 khi A1 sai: mô phỏng trên fold validation bằng cách bỏ ngẫu nhiên 10–30% dòng mỗi prompt rồi chạy lại. Defensive — chỉ cần nếu P2 qua SC3.
- [A1-R1] Đọc lại Rules / Discussion về việc dùng thông tin chéo giữa các dòng test (A3). Bắt buộc trước khi nộp, không chặn việc đo.
- [A1-R1] Phát hiện near-duplicate giữa các prompt khác nhau (chưa kiểm). No evidence yet.
- [A1-R2] Hidden test thiếu hẳn một hoặc vài model (không chỉ thiếu dòng ngẫu nhiên) — chưa đo; ràng buộc sẽ ép nhãn của model vắng mặt lên dòng khác. Defensive — chưa có bằng chứng xảy ra.
- [A1-R2] Thời gian/RAM của gom cụm agglomerative dense trên ~4.000 dòng trong Kaggle — chưa đo; sẽ thấy ở SC6.
- [A1-R3] Làm P2 bền hơn khi số đoạn mỗi model lệch (E9): chọn số cụm bằng ngưỡng khoảng cách thay vì `round(n/18)`, hoặc ràng buộc mềm (phạt trùng nhãn trong cụm thay vì cấm). Chỉ đáng thử nếu người dùng muốn theo hướng `topic`; đánh giá bằng kịch bản lo=0,1/0,3 của `1_r3_imbalance.py`, không phải fold cân bằng.
- [A1-R3] Rules §2.6.b cho phép "external data and models" trừ khi Host cấm — liên quan tới mục transformer ở trên; chưa kiểm phần Host có cấm gì riêng không.

---

## Dispute Log

| Round | Agent | Section | What Changed | Why | Status |
|-------|-------|---------|--------------|-----|--------|
| 1 | [A2-R1] | E3/E5, P1 | Tái lập holdout và đo 5 fold | 157 → 242 đúng trên holdout, tăng ở 5/5 fold | CLOSED |
| 1 | [A2-R1] | E7, SC3 | Bỏ ngưỡng nhiễu 0,04, dùng chênh lệch balanced accuracy theo fold | Hai cách gán ghép cặp, fold 1 có 539 dòng | CLOSED |
| 1 | [A2-R1] | P2, SC4 | Fallback argmax khi TF-IDF rỗng hoặc có hàng 0 | Lỗi cosine tái hiện với “the and” | CLOSED |
| 1 | [A2-R1] | Assumptions | Cấu trúc hidden và điều khoản dùng chung text test | Hai mẫu thiếu dòng vẫn tăng điểm, điều khoản chưa xác nhận | PARKED |
| 2 | [A1-R2] | E3/E5, P1 | Xác minh độc lập dòng 1: 5-fold và holdout khớp từng số (E8) | `1_r2_verify.py` a,b: 157→242; bal 0,299655→0,418225 | CLOSED |
| 2 | [A1-R2] | E7, SC3 | Chấp nhận cổng balanced accuracy theo fold; ghi nhận SC3 đạt trong đo chỉ đọc | 128/43 khớp; gain min +0,085, mean +0,119 | CLOSED |
| 2 | [A1-R2] | P2, SC4 | Đổi fallback toàn tập thành fallback từng dòng cho hàng vector 0; toàn tập chỉ khi từ vựng rỗng, n<18 hoặc k<1 | 1 dòng “the and”: toàn tập 157/540, từng dòng 242/540 | CLOSED |
| 2 | [A1-R2] | P2 | Công tắc: CLI mặc định argmax, notebook `ASSIGNMENT='topic'` | Notebook là nơi duy nhất chạy hidden test | CLOSED |
| 2 | [A1-R2] | SC1 | Sửa trích dẫn 143-155 → 143-147 (149-155 là ghi audit) | Đọc src/baseline.py:139-162 | CLOSED |
| 2 | [A1-R2] | Assumptions | Thiếu dòng ngẫu nhiên: tái lập với rng khác, cùng hướng (+63, +30) | `1_r2_verify.py` phần c | CLOSED |
| 2 | [A2-R2] | P2, SC4 | Chấp nhận fallback từng dòng | Tính lại xác suất sau khi sửa text, 156 → 241 đúng, 540/540 ID khớp khi đảo thứ tự | CLOSED |
| 2 | [A2-R2] | P2 | Chọn phương án argmax mặc định ở cả hai nơi đã được Agent 1 chấp thuận | Test lệch tỷ lệ model, lượt Kaggle 3 dòng chỉ chạy argmax, Rules còn PARKED | CLOSED |
| 3 | [A1-R3] | Assumptions | Xác nhận A1 sai trên hidden test (Overview: số đoạn mỗi model không bằng nhau); thêm E9 | Gain cân bằng +0,119 → lệch +0,026/+0,036, 1/30 lượt âm | CLOSED |
| 3 | [A1-R3] | Assumptions (A3) | Đọc được Rules: chỉ cấm hand labeling/human prediction trên test | Không phải xác nhận rõ ràng; người dùng tự quyết trước khi bật `topic` | PARKED |
| 3 | [A1-R3] | P2, SC4 | Xác minh độc lập phép đo [A2-R2]: 156/241, 540/540 khi đảo thứ tự, 18 dòng stopword = argmax | Chạy hàm `topic` của `1_r3_imbalance.py` | CLOSED |
| 4 | [A2-R4] | Plan #3, Verification 2 | Giữ biến audit và đối chiếu cả split mới với manifest | evaluate chỉ đọc split cũ; mốc trước sửa 2.159/540 | CLOSED |
| 4 | [A2-R4] | Plan #8, SC4–SC5 | Kiểm lệnh cv trên fixture; sửa fixture Hungarian và loại hòa điểm | Fixture 36 dòng cho 17/18 nhãn; fixture 37 dòng khớp 37/37 ID khi đảo thứ tự | CLOSED |
| 4 | [A2-R4] | Verification 5–6 | Kiểm hash v1, so đầu ra Kaggle với input thực tế | Notebook v1 untracked; validator kiểm chính xác số dòng và thứ tự ID | CLOSED |
| 5 | [A1-R5] | Plan #3, Verification 2 | Chấp nhận: giữ `a, b` ở dòng 139, giữ thông báo lỗi, so split mới với manifest | Tự đo: component 2.699/2.699, train 2.159, validation 540/540; `evaluate` không gọi `make_split` | CLOSED |
| 5 | [A1-R5] | Plan #8 | Chấp nhận 8a = test lệnh `cv` trên fixture, 8d = fixture 37 dòng | Tự chạy: fold 36/36/18/36/18 = 144; 37-row đảo thứ tự 37/37; 36-row P đều 2/36 | CLOSED |
| 5 | [A1-R5] | Verification 5–6 | Chấp nhận hash v1 thay git diff; so nhãn Kaggle chỉ khi hash test khớp | `sha256sum` khớp f3ae493a…886f; file untracked | CLOSED |
| 5 | [A1-R5] | Plan | Đặt PLAN_STATUS: CONVERGED | Không còn dòng OPEN; không thêm nội dung chưa duyệt | CLOSED |

**Status values:** `OPEN` = unresolved, needs further debate. `CLOSED` = all agents agree (accepted, conceded, or resolved). `PARKED` = deferred, not blocking convergence.

---

## Implementation Results

**Ngày:** 2026-11-19 · **Người thực hiện:** Agent 1 (Claude) · Theo Plan `PLAN_STATUS: CONVERGED`.

### Files changed
- `src/baseline.py` (+130/−11): tách `connected_groups()` và `check_disjoint()` (chép nguyên văn; `make_split` giữ `a, b` và thông báo lỗi cũ); thêm `_assign_within()`, `assign_by_topic()` (đúng snippet #4); thêm lệnh `cv` (`cross_validate`); `predict` gom xác suất mọi batch, thêm `--assignment {argmax,topic}` mặc định `argmax`, ghi `assignment` vào `.checks.json`.
- `tests/test_baseline.py` (+56): 4 test mới — `cv` trên fixture 8 prompt × 18, n < 18 = argmax, 18 dòng stopword = argmax, fixture 37 dòng (nhãn khác nhau mỗi cụm, 1 nhãn khác argmax, đảo thứ tự khớp theo ID).
- `scripts/build_kaggle_cpu_notebook.py`: biến `ASSIGNMENT = 'argmax'` trong cell chạy, truyền `--assignment` vào `predict`; tham số `--output`.
- `notebooks/kaggle_cpu_standalone_v2.ipynb` (mới, SHA-256 `79bf95c2fcd4971241ece0b4d22fcb8022f8985f1abb8daa333985c7a7b3f59d`); cell nguồn khớp nguyên văn `src/baseline.py`.
- `README-baseline.md`: mục "5-fold CV và chế độ gán nhãn `topic` (debate 1)".

### Commands / tests run
| Verification | Lệnh | Kết quả |
|---|---|---|
| 1 (SC5) | `python -m unittest discover -s tests -v` | 9 test cũ OK sau #2-#3; toàn bộ 13 test OK |
| 2 | so `connected_groups`/`make_split` với `artifacts/baseline/split.csv`; `evaluate ... --output-dir outputs/evaluation_debate1` | component 2.699/2.699, train 2.159/2.159, validation 540/540; `matches_saved_metrics: true` |
| 3 (SC1-SC3) | `python -m src.baseline cv --train train.csv --config configs/baseline.json --output-dir outputs/cv_run1` | 5 fold (539/540/540/540/540), 18/18 lớp hai phía, ba overlap = 0, 18 recall/fold. Argmax 0,299655 ± 0,009488; topic 0,418225; gain min 0,085185, mean 0,118570 — **khớp E8**; `sc3_balanced_folds_pass: true`; 36,7 giây |
| 4 (SC4) | `predict --assignment argmax|topic` trên `test.csv` → `outputs/debate1_predict/`; `validate` cả hai | Hai file cùng nhãn `gemini-3.6-flash, gpt-5.6-terra, gemini-3.7-flash` (giống Kaggle V1); `.checks.json` ghi đúng `assignment` |
| 5 | build v2; `sha256sum` v1 trước/sau; `ast.parse` các cell | v1 vẫn `f3ae493a…886f`; cell parse OK; có `ASSIGNMENT = 'argmax'` |
| 5b | chạy thử notebook v2 local (thay đường dẫn Kaggle bằng repo, `outputs/notebook_v2_smoke`) | `SUCCESS`; Parquet 3 dòng, cùng nhãn |
| 6 (SC6) | Kaggle: người dùng import `kaggle_cpu_standalone_v2.ipynb` vào notebook `khoabi142007/name-that-llm-cpu-tf-idf-baseline`; Agent 1 kiểm tra cell (import mới, `ASSIGNMENT = 'argmax'`, `--assignment`), Input = dữ liệu cuộc thi, Accelerator None, Internet off, Pin to original environment → Save & Run All, Version 2 "v2 debate1 cv+assignment (argmax)" | **Successful**, runtime 56 giây. `submission.checks.json`: 3 dòng, `assignment: argmax`, hash test `4545aa3a…0c57` (khớp V1), hash Parquet `74496ed6…869a` — **trùng V1**. **SC6 đạt.** |
| 7 | Submit to Competition (Version 2, output `submission.parquet`), 2026-11-19, theo yêu cầu người dùng | Trạng thái **Succeeded**, **Public Score 0.300** (balanced accuracy trên public LB). Khớp 5-fold CV argmax 0,2997 ± 0,0095. Private score chưa công bố. |

### Deviations from the plan
- Không có thay đổi thuật toán so với Plan. Thêm bước 5b (chạy thử notebook local) ngoài Plan, chỉ đọc.
- `cv` yêu cầu ≥ 5 nhóm độc lập (thêm `require`), để báo lỗi rõ thay vì exception của scikit-learn.
- Không dán được code vào trình soạn thảo Kaggle từ trình duyệt của app (iframe JupyterLab không nhận clipboard); người dùng tự import notebook v2. Draft trước đó không bị thay đổi.
- Lần mở hộp Submit đầu tiên chọn nhầm `submission.checks.json` (Kaggle báo không tìm thấy file); đã hủy, chọn `submission.parquet` rồi mới nộp. Chỉ có **một** lượt nộp.
