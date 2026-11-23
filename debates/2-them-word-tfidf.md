# Debate: Cải thiện baseline hiện tại — thêm nhánh word TF-IDF giữ dấu câu và chữ hoa

**Created:** 2026-11-19
**Agent 1:** Claude (Opus 5.5)
**Agent 2:** Astra (Codex, gpt-6-astra)
**Max Rounds:** 3
**Status:** CONVERGED

## Context

Sau debate 1 (đã merge vào `main` ở `b9d923a`), baseline là TF-IDF ký tự 3–5 (40.000 đặc trưng) + Logistic Regression `C=1`, gán nhãn bằng argmax. Đo bằng lệnh `cv` (5-fold theo nhóm prompt): balanced accuracy **0,2997 ± 0,0095**. Lượt nộp Kaggle ngày 19/11/2026: **public score 0,300**. CV khớp leaderboard, nên có thể dùng `cv` để chọn cải tiến mà không tốn lượt nộp.

Câu hỏi của debate: **thay đổi nhỏ nhất nào tăng balanced accuracy một cách ổn định trên prompt mới?** Phạm vi: chỉ đặc trưng / bộ phân loại tuyến tính trên CPU. Không bàn transformer, dữ liệu ngoài hay hậu xử lý `topic` (đã chốt ở debate 1).

Trong debate được **chạy script đo chỉ đọc**. Không sửa `src/`, `configs/`, notebook cho tới khi hội tụ.

### Evidence

Tất cả số liệu đo ngày 2026-11-19 bằng `debates/evidence/2_r1_sweep.py`. Script dùng `connected_groups()` (src/baseline.py) và `StratifiedGroupKFold(5, shuffle=True)` giống hệt lệnh `cv`. Mỗi cấu hình chỉ đổi **một** yếu tố so với baseline. `fit` trong `threadpool_limits(1)`, đếm `ConvergenceWarning`. Kết quả tái lập được bằng lệnh ở cuối mỗi bảng.

**E1 — Vòng quét một yếu tố (seed fold 42, cùng 5 fold với `outputs/cv_run1`).** Baseline trong script ra **0,2997**, khớp `cv_run1` (0,299655).

| Cấu hình (một thay đổi) | Bal. acc ± std | Tăng TB | Tăng nhỏ nhất | Fold tốt hơn | Số đặc trưng (fold 1) | Fit fold 1 |
|---|---:|---:|---:|---:|---:|---:|
| + word TF-IDF 1–2 (token giữ dấu câu, chữ hoa) | **0,3338 ± 0,0190** | **+0,0341** | +0,0185 | **5/5** | 80.000 | 19 s |
| `C=10` | 0,3193 ± 0,0106 | +0,0197 | +0,0038 | 5/5 | 40.000 | 48 s |
| `C=3` | 0,3163 ± 0,0127 | +0,0167 | −0,0018 | 4/5 | 40.000 | 20 s |
| LinearSVC `C=0,5` | 0,3067 ± 0,0147 | +0,0070 | −0,0093 | 3/5 | 40.000 | 6 s |
| char 1–5 | 0,3056 ± 0,0099 | +0,0059 | −0,0056 | 4/5 | 40.000 | 15 s |
| **baseline** | 0,2997 ± 0,0095 | 0 | 0 | — | 40.000 | 16 s |
| `char_wb` 2–5 | 0,2923 ± 0,0115 | −0,0074 | −0,0241 | 1/5 | 40.000 | 13 s |
| char `max_features=200000` | 0,2922 ± 0,0084 | −0,0074 | −0,0167 | 1/5 | 116.993 | 20 s |
| char `max_features=None` | 0,2922 ± 0,0084 | −0,0074 | −0,0167 | 1/5 | 116.993 | 20 s |
| `C=0,3` | 0,2759 ± 0,0117 | −0,0237 | −0,0315 | 0/5 | 40.000 | 9 s |

Không có `ConvergenceWarning` ở cấu hình nào. Lệnh: `.venv/Scripts/python.exe debates/evidence/2_r1_sweep.py`.

**E2 — Biến thể của nhánh word (seed fold 42).** Nhánh word dùng `analyzer='word'`, `lowercase=False`, `min_df=2`, `sublinear_tf=True`, `max_features=40000`, `token_pattern=r"(?u)\b\w+\b|[^\w\s]"` (mỗi dấu câu là một token), ghép với nhánh char bằng `FeatureUnion`.

| Cấu hình | Bal. acc ± std | Tăng TB | Tăng nhỏ nhất | Fold tốt hơn | Theo fold |
|---|---:|---:|---:|---:|---|
| char + word 1–2, `C=3` (hai thay đổi) | 0,3419 ± 0,0200 | +0,0423 | +0,0259 | 5/5 | 0,354 / 0,365 / 0,309 / 0,330 / 0,352 |
| **char + word 1–1 (unigram)** | **0,3393 ± 0,0184** | **+0,0397** | +0,0222 | **5/5** | 0,358 / 0,363 / 0,320 / 0,319 / 0,337 |
| char + word 1–2 | 0,3338 ± 0,0190 | +0,0341 | +0,0185 | 5/5 | 0,339 / 0,363 / 0,311 / 0,315 / 0,341 |
| char + word 1–3 | 0,3252 ± 0,0172 | +0,0256 | +0,0148 | 5/5 | 0,332 / 0,348 / 0,298 / 0,315 / 0,333 |
| chỉ word 1–2 (bỏ char) | 0,3223 ± 0,0152 | +0,0227 | +0,0130 | 5/5 | fit 3 s |
| char + word 1–2, **token mặc định** (`\b\w\w+\b`, bỏ dấu câu) | 0,3030 ± 0,0238 | +0,0033 | −0,0185 | 2/5 | |
| baseline | 0,2997 ± 0,0095 | 0 | 0 | — | 0,306 / 0,311 / 0,283 / 0,296 / 0,302 |

Lệnh: `.venv/Scripts/python.exe debates/evidence/2_r1_sweep.py out.json plus_word_1_2 word_only_1_2 plus_word_default_tokens plus_word_1_1 plus_word_1_2_C3 plus_word_1_3`.

~~Nhận xét: gần như toàn bộ mức tăng đến từ **token dấu câu**. Cùng nhánh word nhưng dùng token mặc định chỉ còn +0,0033 và kém baseline ở 3/5 fold. Lưu ý: `lowercase=False` đổi cùng lúc với token pattern trong biến thể này, nên chưa tách được riêng tác động của chữ hoa.~~
[A2-R1] Hai biến thể đều giữ lowercase=False (debates/evidence/2_r1_sweep.py:28-30,46,50). Với chuỗi “A i X AI ai ! ?”, tokenizer đề xuất cho 7 token; tokenizer mặc định cho 2 token “AI”, “ai”. E2 kiểm tác động gộp của dấu câu và từ một ký tự ở nhánh 1–2. Đóng góp riêng của dấu câu, chữ hoa và unigram vẫn chưa đo.
[A1-R2] Chấp nhận, rút lại câu "lowercase=False đổi cùng lúc". Đã kiểm độc lập: (1) `2_r1_sweep.py:29` đặt `'lowercase': False` trong `word()`, còn `:50` chỉ ghi đè `token_pattern=r'(?u)\b\w\w+\b'`, nên hai biến thể đều giữ hoa/thường. (2) `2_r2_checks.py unit`, chuỗi `"A i X AI ai ! ?"` với `lowercase=False`: pattern đề xuất → `[A, i, X, AI, ai, !, ?]` (7 token); `\b\w\w+\b` → `[AI, ai]` (2 token); `\b\w+\b` → `[A, i, X, AI, ai]` (5 token). Vậy hàng "token mặc định" của E2 bỏ cùng lúc dấu câu **và** từ một ký tự, đúng như [A2-R1] nói.

[A2-R1] Tái lập độc lập seed 42 bằng build_pipeline, connected_groups, check_disjoint và fit_checked của src/baseline.py; thay bước đầu trong bộ nhớ bằng FeatureUnion(char, word unigram float32) rồi lấy argmax của predict_proba. Balanced accuracy từng fold: 0,357791 / 0,362963 / 0,320370 / 0,318519 / 0,337037. Trung bình 0,339336 (std 0,018424), so với 0,299655 của cv_run1; cả 5 fold tăng, ít nhất +0,022222. Fold 1 có 40.000 cột char + 11.622 cột word = 51.622. Chấp nhận P1 và lựa chọn unigram cho P2 trên cơ sở phép đo này. Seed 7 và 2026 là kết quả do Agent 1 báo cáo.

**E3 — Kiểm tra thiên lệch do chọn cấu hình.** E1-E2 thử 15 cấu hình trên cùng 5 fold, nên cấu hình tốt nhất có thể tốt nhờ may mắn. Chạy lại trên hai cách chia fold **mới** (seed 7 và 2026, chưa dùng để chọn):

| Seed fold | Baseline | char + word 1–1 | Tăng TB | Tăng nhỏ nhất | Fold tốt hơn | char + word 1–2 | Tăng TB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 (đã dùng để chọn) | 0,2997 | 0,3393 | +0,0397 | +0,0222 | 5/5 | 0,3338 | +0,0341 |
| 7 | 0,3000 | 0,3401 | +0,0400 | +0,0130 | 5/5 | 0,3349 | +0,0348 |
| 2026 | 0,2904 | 0,3319 | +0,0415 | +0,0315 | 5/5 | 0,3312 | +0,0408 |

char + word 1–1 tốt hơn baseline ở **15/15 fold** qua ba cách chia, mức tăng ổn định quanh +0,040. Lệnh: `FOLD_SEED=7 .venv/Scripts/python.exe debates/evidence/2_r1_sweep.py out.json plus_word_1_1 plus_word_1_2` (tương tự với 2026).

**E4 — Tách từng yếu tố trên chính nhánh unigram đề xuất [A1-R2].** Đo ngày 2026-11-20 bằng `debates/evidence/2_r2_checks.py cv`, cùng fold và cùng `threadpool_limits(1)` như E1–E3. Mỗi hàng chỉ đổi **một** tham số so với `uni_proposed` (char + word 1–1, `C=1`). Không có `ConvergenceWarning` ở hàng nào. `uni_proposed` tái lập đúng E3: 0,3393 / 0,3401 / 0,3319.

| Biến thể (một thay đổi so với đề xuất) | Seed 42 | Seed 7 | Seed 2026 | Δ TB so với đề xuất (42 / 7 / 2026) | Fold tốt hơn đề xuất |
|---|---:|---:|---:|---|---:|
| `uni_proposed` (đề xuất) | 0,3393 | 0,3401 | 0,3319 | 0 | — |
| `uni_no_punct`: pattern `\b\w+\b` (bỏ token dấu câu, giữ từ 1 ký tự và hoa/thường) | 0,3160 | 0,3249 | 0,3160 | −0,0233 / −0,0152 / −0,0159 | **1/15** |
| `uni_lowercase`: `lowercase=True` (giữ token dấu câu) | 0,3315 | 0,3416 | 0,3316 | −0,0078 / +0,0015 / −0,0004 | 5/15 |
| `uni_C3`: `C=3` | 0,3393 | 0,3397 | 0,3408 | 0,0000 / −0,0003 / +0,0089 | 9/15 |
| baseline (chỉ char) | 0,2997 | 0,3000 | 0,2904 | −0,0397 / −0,0400 / −0,0415 | 0/15 |

Đọc kết quả: token dấu câu đóng góp khoảng +0,015 đến +0,023 trong tổng mức tăng ~+0,040 (bỏ đi thì kém hơn đề xuất ở 14/15 fold). Phần còn lại (+0,016 đến +0,026, tính bằng `uni_no_punct` − baseline) đến từ chính unigram từ. Chữ hoa/thường không có tác động ổn định (dấu Δ đổi chiều giữa các seed, 5/15 fold). `C=3` trên unigram cũng không ổn định: Δ TB +0,0029, 9/15 fold; fit + predict fold 1 mất 18,1–18,5 s so với 14,0–15,0 s. Lệnh: `FOLD_SEED=<42|7|2026> .venv/Scripts/python.exe debates/evidence/2_r2_checks.py cv`.

[A2-R2] Đối chiếu thiết kế E4 với `debates/evidence/2_r2_checks.py:36-44`: ba biến thể unigram lần lượt chỉ đổi pattern, lowercase hoặc C. Chạy lại `unit` cho 7 token với pattern đề xuất và 5 token khi chỉ bỏ dấu câu, trong đó cả hai đều giữ A, i, X, AI, ai. Từ các điểm trung bình E4 đã ghi, lợi thế khi giữ dấu câu là 0,0233 / 0,0152 / 0,0159. Đây là chênh lệch điểm khi đổi tokenizer trên bộ dữ liệu này. Lượt này kiểm thiết kế và phép trừ; số đếm theo 15 fold vẫn lấy từ lần chạy CV của A1.

### Relevant Files
- `src/baseline.py:180-185` — `build_pipeline(config)`: hiện chỉ có một `TfidfVectorizer` tên bước `'tfidf'`.
- `src/baseline.py:319-320` — `train()` ghi `validation_feature_count` / `full_feature_count` bằng `model.named_steps['tfidf'].vocabulary_`; sẽ hỏng nếu bước `'tfidf'` thành `FeatureUnion`.
- `configs/baseline.json` — cấu hình hiện tại, được `cv`/`train` dùng mặc định (src/baseline.py:452, 471) và được nhúng vào notebook (scripts/build_kaggle_cpu_notebook.py:26).
- `debates/evidence/2_r1_sweep.py` — script đo E1-E3.
- `debates/evidence/2_r2_checks.py` — [A1-R2] script chỉ đọc: `unit` (tokenizer, lỗi `ngram_range` dạng list, dtype của `FeatureUnion`, tổng số cột) và `cv` (E4).
- `debates/1-huong-cai-tien-mo-hinh.md` — debate trước, định nghĩa `cv` và quy tắc nộp bài.

### Constraints
- CPU, Internet OFF, không dữ liệu ngoài, không model tải sẵn.
- Không dùng public leaderboard để chọn cấu hình; chỉ dùng để quan sát sau khi chốt.
- Giữ tái lập được baseline cũ: `cv` với cấu hình cũ phải vẫn ra 0,299655.
- Đầu ra giữ schema `id, model`. Không tự nộp bài.

---

## Assumptions

- [A1-R1] Assumption: Mức tăng của nhánh word không đến từ rò rỉ chủ đề giữa train và validation. Basis: `cv` tách hoàn toàn theo nhóm prompt + text trùng (`prompt_overlap = exact_text_overlap = normalized_text_overlap = 0` ở mọi fold, `outputs/cv_run1/cv_metrics.json`); test cũng là prompt mới.
- [A1-R1] Assumption: ~~Tín hiệu chính là phong cách dùng dấu câu, không phải từ khóa chủ đề.~~ Basis: E2 — bỏ token dấu câu thì mức tăng gần như mất (+0,0341 → +0,0033). Chưa phân tích trọng số đặc trưng để xác nhận trực tiếp.
[A2-R1] Chỉ chấp nhận mức tăng của cấu hình hoàn chỉnh. Chưa có phép so sánh giữ từ một ký tự và chỉ bỏ dấu câu. Giữ giả thuyết cơ chế trong Parking Lot.
[A1-R2] Đã đo đúng phép so sánh [A2-R1] yêu cầu (E4, `uni_no_punct`: giữ từ một ký tự và hoa/thường, chỉ bỏ dấu câu). Kết quả: dấu câu giải thích khoảng một nửa mức tăng (−0,0152 đến −0,0233 khi bỏ, 14/15 fold kém hơn đề xuất), nửa còn lại đến từ unigram từ. Câu gạch ngang "tín hiệu chính là dấu câu" vẫn bị gạch, vì chỉ đúng một nửa. Giả thuyết này không là điều kiện cho đề xuất: đề xuất chỉ dựa vào mức tăng của cấu hình hoàn chỉnh (E2, E3).
- [A1-R1] Assumption: Thiên lệch do chọn cấu hình là nhỏ. Basis: E3 — trên hai cách chia fold không dùng để chọn, mức tăng vẫn +0,040 và +0,0415, 10/10 fold tốt hơn.
[A2-R1] Seed 7 và 2026 đổi cách chia của cùng 2.699 dòng và cùng nhóm prompt (script:74-76). E3 cho biết độ nhạy với cách chia; độ chính xác trên tập prompt thu thập độc lập vẫn chưa đo.
- [A1-R1] Assumption: Thời gian chạy trên Kaggle vẫn chấp nhận được. Basis: fit fold 1 tăng từ 16 s lên 20 s (E1/E2, một luồng); notebook v2 hiện chạy 56 s trên Kaggle. Chưa đo trên Kaggle.
- [A1-R1] Assumption: Tỷ lệ model không đều trên hidden test (Overview Kaggle, debate 1 E9) không làm mất mức tăng, vì metric là balanced accuracy và mô hình vẫn dự đoán từng dòng độc lập (argmax). Chưa mô phỏng riêng.

---

## Success Criteria

- [x] SC1 — Cấu hình mới đạt trên 5-fold seed 42: `python -m src.baseline cv --train train.csv --config configs/char_word.json --output-dir outputs/cv_char_word` → `argmax.balanced_accuracy_mean` = **0,3393** (±0,0001 so với E2), tốt hơn baseline ở 5/5 fold, mức tăng nhỏ nhất > 0.
[A2-R1] Đường dẫn trường trong cv_metrics.json là summary.argmax.balanced_accuracy_mean (src/baseline.py:431-441); tính mức tăng từng fold bằng cách ghép theo fold với outputs/cv_run1/cv_metrics.json. Hai mốc SC2 trong file hiện tại là 0,2996551724 và 0,4182247765.
- [x] SC2 — Baseline cũ không đổi: `python -m src.baseline cv --train train.csv --config configs/baseline.json --output-dir outputs/cv_baseline_recheck` → argmax mean **0,299655**, topic mean **0,418225** (giống `outputs/cv_run1`).
- [x] SC3 — `train` + `evaluate` chạy được với cấu hình mới; `metadata.json` ghi số đặc trưng của từng nhánh; `evaluate` trả `matches_saved_metrics: true`.
[A2-R1] Giữ validation_feature_count và full_feature_count là số nguyên như src/baseline.py:319-320. Khi có word_tfidf, ghi thêm số cột char/word cho từng lần fit, kiểm tổng bằng số cột classifier.coef_ ở đúng model. evaluate kiểm điểm lưu nhưng không kiểm metadata đặc trưng (src/baseline.py:383-399).
[A1-R2] Chấp nhận. Đã kiểm `src/baseline.py:319-320`: hai trường là `len(...vocabulary_)` (số nguyên), và `evaluate` (`:383-399`) chỉ so `metrics == meta['metrics']`, không đọc trường đặc trưng nào. Vì vậy phép kiểm tổng cột phải là một bước xác minh riêng trong SC3: nạp `validation_pipeline.joblib` và `pipeline.joblib`, so tổng số cột char + word ghi trong metadata với `named_steps['classifier'].coef_.shape[1]` của đúng model đó. Không cần thêm `require` lúc chạy: bước kiểm này đủ để xác nhận.
[A2-R2] Đã đối chiếu SC1–SC3. Báo cáo gốc `outputs/cv_run1/cv_metrics.json:296,300` ghi 0,2996551724 (argmax) và 0,4182247765 (topic); cả ba bộ đếm overlap bằng 0 ở cả 5 fold. Chạy lại `2_r2_checks.py unit` cho 86 cột char + 9 cột word = 95 cột hệ số; union không có `vocabulary_`, trong khi `src/baseline.py:319-320` đang đọc thuộc tính này ở cả hai model. Chấp nhận phép đối chiếu metadata riêng cho mỗi model ở SC3. Đây là phép thử trên 4 văn bản của script; SC3 vẫn chờ hai artifact thực tế.
- [x] SC4 — Toàn bộ test pass: `python -m unittest discover -s tests -v` → OK, có thêm test cho pipeline hai nhánh.
- [ ] SC5 — Notebook mới chạy Save & Run All thành công trên Kaggle (CPU, Internet OFF), `submission.parquet` hợp lệ — người dùng xác nhận. Điểm public chỉ ghi nhận, không dùng làm cổng.

---

## Proposal

STATUS: CONVERGED

**Tóm tắt [A1-R1]:** Thêm **một** nhánh word TF-IDF unigram (giữ chữ hoa, mỗi dấu câu là một token) song song với nhánh char hiện tại, ghép bằng `FeatureUnion`. Giữ nguyên Logistic Regression `C=1`. Đây là thay đổi đơn lẻ có mức tăng lớn nhất và ổn định nhất đã đo: +0,040 balanced accuracy, 15/15 fold tốt hơn trên ba cách chia (E2, E3). Không đổi `C` cùng lúc: ~~`C=3` chỉ thêm +0,003 so với word 1–1 (E2) nhưng gấp đôi thời gian fit~~, và chưa kiểm tra trên fold mới.
[A2-R1] Giữ C=1. Hai cấu hình ở script:51-52 khác cả C lẫn ngram_range; hiệu số 0,3419 − 0,3393 = 0,0026 không ước lượng riêng tác động C trên unigram. Cùng bigram, chênh lệch E2 là 0,3419 − 0,3338 = 0,0081. Thời gian script:61-67 còn bao gồm predict. Chưa cần thêm lượt quét để triển khai ứng viên đã chọn.
[A1-R2] Chấp nhận, giữ phần gạch ngang. Đã kiểm: `2_r1_sweep.py:51` là `ngram_range=(1, 1)` + `lr()` (C=1), còn `:52` là `ngram_range` mặc định `(1, 2)` (`:29`) + `lr(C=3.0)`. Hiệu 0,3419 − 0,3393 = 0,0026 trộn hai yếu tố, nên không quy được cho `C`. So cùng bigram: 0,3419 − 0,3338 = +0,0081. `start` ở `:61` đo đến sau `predict` ở `:65`, nên cột "Fit" trong E1 là fit + predict, và tôi chưa từng đo thời gian của `plus_word_1_2_C3`. Câu "gấp đôi thời gian fit" không có số liệu chứng minh. Quyết định giữ `C=1` không đổi.

### P1 — Cấu hình mới, giữ cấu hình cũ [A1-R1]

- **Hiện tại:** `configs/baseline.json` chỉ có khóa `tfidf` (char).
- **Đề xuất:** Tạo `configs/char_word.json` = nội dung `configs/baseline.json` + khóa mới:

```json
"word_tfidf": {
  "analyzer": "word",
  "ngram_range": [1, 1],
  "lowercase": false,
  "min_df": 2,
  "max_features": 40000,
  "sublinear_tf": true,
  "token_pattern": "(?u)\\b\\w+\\b|[^\\w\\s]"
}
```

`configs/baseline.json` giữ nguyên để SC2 tái lập được kết quả cũ.

### P2 — `build_pipeline` hỗ trợ nhánh word tùy chọn [A1-R1]

- **Hiện tại:** `src/baseline.py:180-185` dựng `Pipeline([('tfidf', TfidfVectorizer(...)), ('classifier', LogisticRegression(...))])`.
- **Đề xuất:** Nếu config có `word_tfidf`, bước `'tfidf'` là `FeatureUnion([('char', TfidfVectorizer(**tfidf)), ('word', TfidfVectorizer(**word_tfidf))])`; nếu không, giữ đúng như hiện tại (để pipeline cũ, artifact cũ và SC2 không đổi).
[A2-R1] Chuyển ngram_range từ list sang tuple cho cả hai vectorizer và truyền dtype=np.float32 cho từng nhánh như script:24-30 và src/baseline.py:182-183. Trên môi trường dự án, fit với [1,1] cho InvalidParameterError: ngram_range phải là tuple. SC4 nên fit pipeline tạo từ JSON và kiểm tra kiểu ma trận ghép, cùng số cột bằng tổng kích thước hai vocabulary.
[A1-R2] Chấp nhận. Kiểm trên sklearn 1.7.2 của dự án bằng `2_r2_checks.py unit`: (1) `TfidfVectorizer(ngram_range=[1, 1]).fit(...)` → `InvalidParameterError: The 'ngram_range' parameter of TfidfVectorizer must be an instance of 'tuple'. Got [1, 1] instead.` Hiện `src/baseline.py:182` chỉ chuyển tuple cho `tfidf`, nên `word_tfidf` cũng phải chuyển. (2) dtype của `FeatureUnion.fit_transform`: không truyền dtype → `float64`; cả hai nhánh `float32` → `float32`; **chỉ nhánh char `float32`, nhánh word mặc định → `float64`** (hstack nâng kiểu). Vì vậy phải truyền `dtype=np.float32` cho **cả hai** nhánh, không chỉ giữ nguyên dòng `:183`. (3) Pipeline `('tfidf', FeatureUnion)` đã fit: vocab char 86 + vocab word 9 = 95 = `coef_.shape[1]`. `FeatureUnion` không có thuộc tính `vocabulary_` (`hasattr` → `False`), xác nhận `src/baseline.py:319-320` sẽ lỗi nếu không sửa.
- **Kéo theo:** `src/baseline.py:319-320` đang đọc `model.named_steps['tfidf'].vocabulary_`. Thêm hàm nhỏ `feature_count(model)` trả tổng số đặc trưng (và theo từng nhánh nếu là `FeatureUnion`), dùng ở hai dòng đó.

### P3 — Notebook dùng cấu hình mới [A1-R1]

- **Hiện tại:** `scripts/build_kaggle_cpu_notebook.py:26` đọc cố định `configs/baseline.json`.
- **Đề xuất:** Thêm tham số `--config` (mặc định `configs/baseline.json`), build `notebooks/kaggle_cpu_standalone_v3.ipynb` với `--config configs/char_word.json`. Notebook v1, v2 giữ nguyên.

### Không đề xuất ở vòng này [A1-R1]

Tăng `C`, word 1–2, LinearSVC, đặc trưng phong cách thủ công, transformer → Parking Lot.

---

## Plan

PLAN_STATUS: CONVERGED

[A2-R3] Đã duyệt kế hoạch và đối chiếu với bản triển khai. Các bước xác minh bao phủ SC1–SC5. Bốn tiêu chí đầu đạt tại máy cục bộ; SC5 chờ kết quả chạy trên Kaggle.

[A1-R3] Ngày 2026-11-23, người dùng chọn "Viết Plan và code luôn": Plan này được triển khai ngay mà **chưa qua Astra duyệt**. Astra review Plan và code sau, trong cùng file này.

### Changes

| # | File | What to change | Why |
|---|------|----------------|-----|
| 1 [A1-R3] | `configs/char_word.json` (mới) | Sao nguyên `configs/baseline.json` và thêm khóa `word_tfidf` đúng như P1. `configs/baseline.json` không đổi (SHA-256 trước sửa `9fb7b4ef…98b8ed`). | P1, SC2 |
| 2 [A1-R3] | `src/baseline.py:29` | Thêm import `FeatureUnion` từ `sklearn.pipeline`. | P2 |
| 3 [A1-R3] | `src/baseline.py:180-185` (`build_pipeline`) | Thêm hàm nhỏ `_tfidf(params)`: sao dict, chuyển `ngram_range` sang tuple, trả `TfidfVectorizer(**params, dtype=np.float32)`. Nếu không có `word_tfidf`, bước `'tfidf'` vẫn là `_tfidf(config['tfidf'])`, tương đương code cũ. Nếu có, bước `'tfidf'` là `FeatureUnion([('char', _tfidf(config['tfidf'])), ('word', _tfidf(config['word_tfidf']))])`. Tên bước `'tfidf'` và `'classifier'` giữ nguyên. | P2; tuple và float32 ở **cả hai** nhánh ([A1-R2] `unit`) |
| 4 [A1-R3] | `src/baseline.py` (thêm sau `build_pipeline`) + `:319-320` | Thêm `feature_counts(model)`: trả `{tên nhánh: len(vocabulary_)}` cho `FeatureUnion`, hoặc `{'char': len(vocabulary_)}` cho vectorizer đơn. `validation_feature_count` / `full_feature_count` vẫn là số nguyên (tổng các nhánh); thêm `validation_feature_counts` / `full_feature_counts` (dict theo nhánh). | P2 "kéo theo", SC3 |
| 5 [A1-R3] | `tests/test_baseline.py` | Thêm 1 test: `build_pipeline` từ `configs/char_word.json` fit trên 4 văn bản, rồi kiểm (a) bước `tfidf` là `FeatureUnion`; (b) `transform` ra `float32`; (c) số cột = tổng `feature_counts` = `coef_.shape[1]`; (d) có token dấu câu `!` trong vocab nhánh word. Thêm kiểm `build_pipeline(baseline.json)` vẫn là `TfidfVectorizer` và `feature_counts` chỉ có khóa `char`. | SC4 |
| 6 [A1-R3] | `scripts/build_kaggle_cpu_notebook.py:26,69,72,81-84` | Thêm tham số `--config` (mặc định `configs/baseline.json`) và đọc config từ đó. Câu mô tả trong markdown ghi đúng tên file config và loại đặc trưng (char, hoặc char + word nếu có `word_tfidf`). Build `notebooks/kaggle_cpu_standalone_v3.ipynb` với `--config configs/char_word.json`. Notebook v1/v2 không đổi (kiểm bằng SHA-256). | P3, SC5 |

### Order of Operations
1. [A1-R3] #1–#4 → chạy test cũ (cần pass để chứng minh baseline không đổi).
2. [A1-R3] #5 → chạy toàn bộ suite (SC4).
3. [A1-R3] Chạy song song SC1, SC2 (`cv`) và SC3 (`train` + `evaluate`) trên `train.csv`.
4. [A1-R3] #6 → build notebook v3, kiểm hash v1/v2.
5. [A1-R3] Ghi `## Implementation Results`. Không commit nếu người dùng chưa yêu cầu.

### Verification
1. [A1-R3] SC4: `python -m unittest discover -s tests -v` → OK, có test mới ở #5.
2. [A1-R3] SC2: `cv --config configs/baseline.json --output-dir outputs/cv_baseline_recheck` → `summary.argmax.balanced_accuracy_mean` = 0,2996551724, topic = 0,4182247765.
3. [A1-R3] SC1: `cv --config configs/char_word.json --output-dir outputs/cv_char_word` → argmax mean = 0,3393 (±0,0001), theo fold 0,357791 / 0,362963 / 0,320370 / 0,318519 / 0,337037 như [A2-R1]. Ghép theo fold với `outputs/cv_run1`: 5/5 fold tăng, mức tăng nhỏ nhất > 0. Ba số overlap bằng 0.
4. [A1-R3] SC3: `train --config configs/char_word.json --artifact-dir artifacts/char_word --output-dir outputs/char_word` rồi `evaluate` → `matches_saved_metrics: true`. Với mỗi file `validation_pipeline.joblib` / `pipeline.joblib`, `sum(*_feature_counts)` = `*_feature_count` = `coef_.shape[1]`.
5. [A1-R3] P3: notebook v3 parse được (`ast.parse` từng cell code), chứa `word_tfidf` và `FeatureUnion`; SHA-256 của v1 `f3ae493a…2886f` và v2 `79bf95c2…b3f59d` không đổi. SC5 (chạy trên Kaggle) do người dùng thực hiện.

---

## Parking Lot

- [A1-R1] Tăng `C` (E1: `C=10` +0,0197 nhưng fit 48 s; E2: char+word 1–2 với `C=3` +0,0423). Thử sau khi chốt nhánh word, trên fold mới (seed 7, 2026) để tránh thiên lệch chọn cấu hình.
[A1-R2] Đã đo `C=3` trên nhánh unigram, ba seed (E4): Δ TB +0,0000 / −0,0003 / +0,0089, 9/15 fold. Chưa đủ ổn định để đổi `C=1`; nếu thử tiếp thì quét `C` trên mô hình đã triển khai.
- [A1-R1] Tách riêng tác động của `lowercase=False` và token dấu câu trong nhánh word (E2 đổi hai yếu tố cùng lúc ở biến thể token mặc định).
[A2-R1] PARKED: nếu cần giải thích cơ chế, so hai pattern unigram chỉ khác dấu câu, giữ nguyên lowercase và cách xử lý từ một ký tự. Chỉ đổi pattern ở script:50, lowercase vẫn False.
[A1-R2] Đã đo (E4). Bỏ dấu câu: −0,015 đến −0,023, 14/15 fold kém hơn. Đổi sang `lowercase=True`: −0,0078 / +0,0015 / −0,0004, 5/15 fold, không có tác động ổn định nên giữ `lowercase=False` như đề xuất (khớp nhánh char, `configs/baseline.json`). Mục này không còn việc gì cần làm.
- [A1-R1] Tăng `max_features` của nhánh char làm **giảm** điểm (E1: −0,0074, 1/5 fold tốt hơn). Chưa hiểu nguyên nhân; có thể do `min_df=2` giữ nhiều cụm hiếm gắn với chủ đề.
- [A1-R1] Xem chế độ `topic` (debate 1) có còn mức tăng như cũ với mô hình mới không. Mặc định vẫn argmax.
- [A1-R1] Phân tích recall từng lớp (đặc biệt `glm-5.3`, `claude-fable-5`) với mô hình mới; chưa đo.
- [A1-R1] LinearSVC, đặc trưng phong cách thủ công, transformer — chưa có bằng chứng hoặc ngoài phạm vi CPU nhẹ.

---

## Dispute Log

| Round | Agent | Section | What Changed | Why | Status |
|-------|-------|---------|--------------|-----|--------|
| 1 | [A2-R1] | P1, E2, SC1 | Tái lập độc lập ứng viên unigram, seed 42 | 0,339336; 5/5 fold tăng, gain min 0,022222 | CLOSED |
| 1 | [A2-R1] | E2, Assumptions | Sửa diễn giải phép so tokenizer; tách cơ chế để nghiên cứu sau | Kiểm trực tiếp 7 token so với 2, cả hai giữ hoa/thường | CLOSED |
| 1 | [A2-R1] | Tóm tắt | Giữ C=1, sửa diễn giải mức tăng của C | Hai hàng chênh 0,0026 cũng khác bậc n-gram | CLOSED |
| 1 | [A2-R1] | P2, SC3–4 | Chuẩn hóa tuple/float32, kiểm tổng số cột của từng model đã fit | List [1,1] lỗi khi fit; hai trường metadata hiện là số nguyên | CLOSED |
| 2 | [A1-R2] | E2, Assumptions | Chấp nhận, rút câu "lowercase đổi cùng lúc" | `2_r1_sweep.py:29,50` chỉ đổi pattern; `unit`: 7 / 2 / 5 token | CLOSED |
| 2 | [A1-R2] | Tóm tắt | Chấp nhận, giữ `C=1` | `:51` (1,1) C=1 so với `:52` (1,2) C=3; cùng bigram +0,0081; E4 unigram C=3: +0,0029, 9/15 fold | CLOSED |
| 2 | [A1-R2] | P2, SC3–4 | Chấp nhận; float32 phải đặt ở **cả hai** nhánh; kiểm tổng cột là bước xác minh SC3, không thêm `require` lúc chạy | `unit`: InvalidParameterError với [1,1]; char f32 + word mặc định → float64; 86 + 9 = 95 = `coef_.shape[1]` | CLOSED |
| 2 | [A1-R2] | Evidence E4, Parking Lot | Thêm phép tách dấu câu / chữ hoa / `C` trên unigram, 3 seed; không đổi đề xuất | Bỏ dấu câu −0,015 đến −0,023 (14/15 fold kém); lowercase ±0,008, 5/15 fold; C=3 9/15 fold | CLOSED |
| 2 | [A2-R2] | E4, P2, SC3 | Xác nhận phép so tokenizer và phép kiểm số cột; giữ kết luận hội tụ | Chạy lại unit: 7/5 token khi giữ/bỏ dấu câu; list [1,1] lỗi khi fit; hai nhánh float32 cho float32, chỉ char float32 cho float64; 86 + 9 = 95 | CLOSED |
| 3 | [A1-R3] | Plan, Implementation | Viết Plan (PLAN_STATUS: OPEN) và triển khai ngay theo lựa chọn của người dùng | SC1 0,339336 (5/5 fold tăng, min +0,022222); SC2 0,299655 / 0,418225; SC3 tổng cột = `coef_` ở cả hai model; 14/14 test OK | CLOSED |
| 3 | [A2-R3] | Plan, Implementation | Duyệt P1–P3 và đóng mục review | Đối chiếu 5 fold với baseline, nạp lại 2 model, chạy lại evaluate và 14 test; SC5 chờ Kaggle | CLOSED |

**Status values:** `OPEN` = unresolved, needs further debate. `CLOSED` = all agents agree (accepted, conceded, or resolved). `PARKED` = deferred, not blocking convergence.

---

## Implementation Results

**Ngày:** 2026-11-23 · **Người thực hiện:** Agent 1 (Claude) · [A1-R3] Theo Plan ở trên với `PLAN_STATUS: OPEN`: người dùng chọn triển khai trước khi Astra duyệt Plan. Chưa commit.

### Files changed
- `configs/char_word.json` (mới): giống hệt `configs/baseline.json` cộng khóa `word_tfidf` của P1 (đã kiểm: bỏ `word_tfidf` thì hai dict bằng nhau). `configs/baseline.json` không đổi (SHA-256 `9fb7b4ef…98b8ed`).
- `src/baseline.py`: import `FeatureUnion`; thêm `_tfidf(params)` (chuyển tuple + `dtype=np.float32`); `build_pipeline` dựng `FeatureUnion([('char', …), ('word', …)])` khi có `word_tfidf`, không có thì vẫn là `TfidfVectorizer` như cũ; thêm `feature_counts(model)`; `train()` ghi `*_feature_count` (số nguyên, tổng) và `*_feature_counts` (dict theo nhánh).
- `tests/test_baseline.py`: thêm `test_char_word_config_builds_float32_union_with_punctuation_tokens`.
- `scripts/build_kaggle_cpu_notebook.py`: thêm `--config` (mặc định `configs/baseline.json`); markdown ghi tên config và loại đặc trưng.
- `notebooks/kaggle_cpu_standalone_v3.ipynb` (mới), build bằng `--config configs/char_word.json`, SHA-256 `94c6f17f…86cbc`.

### Commands / results
| Kiểm tra | Lệnh | Kết quả |
|---|---|---|
| SC4 | `python -m unittest discover -s tests -v` | **14/14 OK** (13 test cũ đã pass trước khi thêm test mới) |
| SC2 | `cv --config configs/baseline.json --output-dir outputs/cv_baseline_recheck` | argmax **0,2996551724**, topic **0,4182247765**; khối `argmax` của từng fold giống hệt `outputs/cv_run1` |
| SC1 | `cv --config configs/char_word.json --output-dir outputs/cv_char_word` | argmax **0,3393358876** ± 0,0184; theo fold 0,357791 / 0,362963 / 0,320370 / 0,318519 / 0,337037, khớp [A2-R1] tới 6 chữ số; tăng so với `cv_run1` +0,052107 / +0,051852 / +0,037037 / +0,022222 / +0,035185 (5/5, min +0,022222); overlap (0,0,0) ở mọi fold. Topic: 0,4445 (cũ 0,4182), chỉ ghi nhận, mặc định vẫn argmax |
| SC3 | `train --config configs/char_word.json --artifact-dir artifacts/char_word --output-dir outputs/char_word` rồi `evaluate … --output-dir outputs/evaluation_char_word` | holdout balanced accuracy **0,3444** (baseline `artifacts/baseline`: 0,2907); `matches_saved_metrics: true`. Metadata: validation 40.000 + 11.670 = **51.670** = `coef_.shape[1]` của `validation_pipeline.joblib`; full 40.000 + 12.960 = **52.960** = `coef_.shape[1]` của `pipeline.joblib` |
| P3 | `python scripts/build_kaggle_cpu_notebook.py --config configs/char_word.json --output notebooks/kaggle_cpu_standalone_v3.ipynb` | 2 cell code `ast.parse` OK; `CONFIG` nhúng trong notebook bằng `configs/char_word.json`; có `FeatureUnion`, `feature_counts`. SHA-256 v1 `f3ae493a…2886f`, v2 `79bf95c2…b3f59d` không đổi |
| SC5 | Kaggle Save & Run All notebook v3 | [A1-R3] **Đạt** (người dùng chạy, Agent 1 đọc log Kaggle ngày 2026-11-23): "Successfully ran in 77.5s", CPU, sklearn 1.6.1 (local 1.7.2). Holdout `tfidf_logistic` 0,3444444444 khớp local tới mọi chữ số; `matches_saved_metrics: true`; `submission.parquet` 3 dòng, `valid_labels: true`, `assignment: argmax`. Version này **chưa được nộp**: trang Submissions chỉ có v2 (public 0,300) |
| Public LB (chỉ ghi nhận, không dùng để chọn) | Người dùng nộp "Version 3" ngày 2026-11-23 | [A1-R3] **Public score 0,339** (v2: 0,300), +0,039, khớp CV 0,3393 và mức tăng CV +0,040 |

**Kiểm tra độc lập [A2-R3]**

- SC1–SC2. Ghép theo số fold trong ba báo cáo `cv_run1`, `cv_baseline_recheck`, `cv_char_word`. Trên seed 42, balanced accuracy tăng từ 0,299655 lên 0,339336. Năm mức tăng lần lượt là 0,052107 / 0,051852 / 0,037037 / 0,022222 / 0,035185. Cả ba chỉ số overlap đều bằng 0 ở mỗi fold. Báo cáo baseline chạy lại khớp từng fold cho cả argmax và topic. Lượt duyệt này đối chiếu báo cáo đã lưu.
- SC3–SC4. Nạp hai model trong `artifacts/char_word` và đếm trực tiếp vocabulary. Model validation có 40.000 + 11.670 = 51.670 cột, model full có 40.000 + 12.960 = 52.960 cột. Cả hai khớp metadata và chiều hệ số. Chạy lại `evaluate` cho balanced accuracy 0,344444 và `matches_saved_metrics: true`. Chạy lại toàn bộ suite được 14/14 OK.
- P3 và SC5. Hai cell code của v3 đều phân tích cú pháp được. Cell nguồn khớp nội dung `src/baseline.py` trước điểm gọi CLI; biến `CONFIG` khớp `configs/char_word.json`. Hash v1/v2 khớp mốc trong Plan. SC5 vẫn chờ xác nhận Save & Run All trên Kaggle và kiểm tra tệp submission tại đó.

### Thời gian
`train` với char_word: fit validation 45,6 s, fit full 66,7 s, tổng 123,3 s (baseline: 29,9 / 29,1 / 67,6 s). Lưu ý: SC1, SC2, SC3 chạy song song ba tiến trình trên cùng máy, nên thời gian bị thổi phồng; trong `cv`, fit mỗi fold của char_word là 42–55 s so với 37–43 s của baseline chạy cùng lúc.

### Deviations from the plan
- Không có sai lệch về nội dung. `*_feature_counts` được ghi cho cả cấu hình baseline (dạng `{'char': N}`) thay vì chỉ khi có `word_tfidf`, để không cần thêm nhánh điều kiện; trường số nguyên cũ giữ nguyên ý nghĩa.
