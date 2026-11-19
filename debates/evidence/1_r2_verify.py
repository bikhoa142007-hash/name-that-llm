"""Read-only R2 verification for debate 1 (Agent 1). Writes nothing to the repo.
Usage (repo root): .venv/Scripts/python.exe debates/evidence/1_r2_verify.py
Checks: (a) paired flips on the saved holdout, (b) independent 5-fold argmax vs topic,
(c) random row dropping, (d) zero-vector text: whole-set vs per-row fallback."""
import json, math, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, joblib
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path.cwd()))
from src.baseline import read_input, make_split, build_pipeline, fit_checked  # noqa: E402

config = json.loads(Path('configs/baseline.json').read_text())
frame, _ = read_input('train.csv', labeled=True)
_, _, groups, _ = make_split(frame, config)


def assign(logP, classes, labels):
    pred = np.empty(len(logP), dtype=object)
    for g in np.unique(labels):
        idx = np.where(labels == g)[0]
        reps = math.ceil(len(idx) / len(classes))
        r, c = linear_sum_assignment(-np.tile(logP[idx], (1, reps)))
        pred[idx[r]] = np.asarray(classes)[c % len(classes)]
    return pred


def topic(texts, P, classes, per_row_fallback):
    """Per-row fallback: rows with a zero topic vector keep argmax; others are clustered with k=round(m/18)."""
    argmax = np.asarray(classes)[P.argmax(1)]
    if len(texts) < len(classes):
        return argmax
    try:
        X = TfidfVectorizer(analyzer='word', stop_words='english', min_df=2,
                            sublinear_tf=True).fit_transform(texts).toarray()
    except ValueError:  # empty vocabulary
        return argmax
    ok = np.linalg.norm(X, axis=1) > 0
    if not per_row_fallback and not ok.all():
        return argmax
    m = int(ok.sum())
    k = round(m / len(classes))
    if k < 1:
        return argmax
    out = argmax.copy()
    lab = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average').fit_predict(X[ok])
    out[ok] = assign(np.log(np.clip(P[ok], 1e-12, 1)), classes, lab)
    return out


res = {}
# (a) paired flips on the saved holdout
split = pd.read_csv('artifacts/baseline/split.csv', dtype=str, keep_default_na=False)
val = frame[frame.id.isin(set(split.loc[split.split == 'validation', 'id']))].reset_index(drop=True)
pipe = joblib.load('artifacts/baseline/validation_pipeline.joblib')
P = pipe.predict_proba(val.text.tolist()); cls = list(pipe.classes_); y = val.model.to_numpy()
a = np.asarray(cls)[P.argmax(1)]; t = topic(val.text.tolist(), P, cls, True)
res['holdout'] = {'argmax_correct': int((a == y).sum()), 'topic_correct': int((t == y).sum()),
                  'wrong_to_right': int(((a != y) & (t == y)).sum()), 'right_to_wrong': int(((a == y) & (t != y)).sum()),
                  'k_estimated': round(len(val) / 18)}

# (d) zero-vector text: replace one holdout text with a stopword-only string
z = val.text.tolist(); z[0] = 'the and'
whole = topic(z, P, cls, False); per_row = topic(z, P, cls, True)
res['zero_vector_one_row'] = {'whole_set_fallback_correct': int((whole == y).sum()),
                              'per_row_fallback_correct': int((per_row == y).sum())}

# (c) random row dropping on holdout (independent rng draw, not Astra's exact sample)
rng = np.random.default_rng(7)
for keep in (16, 13):
    idx = np.concatenate([rng.choice(np.where(val.prompt_id == p)[0], size=min(keep, (val.prompt_id == p).sum()), replace=False)
                          for p in sorted(val.prompt_id.unique())])
    aa = np.asarray(cls)[P[idx].argmax(1)]; tt = topic(val.text.iloc[idx].tolist(), P[idx], cls, True)
    res[f'drop_keep{keep}'] = {'n': len(idx), 'k_est': round(len(idx) / 18),
                               'argmax_correct': int((aa == y[idx]).sum()), 'topic_correct': int((tt == y[idx]).sum())}

# (b) independent 5-fold
folds = []
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config['seed'])
for i, (tr, va) in enumerate(cv.split(frame.text, frame.model, groups), 1):
    tr_df, va_df = frame.iloc[tr], frame.iloc[va]
    assert not set(tr_df.prompt_id) & set(va_df.prompt_id) and not set(tr_df.text) & set(va_df.text)
    model = build_pipeline(config)
    with warnings.catch_warnings():
        fit_checked(model, tr_df.text, tr_df.model)
    Pv = model.predict_proba(va_df.text.tolist()); c = list(model.classes_); yv = va_df.model.to_numpy()
    av = np.asarray(c)[Pv.argmax(1)]; tv = topic(va_df.text.tolist(), Pv, c, True)
    folds.append({'fold': i, 'rows': len(va), 'classes_val': int(va_df.model.nunique()),
                  'bal_argmax': round(balanced_accuracy_score(yv, av), 6), 'bal_topic': round(balanced_accuracy_score(yv, tv), 6)})
res['folds'] = folds
d = [f['bal_topic'] - f['bal_argmax'] for f in folds]
res['fold_mean'] = {'argmax': round(float(np.mean([f['bal_argmax'] for f in folds])), 6),
                    'topic': round(float(np.mean([f['bal_topic'] for f in folds])), 6),
                    'gain_mean': round(float(np.mean(d)), 6), 'gain_min': round(float(np.min(d)), 6)}
print(json.dumps(res, indent=1))
