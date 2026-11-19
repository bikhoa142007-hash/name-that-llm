"""Read-only R3 check for debate 1 (Agent 1): does topic assignment still help when per-model counts are unequal?
Kaggle Overview states test passages per model are not equal. Simulation: in each 5-fold validation set, keep each row
of model m with probability r_m ~ Uniform(lo, 1), independently; compare balanced accuracy argmax vs topic.
Usage (repo root): .venv/Scripts/python.exe debates/evidence/1_r3_imbalance.py"""
import json, math, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path.cwd()))
from src.baseline import read_input, make_split, build_pipeline, fit_checked  # noqa: E402


def assign(logP, classes, labels):
    pred = np.empty(len(logP), dtype=object)
    for g in np.unique(labels):
        idx = np.where(labels == g)[0]
        r, c = linear_sum_assignment(-np.tile(logP[idx], (1, math.ceil(len(idx) / len(classes)))))
        pred[idx[r]] = np.asarray(classes)[c % len(classes)]
    return pred


def topic(texts, P, classes):  # same logic as 1_r2_verify.topic(per_row_fallback=True)
    argmax = np.asarray(classes)[P.argmax(1)]
    if len(texts) < len(classes):
        return argmax
    try:
        X = TfidfVectorizer(analyzer='word', stop_words='english', min_df=2, sublinear_tf=True).fit_transform(texts).toarray()
    except ValueError:
        return argmax
    ok = np.linalg.norm(X, axis=1) > 0
    k = round(int(ok.sum()) / len(classes))
    if k < 1:
        return argmax
    out = argmax.copy()
    lab = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average').fit_predict(X[ok])
    out[ok] = assign(np.log(np.clip(P[ok], 1e-12, 1)), classes, lab)
    return out


config = json.loads(Path('configs/baseline.json').read_text())
frame, _ = read_input('train.csv', labeled=True)
_, _, groups, _ = make_split(frame, config)
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config['seed'])
SCEN = {'lo=0.3': 0.3, 'lo=0.1': 0.1}
SEEDS = range(3)
rows = []
for fold, (tr, va) in enumerate(cv.split(frame.text, frame.model, groups), 1):
    model = build_pipeline(config)
    fit_checked(model, frame.text.iloc[tr], frame.model.iloc[tr])
    va_df = frame.iloc[va].reset_index(drop=True)
    P = model.predict_proba(va_df.text.tolist()); cls = list(model.classes_); y = va_df.model.to_numpy()
    for name, lo in SCEN.items():
        for s in SEEDS:
            rng = np.random.default_rng(1000 * fold + s)
            rate = dict(zip(cls, rng.uniform(lo, 1.0, len(cls))))
            keep = rng.random(len(va_df)) < va_df.model.map(rate).to_numpy()
            idx = np.where(keep)[0]
            a = np.asarray(cls)[P[idx].argmax(1)]; t = topic(va_df.text.iloc[idx].tolist(), P[idx], cls)
            counts = np.bincount(np.searchsorted(cls, y[idx]), minlength=len(cls))
            rows.append({'fold': fold, 'scenario': name, 'seed': s, 'n': int(len(idx)),
                         'min_max_per_model': [int(counts.min()), int(counts.max())],
                         'bal_argmax': round(balanced_accuracy_score(y[idx], a), 4),
                         'bal_topic': round(balanced_accuracy_score(y[idx], t), 4)})
summary = {}
for name in SCEN:
    g = np.array([r['bal_topic'] - r['bal_argmax'] for r in rows if r['scenario'] == name])
    summary[name] = {'runs': int(len(g)), 'gain_mean': round(float(g.mean()), 4), 'gain_min': round(float(g.min()), 4),
                     'gain_max': round(float(g.max()), 4), 'runs_negative': int((g < 0).sum())}
print(json.dumps({'summary': summary, 'runs': rows}, indent=0))
