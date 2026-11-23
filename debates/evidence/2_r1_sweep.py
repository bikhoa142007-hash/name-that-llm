"""Read-only R1 sweep for debate 2 (Agent 1): one change at a time vs the current baseline, same 5 folds as `cv`.
Usage (repo root): .venv/Scripts/python.exe debates/evidence/2_r1_sweep.py [out.json]
Writes only the optional JSON path given (default: stdout)."""
import json, sys, time, warnings
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path.cwd()))
from src.baseline import read_input, connected_groups  # noqa: E402

BASE = json.loads(Path('configs/baseline.json').read_text())
T, L = {**BASE['tfidf'], 'ngram_range': tuple(BASE['tfidf']['ngram_range'])}, BASE['logistic_regression']


def char(**kw):
    return TfidfVectorizer(**{**T, **kw}, dtype=np.float32)


def word(**kw):
    return TfidfVectorizer(**{'analyzer': 'word', 'ngram_range': (1, 2), 'lowercase': False, 'min_df': 2,
                              'sublinear_tf': True, 'token_pattern': r"(?u)\b\w+\b|[^\w\s]", **kw}, dtype=np.float32)


def lr(**kw):
    return LogisticRegression(**{**L, **kw})


CONFIGS = {
    'baseline': lambda: Pipeline([('f', char()), ('c', lr())]),
    'char_max200k': lambda: Pipeline([('f', char(max_features=200000)), ('c', lr())]),
    'char_maxNone': lambda: Pipeline([('f', char(max_features=None)), ('c', lr())]),
    'char_1_5': lambda: Pipeline([('f', char(ngram_range=(1, 5))), ('c', lr())]),
    'char_wb_2_5': lambda: Pipeline([('f', char(analyzer='char_wb', ngram_range=(2, 5))), ('c', lr())]),
    'C_0.3': lambda: Pipeline([('f', char()), ('c', lr(C=0.3))]),
    'C_3': lambda: Pipeline([('f', char()), ('c', lr(C=3.0))]),
    'C_10': lambda: Pipeline([('f', char()), ('c', lr(C=10.0, max_iter=2000))]),
    'plus_word_1_2': lambda: Pipeline([('f', FeatureUnion([('ch', char()), ('w', word(max_features=40000))])), ('c', lr())]),
    'linear_svc': lambda: Pipeline([('f', char()), ('c', LinearSVC(C=0.5, class_weight='balanced', max_iter=5000))]),
    # Round-1 follow-ups (run with names on the command line)
    'word_only_1_2': lambda: Pipeline([('f', word(max_features=40000)), ('c', lr())]),
    'plus_word_default_tokens': lambda: Pipeline([('f', FeatureUnion([('ch', char()), ('w', word(max_features=40000, token_pattern=r'(?u)\b\w\w+\b'))])), ('c', lr())]),
    'plus_word_1_1': lambda: Pipeline([('f', FeatureUnion([('ch', char()), ('w', word(max_features=40000, ngram_range=(1, 1)))])), ('c', lr())]),
    'plus_word_1_2_C3': lambda: Pipeline([('f', FeatureUnion([('ch', char()), ('w', word(max_features=40000))])), ('c', lr(C=3.0))]),
    'plus_word_1_3': lambda: Pipeline([('f', FeatureUnion([('ch', char()), ('w', word(max_features=40000, ngram_range=(1, 3)))])), ('c', lr())]),
}


def run(name, frame, folds):
    out = []
    for i, (tr, va) in enumerate(folds, 1):
        model = CONFIGS[name]()
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(frame.text.iloc[tr], frame.model.iloc[tr])
            pred = model.predict(frame.text.iloc[va])
        out.append({'fold': i, 'bal': float(balanced_accuracy_score(frame.model.iloc[va], pred)),
                    'seconds': round(time.perf_counter() - start, 1),
                    'convergence_warnings': sum(issubclass(w.category, ConvergenceWarning) for w in caught),
                    'n_features': int(sum(len(getattr(t, 'vocabulary_', {})) for t in
                                          ([model[0]] if hasattr(model[0], 'vocabulary_') else [s for _, s in model[0].transformer_list])))})
    return name, out


frame, _ = read_input('train.csv', labeled=True)
groups, _ = connected_groups(frame)
folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=int(__import__('os').environ.get('FOLD_SEED', BASE['seed']))).split(frame.text, frame.model, groups))
names = sys.argv[2:] or [n for n in CONFIGS if n in ('baseline', 'char_max200k', 'char_maxNone', 'char_1_5', 'char_wb_2_5',
                                                      'C_0.3', 'C_3', 'C_10', 'plus_word_1_2', 'linear_svc')]
if 'baseline' not in names:
    names = ['baseline'] + names
results = dict(Parallel(n_jobs=6)(delayed(run)(n, frame, folds) for n in names))
base = np.array([f['bal'] for f in results['baseline']])
summary = {}
for name, folds_ in results.items():
    v = np.array([f['bal'] for f in folds_]); d = v - base
    summary[name] = {'mean': round(float(v.mean()), 4), 'std': round(float(v.std()), 4),
                     'gain_mean': round(float(d.mean()), 4), 'gain_min': round(float(d.min()), 4),
                     'folds_better': int((d > 0).sum()), 'per_fold': [round(x, 4) for x in v],
                     'conv_warn': sum(f['convergence_warnings'] for f in folds_),
                     'n_features_fold1': folds_[0]['n_features'], 'fit_s_fold1': folds_[0]['seconds']}
text = json.dumps({'summary': summary}, indent=1)
if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(text)
print(text)
