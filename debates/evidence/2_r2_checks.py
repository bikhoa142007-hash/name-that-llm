"""Read-only R2 checks for debate 2 (Agent 1).
Usage (repo root):
  .venv/Scripts/python.exe debates/evidence/2_r2_checks.py unit          # tokenizer / ngram list / dtype / column-count checks
  FOLD_SEED=42 .venv/Scripts/python.exe debates/evidence/2_r2_checks.py cv  # unigram ablations on 5 grouped folds
Writes nothing; prints JSON."""
import json, os, sys, time, warnings
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path.cwd()))
from src.baseline import read_input, connected_groups  # noqa: E402

BASE = json.loads(Path('configs/baseline.json').read_text())
T, L = {**BASE['tfidf'], 'ngram_range': tuple(BASE['tfidf']['ngram_range'])}, BASE['logistic_regression']
PROPOSED = r"(?u)\b\w+\b|[^\w\s]"
WORD = {'analyzer': 'word', 'ngram_range': (1, 1), 'lowercase': False, 'min_df': 2, 'max_features': 40000,
        'sublinear_tf': True, 'token_pattern': PROPOSED}


def char():
    return TfidfVectorizer(**T, dtype=np.float32)


def union(**kw):
    return FeatureUnion([('ch', char()), ('w', TfidfVectorizer(**{**WORD, **kw}, dtype=np.float32))])


CONFIGS = {
    'baseline': lambda: Pipeline([('f', char()), ('c', LogisticRegression(**L))]),
    'uni_proposed': lambda: Pipeline([('f', union()), ('c', LogisticRegression(**L))]),
    # same unigram branch, drop punctuation tokens only (single-char words and case kept)
    'uni_no_punct': lambda: Pipeline([('f', union(token_pattern=r"(?u)\b\w+\b")), ('c', LogisticRegression(**L))]),
    # same unigram branch, lowercase only (punctuation tokens kept)
    'uni_lowercase': lambda: Pipeline([('f', union(lowercase=True)), ('c', LogisticRegression(**L))]),
    # same unigram branch, C=3 only (parking-lot evidence, not a proposal)
    'uni_C3': lambda: Pipeline([('f', union()), ('c', LogisticRegression(**{**L, 'C': 3.0}))]),
}


def unit():
    s = 'A i X AI ai ! ?'
    out = {'tokens': {p: TfidfVectorizer(lowercase=False, token_pattern=p).build_analyzer()(s)
                      for p in (PROPOSED, r'(?u)\b\w\w+\b', r'(?u)\b\w+\b')}}
    docs = ['Hello , world !', 'Hello world .', 'a b , c !', 'Hello again , world']
    y = [0, 1, 0, 1]
    try:
        TfidfVectorizer(**{**WORD, 'ngram_range': [1, 1], 'min_df': 1}).fit(docs)
        out['list_ngram_range'] = 'fit ok'
    except Exception as e:  # noqa: BLE001
        out['list_ngram_range'] = f'{type(e).__name__}: {str(e)[:160]}'
    fu_default = FeatureUnion([('ch', TfidfVectorizer(**{**T, 'min_df': 1})),
                               ('w', TfidfVectorizer(**{**WORD, 'min_df': 1}))])
    fu_32 = FeatureUnion([('ch', TfidfVectorizer(**{**T, 'min_df': 1}, dtype=np.float32)),
                          ('w', TfidfVectorizer(**{**WORD, 'min_df': 1}, dtype=np.float32))])
    fu_mixed = FeatureUnion([('ch', TfidfVectorizer(**{**T, 'min_df': 1}, dtype=np.float32)),
                             ('w', TfidfVectorizer(**{**WORD, 'min_df': 1}))])
    out['union_dtype'] = {'no_dtype': str(fu_default.fit_transform(docs).dtype),
                          'both_float32': str(fu_32.fit_transform(docs).dtype),
                          'char_float32_word_default': str(fu_mixed.fit_transform(docs).dtype)}
    pipe = Pipeline([('tfidf', fu_32), ('classifier', LogisticRegression(**L))]).fit(docs, y)
    u = pipe.named_steps['tfidf']
    out['column_count'] = {'char_vocab': len(u.transformer_list[0][1].vocabulary_),
                           'word_vocab': len(u.transformer_list[1][1].vocabulary_),
                           'coef_cols': int(pipe.named_steps['classifier'].coef_.shape[1]),
                           'has_vocabulary_attr_on_union': hasattr(u, 'vocabulary_')}
    print(json.dumps(out, indent=1, ensure_ascii=False))


def run(name, frame, folds):
    res = []
    for tr, va in folds:
        model = CONFIGS[name]()
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(frame.text.iloc[tr], frame.model.iloc[tr])
            pred = model.predict(frame.text.iloc[va])
        res.append((float(balanced_accuracy_score(frame.model.iloc[va], pred)),
                    sum(issubclass(w.category, ConvergenceWarning) for w in caught), time.perf_counter() - start))
    return name, res


def cv():
    seed = int(os.environ.get('FOLD_SEED', BASE['seed']))
    frame, _ = read_input('train.csv', labeled=True)
    groups, _ = connected_groups(frame)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(frame.text, frame.model, groups))
    results = dict(Parallel(n_jobs=len(CONFIGS))(delayed(run)(n, frame, folds) for n in CONFIGS))
    ref = {k: np.array([r[0] for r in results[k]]) for k in ('baseline', 'uni_proposed')}
    summary = {}
    for name, res in results.items():
        v = np.array([r[0] for r in res])
        summary[name] = {'mean': round(float(v.mean()), 4), 'std': round(float(v.std()), 4),
                         'vs_baseline_mean': round(float((v - ref['baseline']).mean()), 4),
                         'vs_proposed_mean': round(float((v - ref['uni_proposed']).mean()), 4),
                         'vs_proposed_folds_better': int((v > ref['uni_proposed']).sum()),
                         'per_fold': [round(x, 4) for x in v], 'conv_warn': sum(r[1] for r in res),
                         'seconds_fold1_fit_predict': round(res[0][2], 1)}
    print(json.dumps({'fold_seed': seed, 'summary': summary}, indent=1))


{'unit': unit, 'cv': cv}[sys.argv[1]]()
