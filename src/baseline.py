"""CPU TF-IDF + Logistic Regression. Run: python -m src.baseline --help.

No network calls. Only load joblib artifacts you created/trust: joblib uses pickle.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import time
import unicodedata
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

# Official Overview > Evaluation > Valid Labels, verified 2026-10-26.
# This display order is NOT an order of probability columns in the submission.
LABELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra',
          'gpt-6-astra', 'gemini-3.1-pro-preview', 'gemini-3.6-flash',
          'gemini-3.7-flash', 'gemini-3.8-flash', 'claude-fable-5',
          'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5',
          'claude-sonnet-5', 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def save_csv(path, frame):
    with Path(path).open('x', encoding='utf-8', newline='') as f:
        frame.to_csv(f, index=False)


def environment():
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'libraries': {name: importlib.metadata.version(name) for name in
                          ['numpy', 'scipy', 'pandas', 'scikit-learn', 'pyarrow',
                           'joblib', 'threadpoolctl']}}


def normalized(text):
    # Used only for duplicate grouping, not as model input.
    return ' '.join(unicodedata.normalize('NFC', text).split())


def read_input(path, labeled=False):
    path = Path(path)
    require(path.suffix.lower() == '.csv', 'Input must be a CSV file.')
    with path.open(encoding='utf-8-sig', newline='') as f:
        header = next(csv.reader(f), [])
    require(len(header) == len(set(header)), 'Duplicate column headers are not allowed.')
    frame = pd.read_csv(path, dtype='string', keep_default_na=False,
                        na_values=[''], skip_blank_lines=False)
    required = {'id', 'text'} | ({'model'} if labeled else set())
    require(required <= set(frame), f'Missing columns: {sorted(required - set(frame))}')
    require(not frame['id'].isna().any(), 'IDs contain null values.')
    require(not frame['id'].str.strip().eq('').any(), 'IDs contain blank values.')
    require(not frame['id'].duplicated().any(), 'IDs must be unique.')
    null_count = int(frame['text'].isna().sum())
    frame['text'] = frame['text'].fillna('')
    blank_count = int(frame['text'].str.strip().eq('').sum())
    audit = {'rows': len(frame), 'columns': list(frame), 'null_text': null_count,
             'blank_text_including_null': blank_count,
             'sha256': sha256(path), 'bytes': path.stat().st_size}
    if labeled:
        require(len(frame) > 0, 'Training data is empty.')
        require(blank_count == 0, 'Blank/null training texts are rejected; fix the data source.')
        require(not frame['model'].isna().any(), 'Training labels contain null values.')
        unknown = sorted(set(frame['model']) - set(LABELS))
        missing = sorted(set(LABELS) - set(frame['model']))
        require(not unknown, f'Unknown model labels: {unknown}')
        require(not missing, f'Missing official training classes: {missing}; no synthetic classes added.')
        if 'prompt_id' in frame:
            require(not frame['prompt_id'].isna().any() and
                    not frame['prompt_id'].str.strip().eq('').any(), 'prompt_id contains null/blank values.')
        audit['class_counts'] = {str(k): int(v) for k, v in frame['model'].value_counts().items()}
    elif blank_count:
        warnings.warn(f'{blank_count} blank/null test texts kept as empty strings; no rows dropped.')
    return frame, audit


def make_split(frame, config):
    """Connected components combine prompt groups AND duplicate-text groups.

    A duplicate connecting two prompts must keep both entire prompts together.
    Without prompt_id, only duplicate grouping is available, a weaker guarantee.
    """
    parent = list(range(len(frame)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(b)] = find(a)

    keys = frame['text'].map(normalized)
    for values in ([frame['prompt_id'].tolist()] if 'prompt_id' in frame else []) + [keys.tolist()]:
        seen = {}
        for i, value in enumerate(values):
            if value in seen:
                union(i, seen[value])
            else:
                seen[value] = i
    groups = np.array([find(i) for i in range(len(frame))])
    require(len(set(groups)) >= 2, 'Not enough independent groups for a holdout.')
    splitter = GroupShuffleSplit(n_splits=1, test_size=config['validation_fraction'],
                                random_state=config['seed'])
    train_idx, valid_idx = next(splitter.split(frame['text'], frame['model'], groups))
    a, b = frame.iloc[train_idx], frame.iloc[valid_idx]
    absent = {name: sorted(set(LABELS) - set(part['model'])) for name, part in
              [('train', a), ('validation', b)]}
    require(not any(absent.values()), f'Holdout lacks classes: {absent}; evaluation aborted.')
    require(not (set(groups[train_idx]) & set(groups[valid_idx])), 'Group leakage detected.')
    exact_overlap = len(set(a['text']) & set(b['text']))
    normalized_overlap = len(set(a['text'].map(normalized)) & set(b['text'].map(normalized)))
    prompt_overlap = (len(set(a['prompt_id']) & set(b['prompt_id'])) if 'prompt_id' in frame else None)
    require(exact_overlap == normalized_overlap == 0 and prompt_overlap in (0, None), 'Text/prompt leakage detected.')
    conflicts = frame.assign(key=keys).groupby('key')['model'].nunique()
    audit = {'method': 'GroupShuffleSplit on connected prompt/duplicate components',
             'seed': config['seed'], 'validation_fraction': config['validation_fraction'],
             'group_metadata': 'prompt_id' if 'prompt_id' in frame else 'duplicate text only; prompt/source unavailable',
             'component_count': len(set(groups)), 'train_rows': len(a), 'validation_rows': len(b),
             'train_groups': len(set(groups[train_idx])), 'validation_groups': len(set(groups[valid_idx])),
             'prompt_overlap': prompt_overlap, 'exact_text_overlap': exact_overlap,
             'normalized_text_overlap': normalized_overlap,
             'exact_duplicate_extra_rows': int(frame['text'].duplicated().sum()),
             'normalized_duplicate_extra_rows': int(keys.duplicated().sum()),
             'normalized_duplicate_conflicting_label_groups': int((conflicts > 1).sum()),
             'absent_classes': absent,
             'class_counts': {name: {label: int((part['model'] == label).sum()) for label in LABELS}
                              for name, part in [('train', a), ('validation', b)]}}
    return train_idx, valid_idx, groups, audit


def build_pipeline(config):
    tfidf = dict(config['tfidf'])
    tfidf['ngram_range'] = tuple(tfidf['ngram_range'])
    return Pipeline([('tfidf', TfidfVectorizer(**tfidf, dtype=np.float32)),
                     ('classifier', LogisticRegression(**config['logistic_regression'],
                                                       random_state=config['seed']))])


def fit_checked(model, texts, labels):
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(texts, labels)
    convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    require(not convergence, f'Logistic Regression did not converge: {convergence}')
    classifier = model.named_steps['classifier']
    require(np.any(classifier.n_iter_ > 0) and np.any(classifier.coef_ != 0),
            'Optimizer stopped without learning coefficients; inspect numerical tolerance.')
    return time.perf_counter() - start


def score(labels, predictions):
    return {'balanced_accuracy': float(balanced_accuracy_score(labels, predictions)),
            'accuracy': float(accuracy_score(labels, predictions)),
            'macro_f1': float(f1_score(labels, predictions, labels=LABELS, average='macro', zero_division=0))}


def evaluate_models(frame, train_idx, valid_idx, model, dummy, directory):
    """Evaluate held-out rows only. The full-data inference model is never used here."""
    valid = frame.iloc[valid_idx]
    with threadpool_limits(limits=1):
        pred = model.predict(valid['text'])
    dummy_pred = dummy.predict(np.zeros((len(valid), 1)))
    metrics = {'dummy': score(valid['model'], dummy_pred), 'tfidf_logistic': score(valid['model'], pred)}
    for name, predictions in [('dummy', dummy_pred), ('tfidf_logistic', pred)]:
        report = classification_report(valid['model'], predictions, labels=LABELS,
                                       output_dict=True, zero_division=0)
        save_json(directory / f'{name}_per_class.json', report)
        matrix = pd.DataFrame(confusion_matrix(valid['model'], predictions, labels=LABELS),
                              index=LABELS, columns=LABELS).rename_axis('true_model').reset_index()
        save_csv(directory / f'{name}_confusion_matrix.csv', matrix)
    predictions = valid[['id', 'model']].rename(columns={'model': 'true_model'}).copy()
    predictions['predicted_model'] = pred
    predictions['dummy_model'] = dummy_pred
    save_csv(directory / 'validation_predictions.csv', predictions)
    wrong = valid.loc[np.asarray(valid['model']) != pred, ['id', 'text', 'model']].copy()
    wrong['predicted_model'] = pred[np.asarray(valid['model']) != pred]
    wrong['text'] = wrong['text'].str.slice(0, 180)
    save_json(directory / 'error_examples.json', json.loads(wrong.head(10).to_json(orient='records', force_ascii=False)))
    save_json(directory / 'metrics.json', metrics)
    return metrics


def load_artifact(directory, name):
    directory = Path(directory)
    meta = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    require(meta['environment']['libraries']['scikit-learn'] == importlib.metadata.version('scikit-learn'),
            'scikit-learn version differs from artifact; recreate the pinned environment.')
    require(sha256(directory / name) == meta['artifact_sha256'][name], 'Artifact hash mismatch.')
    # Hash integrity does not make third-party pickle/joblib safe. Trust the producer.
    return joblib.load(directory / name), meta


def train(args):
    start = time.perf_counter()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    frame, input_audit = read_input(args.train, labeled=True)
    train_idx, valid_idx, groups, split_audit = make_split(frame, config)
    artifact, output = Path(args.artifact_dir), Path(args.output_dir)
    require(not artifact.exists() and not output.exists(), 'Use new artifact/output directories; existing runs are never overwritten.')
    artifact.mkdir(parents=True)
    output.mkdir(parents=True)
    print(f'Holdout: {len(train_idx)} train / {len(valid_idx)} validation rows; CPU threads=1', flush=True)
    model = build_pipeline(config)
    fit_seconds = fit_checked(model, frame.iloc[train_idx]['text'], frame.iloc[train_idx]['model'])
    dummy = DummyClassifier(strategy='most_frequent', random_state=config['seed'])
    dummy.fit(np.zeros((len(train_idx), 1)), frame.iloc[train_idx]['model'])
    metrics = evaluate_models(frame, train_idx, valid_idx, model, dummy, output)
    joblib.dump(model, artifact / 'validation_pipeline.joblib', compress=3)
    joblib.dump(dummy, artifact / 'validation_dummy.joblib', compress=3)
    split = frame[['id']].copy()
    split['split'] = 'train'
    split.loc[valid_idx, 'split'] = 'validation'
    split['component'] = groups
    if 'prompt_id' in frame:
        split['prompt_id'] = frame['prompt_id']
    save_csv(artifact / 'split.csv', split)
    save_json(output / 'split_audit.json', split_audit)
    print(json.dumps(metrics), flush=True)
    print('Refitting the same configuration on all labeled rows for inference.', flush=True)
    full = build_pipeline(config)
    full_seconds = fit_checked(full, frame['text'], frame['model'])
    joblib.dump(full, artifact / 'pipeline.joblib', compress=3)
    # Explicitly associate probability columns with the estimator's own class names.
    mapping = {'official_labels': LABELS, 'model_classes': full.classes_.tolist(),
               'probability_column_to_model': {str(i): str(label) for i, label in enumerate(full.classes_)}}
    save_json(artifact / 'label_mapping.json', mapping)
    save_json(artifact / 'config.json', config)
    meta = {'created_utc': datetime.now(timezone.utc).isoformat(), 'environment': environment(),
            'input': input_audit, 'config': config, 'split': split_audit, 'metrics': metrics,
            'validation_feature_count': len(model.named_steps['tfidf'].vocabulary_),
            'full_feature_count': len(full.named_steps['tfidf'].vocabulary_),
            'validation_iterations': model.named_steps['classifier'].n_iter_.tolist(),
            'full_iterations': full.named_steps['classifier'].n_iter_.tolist(),
            'timing_seconds': {'validation_fit': fit_seconds, 'full_fit': full_seconds,
                               'train_total': time.perf_counter() - start},
            'artifact_sha256': {p.name: sha256(p) for p in artifact.iterdir() if p.is_file()}}
    save_json(artifact / 'metadata.json', meta)
    print(f'Saved artifacts: {artifact}', flush=True)


def validate_submission(test, submission):
    require(list(submission.columns) == ['id', 'model'], 'Submission columns must be exactly id, model in that order.')
    require(len(submission) == len(test), 'Submission row count differs from test.')
    require(not submission.isna().any().any(), 'Submission contains nulls.')
    for col in ['id', 'model']:
        require(all(isinstance(x, str) for x in submission[col]), f'{col} must contain strings.')
    require(not submission['id'].duplicated().any(), 'Submission IDs are duplicated.')
    require(submission['id'].tolist() == test['id'].tolist(), 'Submission IDs/order differ from test.')
    require(set(submission['model']) <= set(LABELS), 'Submission contains invalid labels.')
    return {'rows': len(submission), 'columns': list(submission), 'valid_labels': True,
            'id_order_equal': True, 'no_nulls': True, 'unique_ids': True,
            'string_values': True, 'read_back_checked': True}


def predict(args):
    start = time.perf_counter()
    output = Path(args.output)
    require(output.suffix == '.parquet', 'Output must use .parquet.')
    require(not output.exists() and not output.with_suffix('.checks.json').exists(), 'Prediction output already exists; choose a new path.')
    require(args.batch_size > 0, 'batch-size must be positive.')
    test, audit = read_input(args.test)
    pipeline, meta = load_artifact(args.artifact_dir, 'pipeline.joblib')
    require(set(pipeline.classes_) == set(LABELS), 'Artifact lacks official classes.')
    mapping = json.loads((Path(args.artifact_dir) / 'label_mapping.json').read_text(encoding='utf-8'))
    require(mapping['model_classes'] == pipeline.classes_.tolist(), 'Saved class mapping is inconsistent.')
    predictions = []
    with threadpool_limits(limits=1):
        for offset in range(0, len(test), args.batch_size):
            probabilities = pipeline.predict_proba(test['text'].iloc[offset:offset + args.batch_size])
            require(np.isfinite(probabilities).all() and (probabilities >= 0).all() and
                    (probabilities <= 1).all() and np.allclose(probabilities.sum(axis=1), 1),
                    'Invalid model probabilities.')
            predictions.extend(pipeline.classes_[probabilities.argmax(axis=1)].tolist())
    result = pd.DataFrame({'id': test['id'].astype('string'), 'model': pd.Series(predictions, dtype='string')})
    # Validate in memory and AGAIN after reading the exported file.
    validate_submission(test, result)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as f:
        result.to_parquet(f, index=False, engine='pyarrow')
    checked = validate_submission(test, pd.read_parquet(output))
    checked.update(input=audit, output_sha256=sha256(output),
                   seconds=time.perf_counter() - start,
                   probability_class_order=pipeline.classes_.tolist(),
                   artifact_model_sha256=meta['artifact_sha256']['pipeline.joblib'])
    save_json(output.with_suffix('.checks.json'), checked)
    print(json.dumps(checked, ensure_ascii=False), flush=True)


def evaluate(args):
    frame, audit = read_input(args.train, labeled=True)
    model, meta = load_artifact(args.artifact_dir, 'validation_pipeline.joblib')
    dummy, _ = load_artifact(args.artifact_dir, 'validation_dummy.joblib')
    require(audit['sha256'] == meta['input']['sha256'], 'Training CSV changed; cannot reuse the recorded holdout.')
    split_path = Path(args.artifact_dir) / 'split.csv'
    require(sha256(split_path) == meta['artifact_sha256']['split.csv'], 'Split manifest changed.')
    split = pd.read_csv(split_path, dtype={'id': 'string'})
    require(split['id'].tolist() == frame['id'].tolist(), 'Split ID order mismatch.')
    directory = Path(args.output_dir)
    require(not directory.exists(), 'Evaluation output directory already exists.')
    directory.mkdir(parents=True)
    metrics = evaluate_models(frame, np.flatnonzero(split['split'].eq('train')),
                              np.flatnonzero(split['split'].eq('validation')), model, dummy, directory)
    require(metrics == meta['metrics'], 'Reloaded validation metrics differ from the training run.')
    print(json.dumps({'metrics': metrics, 'matches_saved_metrics': True}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('train', help='One grouped holdout, then full-data refit; never reads test.')
    p.add_argument('--train', required=True)
    p.add_argument('--config', default='configs/baseline.json')
    p.add_argument('--artifact-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.set_defaults(func=train)
    p = commands.add_parser('evaluate', help='Reload the held-out model and original split, without refitting.')
    p.add_argument('--train', required=True)
    p.add_argument('--artifact-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.set_defaults(func=evaluate)
    p = commands.add_parser('predict', help='Offline batched inference using the full-data fitted pipeline.')
    p.add_argument('--test', required=True)
    p.add_argument('--artifact-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--batch-size', type=int, default=256)
    p.set_defaults(func=predict)
    p = commands.add_parser('validate', help='Independently read and validate an existing Parquet output.')
    p.add_argument('--test', required=True)
    p.add_argument('--submission', required=True)
    p.set_defaults(func=lambda a: print(json.dumps(validate_submission(read_input(a.test)[0], pd.read_parquet(a.submission)))))
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
