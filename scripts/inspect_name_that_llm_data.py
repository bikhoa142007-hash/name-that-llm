"""Read-only CPU inspection. No training, network, credentials, or submissions.

Usage: python scripts/inspect_name_that_llm_data.py --data-dir . --output-dir reports
Requires pandas; pyarrow for Parquet; matplotlib for optional PNG figures.
All outputs are created exclusively; existing outputs receive numeric suffixes.
"""
import argparse
import hashlib
import importlib.metadata
import itertools
import json
import platform
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LABELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra',
          'gpt-6-astra', 'gemini-3.1-pro-preview', 'gemini-3.6-flash',
          'gemini-3.7-flash', 'gemini-3.8-flash', 'claude-fable-5',
          'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5',
          'claude-sonnet-5', 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']
PATTERNS = {
    'markdown_marker': r'(?m)(^\s{0,3}#{1,6}\s|\*\*[^\n]+?\*\*|\[[^\]]+\]\([^)]+\))',
    'code_fence': r'```|~~~',
    'bullet_list': r'(?m)^\s*(?:[-*+] |\d+[.)] )',
    'url': r'https?://\S+|www\.\S+',
    'non_ascii': r'[^\x00-\x7f]',
    'curly_quotes': r'[“”‘’]',
    'em_dash': r'—',
    'refusal_en': r"\b(?:I (?:cannot|can't|won't|am unable to)|I’m unable to|I'm unable to|I (?:must|have to) (?:decline|refuse))\b",
    'model_name_mention': r'\b(?:GPT[- ]?\d[\w.-]*|ChatGPT|Gemini|Claude|Qwen|GLM[- ]?\d[\w.-]*|Kimi)\b',
    'self_name_candidate_en': r'\b(?:I am|I’m|I\x27m|as)\s+(?:an?\s+)?(?:ChatGPT|GPT[- ]?\d[\w.-]*|Gemini|Claude|Qwen|GLM[- ]?\d[\w.-]*|Kimi)\b',
}


def fresh(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for i in itertools.count():
        candidate = path if i == 0 else path.with_name(f'{path.stem}_{i}{path.suffix}')
        if not candidate.exists():
            return candidate


def write_json(path, data):
    path = fresh(path)
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
    return str(path.resolve())


def normalize(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def summary(values):
    s = pd.Series(values, dtype='float64')
    if s.empty:
        return {}
    return {k: float(v) for k, v in {'min': s.min(), 'median': s.median(),
            'mean': s.mean(), 'p90': s.quantile(.90), 'p95': s.quantile(.95),
            'max': s.max()}.items()}


def dup(values):
    counts = Counter(values)
    return {'groups': sum(n > 1 for n in counts.values()),
            'rows_in_groups': sum(n for n in counts.values() if n > 1),
            'extra_rows': sum(n - 1 for n in counts.values() if n > 1)}


def analyze(df):
    info = {'rows': len(df), 'columns': list(df.columns), 'schema': {},
            'samples': json.loads(df.head(5).to_json(orient='records', force_ascii=False))}
    for sample in info['samples']:
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 180:
                sample[key] = value[:180] + '…'
    for col in df:
        info['schema'][col] = {'dtype': str(df[col].dtype),
                              'missing': int(df[col].isna().sum()),
                              'unique_non_null': int(df[col].nunique(dropna=True))}
    if 'id' in df:
        ids = df['id'].dropna().astype(str).tolist()
        info['id'] = {'missing': int(df['id'].isna().sum()),
                      'blank': sum(not x.strip() for x in ids), **dup(ids)}
    if 'text' in df:
        texts = df['text'].dropna().astype(str).tolist()
        info['text'] = {'null': int(df['text'].isna().sum()),
                        'empty': sum(t == '' for t in texts),
                        'whitespace_only': sum(t != '' and not t.strip() for t in texts),
                        'characters': summary([len(t) for t in texts]),
                        'words_whitespace': summary([len(t.split()) for t in texts]),
                        'exact_duplicates': dup(texts),
                        'nfc_whitespace_duplicates': dup([normalize(t) for t in texts]),
                        'patterns': {name: sum(bool(re.search(pat, t, re.I)) for t in texts)
                                     for name, pat in PATTERNS.items()}}
        if 'model' in df:
            labeled = df.dropna(subset=['text', 'model']).copy()
            info['conflicting_labels'] = {}
            for mode in ['exact', 'nfc_whitespace']:
                keys = labeled['text'].astype(str).map(normalize) if mode != 'exact' else labeled['text']
                grouped = labeled.assign(key=keys).groupby('key')['model'].nunique()
                info['conflicting_labels'][mode] = int((grouped > 1).sum())
            info['classes'] = {}
            for label, group in df.groupby('model', dropna=True):
                ts = group['text'].dropna().astype(str)
                info['classes'][str(label)] = {'count': len(group),
                    'characters': summary(ts.map(len).tolist()),
                    'words_whitespace': summary(ts.map(lambda t: len(t.split())).tolist())}
            info['unknown_labels'] = sorted(set(df['model'].dropna().astype(str)) - set(LABELS))
            info['absent_labels'] = sorted(set(LABELS) - set(df['model'].dropna().astype(str)))
    if {'prompt_id', 'model'}.issubset(df.columns):
        grouped = df.groupby('prompt_id', dropna=True)
        info['prompts'] = {'count': int(df['prompt_id'].nunique()),
            'rows_per_prompt': summary(grouped.size().tolist()),
            'row_count_histogram': {str(k): int(v) for k, v in grouped.size().value_counts().items()},
            'duplicate_prompt_model_extra_rows': int(df.duplicated(['prompt_id', 'model']).sum()),
            'incomplete': {str(p): sorted(set(LABELS) - set(g['model'].dropna()))
                           for p, g in grouped if set(g['model'].dropna()) != set(LABELS)}}
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--files', nargs='+', default=['train.csv', 'test.csv', 'sample_submission.parquet'],
                        help='Exact basenames searched recursively; secrets and arbitrary code are never read.')
    args = parser.parse_args()
    root, output = args.data_dir.resolve(), args.output_dir.resolve()
    if any(Path(n).name != n or Path(n).suffix.lower() not in {'.csv', '.parquet'} for n in args.files):
        parser.error('--files must contain CSV/Parquet basenames only')
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.name in args.files
                   and '.git' not in p.parts and not p.is_symlink())
    result = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'command': sys.argv, 'scope': 'all rows of explicitly selected files; no sampling',
              'python': platform.python_version(), 'libraries': {}, 'files': {},
              'relationships': [], 'errors': [], 'figures': [], 'patterns': PATTERNS}
    for name in ['pandas', 'numpy', 'pyarrow', 'matplotlib']:
        try:
            result['libraries'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result['libraries'][name] = 'not installed'
    inventory, frames = [], {}
    for path in paths:
        with path.open('rb') as stream:
            sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        row = {'file': str(path), 'source_role': 'local; provenance unverified; ' +
               {'train.csv': 'labeled training candidate', 'test.csv': 'public test candidate',
                'sample_submission.parquet': 'format example; not ground truth'}.get(path.name, 'unspecified'),
               'bytes': path.stat().st_size, 'sha256': sha, 'rows': '', 'columns': '', 'notes': ''}
        try:
            # Preserve literal NA-like strings. CSV has no intrinsic null type:
            # blank cells are missing; whitespace-only text is counted separately.
            df = (pd.read_csv(path, dtype='string', keep_default_na=False, na_values=[''])
                  if path.suffix == '.csv' else pd.read_parquet(path))
            info = analyze(df)
            result['files'][str(path)] = info
            frames[str(path)] = df
            row.update(rows=len(df), columns=len(df.columns), notes='full file read')
        except Exception as exc:
            row['notes'] = f'{type(exc).__name__}: {exc}'
            result['errors'].append({'file': str(path), 'error': row['notes']})
        inventory.append(row)
    hashes = {}
    for row in inventory:
        hashes.setdefault(row['sha256'], []).append(row['file'])
    result['duplicate_files_by_sha256'] = [group for group in hashes.values() if len(group) > 1]
    for (pa, a), (pb, b) in itertools.combinations(frames.items(), 2):
        relation = {'a': pa, 'b': pb}
        if 'id' in a and 'id' in b:
            aa, bb = set(a['id'].dropna()), set(b['id'].dropna())
            relation.update(id_overlap=len(aa & bb), id_sets_equal=aa == bb,
                            a_only_ids=len(aa - bb), b_only_ids=len(bb - aa))
        if 'text' in a and 'text' in b:
            for mode, fn in [('exact', str), ('nfc_whitespace', normalize)]:
                relation[f'text_overlap_{mode}'] = len(set(a['text'].dropna().map(fn)) & set(b['text'].dropna().map(fn)))
        result['relationships'].append(relation)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for path, df in frames.items():
            if 'text' not in df or len(df) < 30:
                continue
            text = df['text'].dropna().astype(str)
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].hist(text.map(len), bins=25, color='#397caa')
            axes[0].set(xlabel='Characters', ylabel='Passages', title=f'{Path(path).name}: character length')
            axes[1].hist(text.map(lambda t: len(t.split())), bins=25, color='#439d82')
            axes[1].set(xlabel='Whitespace-separated words (not tokens)', ylabel='Passages', title='Approximate word length')
            fig.tight_layout()
            target = fresh(output / 'figures' / f'{Path(path).stem}_lengths.png')
            fig.savefig(target, dpi=150)
            plt.close(fig)
            result['figures'].append(str(target))
            if 'model' in df:
                fig, ax = plt.subplots(figsize=(10, 7))
                counts = df['model'].value_counts().sort_index()
                ax.barh(counts.index, counts.values, color='#397caa')
                ax.set(xlabel='Labeled passages', title='Local train class counts; not hidden test distribution')
                for i, n in enumerate(counts):
                    ax.text(n + .5, i, str(n), va='center', fontsize=8)
                fig.tight_layout()
                target = fresh(output / 'figures' / f'{Path(path).stem}_classes.png')
                fig.savefig(target, dpi=150)
                plt.close(fig)
                result['figures'].append(str(target))
    except ImportError as exc:
        result['errors'].append({'stage': 'figures', 'error': str(exc)})
    target = fresh(output / 'name_that_llm_data_inventory.csv')
    with target.open('x', encoding='utf-8-sig', newline='') as f:
        pd.DataFrame(inventory).to_csv(f, index=False)
    result['inventory_path'] = str(target)
    target = write_json(output / 'name_that_llm_data_statistics.json', result)
    print(json.dumps({'statistics': target, 'inventory': result['inventory_path'],
                      'files_read': len(frames), 'errors': result['errors']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
