"""Build the offline Kaggle CPU pool experiment, or smoke-test the generated notebook."""
import argparse
import ast
import json
import shutil
import time
from pathlib import Path

from build_kaggle_cpu_notebook import cell

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / 'notebooks/kaggle_pool_cpu_1b2_final.ipynb'
FOLD_HASH = 'e8eb76023a9679528b5f471ab0cdb6f00e7292b177ce5cea69fc2d8a15393106'
POOL_HASH = 'fe6ece20e6f9f2fe5fb3c47f2aaa9749394c1056bc688a8eab865e0dc2bb56df'

SETUP = '''FINAL_ONLY = True
RUN_K3 = False

from pathlib import Path
import json
import math
import os
import time
import warnings
from concurrent.futures import as_completed
from joblib.externals.loky import ProcessPoolExecutor
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

START = time.monotonic()
INPUT = Path('/kaggle/input')
WORK = Path('/kaggle/working')
EXPECTED_FOLDS = __FOLD_HASH__
EXPECTED_POOL = __POOL_HASH__
SMALL, LARGE = 1100, 2500
K0_TARGET, K0_TOLERANCE = 0.43267, 0.005
MIN_GAIN = 0.005
K2_TARGET = 0.47157594
K3_DISABLED = 'disabled: LR on 231k rows exceeds the CPU session budget (pilot > 4.3 h)'
BASELINE = 0.33929903
CONFIG = __CONFIG__
'''

RESOURCES = '''def available_gib():
    # Kaggle can expose host RAM in /proc even when a cgroup limits the session.
    if os.name == 'nt':
        from ctypes import create_string_buffer, windll
        import struct
        buf = create_string_buffer(64)
        struct.pack_into('I', buf, 0, 64)
        require(windll.kernel32.GlobalMemoryStatusEx(buf), 'Cannot measure available RAM')
        return struct.unpack_from('Q', buf, 16)[0] / 2**30
    import psutil
    available = psutil.virtual_memory().available
    mounts = [(Path('/sys/fs/cgroup'), 'memory.max', 'memory.current'),
              (Path('/sys/fs/cgroup/memory'), 'memory.limit_in_bytes', 'memory.usage_in_bytes')]
    for root, limit_file, usage_file in mounts:
        if (root / limit_file).exists():
            limit = (root / limit_file).read_text().strip()
            if limit != 'max':
                available = min(available, int(limit) - int((root / usage_file).read_text()))
    return max(0, available) / 2**30


def peak_gib():
    if os.name == 'nt':
        import ctypes
        counters = (ctypes.c_size_t * 12)()
        size = 8 + 8 * ctypes.sizeof(ctypes.c_size_t)
        require(ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1),
                ctypes.byref(counters), size), 'Cannot read peak working set')
        return ctypes.c_size_t.from_buffer(counters, 8).value / 2**30
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def session_seconds_left():
    elapsed = time.monotonic() - START
    if os.name != 'nt':
        import psutil
        elapsed = max(elapsed, time.time() - psutil.Process().create_time())
    return 12 * 3600 - elapsed


def worker_count(free, peak):
    if os.name == 'nt' and free < 6:
        print('Local RAM below 6 GiB: close unused applications; using at most one worker.')
        return 1
    reserve = 2 if os.name == 'nt' else 4
    jobs = min(3, 5, math.floor((free - reserve) / peak))
    return max(0, jobs)


def checkpoint():
    report['elapsed_seconds'] = time.monotonic() - START
    path = WORK / ('metrics_final.json' if FINAL_ONLY else 'metrics.json')
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)
'''

INPUTS = '''def find_one(filename, dataset=None):
    paths = sorted(p for p in INPUT.rglob(filename) if p.is_file()
                   and (dataset is None or dataset in p.relative_to(INPUT).parts[:-1]))
    require(len(paths) == 1, f'{filename}: expected one file under {INPUT}, found {len(paths)}: {paths}')
    return paths[0]


names = ['train.csv', 'test.csv', 'ntl_open_pool.parquet', 'folds.csv', 'oof_tfidf.csv',
         'sample_ids_A.csv', 'sample_ids_B.csv']
paths = {name: find_one(name, 'ntl-cv-assets' if name in names[3:] else None) for name in names}
if FINAL_ONLY:
    for name in ('metrics.json', 'sample_ids_K2.csv'):
        paths[name] = find_one(name, 'ntl-1b2-v1')
WORK.mkdir(parents=True, exist_ok=True)
require(not any(WORK.glob('oof_pool_K*.csv')) and not any(WORK.glob('sample_ids_K*.csv'))
        and not any((WORK / f).exists() for f in ('metrics.json', 'metrics_final.json', 'submission.parquet', 'folds.csv')),
        'Start with a fresh output directory to avoid mixing separate runs.')
report = {'environment': environment(), 'inputs': {n: {'path': str(p), 'sha256': sha256(p)}
          for n, p in paths.items()}, 'config': CONFIG, 'baseline_oof': BASELINE,
          'variants': {k: {'status': 'pending'} for k in ('K0', 'K1', 'K2', 'K3')}}
print(report['environment'])
checkpoint()
require(report['inputs']['folds.csv']['sha256'] == EXPECTED_FOLDS, 'folds.csv SHA-256 mismatch')
require(report['inputs']['ntl_open_pool.parquet']['sha256'] == EXPECTED_POOL, 'Pool SHA-256 differs from v19')
train, _ = read_input(paths['train.csv'], labeled=True)
test, _ = read_input(paths['test.csv'])
manifest = pd.read_csv(paths['folds.csv'], dtype={'id':'string', 'model':'string', 'prompt_id':'string'})
pd.testing.assert_frame_equal(manifest[['id','prompt_id','model']], train[['id','prompt_id','model']])
folds = manifest.fold
groups, train_keys = connected_groups(train)
require(set(folds) == set(range(1, 6)) and folds.groupby(groups).nunique().max() == 1, 'Invalid real folds')
for fold in range(1, 6):
    require(set(train.model[folds.eq(fold)]) == set(LABELS), f'Incomplete class coverage in fold {fold}')
base_oof = pd.read_csv(paths['oof_tfidf.csv'], dtype={'id':'string'})
require(list(base_oof) == ['id', *LABELS] and base_oof.id.equals(train.id), 'Invalid baseline OOF layout')
baseline_p = base_oof[LABELS].to_numpy()
require(np.isfinite(baseline_p).all() and (baseline_p >= 0).all()
        and np.allclose(baseline_p.sum(axis=1), 1), 'Invalid baseline probabilities')
base_score = balanced_accuracy_score(train.model, base_oof[LABELS].idxmax(axis=1))
require(abs(base_score - BASELINE) < 1e-8, 'Baseline OOF mismatch')
pool = pd.read_parquet(paths['ntl_open_pool.parquet'],
                       columns=['id', 'model', 'text', 'gen_reasoning', 'match_weight']).set_index('id')
require(pool.index.is_unique and not pool.index.hasnans and set(pool.model) == set(LABELS)
        and pool.text.notna().all(), 'Invalid pool IDs, labels or texts')
keys = set(train_keys)
duplicates = sum(normalized(t) in keys for t in pool.text)
report['normalized_text_duplicates'] = duplicates
checkpoint()
require(duplicates == 0, f'{duplicates} pool rows duplicate real train texts')
eligible = pool.loc[pool.model.ne('kimi-k3') | pool.gen_reasoning.eq('high')]
for variant in ('A', 'B'):
    ids = pd.read_csv(paths[f'sample_ids_{variant}.csv'], dtype='string').set_index('id')
    require(ids.index.is_unique and ids.index.isin(pool.index).all(), f'Invalid sample IDs in {variant}')
    require(np.array_equal(ids.model, pool.loc[ids.index, 'model']), f'Sample label mismatch in {variant}')
    require(set(ids.model) == set(LABELS) and ids.model.value_counts().eq(SMALL).all(),
            f'Invalid class counts in {variant}')
    if variant == 'B':
        require(ids.index.isin(eligible.index).all(), 'Ineligible K0 IDs')
        k0_ids = ids.index
import shutil
shutil.copyfile(paths['folds.csv'], WORK / 'folds.csv')
print(f'{len(train)} real rows, {len(pool)} pool rows, {len(eligible)} eligible; {duplicates} duplicate texts')
'''

FUNCTIONS = '''def sample_ids(name):
    if name == 'K0':
        return k0_ids
    if name == 'K3':
        return eligible.index
    rng = np.random.default_rng(42)
    size = SMALL if name == 'K1' else LARGE
    cap = float(eligible.match_weight.quantile(.99))
    chosen = []
    for label in LABELS:
        rows = eligible.loc[eligible.model.eq(label)]
        probabilities = None
        if name == 'K2':
            weights = rows.match_weight.clip(upper=cap).to_numpy(dtype=float)
            require(np.isfinite(weights).all() and (weights >= 0).all() and (weights > 0).sum() >= size,
                    f'Invalid or insufficient weights for {label}')
            probabilities = weights / weights.sum()
        pos = rng.choice(len(rows), size=size, replace=False, p=probabilities)
        chosen.extend(rows.index[pos])
    return pd.Index(chosen, name='id')


def fit_predict(real, extra, valid, config, number, prediction_batch=512):
    start = time.monotonic()
    training = pd.concat([real, extra], ignore_index=True)
    with threadpool_limits(1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model = build_pipeline(config)
        fit_start = time.monotonic()
        model.fit(training.text, training.model)
        fit_seconds = time.monotonic() - fit_start
        require(np.any(model.named_steps['classifier'].coef_ != 0), 'Empty fit')
        order = pd.Index(model.classes_).get_indexer(LABELS)
        require((order >= 0).all(), 'Missing probability columns')
        batches = [model.predict_proba(valid.text.iloc[i:i+prediction_batch])[:, order]
                   for i in range(0, len(valid), prediction_batch)]
    p = np.vstack(batches) if batches else np.empty((0, len(LABELS)))
    require(np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(axis=1), 1), 'Invalid probabilities')
    result = pd.DataFrame(p, columns=LABELS, index=valid.id)
    stats = dict(fold=number, seconds=time.monotonic()-start, fit_seconds=fit_seconds, worker_peak_gib=peak_gib(),
                 real_train=len(real), pool_train=len(extra), predicted_rows=len(valid),
                 convergence_warnings=sum(issubclass(w.category, ConvergenceWarning) for w in caught))
    if number:
        stats['score'] = float(balanced_accuracy_score(valid.model, result.idxmax(axis=1)))
    return result, stats


def run_variant(name):
    info = report['variants'][name]
    start = time.monotonic()
    if name == 'K3':
        if not RUN_K3:
            info.update(status='skipped', reason=K3_DISABLED)
            return
        scores = [report['variants'][k]['oof'] for k in ('K0','K1','K2')]
        gain = scores[2] - max(scores[:2])
        if gain < MIN_GAIN:
            info.update(status='skipped', reason=f'K2 gain {gain:.8f} below {MIN_GAIN}')
            return
        if session_seconds_left() <= 3600:
            info.update(status='skipped', reason='Session already within final hour')
            return
    ids = sample_ids(name)
    extra = eligible.loc[ids, ['text','model']]
    extra[['model']].to_csv(WORK / f'sample_ids_{name}.csv')
    info.update(status='running', rows=len(extra), counts=extra.model.value_counts().to_dict(), folds=[])
    checkpoint()
    def submit(executor, n):
        mask = folds.eq(n)
        return executor.submit(fit_predict, train.loc[~mask, ['text','model']], extra,
                               train.loc[mask, ['id','text','model']], CONFIG, n)
    free_before = available_gib()
    print(f'{name}: {free_before:.2f} GiB available before pilot', flush=True)
    if os.name != 'nt':
        require(free_before > 4, 'Insufficient RAM for solo pilot with 4 GiB reserve')
    pilot_start = time.monotonic()
    with ProcessPoolExecutor(max_workers=1) as executor:
        first, stats = submit(executor, 1).result()
    pilot_seconds = time.monotonic() - pilot_start
    info['folds'].append(stats)
    free = available_gib()
    jobs = worker_count(free, stats['worker_peak_gib'])
    info.update(pilot_wall_seconds=pilot_seconds, free_gib_after_pilot=free, n_jobs=jobs)
    checkpoint()
    if jobs < 1:
        info.update(status='skipped', reason='No worker fits in measured RAM budget',
                    seconds=time.monotonic()-start)
        if name != 'K3':
            checkpoint()
            raise RuntimeError(f'{name}: {info["reason"]}')
        return
    if name == 'K3':
        budget = session_seconds_left() - 3600
        estimate = pilot_seconds * math.ceil(4 / jobs) * 1.8 + pilot_seconds
        info.update(estimated_remaining_seconds=estimate, budget_seconds=budget)
        if estimate >= budget:
            info.update(status='skipped', reason='Estimated remaining folds exceed session budget',
                        seconds=time.monotonic()-start)
            return
    parts = [first]
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = [submit(executor, n) for n in range(2, 6)]
        for future in as_completed(futures):
            part, stats = future.result()
            parts.append(part)
            info['folds'].append(stats)
            print(name, stats, flush=True)
            checkpoint()
    oof = pd.concat(parts, verify_integrity=True)
    require(len(oof) == len(train) and train.id.isin(oof.index).all(), 'Incomplete OOF coverage')
    oof = oof.loc[train.id]
    score = float(balanced_accuracy_score(train.model, oof.idxmax(axis=1)))
    oof.to_csv(WORK / f'oof_pool_{name}.csv')
    info['folds'].sort(key=lambda f: f['fold'])
    info.update(status='complete', oof=score, delta=score-BASELINE, seconds=time.monotonic()-start,
                convergence_warnings=sum(f['convergence_warnings'] for f in info['folds']))
    print(name, 'OOF', score, flush=True)
    if name == 'K0' and abs(score - K0_TARGET) > K0_TOLERANCE:
        info.update(status='failed', reason='K0 reproduction outside tolerance')
        for k in ('K1','K2','K3'):
            report['variants'][k].update(status='skipped', reason='K0 reproduction failed')
        checkpoint()
        raise RuntimeError(f'K0 OOF {score:.8f} outside tolerance. Versions: {report["environment"]}')
'''

FINAL = '''if FINAL_ONLY:
    previous = json.loads(paths['metrics.json'].read_text(encoding='utf-8'))
    variants = previous['variants']
    k2 = variants['K2']
    require(k2['status'] == 'complete' and abs(k2['oof'] - K2_TARGET) <= 1e-8,
            'Historical K2 status or OOF mismatch')
    report['variants'] = {k: dict(v) for k, v in variants.items()}
    if not RUN_K3:
        report['variants']['K3'] = dict(status='skipped', reason=K3_DISABLED)
else:
    for name in ('K0', 'K1', 'K2', 'K3'):
        run_variant(name)
        checkpoint()
    variants = report['variants']

complete = [k for k, v in variants.items() if v['status'] == 'complete']
winner = max(complete, key=lambda k: variants[k]['oof'])
if FINAL_ONLY:
    require(winner == 'K2', f'Historical winner must be K2, found {winner}')
    sample_path = paths['sample_ids_K2.csv']
else:
    sample_path = WORK / f'sample_ids_{winner}.csv'
saved = pd.read_csv(sample_path, dtype='string')
require(saved.id.is_unique and saved.id.isin(eligible.index).all(), 'Duplicate or ineligible pool ID')
require(np.array_equal(saved.model, eligible.loc[saved.id, 'model']), 'Pool sample label mismatch')
if FINAL_ONLY:
    require(len(saved) == LARGE * len(LABELS) and set(saved.model) == set(LABELS)
            and saved.model.value_counts().eq(LARGE).all(), 'Invalid K2 sample size or per-label counts')
estimate = float(variants[winner]['pilot_wall_seconds'])
remaining = session_seconds_left()
report.update(winner=winner, winner_oof=variants[winner]['oof'], final_only=FINAL_ONLY,
              refit_estimated_seconds=estimate, session_seconds_before_refit=remaining)
checkpoint()
require(0 < estimate < remaining, f'Refit needs about {estimate:.0f}s; session has {remaining:.0f}s')
with ProcessPoolExecutor(max_workers=1) as executor:
    probs, fit_stats = executor.submit(fit_predict, train[['text','model']],
        eligible.loc[saved.id, ['text','model']], test[['id','text']], CONFIG, 0).result()
submission = probs.idxmax(axis=1).rename('model').reset_index()
validate_submission(test, submission)
submission.to_parquet(WORK / 'submission.parquet', index=False)
main(['validate', '--test', str(paths['test.csv']), '--submission', str(WORK / 'submission.parquet')])
report.update(refit=fit_stats, submission_validated=True, submission_sha256=sha256(WORK / 'submission.parquet'))
checkpoint()
print('Validated submission.parquet. Selected by pooled OOF:', winner)
'''


def build():
    baseline = (ROOT / 'src/baseline.py').read_text(encoding='utf-8')
    config = json.loads((ROOT / 'configs/char_word.json').read_text())
    # Execute the complete source as an imported module, preserving its CLI guard.
    assert "'''" not in baseline
    embedded = ("baseline_source = r'''" + baseline + "'''\n"
                "baseline_module = {'__name__': 'embedded_baseline'}\n"
                "exec(compile(baseline_source, 'src/baseline.py', 'exec'), baseline_module)\n"
                "for symbol in ('LABELS','normalized','read_input','connected_groups','build_pipeline',"
                "'require','sha256','environment','validate_submission','main'):\n"
                "    globals()[symbol] = baseline_module[symbol]\n")
    setup = SETUP.replace('__FOLD_HASH__', repr(FOLD_HASH)).replace('__POOL_HASH__', repr(POOL_HASH))
    cells = [cell('code', 'setup', setup.replace('__CONFIG__', repr(config))),
             cell('markdown', 'intro', '# Name That LLM · Final CPU refit\n\n'
                  'Attach the competition, Open Pool v19, ntl-cv-assets and ntl-1b2-v1. Select CPU and Internet OFF. '
                  'Run all cells in a fresh session. Default outputs include metrics_final.json and submission.parquet.'),
             cell('code', 'baseline', embedded), cell('code', 'resources', RESOURCES),
             cell('code', 'inputs', INPUTS), cell('code', 'experiment', FUNCTIONS),
             cell('code', 'run', FINAL),
             cell('markdown', 'credit', 'Pool source: **Name That LLM Open Pool**, version 19, by '
                  '**crtproduction**. [Dataset](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool) '
                  '· [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).')]
    for c in cells:
        if c['cell_type'] == 'code':
            ast.parse(''.join(c['source']))
    notebook = dict(cells=cells, nbformat=4, nbformat_minor=5, metadata={
        'kernelspec': {'name':'python3','language':'python','display_name':'Python 3'},
        'language_info': {'name':'python'}, 'kaggle': {'isInternetEnabled':False,'isGpuEnabled':False}})
    with BOOK.open('w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(BOOK)

def smoke_test():
    """Execute the saved notebook's Python cells on small, separately mounted inputs."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import balanced_accuracy_score
    from hashlib import sha256

    run = ROOT / 'outputs' / f'pool_1b2_smoke_{time.time_ns()}'
    mounts = run / 'input'
    comp = mounts / 'competitions/name-that-llm'
    assets = mounts / 'datasets/private/ntl-cv-assets'
    pooldir = mounts / 'datasets/crtproduction/name-that-llm-open-pool'
    for directory in (comp, assets, pooldir):
        directory.mkdir(parents=True)
    real = pd.read_csv(ROOT / 'train.csv', dtype='string')
    original = ROOT / 'outputs/cv_assets'
    folds = pd.read_csv(original / 'folds.csv', dtype={'id':'string','prompt_id':'string'})
    prompts = folds[['fold','prompt_id']].drop_duplicates().groupby('fold').head(3)
    mask = folds.prompt_id.isin(prompts.prompt_id)
    folds.loc[mask].to_csv(assets / 'folds.csv', index=False)
    real.loc[mask].to_csv(comp / 'train.csv', index=False)
    shutil.copyfile(ROOT / 'test.csv', comp / 'test.csv')
    base = pd.read_csv(original / 'oof_tfidf.csv').loc[mask]
    base.to_csv(assets / 'oof_tfidf.csv', index=False)
    base_score = balanced_accuracy_score(real.loc[mask, 'model'], base.drop(columns='id').idxmax(axis=1))
    pool_path = 'ntl_open_pool.parquet'
    source_pool = ROOT / 'data/open-pool' / pool_path
    pool_index = pd.read_parquet(source_pool, columns=['id', 'model']).set_index('id')
    b = pd.read_csv(ROOT / 'outputs/pool_check/sample_ids_B.csv').groupby('model').head(10)
    b.to_csv(assets / 'sample_ids_B.csv', index=False)
    other = pool_index.loc[~pool_index.index.isin(b.id)].groupby('model').head(40)
    keep = list(b.id) + list(other.index)
    tiny = pd.read_parquet(source_pool, filters=[('id', 'in', keep)],
                           columns=['id','model','text','match_weight','gen_reasoning'])
    tiny = tiny.set_index('id').loc[keep].reset_index()
    tiny.groupby('model').tail(10)[['id','model']].to_csv(assets / 'sample_ids_A.csv', index=False)
    tiny.to_parquet(pooldir / pool_path, index=False)
    del pool_index
    def digest(path):
        return sha256(path.read_bytes()).hexdigest()

    work = run / 'working'
    def execute(output_dir, final_only):
        namespace = {'__name__': '__main__'}
        for item in json.loads(BOOK.read_text(encoding='utf-8'))['cells']:
            if item['cell_type'] != 'code':
                continue
            exec(compile(''.join(item['source']), item['id'], 'exec'), namespace)
            if item['id'] == 'setup':
                namespace.update(INPUT=mounts, WORK=output_dir, FINAL_ONLY=final_only,
                                 EXPECTED_FOLDS=digest(assets / 'folds.csv'),
                                 EXPECTED_POOL=digest(pooldir / pool_path), SMALL=10, LARGE=15,
                                 BASELINE=base_score, K0_TOLERANCE=1)
        return namespace

    namespace = execute(work, False)
    result = json.loads((work / 'metrics.json').read_text())
    for name, info in result['variants'].items():
        if name == 'K3':
            assert info == dict(status='skipped', reason=namespace['K3_DISABLED'])
            assert not list(work.glob('*_K3.csv'))
            continue
        assert info['status'] == 'complete' and len(info['folds']) == 5
        oof = pd.read_csv(work / f'oof_pool_{name}.csv', dtype={'id':'string'})
        assert list(oof) == ['id', *namespace['LABELS']] and oof.id.tolist() == folds.id[mask].tolist()
        assert np.isclose(balanced_accuracy_score(real.model[mask], oof.drop(columns='id').idxmax(axis=1)),
                          info['oof'])
        sampled = pd.read_csv(work / f'sample_ids_{name}.csv')
        assert len(sampled) == info['rows'] and sampled.id.is_unique
    assert result['submission_validated'] and result['winner'] == max(
        ('K0','K1','K2'), key=lambda k: result['variants'][k]['oof'])
    assert digest(work / 'folds.csv') == digest(assets / 'folds.csv')
    prior = mounts / 'datasets/private/ntl-1b2-v1'
    prior.mkdir(parents=True)
    shutil.copyfile(ROOT / 'outputs/kaggle_1b2_v1/metrics.json', prior / 'metrics.json')
    for filename in ('sample_ids_K2.csv', 'folds.csv'):
        shutil.copyfile(work / filename, prior / filename)
    final = run / 'final'
    execute(final, True)
    final_result = json.loads((final / 'metrics_final.json').read_text())
    assert final_result['winner'] == 'K2' and final_result['submission_validated']
    assert final_result['refit']['real_train'] == int(mask.sum())
    assert final_result['refit']['pool_train'] == 15 * 18
    assert final_result['submission_sha256'] == digest(final / 'submission.parquet')
    assert not list(final.glob('oof_pool_*.csv')) and not list(final.glob('sample_ids_K*.csv'))
    print('Both smoke tests passed:', run)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-test', action='store_true', help='Test the existing notebook on tiny fixtures')
    if parser.parse_args().smoke_test:
        smoke_test()
    else:
        build()
