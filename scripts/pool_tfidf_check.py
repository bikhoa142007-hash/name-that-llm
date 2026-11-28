"""Phase 1b-1: compare two balanced open-pool samples on the locked real folds."""
import json
import multiprocessing as mp
import shutil
import time

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from export_cv_assets import ROOT, available_gib, measure, memory
from src.baseline import (LABELS, build_pipeline, connected_groups, environment, fit_checked,
                          normalized, read_input, require, save_json, sha256)

ASSETS = ROOT / 'outputs/cv_assets'
POOL = ROOT / 'data/open-pool/ntl_open_pool.parquet'
OUTPUT = ROOT / 'outputs/pool_check'
N = 1100
SEED = 42


def prepare_samples(train, report):
    columns = ['id', 'text', 'model', 'prompt_group', 'prompt_canon', 'match_weight',
               'gen_reasoning', 'flag_kb_proxy', 'max_cos_train', 'max_sim_train_prompt']
    pool = pd.read_parquet(POOL, columns=columns)
    require(pool.id.notna().all() and pool.id.is_unique, 'Missing or repeated pool ID')
    require(pool.text.notna().all() and set(pool.model) == set(LABELS), 'Invalid pool data')
    text_keys = set(train.text.map(normalized))
    duplicate = pool.text.map(lambda text: normalized(text) in text_keys)
    prompts = pool.prompt_canon.fillna('').map(normalized)
    prompt_col = next((col for col in ('prompt', 'prompt_text', 'prompt_canon') if col in train), None)
    excluded = duplicate.copy()
    prompt_matches = None
    if prompt_col:
        matches = prompts.ne('') & prompts.isin(train[prompt_col].dropna().map(normalized))
        prompt_matches = int(matches.sum())
        excluded |= matches
    report['pool'] = dict(source='crtproduction/name-that-llm-open-pool v19, CC BY-SA 4.0',
                          sha256=sha256(POOL), rows=len(pool), exact_text_matches=int(duplicate.sum()),
                          prompt_canon_present=int(prompts.ne('').sum()), train_columns=list(train),
                          train_prompt_column=prompt_col, exact_prompt_matches=prompt_matches,
                          max_cos_train_ge_025=int(pool.max_cos_train.ge(.25).sum()),
                          max_sim_train_prompt_max=float(pool.max_sim_train_prompt.max()),
                          excluded=int(excluded.sum()), eligible=int((~excluded).sum()))
    eligible = pool.loc[~excluded]
    samples = {}
    for variant in ('A', 'B'):
        candidates = eligible
        weights = None
        cap = None
        if variant == 'B':
            candidates = eligible.loc[eligible.model.ne('kimi-k3') | eligible.gen_reasoning.eq('high')]
            weights = candidates.match_weight
            require(np.isfinite(weights).all() and weights.ge(0).all(), 'Invalid match weights')
            cap = float(weights.quantile(.99))
            weights = weights.clip(upper=cap)
        rng = np.random.default_rng(SEED)
        parts = []
        for label in LABELS:
            rows = candidates.loc[candidates.model.eq(label)]
            probabilities = None
            if weights is not None:
                w = weights.loc[rows.index].to_numpy(dtype=float)
                require(np.count_nonzero(w) >= N, f'Insufficient positive weights for {label}')
                probabilities = w / w.sum()
            # Sequential weighted draws redistribute probability over remaining rows.
            positions = rng.choice(len(rows), size=N, replace=False, p=probabilities)
            parts.append(rows.iloc[positions])
        sample = pd.concat(parts, ignore_index=True)
        samples[variant] = sample[['id', 'model', 'text']].copy()
        report[variant] = dict(seed=SEED, candidates=len(candidates), rows=len(sample),
                               sampling='uniform' if variant == 'A' else 'weighted, global eligible p99; kimi high',
                               weight_cap=cap, class_counts=sample.model.value_counts().to_dict(),
                               proxy_rows=int(sample.flag_kb_proxy.sum()), folds=[])
    return samples


def fit_fold(config, training, validation, sender):
    # A fresh process gives an OS-measured peak for each fold and a hard timeout.
    start = time.perf_counter()
    with threadpool_limits(1):
        model = build_pipeline(config)
        fit_checked(model, training.text, training.model)
        order = pd.Index(model.classes_).get_indexer(LABELS)
        require((order >= 0).all(), 'Missing probability columns')
        probs = model.predict_proba(validation)[:, order]
    sender.send((probs, memory().peak / 2**20, time.perf_counter() - start))
    sender.close()


def timed_fold(config, training, validation):
    start = time.perf_counter()
    receiver, sender = mp.Pipe(duplex=False)
    proc = mp.Process(target=fit_fold, args=(config, training, validation, sender))
    proc.start()
    sender.close()
    try:
        require(receiver.poll(max(0, 900 - (time.perf_counter() - start))),
                f'Fold timed out after {time.perf_counter() - start:.1f} s')
        result = receiver.recv()
        proc.join()
        require(proc.exitcode == 0, f'Fold process exited with {proc.exitcode}')
        return result
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join()
        receiver.close()
        proc.close()


def evaluate(variant, sample, train, folds, config, baseline, staged, report):
    stats = report[variant]
    oof = pd.DataFrame(np.nan, index=train.index, columns=LABELS)
    sample[['id', 'model']].to_csv(staged / f'sample_ids_{variant}.csv', index=False)
    for fold in range(1, 6):
        valid = folds.fold.eq(fold)
        training = pd.concat([train.loc[~valid, ['text', 'model']], sample[['text', 'model']]],
                             ignore_index=True)
        name = f'{variant}_fold_{fold}'
        print(f'Start {name}', flush=True)
        with measure(report['steps'], name):
            probs, worker_peak, fit_predict_s = timed_fold(config, training, train.text[valid])
            require(np.isfinite(probs).all() and (probs >= 0).all() and (probs <= 1).all()
                    and np.allclose(probs.sum(axis=1), 1), 'Invalid probabilities')
            oof.loc[valid] = probs
            score = float(balanced_accuracy_score(train.model[valid], np.asarray(LABELS)[probs.argmax(1)]))
        step = report['steps'][name]
        row = dict(fold=fold, real_train=int((~valid).sum()), real_validation=int(valid.sum()),
                   score=score, delta=score-baseline['scores'][fold-1], worker_peak_mib=worker_peak,
                   parent_sampled_peak_mib=step['sampled_peak_mib'], seconds=step['seconds'],
                   fit_predict_seconds=fit_predict_s)
        stats['folds'].append(row)
        print(name, json.dumps(row), flush=True)
        require(step['seconds'] <= 900, f'{name} exceeded 900 s')
    require(oof.notna().all().all(), 'Incomplete OOF')
    path = staged / f'oof_pool_{variant}.csv'
    oof.set_index(train.id).to_csv(path)
    saved = pd.read_csv(path, dtype={'id': 'string'})
    require(list(saved) == ['id', *LABELS] and saved.id.equals(train.id), 'Invalid OOF layout')
    np.testing.assert_allclose(saved[LABELS], oof, rtol=1e-12)
    stats['oof'] = float(balanced_accuracy_score(train.model, saved[LABELS].idxmax(axis=1)))
    stats['delta'] = stats['oof'] - baseline['oof']
    stats['folds_improved'] = sum(row['delta'] > 0 for row in stats['folds'])
    print(f"{variant} OOF {stats['oof']:.8f}", flush=True)


def main():
    start = time.perf_counter()
    free = available_gib()
    print(f'Available RAM: {free:.3f} GiB', flush=True)
    require(free >= 6, 'Below 6 GiB. Close unused apps before training.')
    require(not OUTPUT.exists(), f'Output already exists: {OUTPUT}')
    report = dict(available_gib=free, steps={}, environment=environment())
    with measure(report['steps'], 'prepare'):
        train, audit = read_input(ROOT / 'train.csv', labeled=True)
        folds = pd.read_csv(ASSETS / 'folds.csv', dtype={'id':'string', 'prompt_id':'string', 'model':'string'})
        pd.testing.assert_frame_equal(folds[['id', 'prompt_id', 'model']], train[['id', 'prompt_id', 'model']])
        groups, _ = connected_groups(train)
        require(set(folds.fold) == set(range(1, 6)) and folds.groupby(groups).fold.nunique().max() == 1,
                'Invalid fold assignments')
        report['folds_sha256'] = sha256(ASSETS / 'folds.csv')
        require(report['folds_sha256'] == 'e8eb76023a9679528b5f471ab0cdb6f00e7292b177ce5cea69fc2d8a15393106',
                'Fold manifest differs from verified Phase 1')
        baseline = json.loads((ASSETS / 'metrics.json').read_text())
        require(audit['sha256'] == baseline['train_sha256'] and round(baseline['oof'], 8) == .33929903,
                'Baseline mismatch')
        report['baseline'] = {key: baseline[key] for key in ('train_sha256', 'oof', 'scores')}
        config = json.loads((ROOT / 'configs/char_word.json').read_text())
        samples = prepare_samples(train, report)
    print(json.dumps(report['pool']), flush=True)
    staged = OUTPUT.with_name(OUTPUT.name + '.tmp')
    staged.mkdir()
    try:
        for variant, sample in samples.items():
            evaluate(variant, sample, train, folds, config, baseline, staged, report)
        report['parent_peak_mib'] = memory().peak / 2**20
        report['seconds'] = time.perf_counter() - start
        save_json(staged / 'metrics.json', report)
        staged.rename(OUTPUT)
    except BaseException:
        shutil.rmtree(staged)
        raise


if __name__ == '__main__':
    main()
