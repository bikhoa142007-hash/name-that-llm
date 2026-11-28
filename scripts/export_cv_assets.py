"""Export the verified v3 CV assets on Windows, from the repository root."""
import argparse
import ctypes
import json
import shutil
import struct
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baseline import (LABELS, build_pipeline, check_disjoint, connected_groups,
                          environment, fit_checked, read_input, require, save_json)

EXPECTED = (0.357791, 0.362963, 0.320370, 0.318519, 0.337037)


def available_gib():
    status = ctypes.create_string_buffer(64)  # MEMORYSTATUSEX
    struct.pack_into('I', status, 0, len(status))
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(status):
        raise ctypes.WinError()
    return struct.unpack_from('Q', status, 16)[0] / 2**30


class MemoryCounters(ctypes.Structure):
    _fields_ = [('size', ctypes.c_uint32), ('faults', ctypes.c_uint32),
                ('peak', ctypes.c_size_t), ('working_set', ctypes.c_size_t),
                ('other', ctypes.c_size_t * 6)]


def memory():
    counters = MemoryCounters()
    counters.size = ctypes.sizeof(counters)
    if not ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.c_void_p(-1), ctypes.byref(counters), counters.size):
        raise ctypes.WinError()
    return counters


@contextmanager
def measure(results, name):
    start = time.perf_counter()
    peak = memory().working_set
    done = threading.Event()

    def sample():
        nonlocal peak
        while not done.wait(0.1):
            peak = max(peak, memory().working_set)

    worker = threading.Thread(target=sample)
    worker.start()
    try:
        yield
    finally:
        done.set()
        worker.join()
        results[name] = dict(seconds=round(time.perf_counter() - start, 3),
                             sampled_peak_mib=round(max(peak, memory().working_set) / 2**20, 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'cv_assets')
    output = parser.parse_args().output_dir.resolve()
    start = time.perf_counter()
    free = available_gib()
    print(f'Available RAM: {free:.3f} GiB', flush=True)
    require(free >= 6, 'Below 6 GiB. Close unused apps before training.')
    require(not output.exists(), f'Output already exists: {output}')
    steps = {}
    with measure(steps, 'prepare'):
        config = json.loads((ROOT / 'configs/char_word.json').read_text())
        require(config['seed'] == 42, 'Unexpected baseline seed')
        frame, audit = read_input(ROOT / 'train.csv', labeled=True)
        groups, _ = connected_groups(frame)
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        folds = frame[['id', 'prompt_id', 'model']].copy()
        folds['fold'] = 0
        oof = np.full((len(frame), len(LABELS)), np.nan)
    scores = []
    for fold, (train, valid) in enumerate(splitter.split(frame.text, frame.model, groups), 1):
        with measure(steps, f'fold_{fold}'):
            check_disjoint(frame, groups, train, valid)
            model = build_pipeline(config)
            fit_checked(model, frame.text.iloc[train], frame.model.iloc[train])
            with threadpool_limits(1):
                probabilities = model.predict_proba(frame.text.iloc[valid])
            order = pd.Index(model.classes_).get_indexer(LABELS)
            require((order >= 0).all(), 'Missing class in probability columns')
            oof[valid] = probabilities[:, order]
            folds.loc[valid, 'fold'] = fold
            score = balanced_accuracy_score(frame.model.iloc[valid],
                                            np.asarray(LABELS)[oof[valid].argmax(axis=1)])
            scores.append(float(score))
        print(f'Fold {fold}: {score:.6f} {steps[f"fold_{fold}"]}', flush=True)
        require(round(score, 6) == EXPECTED[fold - 1], 'Fold score differs from reference')
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_name(output.name + '.tmp')
    staged.mkdir()
    try:
        with measure(steps, 'export_verify'):
            folds.to_csv(staged / 'folds.csv', index=False)
            pd.DataFrame(oof, index=frame.id, columns=LABELS).to_csv(staged / 'oof_tfidf.csv')
            saved = pd.read_csv(staged / 'folds.csv', dtype=frame.dtypes.drop('text').to_dict())
            pd.testing.assert_frame_equal(folds, saved)
            probs = pd.read_csv(staged / 'oof_tfidf.csv', dtype={'id': 'string'})
            require(list(probs) == ['id', *LABELS] and probs.id.equals(frame.id),
                    'Unexpected OOF columns or ID order')
            p = probs[LABELS].to_numpy()
            require(np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
                    and np.allclose(p.sum(axis=1), 1), 'Invalid OOF probabilities')
            pred = np.asarray(LABELS)[p.argmax(axis=1)]
            for fold, expected in enumerate(EXPECTED, 1):
                mask = saved.fold.eq(fold)
                require(round(balanced_accuracy_score(saved.model[mask], pred[mask]), 6)
                        == expected, 'Exported fold score differs from reference')
            overall = float(balanced_accuracy_score(saved.model, pred))
            require(abs(overall - 0.3393) <= 0.0001, f'OOF gate failed: {overall:.8f}')
        report = dict(oof=overall, scores=scores, steps=steps, available_gib=free,
                      process_peak_mib=memory().peak / 2**20, seconds=time.perf_counter()-start,
                      train_sha256=audit['sha256'], environment=environment())
        save_json(staged / 'metrics.json', report)
        staged.rename(output)
    except BaseException:
        shutil.rmtree(staged)
        raise
    print(f'OOF {overall:.8f}; assets verified in {output}')


if __name__ == '__main__':
    main()
