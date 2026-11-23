"""Build a self-contained Kaggle notebook from the tested CPU baseline source.

No data or credentials are embedded. Existing output notebooks are never replaced.
Run from repository root: python scripts/build_kaggle_cpu_notebook.py
"""
import argparse
import ast
import json
from pathlib import Path


def cell(kind, identity, source):
    result = {'cell_type': kind, 'id': identity, 'metadata': {},
              'source': source.splitlines(keepends=True)}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    source = (root / 'src/baseline.py').read_text(encoding='utf-8')
    # Keep function definitions intact, but do not invoke the CLI on import.
    source = source.rsplit("if __name__ == '__main__':", 1)[0]
    ast.parse(source)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='notebooks/kaggle_cpu_standalone.ipynb',
                        help='Path relative to repository root; must not exist yet.')
    parser.add_argument('--config', default='configs/baseline.json',
                        help='Config embedded in the notebook, relative to repository root.')
    args = parser.parse_args()
    config = json.loads((root / args.config).read_text(encoding='utf-8'))
    features = 'character + word TF-IDF' if 'word_tfidf' in config else 'character TF-IDF'
    run = '''from pathlib import Path
import tempfile
import shutil
import json
import pandas as pd

DATA_DIR = Path('/kaggle/input/competitions/name-that-llm')
# Older Kaggle mounts can omit the competitions/ segment.
if not (DATA_DIR / 'train.csv').is_file():
    DATA_DIR = Path('/kaggle/input/name-that-llm')
assert (DATA_DIR / 'train.csv').is_file(), 'Attach Name That LLM competition data first'
assert (DATA_DIR / 'test.csv').is_file(), 'Missing competition test.csv'
WORK_DIR = Path('/kaggle/working')
RUN_DIR = Path(tempfile.mkdtemp(prefix='baseline-cpu-', dir=WORK_DIR))
# 'argmax' (default) or 'topic'. Switch to 'topic' only after you confirm the Rules allow it;
# its gain shrinks when per-model test counts are unequal (debates/1-huong-cai-tien-mo-hinh.md, E9).
ASSIGNMENT = 'argmax'
CONFIG = __CONFIG__
config_path = RUN_DIR / 'config.json'
config_path.write_text(json.dumps(CONFIG, indent=2), encoding='utf-8')
print('Environment:', json.dumps(environment(), indent=2))
print('Competition input:', DATA_DIR)
print('Run outputs:', RUN_DIR)
main(['train', '--train', str(DATA_DIR / 'train.csv'), '--config', str(config_path),
      '--artifact-dir', str(RUN_DIR / 'artifacts'), '--output-dir', str(RUN_DIR / 'evaluation')])
main(['evaluate', '--train', str(DATA_DIR / 'train.csv'),
      '--artifact-dir', str(RUN_DIR / 'artifacts'), '--output-dir', str(RUN_DIR / 'reload_check')])
main(['predict', '--test', str(DATA_DIR / 'test.csv'),
      '--artifact-dir', str(RUN_DIR / 'artifacts'), '--output', str(RUN_DIR / 'submission.parquet'),
      '--assignment', ASSIGNMENT])
# Copy only the newly validated prediction file into the required output location.
shutil.copyfile(RUN_DIR / 'submission.parquet', WORK_DIR / 'submission.parquet')
shutil.copyfile(RUN_DIR / 'submission.checks.json', WORK_DIR / 'submission.checks.json')
main(['validate', '--test', str(DATA_DIR / 'test.csv'),
      '--submission', str(WORK_DIR / 'submission.parquet')])
display(pd.read_parquet(WORK_DIR / 'submission.parquet').head())
print('SUCCESS: CPU baseline finished; submission.parquet was read back and validated.')
print('This run does not submit anything to the competition.')
'''.replace('__CONFIG__', repr(config))
    ast.parse(run)
    book = {'cells': [
        cell('markdown', 'introduction', '# Name That LLM — CPU TF-IDF baseline\n\n'
             f'Self-contained notebook generated from src/baseline.py and {args.config}. '
             'CPU only; Internet OFF; attach Name That LLM competition data. '
             'No external datasets, model downloads, paid APIs or automatic submissions.\n\n'
             f'Trains DummyClassifier and {features} + Logistic Regression on a grouped '
             'holdout, refits on all labeled data, then predicts the current test file. '
             'The public example row count is never hardcoded. Dependencies are supplied by '
             'the Kaggle runtime; training and inference use that same environment.\n'),
        cell('code', 'baseline-source', source), cell('code', 'run-baseline', run)],
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                     'language_info': {'name': 'python'},
                     'kaggle': {'isInternetEnabled': False, 'isGpuEnabled': False}},
        'nbformat': 4, 'nbformat_minor': 5}
    output = root / args.output
    with output.open('x', encoding='utf-8') as f:
        json.dump(book, f, ensure_ascii=False, indent=1)
    print(output)


if __name__ == '__main__':
    main()
