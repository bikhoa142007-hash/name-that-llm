"""Small synthetic fixtures for invariants only; never competition training data."""
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import pandas as pd

from src.baseline import LABELS, build_pipeline, fit_checked, make_split, read_input, validate_submission


class BaselineChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def csv(self, frame, name='input.csv'):
        path = self.root / name
        frame.to_csv(path, index=False)
        return path

    def test_missing_column_and_duplicate_id_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing columns'):
            read_input(self.csv(pd.DataFrame({'id': ['a']})))
        with self.assertRaisesRegex(ValueError, 'unique'):
            read_input(self.csv(pd.DataFrame({'id': ['a', 'a'], 'text': ['x', 'y']})))

    def test_null_blank_ids_and_duplicate_headers_rejected(self):
        for value in ['', '  ']:
            with self.assertRaisesRegex(ValueError, 'IDs contain'):
                read_input(self.csv(pd.DataFrame({'id': [value], 'text': ['x']})))
        path = self.root / 'duplicate.csv'
        path.write_text('id,text,text\na,x,y\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Duplicate column'):
            read_input(path)

    def test_null_test_rows_preserved_and_literal_na_kept(self):
        path = self.csv(pd.DataFrame({'id': ['01', '02', '03'], 'text': [None, ' ', 'NA']}))
        with warnings.catch_warnings(record=True) as caught:
            frame, audit = read_input(path)
        self.assertEqual(frame['id'].tolist(), ['01', '02', '03'])
        self.assertEqual(frame['text'].tolist(), ['', ' ', 'NA'])
        self.assertEqual(audit['blank_text_including_null'], 2)
        self.assertEqual(len(caught), 1)

    def test_missing_classes_and_blank_training_text_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing official training classes'):
            read_input(self.csv(pd.DataFrame({'id': ['a'], 'text': ['sample'], 'model': [LABELS[0]]})), True)
        with self.assertRaisesRegex(ValueError, 'Blank/null training'):
            read_input(self.csv(pd.DataFrame({'id': ['a'], 'text': [''], 'model': [LABELS[0]]})), True)

    def test_union_of_prompt_and_normalized_duplicates_cannot_leak(self):
        rows = [{'id': f'{p}-{k}', 'prompt_id': f'p{p}', 'text': f'fixture {p} model {k}', 'model': label}
                for p in range(8) for k, label in enumerate(LABELS)]
        frame = pd.DataFrame(rows)
        frame.loc[0, 'text'] = 'caf\u00e9 duplicate'
        frame.loc[len(LABELS) + 1, 'text'] = 'cafe\u0301   duplicate'
        config = {'seed': 42, 'validation_fraction': .25}
        a, b, groups, audit = make_split(frame, config)
        self.assertEqual(groups[0], groups[len(LABELS) + 1])
        self.assertFalse(set(groups[a]) & set(groups[b]))
        self.assertEqual(audit['normalized_duplicate_extra_rows'], 1)
        self.assertEqual(audit['normalized_duplicate_conflicting_label_groups'], 1)
        self.assertEqual(audit['prompt_overlap'], 0)

    def test_pipeline_does_not_learn_validation_vocabulary(self):
        config = json.loads(Path('configs/baseline.json').read_text(encoding='utf-8'))
        model = build_pipeline(config)
        model.fit(['alpha alpha', 'alpha beta', 'beta beta', 'beta gamma'], ['a', 'a', 'b', 'b'])
        before = dict(model.named_steps['tfidf'].vocabulary_)
        model.predict(['\u03a9\u03a9\u03a9 validation exclusive'])
        self.assertEqual(before, model.named_steps['tfidf'].vocabulary_)
        self.assertNotIn('\u03a9\u03a9\u03a9', before)

    def test_submission_rejects_reordered_ids_columns_and_bad_label(self):
        test = pd.DataFrame({'id': ['001', '002']})
        good = pd.DataFrame({'id': ['001', '002'], 'model': LABELS[:2]})
        self.assertTrue(validate_submission(test, good)['valid_labels'])
        for bad in [good.iloc[::-1], good[['model', 'id']], good.assign(model='not-a-model'),
                    good.assign(model=None), good.assign(id=['001', '001'])]:
            with self.assertRaises(ValueError):
                validate_submission(test, bad)

    def test_optimizer_stopping_without_learning_is_rejected(self):
        config = json.loads(Path('configs/baseline.json').read_text(encoding='utf-8'))
        config['logistic_regression']['tol'] = 1e9
        model = build_pipeline(config)
        with self.assertRaisesRegex(ValueError, 'without learning'):
            fit_checked(model, ['alpha alpha', 'alpha beta', 'beta beta', 'beta gamma'], ['a', 'a', 'b', 'b'])

    def test_empty_and_one_row_test_round_trip(self):
        for n in [0, 1]:
            frame = pd.DataFrame({'id': pd.Series(['0001'][:n], dtype='string'),
                                  'text': pd.Series(['one example'][:n], dtype='string')})
            test, _ = read_input(self.csv(frame))
            output = pd.DataFrame({'id': test['id'], 'model': pd.Series([LABELS[0]][:n], dtype='string')})
            path = self.root / f'{n}.parquet'
            output.to_parquet(path, index=False)
            self.assertEqual(validate_submission(test, pd.read_parquet(path))['rows'], n)


if __name__ == '__main__':
    unittest.main()
