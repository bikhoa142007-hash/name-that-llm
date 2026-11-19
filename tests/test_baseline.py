"""Small synthetic fixtures for invariants only; never competition training data."""
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.baseline import (LABELS, assign_by_topic, build_pipeline, fit_checked, main, make_split,
                          read_input, validate_submission)


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

    def test_cv_command_on_grouped_fixture(self):
        rows = [{'id': f'{p}-{k}', 'prompt_id': f'p{p}', 'text': f'fixture {p} model {k}', 'model': label}
                for p in range(8) for k, label in enumerate(LABELS)]
        frame = pd.DataFrame(rows)
        frame.loc[0, 'text'] = 'café duplicate'
        frame.loc[len(LABELS) + 1, 'text'] = 'café   duplicate'
        out = self.root / 'cv'
        main(['cv', '--train', str(self.csv(frame)), '--config', 'configs/baseline.json', '--output-dir', str(out)])
        result = json.loads((out / 'cv_metrics.json').read_text(encoding='utf-8'))
        self.assertEqual(len(result['folds']), 5)
        self.assertEqual(sum(f['validation_rows'] for f in result['folds']), 8 * len(LABELS))
        for fold in result['folds']:
            self.assertEqual(fold['absent_classes'], {'train': [], 'validation': []})
            self.assertEqual((fold['prompt_overlap'], fold['exact_text_overlap'], fold['normalized_text_overlap']), (0, 0, 0))
            self.assertEqual(set(fold['argmax_recall_per_class']), set(LABELS))
        self.assertIn('sc3_balanced_folds_pass', result['summary'])

    def test_topic_assignment_falls_back_below_class_count(self):
        probabilities = np.eye(len(LABELS))[:5]
        np.testing.assert_array_equal(assign_by_topic(['a'] * 5, probabilities, LABELS), np.array(LABELS[:5]))

    def test_topic_assignment_stopword_only_texts_use_argmax(self):
        rng = np.random.default_rng(0)
        probabilities = rng.dirichlet(np.ones(len(LABELS)), size=len(LABELS))
        expected = np.asarray(LABELS)[probabilities.argmax(axis=1)]
        np.testing.assert_array_equal(assign_by_topic(['the and'] * len(LABELS), probabilities, LABELS), expected)

    def test_topic_assignment_distinct_labels_per_cluster_and_order_invariant(self):
        k = len(LABELS)

        def block():
            p = np.full((k, k), 0.01)
            np.fill_diagonal(p, 0.83)
            return p
        first, second = block(), block()
        first[0, :2], first[1, :2] = [0.55, 0.29], [0.45, 0.39]  # unique optimum; row 1 argmax differs
        stop = np.full((1, k), 0.01)
        stop[0, 0] = 1 - 0.01 * (k - 1)
        probabilities = np.vstack([stop, first, second])
        texts = (['the and'] + [f'galaxy nebula orbit telescope item{i}' for i in range(k)]
                 + [f'recipe flour butter oven dish{i}' for i in range(k)])
        ids = np.arange(len(texts))
        result = assign_by_topic(texts, probabilities, LABELS)
        argmax = np.asarray(LABELS)[probabilities.argmax(axis=1)]
        self.assertEqual(result[0], argmax[0])
        self.assertEqual(len(set(result[1:k + 1])), k)
        self.assertEqual(len(set(result[k + 1:])), k)
        self.assertEqual(int((result != argmax).sum()), 1)
        order = ids[::-1]
        reversed_result = assign_by_topic([texts[i] for i in order], probabilities[order], LABELS)
        self.assertEqual(dict(zip(order, reversed_result)), dict(zip(ids, result)))


if __name__ == '__main__':
    unittest.main()
