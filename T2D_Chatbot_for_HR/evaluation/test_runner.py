import json
import tempfile
import unittest
from pathlib import Path

from run_ragas import load_dataset, reconstruct_contexts, report


class EvaluationTests(unittest.TestCase):
    def test_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'dataset.json'
            row = dict(id='one', user_input='Question', reference='Answer', category='answerable', reference_sources=[])
            path.write_text(json.dumps([row, row]), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_dataset(path)

    def test_contexts_follow_api_order_and_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            version = 'a' * 32
            snapshot = root / 'versions' / version
            snapshot.mkdir(parents=True)
            chunks = [{'metadata': {'source_file': 'a.pdf', 'chunk_id': i}, 'text': text}
                      for i, text in enumerate(['first', 'second', 'excluded'])]
            (snapshot / 'chunks.json').write_text(json.dumps(chunks))
            result = {'index_version': version, 'sources': [
                {'document': 'a.pdf', 'chunk_id': 1}, {'document': 'a.pdf', 'chunk_id': 0}]}
            self.assertEqual(reconstruct_contexts(result, root),
                             ['[Source: a.pdf]\nsecond', '[Source: a.pdf]\nfirst'])

    def test_failed_metrics_are_not_reported_as_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'records').mkdir()
            row = dict(id='one', category='answerable')
            record = dict(row, metrics={'faithfulness': 0.5}, metric_errors={'context_recall': 'TimeoutError'},
                          metric_skips={}, latency_seconds=2)
            (root / 'records' / 'one.json').write_text(json.dumps(record))
            summary = report(root, [row], {'judge': 'test', 'index_version': 'a' * 32})
            self.assertEqual(summary['metrics']['faithfulness']['mean'], 0.5)
            self.assertIsNone(summary['metrics']['context_recall']['mean'])
            self.assertEqual(summary['metrics']['context_recall']['errors'], 1)
            self.assertFalse(summary['complete'])


if __name__ == '__main__':
    unittest.main()
