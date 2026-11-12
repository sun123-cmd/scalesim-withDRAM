"""Native stage-queue regression tests; set REPLAY_BINARY and REPLAY_CONFIG."""
import csv
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml


class SharedReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = Path(os.environ.get('REPLAY_BINARY', 'integration/build/ramulator2-replay')).resolve()
        cls.config = Path(os.environ.get('REPLAY_CONFIG', 'results/smoke/ramulator2/config.yaml')).resolve()
        if not cls.binary.is_file() or not cls.config.is_file():
            raise unittest.SkipTest('Build replay and provide REPLAY_BINARY and REPLAY_CONFIG')

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def run_trace(self, text, policy=None, allocations=None, success=True):
        trace = self.directory / 'requests.trace'
        trace.write_text(text)
        responses = self.directory / 'responses.csv'
        stats = self.directory / 'stats.yaml'
        command = [str(self.binary), str(self.config), str(trace), str(responses),
                   str(stats), '1000000', '4']
        if policy:
            command.append(policy)
        if allocations is not None:
            allocation = self.directory / 'allocations.csv'
            allocation.write_text(allocations)
            command.append(str(allocation))
        process = subprocess.run(command, capture_output=True, text=True)
        if not success:
            self.assertNotEqual(process.returncode, 0)
            return process.stderr
        self.assertEqual(process.returncode, 0, process.stderr)
        with responses.open() as response_file:
            rows = list(csv.DictReader(response_file))
        return rows, yaml.safe_load(stats.read_text())['replay']

    def test_legacy_four_column_trace(self):
        responses, stats = self.run_trace('0 0 R 0\n1 0 W 64\n')
        self.assertEqual(len(responses), 2)
        self.assertEqual(stats['completed'], 2)
        self.assertNotIn('stages', stats)

    def test_shared_sources_and_byte_conservation(self):
        text = ''.join(f'{i} 0 R {i*64} {i%2*3}\n' for i in range(40))
        responses, stats = self.run_trace(text, 'shared')
        self.assertEqual({int(row['id']) for row in responses}, set(range(40)))
        self.assertEqual(set(stats['stages']), {0, 3})
        for stage in stats['stages'].values():
            self.assertEqual(stage['completed_requests'], 20)
            self.assertEqual(stage['read_bytes'], 20*stats['transaction_bytes'])

    def test_stage_round_robin_avoids_frontend_head_of_line_blocking(self):
        text = ''.join(f'{i} 0 R {i*64} 0\n' for i in range(200)) + '200 0 R 1048576 1\n'
        responses, stats = self.run_trace(text, 'shared')
        completed = {int(row['id']): int(row['completion_cycle']) for row in responses}
        self.assertLess(completed[200], max(completed[i] for i in range(200)))
        self.assertEqual(stats['completed'], 201)

    def test_static_bandwidth_pacing(self):
        text = ''.join(f'{i} 0 R {i*64} {i%2}\n' for i in range(40))
        _, shared = self.run_trace(text, 'shared')
        _, static = self.run_trace(text, 'static', 'stage_id,bandwidth_gbps\n0,0.05\n1,0.05\n')
        self.assertGreater(static['cycles'], shared['cycles'])
        self.assertEqual(static['completed'], 40)
        self.assertEqual(static['stages'][0]['allocated_bandwidth_gbps'], 0.05)

    def test_static_requires_every_active_stage(self):
        error = self.run_trace('0 0 R 0 1\n', 'static', 'stage_id,bandwidth_gbps\n0,1\n', False)
        self.assertIn('Missing bandwidth', error)

    def test_negative_source_rejected(self):
        error = self.run_trace('0 0 R 0 -1\n', 'shared', success=False)
        self.assertIn('Invalid', error)

    def test_legacy_rejects_unexpected_stage_field(self):
        error = self.run_trace('0 0 R 0 1\n', success=False)
        self.assertIn('Invalid', error)


if __name__ == '__main__':
    unittest.main()
