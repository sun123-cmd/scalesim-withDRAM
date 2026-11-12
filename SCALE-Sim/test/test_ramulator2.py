"""Regression coverage for timed DRAM conversion and checked latency feedback."""
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scalesim.memory.ramulator2_port import Ramulator2Port
from scalesim.ramulator2 import prepare_requests, save_latencies


class TestRamulator2(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        np.savetxt(self.path / 'IFMAP_DRAM_TRACE.csv',
                   [[-1, 0, 1, -1], [1, 64, 65, -1]], delimiter=',')
        np.savetxt(self.path / 'FILTER_DRAM_TRACE.csv', [[0, 128, -1]], delimiter=',')
        np.savetxt(self.path / 'OFMAP_DRAM_TRACE.csv', [[3, 192, -1]], delimiter=',')

    def prepare(self):
        return prepare_requests(self.path, self.path / 'requests.trace', 64, 0.5, 1000)

    def test_padding_zero_address_burst_coalescing_and_clock_conversion(self):
        matrices, mapping, accelerator_ns = self.prepare()
        self.assertEqual(len(mapping), 4)
        lines = (self.path / 'requests.trace').read_text().splitlines()
        self.assertEqual(lines[0], '0 0 R 0')
        self.assertEqual([int(line.split()[1]) for line in lines], [0, 2, 4, 10])
        self.assertEqual(accelerator_ns, 1)
        self.assertEqual(matrices['ifmap'][0].shape, (2, 3))

    def test_out_of_order_completions_map_to_original_rows(self):
        matrices, mapping, accelerator_ns = self.prepare()
        with (self.path / 'responses.csv').open('w') as file:
            writer = csv.writer(file)
            writer.writerow(['id', 'requested_cycle', 'completion_cycle'])
            for request_id in reversed(range(len(mapping))):
                cycle = mapping[request_id][3]
                writer.writerow([request_id, cycle, cycle + 5])
        save_latencies(self.path, matrices, mapping, self.path / 'responses.csv',
                       0.5, accelerator_ns)
        port = Ramulator2Port(self.path / 'ifmap.npz', 1)
        result = port.service_reads(matrices['ifmap'][0], np.array([10, 20]))
        np.testing.assert_array_equal(result, [[13], [23]])
        port.verify_complete()

    def test_large_latencies_are_preserved(self):
        np.savez(self.path / 'latency.npz', requests=[[0]], latencies=[20001])
        port = Ramulator2Port(self.path / 'latency.npz', 1)
        np.testing.assert_array_equal(port.service_reads([[0]], [[5]]), [[20006]])
        port.verify_complete()

    def test_per_burst_latency_survives_row_repacking(self):
        np.savetxt(self.path / 'IFMAP_DRAM_TRACE.csv', [[-1, 0, 64, -1]], delimiter=',')
        matrices, mapping, accelerator_ns = self.prepare()
        with (self.path / 'responses.csv').open('w') as file:
            writer = csv.writer(file)
            writer.writerow(['id', 'requested_cycle', 'completion_cycle'])
            for request_id, (_, _, _, cycle, address) in enumerate(mapping):
                writer.writerow([request_id, cycle, cycle + (4 if address == 0 else 20)])
        save_latencies(self.path, matrices, mapping, self.path / 'responses.csv', 0.5, accelerator_ns)
        port = Ramulator2Port(self.path / 'ifmap.npz', 1)
        np.testing.assert_array_equal(port.service_reads([[0, -1]], [[0]]), [[2]])
        np.testing.assert_array_equal(port.service_reads([[64, -1]], [[0]]), [[10]])
        port.verify_complete()

    def test_clock_alignment_is_included_in_latency(self):
        matrices = {'ifmap': (np.array([[0], [64]]), np.array([-2, -1])),
                    'filter': (np.array([[-1]]), np.array([0])),
                    'ofmap': (np.array([[-1]]), np.array([0]))}
        mapping = [('ifmap', 0, -2, 0, 0), ('ifmap', 1, -1, 2, 64)]
        (self.path / 'responses.csv').write_text(
            'id,requested_cycle,completion_cycle\n0,0,1\n1,2,3\n')
        save_latencies(self.path, matrices, mapping, self.path / 'responses.csv', 0.833, 1.0)
        port = Ramulator2Port(self.path / 'ifmap.npz', 1)
        np.testing.assert_array_equal(port.service_reads([[0], [64]], [[0], [1]]), [[1], [3]])
        port.verify_complete()

    def test_partial_write_rows_can_be_repacked(self):
        np.savez(self.path / 'latency.npz', requests=[[0, 1, -1], [2, 3, -1]],
                 latencies=[10, 20], word_latencies=[[10, 11, 0], [20, 21, 0]])
        port = Ramulator2Port(self.path / 'latency.npz', 0)
        np.testing.assert_array_equal(port.service_writes([[0, 1, 2]], [[5]]), [[25]])
        np.testing.assert_array_equal(port.service_writes([[3, -1, -1]], [[30]]), [[51]])
        port.verify_complete()

    def test_mismatched_requests_fail(self):
        np.savez(self.path / 'latency.npz', requests=[[0]], latencies=[10])
        port = Ramulator2Port(self.path / 'latency.npz', 1)
        with self.assertRaises(ValueError):
            port.service_reads([[1]], [[0]])
        with self.assertRaises(ValueError):
            port.verify_complete()

    def test_missing_completion_fails(self):
        matrices, mapping, accelerator_ns = self.prepare()
        (self.path / 'responses.csv').write_text('id,requested_cycle,completion_cycle\n')
        with self.assertRaises(ValueError):
            save_latencies(self.path, matrices, mapping, self.path / 'responses.csv',
                           0.5, accelerator_ns)


if __name__ == '__main__':
    unittest.main()
