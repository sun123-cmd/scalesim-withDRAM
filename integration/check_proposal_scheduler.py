"""Regression tests for timestep pipeline event ordering and finite windows."""
import unittest

from proposal_scheduler import simulate_baseline, simulate_pipeline


class SchedulerTests(unittest.TestCase):
    def pipeline(self, **kwargs):
        arguments = dict(prefix_durations=[1.0], denoise_durations=[.5],
                         timestep_splits=[10], requests=40, warmup=5, drain=5)
        arguments.update(kwargs)
        return simulate_pipeline(**arguments)

    def test_timestep_order_and_nonoverlap(self):
        result = self.pipeline(denoise_durations=[.5, .7], timestep_splits=[4, 6])
        for request in result['requests']:
            events = sorted((task for task in result['tasks'] if task['kind'] == 'denoise'
                             and request['request_id'] in task['request_ids']),
                            key=lambda task: task['timestep'])
            self.assertEqual([event['timestep'] for event in events], list(range(10)))
            self.assertTrue(all(a['end'] <= b['start'] for a,b in zip(events, events[1:])))
        self.assertLess(result['metrics']['little_law']['relative_consistency_residual'], 1e-12)

    def test_microbatch_fill_wait_is_in_latency(self):
        result = self.pipeline(microbatch=2)
        a, b = result['requests'][:2]
        self.assertEqual(a['completion'], b['completion'])
        self.assertEqual(a['admission'], 0)
        self.assertEqual(b['admission'], 1)
        self.assertEqual(a['completion'], 7)
        self.assertEqual(a['completion']-a['admission'], b['completion']-b['admission']+1)

    def test_capacity_is_respected(self):
        result = self.pipeline(microbatch=2, max_inflight_per_replica=2)
        events = []
        for request in result['requests']:
            events.extend([(request['admission'], 1), (request['completion'], -1)])
        current = maximum = 0
        for _, delta in sorted(events):
            current += delta
            maximum = max(maximum, current)
        self.assertEqual(maximum, 2)

    def test_replicas_route_round_robin(self):
        result = self.pipeline(replicas=2)
        self.assertTrue(all(record['replica'] == record['request_id'] % 2
                            for record in result['requests']))
        self.assertEqual(result['requests'][0]['completion'], result['requests'][1]['completion'])

    def test_handoff_charged(self):
        result = self.pipeline(prefix_durations=[1, 1], denoise_durations=[1, 1],
                               timestep_splits=[5, 5], handoff_seconds=2)
        self.assertEqual(result['requests'][0]['completion'], 18)

    def test_provider_can_override_tasks(self):
        seen = []
        def provider(kind, stage, ids, step, start):
            seen.append((kind, step))
            return 2 if kind == 'prefix' else 1
        result = self.pipeline(duration_provider=provider)
        self.assertEqual(result['requests'][0]['completion'], 12)
        self.assertEqual(len(seen), 40*11)

    def test_baseline_sequential_whole_chip(self):
        result = simulate_baseline(2, 1, requests=40, warmup=5, drain=5)
        self.assertEqual(result['metrics']['latency_mean_seconds'], 12)
        self.assertEqual(result['metrics']['output_interval_seconds'], 12)
        self.assertEqual(result['requests'][1]['admission'], 12)

    def test_pipeline_can_overlap_separate_resources(self):
        pipeline = self.pipeline()
        baseline = simulate_baseline(1, .5, requests=40, warmup=5, drain=5)
        self.assertLess(pipeline['metrics']['output_interval_seconds'],
                        baseline['metrics']['output_interval_seconds'])

    def test_bad_configuration(self):
        for kwargs in ({'microbatch':3}, {'max_inflight_per_replica':0},
                       {'prefix_durations':[0]}, {'timestep_splits':[0]},
                       {'warmup':39}, {'replicas':True}, {'handoff_seconds':-1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.pipeline(**kwargs)

    def test_deterministic(self):
        self.assertEqual(self.pipeline(), self.pipeline())


if __name__ == '__main__':
    unittest.main()
