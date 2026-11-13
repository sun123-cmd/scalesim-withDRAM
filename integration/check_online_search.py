"""Bound search is cross-checked against exhaustive online-result fixtures."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from online_search import SearchEvaluationError, enumerate_candidates, search
from proposal_config import Hardware


def candidates(rows):
    return [dict(id=str(index), optimistic_l=low_l, optimistic_s=low_s,
                 bounds_admissible=True, parameters={'actual': [latency, interval]})
            for index, (low_l, low_s, latency, interval) in enumerate(rows)]


def evaluate(candidate):
    latency, interval = candidate['parameters']['actual']
    return {'metrics': {'latency_mean_seconds': latency, 'output_interval_seconds': interval}}


class OnlineSearchTests(unittest.TestCase):
    def test_fast_result_is_checkpointed_and_refilled_while_first_is_running(self):
        values = candidates([(0, 0, 3, 3), (0, 0, 2, 2), (0, 0, 1, 1)])
        release = threading.Event()
        replacement_started = threading.Event()
        def evaluator(candidate):
            if candidate['id'] == '0':
                if not release.wait(5):
                    raise RuntimeError('slow evaluator was not released')
            elif candidate['id'] == '2':
                replacement_started.set()
            return evaluate(candidate)
        with tempfile.TemporaryDirectory() as temporary, ThreadPoolExecutor(max_workers=1) as caller:
            checkpoint = Path(temporary) / 'checkpoint.json'
            future = caller.submit(search, values, evaluator, workers=2, batch_size=8,
                                   resume_path=checkpoint)
            try:
                self.assertTrue(replacement_started.wait(3), 'worker waited for the slow batch leader')
                data = json.loads(checkpoint.read_text())
                self.assertIn('1', data['evaluated'])
                self.assertNotIn('0', data['evaluated'])
            finally:
                release.set()
            result = future.result(timeout=3)
            self.assertEqual(result['evaluated_count'], 3)
            self.assertEqual(result['throughput_optimum']['id'], '2')

    def test_failure_stops_refills_but_checkpoints_inflight_success_for_resume(self):
        import online_search
        values = candidates([(0, 0, 1, 1), (0, 0, 2, 2), (0, 0, 3, 3)])
        release = threading.Event()
        failure_saved = threading.Event()
        calls = []
        def evaluator(candidate):
            calls.append(candidate['id'])
            if candidate['id'] == '0':
                if not release.wait(5):
                    raise RuntimeError('slow evaluator was not released')
            elif candidate['id'] == '1':
                raise RuntimeError('native execution failed')
            return evaluate(candidate)
        original_save = online_search._save
        def save(path, state):
            original_save(path, state)
            if state['failures']:
                failure_saved.set()
        with tempfile.TemporaryDirectory() as temporary, ThreadPoolExecutor(max_workers=1) as caller:
            checkpoint = Path(temporary) / 'checkpoint.json'
            with patch('online_search._save', side_effect=save):
                future = caller.submit(search, values, evaluator, workers=2, batch_size=8,
                                       resume_path=checkpoint)
                try:
                    self.assertTrue(failure_saved.wait(3))
                    self.assertEqual(set(calls), {'0', '1'})
                finally:
                    release.set()
                with self.assertRaisesRegex(SearchEvaluationError, 'native execution failed'):
                    future.result(timeout=3)
            data = json.loads(checkpoint.read_text())
            self.assertEqual(set(data['evaluated']), {'0'})
            self.assertEqual(set(data['failures']), {'1'})
            retried = []
            def resume(candidate):
                retried.append(candidate['id'])
                return evaluate(candidate)
            result = search(values, resume, workers=2, resume_path=checkpoint)
            self.assertEqual(set(retried), {'1', '2'})
            self.assertEqual(result['evaluated_count'], 3)
            self.assertEqual(json.loads(checkpoint.read_text())['failures'], {})

    def test_inflight_budget_reservation_prevents_refill_overshoot(self):
        values = candidates([(0, 0, 2, 2)] * 6)
        calls = []
        def evaluator(candidate):
            calls.append(candidate['id'])
            return evaluate(candidate)
        result = search(values, evaluator, workers=2, batch_size=8, max_evaluations=3)
        self.assertEqual(len(calls), 3)
        self.assertEqual(result['evaluated_count'], 3)
        self.assertEqual(result['stopping_reason'], 'evaluation_budget')

    def test_pruning_does_not_discard_already_started_results(self):
        values = candidates([(0, 0, 20, 20), (1, 1, 2, 2), (10, 10, 30, 30)])
        release = threading.Event()
        saved_fast = threading.Event()
        calls = []
        import online_search
        original_save = online_search._save
        def evaluator(candidate):
            calls.append(candidate['id'])
            if candidate['id'] == '0' and not release.wait(5):
                raise RuntimeError('slow evaluator was not released')
            return evaluate(candidate)
        def save(path, state):
            original_save(path, state)
            if '1' in state['evaluated']:
                saved_fast.set()
        with ThreadPoolExecutor(max_workers=1) as caller, patch('online_search._save', side_effect=save):
            future = caller.submit(search, values, evaluator, workers=2, batch_size=2)
            try:
                self.assertTrue(saved_fast.wait(3))
                self.assertNotIn('2', calls)
            finally:
                release.set()
            result = future.result(timeout=3)
        self.assertEqual({x['id'] for x in result['evaluated']}, {'0', '1'})
        self.assertEqual([x['id'] for x in result['pareto_frontier']], ['1'])
        self.assertEqual(result['bound_certificate'], 'exact')

    def test_throughput_improvement_revisits_previously_ineligible_latency_candidate(self):
        import online_search
        values = candidates([(0, 0, 100, 10), (1, 1, 100, 9.9),
                             (1, 2, 1, 10.9), (50, 10.8, 50, 10.8)])
        release = threading.Event()
        latency_saved = threading.Event()
        calls = []
        original_save = online_search._save
        def save(path, state):
            original_save(path, state)
            if '2' in state['evaluated']:
                latency_saved.set()
        def evaluator(candidate):
            calls.append(candidate['id'])
            if candidate['id'] == '1' and not release.wait(5):
                raise RuntimeError('slow evaluator was not released')
            return evaluate(candidate)
        with tempfile.TemporaryDirectory() as temporary, ThreadPoolExecutor(max_workers=1) as caller:
            checkpoint = Path(temporary) / 'checkpoint.json'
            search(values, evaluate, mode='throughput', workers=1, max_evaluations=1,
                   resume_path=checkpoint)
            with patch('online_search._save', side_effect=save):
                future = caller.submit(search, values, evaluator, mode='throughput', workers=2,
                                       batch_size=2, resume_path=checkpoint)
                try:
                    self.assertTrue(latency_saved.wait(3))
                    self.assertNotIn('3', calls)
                finally:
                    release.set()
                result = future.result(timeout=3)
        self.assertEqual(result['throughput_optimum']['id'], '1')
        self.assertEqual(result['latency_optimum_within_10_percent_S']['id'], '3')
        self.assertEqual([x['id'] for x in result['latency_equivalent_2_percent']], ['3'])
        self.assertTrue(result['equivalence_sets_complete'])

    def test_error_margin_preserves_latency_tradeoff(self):
        values = candidates([(1, 1, 10, 10), (9.1, 20, 9.2, 21)])
        result = search(values, evaluate, batch_size=1, error_margin=.1)
        self.assertEqual({item['id'] for item in result['pareto_frontier']}, {'0', '1'})

    def test_independent_stage_allocations_include_idle_cores(self):
        from proposal_config import allocate, Schedule
        allocation = allocate(Hardware(32), Schedule(kp=2, kd=2,
                              prefix_stage_cores=4, dit_stage_cores=2))
        self.assertEqual(allocation['prefix_cores'], [4, 4])
        self.assertEqual(allocation['dit_cores'], [2, 2])
        self.assertEqual(allocation['idle_cores'], 20)
        values = enumerate_candidates(Hardware(8), lambda _: {'optimistic_l': 0, 'optimistic_s': 0})
        for candidate in values:
            schedule = candidate['parameters']['schedule']
            if candidate['parameters']['mode'] == 'pipeline':
                self.assertIn(schedule['prefix_stage_cores'], (1, 2, 4, 8))
                self.assertIn(schedule['dit_stage_cores'], (1, 2, 4, 8))

    def test_pareto_matches_exhaustive_frontier(self):
        rows = [(1, 1, 2, 8), (2, 2, 5, 5), (3, 3, 8, 3),
                (10, 10, 12, 12), (11, 11, 13, 13)]
        result = search(candidates(rows), evaluate, workers=2, batch_size=1)
        self.assertEqual({item['id'] for item in result['pareto_frontier']}, {'0', '1', '2'})
        self.assertLess(result['evaluated_count'], len(rows))
        self.assertEqual(result['bound_certificate'], 'exact')
        self.assertTrue(result['pareto_frontier_complete'])

    def test_secondary_audit_finds_equivalent_and_latency_constrained_optimum(self):
        rows = [(1, 1, 20, 10), (2, 10.1, 18, 10.1),
                (3, 10.8, 3, 10.8), (30, 30, 40, 40)]
        result = search(candidates(rows), evaluate, mode='throughput', batch_size=1)
        self.assertEqual(result['throughput_optimum']['id'], '0')
        self.assertEqual(result['latency_optimum_within_10_percent_S']['id'], '2')
        self.assertEqual({item['id'] for item in result['throughput_equivalent_2_percent']}, {'0', '1'})
        self.assertEqual(result['primary_stopping_reason'], 'throughput_bound')
        self.assertTrue(result['equivalence_sets_complete'])
        self.assertFalse(result['pareto_frontier_complete'])

    def test_equal_pareto_candidates_are_preserved(self):
        rows = [(1, 1, 10, 1), (1, 10, 1, 10), (1, 10, 1, 10)]
        result = search(candidates(rows), evaluate, batch_size=1)
        self.assertEqual(len(result['pareto_frontier']), 3)

    def test_budget_exhaustion_has_no_certificate(self):
        result = search(candidates([(0, 0, 1, 1)] * 4), evaluate,
                        batch_size=3, max_evaluations=2)
        self.assertEqual(result['evaluated_count'], 2)
        self.assertEqual(result['stopping_reason'], 'evaluation_budget')
        self.assertFalse(result['equivalence_sets_complete'])
        self.assertEqual(result['bound_certificate'], 'conditional_or_incomplete')

    def test_resume_evaluates_successes_once(self):
        values = candidates([(0, 0, 2, 2), (0, 0, 1, 1), (0, 0, 3, 3)])
        calls = []
        def evaluator(candidate):
            calls.append(candidate['id'])
            return evaluate(candidate)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'checkpoint.json'
            search(values, evaluator, resume_path=checkpoint, max_evaluations=1)
            result = search(values, evaluator, resume_path=checkpoint)
            self.assertEqual(len(calls), len(set(calls)))
            self.assertEqual(result['evaluated_count'], 3)
            self.assertEqual(result['throughput_optimum']['S'], 1)
            with self.assertRaises(ValueError):
                search(values, evaluator, resume_path=checkpoint, error_margin=.1)

    def test_failure_checkpoint_preserves_successes_and_retries_failure(self):
        values = candidates([(0, 0, 1, 1), (0, 0, 2, 2)])
        def fail(candidate):
            if candidate['id'] == '1':
                raise RuntimeError('simulation cycle limit')
            return evaluate(candidate)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'checkpoint.json'
            with self.assertRaises(SearchEvaluationError):
                search(values, fail, batch_size=2, resume_path=checkpoint)
            data = json.loads(checkpoint.read_text())
            self.assertEqual(set(data['evaluated']), {'0'})
            self.assertEqual(set(data['failures']), {'1'})
            result = search(values, evaluate, resume_path=checkpoint)
            self.assertEqual(result['evaluated_count'], 2)

    def test_bound_violation_fails_closed(self):
        with self.assertRaises(SearchEvaluationError):
            search(candidates([(10, 10, 1, 1)]), evaluate)

    def test_uncertified_table_is_reported_conditional_when_pruned(self):
        values = candidates([(1, 1, 2, 2), (10, 10, 20, 20)])
        for candidate in values:
            candidate['bounds_admissible'] = False
        result = search(values, evaluate, batch_size=1)
        self.assertEqual(result['bound_certificate'], 'conditional_or_incomplete')
        self.assertFalse(result['equivalence_sets_complete'])

    def test_complete_exhaustive_results_certify_without_bounds(self):
        values = candidates([(0, 0, 1, 1), (0, 0, 2, 2)])
        for candidate in values:
            candidate['bounds_admissible'] = False
        result = search(values, evaluate, batch_size=2)
        self.assertEqual(result['bound_certificate'], 'exact')

    def test_epsilon_certificate_is_explicit(self):
        values = candidates([(1, 1, 2, 2), (20, 20, 30, 30)])
        result = search(values, evaluate, batch_size=1, error_margin=.1)
        self.assertEqual(result['bound_certificate'], 'epsilon')

    def test_zero_interval_completion_burst_is_a_valid_finite_window(self):
        values = candidates([(0, 0, 2, 0), (0, 0, 1, 0), (0, 0, 3, 1)])
        result = search(values, evaluate, batch_size=1)
        self.assertEqual(result['throughput_optimum']['S'], 0)
        self.assertEqual(result['latency_optimum_within_10_percent_S']['L'], 1)

    def test_invalid_input_rejected(self):
        for options in ({'workers': 0}, {'batch_size': 0}, {'error_margin': 1}, {'mode': 'other'}):
            with self.assertRaises(ValueError):
                search(candidates([(0, 0, 1, 1)]), evaluate, **options)
        with self.assertRaises(ValueError):
            search(candidates([(float('nan'), 0, 1, 1)]), evaluate)
        with self.assertRaises(ValueError):
            search([dict(id='x', optimistic_l=0, optimistic_s=0)] * 2, evaluate)

    def test_stage_slowdowns_preserve_evaluation(self):
        def evaluator(candidate):
            return dict(evaluate(candidate), stage_timings=[{'actual_seconds': 2, 'uncontended_seconds': 1}])
        result = search(candidates([(0, 0, 1, 1)]), evaluator)
        self.assertEqual(result['evaluated'][0]['result']['stage_timings'][0]['actual_over_uncontended'], 2)

    def test_randomized_search_matches_exhaustive_selections(self):
        import random
        for seed in range(20):
            rng = random.Random(seed)
            rows = []
            for _ in range(30):
                latency, interval = rng.uniform(1, 20), rng.uniform(1, 20)
                rows.append((latency * rng.random(), interval * rng.random(), latency, interval))
            values = candidates(rows)
            expected_pareto = {str(index) for index, (_, _, latency, interval) in enumerate(rows)
                               if not any(other[2] <= latency and other[3] <= interval
                                          and (other[2] < latency or other[3] < interval) for other in rows)}
            best_s = min(row[3] for row in rows)
            best_l = min(row[2] for row in rows if row[3] <= best_s * 1.1)
            for mode in ('pareto', 'throughput'):
                result = search(values, evaluate, workers=3, batch_size=3, mode=mode)
                self.assertEqual(result['throughput_optimum']['S'], best_s)
                self.assertEqual(result['latency_optimum_within_10_percent_S']['L'], best_l)
                if mode == 'pareto':
                    self.assertEqual({item['id'] for item in result['pareto_frontier']}, expected_pareto)
                self.assertEqual({item['id'] for item in result['throughput_equivalent_2_percent']},
                                 {str(index) for index, row in enumerate(rows) if row[3] <= best_s * 1.02})
                self.assertEqual({item['id'] for item in result['latency_equivalent_2_percent']},
                                 {str(index) for index, row in enumerate(rows)
                                  if row[3] <= best_s * 1.1 and row[2] <= best_l * 1.02})

    def test_checkpoint_exclusive_lock(self):
        import fcntl
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'checkpoint.json'
            with Path(str(path) + '.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(ValueError):
                    search(candidates([(0, 0, 1, 1)]), evaluate, resume_path=path)

    def test_enumeration_shared_dram_beta_dedup_and_baselines(self):
        values = enumerate_candidates(Hardware(32, 2), lambda parameters:
                                      {'optimistic_l': 0, 'optimistic_s': 0})
        self.assertEqual(len({item['id'] for item in values}), len(values))
        baselines = [item for item in values if item['parameters']['mode'] == 'baseline']
        self.assertEqual(len(baselines), 12)
        self.assertTrue(all(item['parameters']['schedule']['beta'] == .5 for item in values))


if __name__ == '__main__':
    unittest.main()
