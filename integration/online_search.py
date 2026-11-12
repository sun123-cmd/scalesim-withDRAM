"""Deterministic batched bound search over online simulation results.

Bounds use seconds. Certificates require an admissible cost table and a complete
candidate space. The equivalence audit finishes all candidates that could enter
the requested 2% sets or the latency optimum under the 10% throughput constraint.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import tempfile


class SearchEvaluationError(RuntimeError):
    pass


class BoundViolationError(ValueError):
    pass


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _metrics(result):
    metrics = result.get('metrics', result)
    latency = metrics.get('latency_mean_seconds', metrics.get('L'))
    interval = metrics.get('output_interval_seconds', metrics.get('S'))
    if (isinstance(latency, bool) or isinstance(interval, bool)
            or not isinstance(latency, (int, float)) or not isinstance(interval, (int, float))
            or not math.isfinite(latency) or not math.isfinite(interval)
            or latency <= 0 or interval < 0):
        raise ValueError('Online evaluation must return finite positive L and nonnegative S')
    return float(latency), float(interval)


def dominates(first, second):
    return first[0] <= second[0] and first[1] <= second[1] and first != second


def _selections(evaluated):
    values = list(evaluated.values())
    if not values:
        return [], None, None, [], []
    frontier = [item for item in values if not any(
        dominates((other['L'], other['S']), (item['L'], item['S'])) for other in values)]
    frontier.sort(key=lambda item: (item['L'], item['S'], item['id']))
    throughput = min(values, key=lambda item: (item['S'], item['L'], item['id']))
    eligible = [item for item in values if item['S'] <= throughput['S'] * 1.1]
    latency = min(eligible, key=lambda item: (item['L'], item['S'], item['id']))
    throughput_equivalent = sorted((item for item in values if item['S'] <= throughput['S'] * 1.02),
                                  key=lambda item: (item['S'], item['L'], item['id']))
    latency_equivalent = sorted((item for item in eligible if item['L'] <= latency['L'] * 1.02),
                               key=lambda item: (item['L'], item['S'], item['id']))
    return frontier, throughput, latency, throughput_equivalent, latency_equivalent


def _save(path, state):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            file.write(json.dumps(state, indent=2, allow_nan=False) + '\n')
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _search(candidates, evaluator, workers=150, batch_size=150, error_margin=0.0,
           mode='pareto', resume_path=None, max_evaluations=None,
           candidate_space_complete=True):
    """Evaluate each selected candidate once, preserving completed results on resume.

    evaluator(candidate) must run the candidate's online simulator and return
    metrics (latency_mean_seconds, output_interval_seconds) in seconds. Optional
    ``stage_timings`` entries with actual_seconds and uncontended_seconds receive
    an actual_over_uncontended field. Candidate ``bounds_admissible`` defaults to
    False, which makes bound-based certificates conditional.
    """
    if type(workers) is not int or type(batch_size) is not int or min(workers, batch_size) < 1:
        raise ValueError('workers and batch_size must be positive integers')
    if not math.isfinite(error_margin) or not 0 <= error_margin < 1:
        raise ValueError('error_margin must be in [0,1)')
    if type(candidate_space_complete) is not bool:
        raise ValueError('candidate_space_complete must be boolean')
    if mode not in ('throughput', 'pareto'):
        raise ValueError('mode must be throughput or pareto')
    if max_evaluations is not None and (type(max_evaluations) is not int or max_evaluations < 1):
        raise ValueError('max_evaluations must be a positive integer')
    ordered = []
    ids = set()
    for source in candidates:
        candidate = dict(source)
        if not isinstance(candidate.get('id'), str) or not candidate['id'] or candidate['id'] in ids:
            raise ValueError('Each candidate requires a unique nonempty string id')
        ids.add(candidate['id'])
        for field in ('optimistic_l', 'optimistic_s'):
            value = candidate.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{field} must be finite and nonnegative')
        ordered.append(candidate)
    if not ordered:
        raise ValueError('Candidate space is empty')
    ordered.sort(key=lambda candidate: (candidate['optimistic_s'], candidate['optimistic_l'], candidate['id']))
    configuration = {'candidates': ordered, 'error_margin': error_margin, 'mode': mode,
                     'candidate_space_complete': bool(candidate_space_complete)}
    fingerprint = hashlib.sha256(_json(configuration).encode()).hexdigest()
    state = {'version': 1, 'fingerprint': fingerprint, 'evaluated': {}, 'failures': {}}
    if resume_path is not None and Path(resume_path).exists():
        state = json.loads(Path(resume_path).read_text())
        if state.get('version') != 1 or state.get('fingerprint') != fingerprint:
            raise ValueError('Checkpoint belongs to a different candidate space or search policy')
        for identifier, record in state['evaluated'].items():
            if identifier not in ids or record['id'] != identifier:
                raise ValueError('Checkpoint contains an unknown candidate')
            _metrics(record)
    evaluated = state['evaluated']
    primary_reason = 'candidate_space_exhausted'
    phase = 'primary'
    factor = 1 - error_margin
    budget_exhausted = False

    def bounds(candidate):
        return candidate['optimistic_l'], candidate['optimistic_s']

    def can_enter_requested_sets(candidate, best_s, best_l):
        low_l, low_s = bounds(candidate)
        low_l, low_s = low_l * factor, low_s * factor
        return (low_s <= best_s['S'] * 1.02
                or (low_s <= best_s['S'] * 1.1 and low_l <= best_l['L'] * 1.02))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        while True:
            pending = [candidate for candidate in ordered if candidate['id'] not in evaluated]
            if not pending:
                break
            frontier, best_s, best_l, _, _ = _selections(evaluated)
            if phase == 'primary' and best_s is not None:
                if mode == 'throughput':
                    stop = pending[0]['optimistic_s'] >= best_s['S'] * factor
                else:
                    stop = all(any(point['L'] * factor <= candidate['optimistic_l']
                                   and point['S'] * factor <= candidate['optimistic_s']
                                   and (point['L'] * factor < candidate['optimistic_l']
                                        or point['S'] * factor < candidate['optimistic_s'])
                                   for point in frontier) for candidate in pending)
                if stop:
                    primary_reason = 'throughput_bound' if mode == 'throughput' else 'pareto_bound'
                    phase = 'equivalence_audit'
            if phase == 'equivalence_audit':
                pending = [candidate for candidate in pending if can_enter_requested_sets(candidate, best_s, best_l)]
                if not pending:
                    break
            remaining_budget = len(pending) if max_evaluations is None else max_evaluations - len(evaluated)
            if remaining_budget <= 0:
                budget_exhausted = True
                break
            selected = pending[:min(batch_size, remaining_budget)]
            futures = [(candidate, pool.submit(evaluator, candidate)) for candidate in selected]
            errors = []
            for candidate, future in futures:
                identifier = candidate['id']
                try:
                    result = json.loads(_json(future.result()))
                    latency, interval = _metrics(result)
                    if candidate.get('bounds_admissible', False) and (
                            latency + 1e-12 < candidate['optimistic_l']
                            or interval + 1e-12 < candidate['optimistic_s']):
                        raise BoundViolationError(f'Observed result violates certified bounds for {identifier}')
                    timings = result.get('stage_timings', [])
                    for timing in timings:
                        uncontended = timing['uncontended_seconds']
                        actual = timing['actual_seconds']
                        if not math.isfinite(uncontended) or uncontended <= 0:
                            raise ValueError('Stage uncontended time must be positive')
                        if not math.isfinite(actual) or actual < 0:
                            raise ValueError('Stage actual time must be finite and nonnegative')
                        timing['actual_over_uncontended'] = actual / uncontended
                    evaluated[identifier] = {'id': identifier, 'L': latency, 'S': interval,
                                             'parameters': candidate.get('parameters', {}), 'result': result}
                    state['failures'].pop(identifier, None)
                except Exception as error:
                    state['failures'][identifier] = {'type': type(error).__name__, 'message': str(error)}
                    errors.append((identifier, error))
                _save(resume_path, state)
            if errors:
                identifier, error = errors[0]
                raise SearchEvaluationError(f'Online evaluation failed for {identifier}: {error}') from error
    frontier, best_s, best_l, s_equivalent, l_equivalent = _selections(evaluated)
    remaining = [candidate for candidate in ordered if candidate['id'] not in evaluated]
    certified = (candidate_space_complete and not budget_exhausted and not state['failures']
                 and (not remaining or all(candidate.get('bounds_admissible', False) for candidate in ordered)))
    result = {'mode': mode, 'error_margin': error_margin, 'primary_stopping_reason': primary_reason,
              'stopping_reason': 'evaluation_budget' if budget_exhausted else 'bounds_or_exhaustion',
              'candidate_count': len(ordered), 'evaluated_count': len(evaluated),
              'remaining_count': len(remaining), 'candidate_space_complete': candidate_space_complete,
              'bound_certificate': 'epsilon' if certified and error_margin and remaining else 'exact' if certified else 'conditional_or_incomplete',
              'pareto_frontier': frontier, 'throughput_optimum': best_s,
              'latency_optimum_within_10_percent_S': best_l,
              'throughput_equivalent_2_percent': s_equivalent,
              'latency_equivalent_2_percent': l_equivalent,
              'equivalence_sets_complete': certified,
              'pareto_frontier_complete': certified and (not remaining or (mode == 'pareto' and error_margin == 0)),
              'evaluated': [evaluated[candidate['id']] for candidate in ordered if candidate['id'] in evaluated]}
    _save(resume_path, state)
    return result


def search(candidates, evaluator, workers=150, batch_size=150, error_margin=0.0,
           mode='pareto', resume_path=None, max_evaluations=None,
           candidate_space_complete=True):
    """Run batched search with exclusive ownership of its atomic checkpoint."""
    arguments = (candidates, evaluator, workers, batch_size, error_margin, mode,
                 resume_path, max_evaluations, candidate_space_complete)
    if resume_path is None:
        return _search(*arguments)
    import fcntl
    path = Path(resume_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(path) + '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('A search already owns this checkpoint') from error
        try:
            return _search(*arguments)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def enumerate_candidates(hardware, bound_fn, include_uneven=False):
    """Deduplicate integer stage allocations under shared DRAM (beta-independent)."""
    from proposal_config import Schedule, allocate
    schedules = []
    seen = set()
    dit_counts = (1, 2, 3, 4, 5, 10) if include_uneven else (1, 2, 5, 10)
    for kp, kd, numerator, batch, replicas in itertools.product(
            (1, 2, 3, 4, 6), dit_counts, range(1, 16), (1, 2, 5, 10), (1, 2, 4)):
        schedule = Schedule(kp, kd, batch, replicas, numerator, .5)
        try:
            allocation = allocate(hardware, schedule)
        except ValueError:
            continue
        key = ('pipeline', tuple(allocation['prefix_cores']), tuple(allocation['dit_cores']),
               tuple(allocation['steps']), batch, replicas)
        if key in seen:
            continue
        seen.add(key)
        schedules.append({'mode': 'pipeline', 'schedule': asdict(schedule)})
    for batch, replicas in itertools.product((1, 2, 5, 10), (1, 2, 4)):
        if hardware.cores // replicas < 1:
            continue
        schedules.append({'mode': 'baseline', 'tm_batch': batch,
                          'schedule': asdict(Schedule(kp=1, kd=1, b=batch, replicas=replicas))})
    candidates = []
    for parameters in schedules:
        parameters = dict(parameters, hardware=asdict(hardware))
        bound = bound_fn(parameters)
        identifier = hashlib.sha256(_json(parameters).encode()).hexdigest()[:20]
        candidates.append({'id': identifier, 'parameters': parameters,
                           'optimistic_l': bound['optimistic_l'], 'optimistic_s': bound['optimistic_s'],
                           'bounds_admissible': bound.get('bounds_admissible', False),
                           'bound_metadata': {key: value for key, value in bound.items()
                                              if key not in ('optimistic_l', 'optimistic_s', 'bounds_admissible')}})
    return candidates
