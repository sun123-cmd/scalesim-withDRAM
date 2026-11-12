"""Deterministic saturated-admission timestep pipelines with finite WIP credits."""

import heapq
import math


def _positive(values, name):
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError(f'{name} must contain finite positive durations')


def _percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _metrics(records, tasks, warmup, drain):
    ordered = sorted(records, key=lambda item: (item['completion'], item['request_id']))
    if type(warmup) is not int or type(drain) is not int or min(warmup, drain) < 0:
        raise ValueError('Warmup and drain counts must be nonnegative integers')
    stop = len(ordered) - drain
    selected = ordered[warmup:stop]
    if len(selected) < 2:
        raise ValueError('At least two measured completions are required')
    start, end = selected[0]['completion'], selected[-1]['completion']
    if end <= start:
        raise ValueError('Measured completion window must have positive duration')
    latencies = [item['completion'] - item['admission'] for item in selected]
    span = end - start
    # Observation is (start,end]. All ties on either boundary use the same
    # inclusion rule, independent of request-id trimming inside a microbatch.
    finished = [item for item in records if start < item['completion'] <= end]
    occupancy_area = sum(max(0.0, min(end, item['completion']) - max(start, item['admission']))
                         for item in records)
    full_residence = sum(item['completion'] - item['admission'] for item in finished)
    before_window = sum(max(0.0, start - item['admission']) for item in finished)
    surviving_area = sum(max(0.0, end - max(start, item['admission']))
                         for item in records if item['completion'] > end)
    corrected_area = full_residence - before_window + surviving_area
    mean_inflight = occupancy_area / span
    residual = abs(occupancy_area - corrected_area) / max(occupancy_area, 1e-30)
    throughput = len(finished) / span
    little_mean_latency = full_residence / len(finished) if finished else None
    # Resource intervals are sorted independently because replicas execute in parallel.
    resources = {}
    for task in tasks:
        resources.setdefault((task['replica'], task['kind'], task['stage']), []).append(task)
    for intervals in resources.values():
        intervals.sort(key=lambda task: task['start'])
        for previous, current in zip(intervals, intervals[1:]):
            if previous['end'] > current['start'] + 1e-12:
                raise AssertionError('A stage resource executes overlapping tasks')
    count = len(selected)
    quarter = max(1, count // 4)
    first_mean = sum(latencies[:quarter]) / quarter
    last_mean = sum(latencies[-quarter:]) / quarter
    drift = abs(last_mean - first_mean) / max(first_mean, 1e-30)
    return {
        'latency_mean_seconds': sum(latencies) / count,
        'latency_p95_seconds': _percentile(latencies, .95),
        'output_interval_seconds': span / (count - 1),
        'measured_requests': count,
        'observation_window_seconds': [start, end],
        'latency_drift_fraction': drift,
        'steady_state_flag': drift < .02,
        'little_law': {
            'window_convention': '(first measured completion,last measured completion]',
            'mean_inflight': mean_inflight,
            'throughput_per_second': throughput,
            'completed_mean_full_residence_seconds': little_mean_latency,
            'boundary_correction_mean_inflight': (-before_window + surviving_area) / span,
            'relative_consistency_residual': residual,
            'uncorrected_relative_difference': abs(mean_inflight - throughput * little_mean_latency)
                / max(mean_inflight, 1e-30) if finished else None,
        },
    }


def simulate_pipeline(prefix_durations, denoise_durations, timestep_splits,
                      microbatch=1, replicas=1, requests=200, warmup=50, drain=20,
                      handoff_seconds=0.0, max_inflight_per_replica=None,
                      duration_provider=None):
    """Simulate requests admitted through finite per-replica WIP credits.

    Durations are seconds. Each denoise duration describes one timestep of one
    complete microbatch on that stage. Request latency starts at prefix-stage-0
    admission and includes queues, batch filling, and stage transfers.

    duration_provider(kind, stage, request_ids, timestep, start_time) may return a
    task duration. Prefix tasks use timestep=None and one request id. Denoise
    tasks use the globally increasing timestep and the full microbatch ids.
    """
    _positive(prefix_durations, 'prefix_durations')
    _positive(denoise_durations, 'denoise_durations')
    if len(denoise_durations) != len(timestep_splits):
        raise ValueError('Each denoise stage needs one timestep split')
    if any(type(value) is not int or value <= 0 for value in timestep_splits):
        raise ValueError('Timestep splits must be positive integers')
    for value in (microbatch, replicas, requests):
        if type(value) is not int or value <= 0:
            raise ValueError('Batch, replica, and request counts must be positive integers')
    if requests % (microbatch * replicas):
        raise ValueError('Requests must be divisible by microbatch * replicas')
    if not math.isfinite(handoff_seconds) or handoff_seconds < 0:
        raise ValueError('Handoff duration must be finite and nonnegative')
    capacity = max_inflight_per_replica
    if capacity is None:
        capacity = len(prefix_durations) + microbatch * (len(timestep_splits) + 1)
    if type(capacity) is not int or capacity < microbatch:
        raise ValueError('WIP capacity must hold at least one microbatch')
    tasks, records, transfers = [], [], []

    def execute(kind, stage, ids, step, start, duration, replica):
        if duration_provider is not None:
            duration = duration_provider(kind, stage, list(ids), step, start)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('Task duration provider must return a finite positive duration')
        task = {'kind': kind, 'stage': stage, 'replica': replica,
                'request_ids': list(ids), 'timestep': step, 'start': start, 'end': start + duration}
        tasks.append(task)
        return task['end']

    for replica in range(replicas):
        prefix_available = [0.0] * len(prefix_durations)
        denoise_available = [0.0] * len(denoise_durations)
        active_completions = []
        ids = list(range(replica, requests, replicas))
        for offset in range(0, len(ids), microbatch):
            batch = ids[offset:offset + microbatch]
            admissions, ready = {}, []
            for index, request_id in enumerate(batch):
                admission = prefix_available[0]
                while active_completions and active_completions[0] <= admission:
                    heapq.heappop(active_completions)
                while len(active_completions) + index >= capacity:
                    admission = max(admission, heapq.heappop(active_completions))
                    while active_completions and active_completions[0] <= admission:
                        heapq.heappop(active_completions)
                admissions[request_id] = admission
                completion = admission
                for stage, duration in enumerate(prefix_durations):
                    start = max(completion, prefix_available[stage])
                    completion = execute('prefix', stage, [request_id], None, start, duration, replica)
                    prefix_available[stage] = completion
                    if stage + 1 < len(prefix_durations):
                        transfers.append({'replica': replica, 'request_ids': [request_id],
                                          'after_kind': 'prefix', 'after_stage': stage,
                                          'start': completion, 'end': completion + handoff_seconds})
                        completion += handoff_seconds
                ready.append(completion)
            completion = max(ready)
            if handoff_seconds:
                transfers.append({'replica': replica, 'request_ids': batch,
                                  'after_kind': 'prefix', 'after_stage': len(prefix_durations)-1,
                                  'start': completion, 'end': completion + microbatch * handoff_seconds})
                completion += microbatch * handoff_seconds
            step = 0
            for stage, (duration, iterations) in enumerate(zip(denoise_durations, timestep_splits)):
                for _ in range(iterations):
                    start = max(completion, denoise_available[stage])
                    completion = execute('denoise', stage, batch, step, start, duration, replica)
                    denoise_available[stage] = completion
                    step += 1
                if stage + 1 < len(denoise_durations):
                    transfers.append({'replica': replica, 'request_ids': batch,
                                      'after_kind': 'denoise', 'after_stage': stage,
                                      'start': completion, 'end': completion + microbatch * handoff_seconds})
                    completion += microbatch * handoff_seconds
            for request_id in batch:
                records.append({'request_id': request_id, 'replica': replica,
                                'admission': admissions[request_id], 'completion': completion,
                                'timesteps': list(range(sum(timestep_splits)))})
                heapq.heappush(active_completions, completion)
    metrics = _metrics(records, tasks, warmup, drain)
    return {'method': 'bounded-WIP deterministic timestep pipeline',
            'admission_policy': 'saturated prefix-stage-0 admission with per-replica credits',
            'max_inflight_per_replica': capacity,
            'queue_policy': 'FIFO stage resources and a microbatch collector',
            'handoff_model': 'per-request KV transfer time with microbatch byte scaling and independent links',
            'requests': sorted(records, key=lambda item: item['request_id']),
            'tasks': sorted(tasks, key=lambda task: (task['start'], task['replica'], task['kind'], task['stage'])),
            'transfers': transfers, 'metrics': metrics}


def simulate_baseline(prefix_duration, denoise_duration, steps=10, batch=1,
                      replicas=1, requests=200, warmup=50, drain=20,
                      handoff_seconds=0.0):
    """Whole-chip-per-replica sequential prefix and denoise microbatches."""
    _positive([prefix_duration, denoise_duration], 'baseline durations')
    for value in (steps, batch, replicas, requests):
        if type(value) is not int or value <= 0:
            raise ValueError('Baseline counts must be positive integers')
    if requests % (batch * replicas):
        raise ValueError('Requests must be divisible by batch * replicas')
    if not math.isfinite(handoff_seconds) or handoff_seconds < 0:
        raise ValueError('Handoff duration must be finite and nonnegative')
    records, tasks = [], []
    for replica in range(replicas):
        clock = 0.0
        ids = list(range(replica, requests, replicas))
        for offset in range(0, len(ids), batch):
            batch_ids = ids[offset:offset+batch]
            admission = clock
            clock += prefix_duration + steps * denoise_duration + handoff_seconds
            tasks.append({'kind': 'baseline', 'stage': 0, 'replica': replica,
                          'request_ids': batch_ids, 'timestep': None,
                          'start': admission, 'end': clock})
            for request_id in batch_ids:
                records.append({'request_id': request_id, 'replica': replica,
                                'admission': admission, 'completion': clock,
                                'timesteps': list(range(steps))})
    return {'method': 'whole-chip sequential microbatch baseline', 'requests': records,
            'tasks': tasks, 'metrics': _metrics(records, tasks, warmup, drain)}
