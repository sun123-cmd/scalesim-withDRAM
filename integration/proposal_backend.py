"""Proposal roofline costs with explicit ideal reuse and utilization assumptions."""
import math


def operation_traffic(operation, word_bytes=2):
    m, n, k, count = (operation[key] for key in ('m', 'n', 'k', 'count'))
    attention = operation['weight_copies'] == 0
    weights = k * n * operation['weight_copies'] * word_bytes
    # MQA K/V is shared across eight query heads within a request.
    shared_mqa = attention and operation['name'].startswith(('prefix.', 'denoise.'))
    rhs_count = count // 8 if shared_mqa else count
    activations = (m * k + m * n) * count * word_bytes
    if attention:
        activations += k * n * rhs_count * word_bytes
    return {'weight_bytes': weights, 'activation_bytes': activations,
            'total_bytes': weights + activations}


def stage_cost(stage, hardware, utilization_correction=1.0, operator_overhead_us=0.0, calibration=None):
    seconds = 0.0
    flops = 0
    traffic = dict.fromkeys(('weight_bytes', 'activation_bytes', 'total_bytes'), 0)
    rows = []
    for operation in stage['operations']:
        m, n, k, count = (operation[key] for key in ('m', 'n', 'k', 'count'))
        work = 2 * m * n * k * count
        u = (k / (hardware.array_height * math.ceil(k / hardware.array_height))
             * n / (hardware.array_width * math.ceil(n / hardware.array_width))
             * m / (m + hardware.array_height + hardware.array_width)
             * utilization_correction)
        u = min(1.0, max(u, 1e-12))
        peak = stage['cores'] * 2 * hardware.array_height * hardware.array_width * hardware.frequency_ghz * 1e9
        compute = work / (peak * u)
        if calibration is not None:
            from pi05_runner import operation_tiles
            tiles = operation_tiles(operation, calibration['tile_size'])
            cycles = sum(math.ceil(count / stage['cores']) * calibration['cycles'][shape]
                         for shape, count in tiles.items())
            compute = cycles / (hardware.frequency_ghz * 1e9)
        byte_counts = operation_traffic(operation, hardware.word_bytes)
        memory = byte_counts['total_bytes'] / (stage['bandwidth_gbps'] * 1e9)
        duration = max(compute, memory) + count * (hardware.dram_latency_ns * 1e-9 + operator_overhead_us * 1e-6)
        seconds += duration
        flops += work
        for key in traffic:
            traffic[key] += byte_counts[key]
        rows.append({'name': operation['name'], 'flops': work, 'utilization': u,
                     'compute_seconds': compute, 'memory_seconds': memory,
                     'utilization_source': 'analytic roofline formula',
                     'compute_source': 'SCALE-Sim measured tile waves' if calibration else 'analytic utilization formula',
                     'modeled_array_utilization': work / (peak * compute),
                     'duration_seconds': duration, **byte_counts})
    return {'name': stage['name'], 'kind': stage['kind'], 'seconds_per_service': seconds,
            'flops_per_service': flops, 'cores': stage['cores'],
            'bandwidth_gbps': stage['bandwidth_gbps'], **traffic, 'operators': rows}
