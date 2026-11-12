"""Run a proposal-aligned pi0.5 timestep scheduling experiment."""
import argparse
import csv
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys

from proposal_backend import stage_cost
from proposal_config import HARDWARE_POINTS, Hardware, Schedule, build_stage_workloads
from proposal_scheduler import simulate_baseline, simulate_pipeline


def evaluate(hardware, schedule, correction=1.0, overhead=0.0, calibration=None):
    workload = build_stage_workloads(hardware, schedule)
    costs = [stage_cost(stage, hardware, correction, overhead, calibration) for stage in workload['stages']]
    return workload, costs


def balance_beta(hardware, schedule, correction=1.0, overhead=0.0, calibration=None):
    low, high = 1e-4, 1 - 1e-4
    for _ in range(32):
        middle = (low + high) / 2
        _, costs = evaluate(hardware, replace(schedule, beta=middle), correction, overhead, calibration)
        prefix = max(item['seconds_per_service'] for item in costs if item['kind'] == 'prefix')
        dit = max(item['seconds_per_service'] * steps / schedule.b for item, steps in
                  zip([item for item in costs if item['kind'] == 'dit'],
                      build_stage_workloads(hardware, schedule)['allocation']['steps']))
        if dit > prefix:
            low = middle
        else:
            high = middle
    return replace(schedule, beta=(low + high) / 2)


def memory_probe(root, output, workload, policy, packets, replay):
    """Profile simultaneous per-stage streaming demand in one native memory system."""
    import yaml
    output.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get('RAMULATOR_CHANNELS')
    os.environ['RAMULATOR_CHANNELS'] = str(workload['hardware']['channels'])
    try:
        namespace = runpy.run_path(str(root / 'examples/proposal_lpddr5.py'))
    finally:
        if previous is None:
            os.environ.pop('RAMULATOR_CHANNELS', None)
        else:
            os.environ['RAMULATOR_CHANNELS'] = previous
    config = {'frontend': {'impl': 'External', 'clock_ratio': 1},
              'memory_system': namespace['memory_system'].to_config()}
    (output / 'memory.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    stages = workload['stages']
    allocations = output / 'allocations.csv'
    # A streaming stress window exercises physical queues and source fairness.
    # It is a diagnostic alongside the full-request roofline schedule.
    with allocations.open('w') as file:
        file.write('stage_id,bandwidth_gbps\n')
        for replica in range(workload['schedule']['replicas']):
            for index, stage in enumerate(stages):
                file.write(f"{replica * len(stages) + index},{stage['bandwidth_gbps']}\n")
    stage_count = len(stages) * workload['schedule']['replicas']
    tx_bytes = 32  # LPDDR5 x16 controller transaction, validated against replay statistics.
    trace = output / 'requests.trace'
    with trace.open('w') as file:
        request_id = 0
        for index in range(packets):
            for stage in range(stage_count):
                address = (stage * packets + index) * tx_bytes
                operation = 'W' if index % 4 == 3 else 'R'
                file.write(f'{request_id} 0 {operation} {address} {stage}\n')
                request_id += 1
    policies = ('static', 'shared') if policy == 'both' else (policy,)
    result = {'method': 'simultaneous stage-tagged streaming diagnostic',
              'physical_model': namespace['proposal_metadata'], 'policies': {}}
    for selected in policies:
        stats_path = output / f'{selected}.yaml'
        command = [str(replay.resolve()), str(output / 'memory.yaml'), str(trace),
                   str(output / f'{selected}.csv'), str(stats_path), '10000000', '16', selected]
        if selected == 'static':
            command.append(str(allocations))
        subprocess.run(command, check=True)
        stats = yaml.safe_load(stats_path.read_text())
        if stats['replay']['requests'] != stats['replay']['completed']:
            raise ValueError('Native profile did not complete all stage requests')
        if stats['replay']['transaction_bytes'] != tx_bytes:
            raise ValueError('Update the profile packet size to match the DRAM transaction')
        controllers = stats['memory_system']['controller']
        if isinstance(controllers, dict):
            controllers = [controllers]
        for stage_id, measured in stats['replay']['stages'].items():
            stage_index = int(stage_id) % len(stages)
            measured['name'] = stages[stage_index]['name']
            measured['replica'] = int(stage_id) // len(stages)
            target = stages[stage_index]['bandwidth_gbps']
            measured['proposal_allocation_gbps'] = target
            measured['actual_over_proposal_allocation'] = measured['actual_bandwidth_gbps'] / target
            hits = sum(controller.get(f'read_row_hits_core_{stage_id}', 0) for controller in controllers)
            misses = sum(controller.get(f'read_row_misses_core_{stage_id}', 0) for controller in controllers)
            conflicts = sum(controller.get(f'read_row_conflicts_core_{stage_id}', 0) for controller in controllers)
            measured['read_row_hit_rate'] = hits / (hits + misses + conflicts) if hits + misses + conflicts else None
        total_bytes = sum(stage['read_bytes'] + stage['write_bytes'] for stage in stats['replay']['stages'].values())
        actual_bandwidth = total_bytes / (stats['replay']['cycles'] * stats['replay']['tCK_ns'])
        stats['replay']['aggregate_actual_bandwidth_gbps'] = actual_bandwidth
        stats['replay']['utilization_of_native_peak'] = actual_bandwidth / namespace['proposal_metadata']['modeled_peak_gbps']
        result['policies'][selected] = stats
    return result


def calibrate_tiles(root, output, workload, args):
    manifest = output / 'calibration_manifest.json'
    operations = [operation for stage in workload['stages'] for operation in stage['operations']]
    # Include all four sequential baselines so their edge shapes are calibrated too.
    for batch in (1, 2, 5, 10):
        base = build_stage_workloads(Hardware(args.cores, args.channels), Schedule(b=batch, kp=1, kd=1))
        for stage in base['stages']:
            for source in stage['operations']:
                operation = dict(source)
                if stage['kind'] == 'prefix':
                    if operation['weight_copies'] == 0:
                        operation['count'] *= batch
                    else:
                        operation['m'] *= batch
                operations.append(operation)
    manifest.write_text(json.dumps({'model': 'pi0.5 horizon16 proposal calibration', 'operations': operations}))
    destination = output / 'scale_calibration'
    command = [sys.executable, str(root / 'integration/pi05_runner.py'), '--manifest', str(manifest),
               '--output', str(destination), '--tile-size', '32', '--array-size', '32', '--total-sram-kib', '1024',
               '--word-bytes', '2', '--replay', str(args.replay.resolve()),
               '--dram-config', str(root / 'examples/proposal_lpddr5.py')]
    subprocess.run(command, check=True)
    result = json.loads((destination / 'summary.json').read_text())
    tile_results = json.loads(Path(result['tile_summary']).read_text())
    rows = []
    for shape, layer in zip(result['tile_shapes'], tile_results['layers']):
        cycles = layer['baseline']['compute_cycles_including_stalls'] - layer['baseline']['stall_cycles']
        rows.append({'shape': list(shape.values()), 'compute_cycles': cycles})
    (output / 'calibration.json').write_text(json.dumps({'array_size': 32, 'total_sram_kib': 1024,
                                                       'word_bytes': 2, 'tiles': rows}, indent=2))
    return {'tile_size': 32, 'cycles': {tuple(row['shape']): row['compute_cycles'] for row in rows}}


def run_point(hardware, schedule, args, output):
    output.mkdir(parents=True, exist_ok=True)
    if args.beta == 'auto':
        schedule = balance_beta(hardware, schedule, args.utilization_correction, args.operator_overhead_us)
    workload, costs = evaluate(hardware, schedule, args.utilization_correction, args.operator_overhead_us)
    calibration = None
    if args.calibrate_tiles:
        calibration = calibrate_tiles(Path(__file__).resolve().parents[1], output, workload, args)
        if args.beta == 'auto':
            schedule = balance_beta(hardware, schedule, args.utilization_correction,
                                    args.operator_overhead_us, calibration)
        workload, costs = evaluate(hardware, schedule, args.utilization_correction,
                                   args.operator_overhead_us, calibration)
    prefix = [item['seconds_per_service'] for item in costs if item['kind'] == 'prefix']
    dit = [item['seconds_per_service'] for item in costs if item['kind'] == 'dit']
    link_count = schedule.replicas * (schedule.kp + schedule.kd - 1)
    handoff = workload['kv_bytes_per_request'] / (hardware.handoff_gbps * 1e9 / link_count)
    timeline = simulate_pipeline(prefix, dit, workload['allocation']['steps'],
                                 schedule.b, schedule.replicas, args.requests, args.warmup, args.drain,
                                 handoff, args.max_inflight)
    # Sequential baselines use every core and all bandwidth assigned to a replica.
    baselines = {}
    for batch in (1, 2, 5, 10):
        base_workload = build_stage_workloads(hardware, replace(schedule, kp=1, kd=1, b=batch))
        durations = []
        for side in ('prefix_stages', 'dit_stages'):
            stage = dict(base_workload[side][0])
            stage['cores'] = hardware.cores // schedule.replicas
            stage['bandwidth_gbps'] = hardware.bandwidth_gbps / schedule.replicas
            stage['operations'] = [dict(operation) for operation in stage['operations']]
            if side == 'prefix_stages':
                for operation in stage['operations']:
                    if operation['weight_copies'] == 0:
                        operation['count'] *= batch
                    else:
                        operation['m'] *= batch
            durations.append(stage_cost(stage, hardware, args.utilization_correction,
                                        args.operator_overhead_us, calibration)['seconds_per_service'])
        baselines[str(batch)] = simulate_baseline(*durations, batch=batch, replicas=schedule.replicas,
                                                  requests=args.requests, warmup=args.warmup, drain=args.drain)['metrics']
    result = {'backend': 'SCALE-Sim tile compute + ideal traffic roofline timeline' if calibration else 'proposal roofline costs with deterministic request timeline',
              'workload': workload, 'stage_costs': costs, 'metrics': timeline['metrics'],
              'baseline_metrics_by_batch': baselines,
              'assumptions': {'traffic': 'ideal per-microbatch weight reuse and MQA KV-head sharing',
                              'residency': 'cold', 'utilization_correction': args.utilization_correction,
                              'operator_overhead_us': args.operator_overhead_us,
                              'dram_latency_model': 'fixed per operator occurrence added to roofline max',
                              'stage_memory_allocation': 'proportional static bandwidth by side and core count',
                              'handoff': 'total handoff bandwidth split equally among replica stage boundaries; microbatch bytes scale with b',
                              'handoff_links': link_count,
                              'native_profile_scope': 'standalone simultaneous-stream calibration diagnostic',
                              'admission_policy': timeline['admission_policy'],
                              'max_inflight_per_replica': timeline['max_inflight_per_replica']}}
    if args.profile_memory:
        result['native_memory_profile'] = memory_probe(Path(__file__).resolve().parents[1],
                                                        output / 'memory_profile', workload,
                                                        args.memory_policy, args.profile_packets, args.replay)
    (output / 'timeline.json').write_text(json.dumps(timeline, indent=2) + '\n')
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cores', type=int, default=32)
    parser.add_argument('--channels', type=int, default=2)
    parser.add_argument('--kp', type=int, default=2)
    parser.add_argument('--kd', type=int, default=2)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--replicas', type=int, default=1)
    parser.add_argument('--dit-core-numerator', type=int, default=4)
    parser.add_argument('--beta', default='0.5')
    parser.add_argument('--requests', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--drain', type=int, default=20)
    parser.add_argument('--max-inflight', type=int)
    parser.add_argument('--utilization-correction', type=float, default=1.0)
    parser.add_argument('--operator-overhead-us', type=float, default=0.0)
    parser.add_argument('--all-hardware', action='store_true')
    parser.add_argument('--profile-memory', action='store_true')
    parser.add_argument('--calibrate-tiles', action='store_true')
    parser.add_argument('--memory-policy', choices=('static', 'shared', 'both'), default='both')
    parser.add_argument('--profile-packets', type=int, default=256)
    parser.add_argument('--replay', type=Path, default=root / 'integration/build/ramulator2-replay')
    parser.add_argument('--output', type=Path, default=root / 'results/proposal')
    args = parser.parse_args()
    if (not math.isfinite(args.utilization_correction) or args.utilization_correction <= 0
            or not math.isfinite(args.operator_overhead_us) or args.operator_overhead_us < 0
            or args.profile_packets <= 0):
        parser.error('Correction and packet count must be positive and operator overhead nonnegative')
    try:
        beta = .5 if args.beta == 'auto' else float(args.beta)
        schedule = Schedule(args.kp, args.kd, args.batch, args.replicas, args.dit_core_numerator, beta)
        points = HARDWARE_POINTS if args.all_hardware else (Hardware(args.cores, args.channels),)
        results = []
        for hardware in points:
            output = args.output / hardware.point_id if args.all_hardware else args.output
            try:
                result = run_point(hardware, schedule, args, output)
            except ValueError as error:
                if not args.all_hardware:
                    raise
                results.append({'point': hardware.point_id, 'valid': False, 'reason': str(error)})
                print(hardware.point_id, 'invalid:', str(error))
                continue
            results.append({'point': hardware.point_id, **result['metrics']})
            print(hardware.point_id, 'L=', result['metrics']['latency_mean_seconds'],
                  'S=', result['metrics']['output_interval_seconds'])
        if args.all_hardware:
            (args.output / 'hardware_summary.json').write_text(json.dumps(results, indent=2) + '\n')
            with (args.output / 'hardware_summary.csv').open('w') as file:
                writer = csv.DictWriter(file, fieldnames=('point', 'valid', 'reason', 'latency_mean_seconds',
                    'latency_p95_seconds', 'output_interval_seconds', 'steady_state_flag'), extrasaction='ignore')
                writer.writeheader()
                writer.writerows(dict(valid=True, **row) if 'valid' not in row else row for row in results)
    except (ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
