"""Compare complete native reports and bounded wall time for two executables.

These are correctness/performance microbenchmarks, never DSE measurements.
An optional compiled real program supplies its first complete prefix operation.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import yaml

from check_online_native import fixture_config
from online_workload import isolated_config, target_memory_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--optimized', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--actual-program', type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for target in (136.6, 819):
        cases.append((f'fixture_{target}', fixture_config(
            target_gbps=target, max_inflight=64, warmup=20, measure=50)))
        if args.actual_program:
            compiled = yaml.safe_load(args.actual_program.read_text())
            config = isolated_config(compiled, 0, target_memory_config(target),
                                     warmup_requests=0, measure_requests=1)
            # isolated_config retains the compiled memory keys if supplied.
            config.update(target_memory_config(target))
            config['online']['stages'][0]['operations'] = [
                copy.deepcopy(compiled['online']['stages'][0]['operations'][0])]
            config['online'].update(calibration_batch=1, batch=1,
                                    queue_capacity=2, request_slots=1)
            config['online'].pop('event_log', None)
            cases.append((f'actual_prefix_operation_{target}', config))
    results = []
    for name, config in cases:
        path = args.output / name
        path.mkdir(exist_ok=True)
        program = path / 'program.yaml'
        program.write_text(yaml.safe_dump(config, sort_keys=False))
        timings = {'baseline': [], 'optimized': []}
        expected = None
        for repeat in range(args.repeats):
            order = [('baseline', args.baseline), ('optimized', args.optimized)]
            if repeat % 2:
                order.reverse()
            for label, binary in order:
                report = path / f'{label}_{repeat}.yaml'
                env = dict(os.environ)
                env['LD_LIBRARY_PATH'] = str(binary.resolve().parents[2] / 'ramulator2')
                start = time.monotonic()
                process = subprocess.run([str(binary.resolve()), str(program), str(report)],
                                         capture_output=True, text=True, timeout=args.timeout,
                                         env=env)
                wall = time.monotonic() - start
                (path / f'{label}_{repeat}.log').write_text(process.stdout + process.stderr)
                process.check_returncode()
                value = yaml.safe_load(report.read_text())
                if expected is None:
                    expected = value
                if value != expected:
                    raise AssertionError(f'{name}: complete native report differs for {label}/{repeat}')
                timings[label].append(wall)
        result = {'case': name, 'complete_report_equal': True,
                  'wall_seconds': timings,
                  'median_speedup': statistics.median(timings['baseline']) /
                                    statistics.median(timings['optimized']),
                  'online': expected['online']}
        results.append(result)
        (args.output / 'equivalence_summary.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps({key: value for key, value in result.items() if key != 'online'}), flush=True)


if __name__ == '__main__':
    main()
