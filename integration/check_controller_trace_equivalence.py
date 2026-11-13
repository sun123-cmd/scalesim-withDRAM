"""Exact replay comparison with seeded read/write bursts and refresh gaps."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import time

import yaml
from online_workload import target_memory_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--optimized', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for target in (136.6, 819):
        for seed in (7, 23):
            directory = args.output / f'{target}_{seed}'
            directory.mkdir(exist_ok=True)
            config = directory / 'memory.yaml'
            memory = target_memory_config(target)
            # ModuloInterleave is registered in the online executable only.
            # For these power-of-two (2/16) channel counts this built-in mapper
            # is exactly the same transaction-interleaving mapping.
            memory['memory_system']['channel_mapper'] = {'impl': 'CacheLineInterleave'}
            config.write_text(yaml.safe_dump(memory, sort_keys=False))
            rng = random.Random(seed)
            lines = []
            for index in range(4096):
                cycle = index // 32 * 8 if index < 2048 else 20000 + (index - 2048) // 16 * 256
                address = rng.randrange(256 if rng.random() < .6 else 1 << 22) * 128
                operation = 'R' if rng.random() < .65 else 'W'
                lines.append(f'{index} {cycle} {operation} {address} {index % 4}')
            trace = directory / 'requests.trace'
            trace.write_text('\n'.join(lines) + '\n')
            expected = None
            times = {}
            for label, binary in [('baseline', args.baseline), ('optimized', args.optimized)]:
                binary = binary.resolve()
                responses, stats = directory / f'{label}.csv', directory / f'{label}.yaml'
                env = dict(os.environ, LD_LIBRARY_PATH=str(binary.parents[2] / 'ramulator2'))
                start = time.monotonic()
                run = subprocess.run([str(binary), str(config), str(trace), str(responses),
                                      str(stats), '2000000', '64', 'shared'],
                                     capture_output=True, text=True, env=env, timeout=30)
                times[label] = time.monotonic() - start
                (directory / f'{label}.log').write_text(run.stdout + run.stderr)
                run.check_returncode()
                value = (responses.read_text(), yaml.safe_load(stats.read_text()))
                if expected is None:
                    expected = value
                assert value == expected, f'Request completions or full DRAM stats differ: {target}/{seed}'
            result = {'target': target, 'seed': seed, 'requests': 4096,
                      'completion_csv_and_all_stats_equal': True,
                      'cycles': expected[1]['replay']['cycles'], 'wall_seconds': times}
            results.append(result)
            print(json.dumps(result), flush=True)
    (args.output / 'trace_equivalence_summary.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
