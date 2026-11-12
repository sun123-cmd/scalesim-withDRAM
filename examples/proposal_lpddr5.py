"""LPDDR5-6400 proxy with explicit physical bandwidth metadata.

One native controller models one x16, 6.4 GT/s interface (12.8 decimal GB/s).
The proposal LPDDR5X budget of 68.3 GB/s per logical channel is stored as a
separate target. RAMULATOR_CHANNELS selects native controllers, a power of two.
"""
import os
import ramulator

native_channels = int(os.environ.get('RAMULATOR_CHANNELS', '2'))
if native_channels not in (1, 2, 4, 8):
    raise ValueError('RAMULATOR_CHANNELS must be 1, 2, 4, or 8')

proposal_metadata = {
    'proposal_standard': 'LPDDR5X',
    'proposal_peak_gbps_per_logical_channel': 68.3,
    'modeled_standard': 'LPDDR5',
    'modeled_timing': 'LPDDR5_6400',
    'native_controller_width_bits': 16,
    'native_controller_peak_gbps': 12.8,
    'native_controllers': native_channels,
    'modeled_peak_gbps': native_channels * 12.8,
    'capacity_bytes': native_channels * (2 ** 30),
}
controller = ramulator.controller.LPDDR5(
    dram=ramulator.dram.LPDDR5(org_preset='LPDDR5_8Gb_x16', timing_preset='LPDDR5_6400'),
    scheduler=ramulator.scheduler.FRFCFS(),
    refresh_manager=ramulator.refresh_manager.AllBank(),
    row_policy=ramulator.row_policy.Open(),
    addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(),
)
memory_system = ramulator.memory_system.GenericDRAM(
    clock_ratio=1,
    controllers=[controller] * native_channels,
    channel_mapper=ramulator.channel_mapper.CacheLineInterleave(),
)
