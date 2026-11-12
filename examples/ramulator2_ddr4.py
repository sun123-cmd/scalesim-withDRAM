"""DDR4 configuration consumed by python -m scalesim.ramulator2."""
import ramulator

controller = ramulator.controller.GenericDDR(
    dram=ramulator.dram.DDR4(org_preset="DDR4_8Gb_x8", timing_preset="DDR4_2400R", rank=1),
    scheduler=ramulator.scheduler.FRFCFS(),
    refresh_manager=ramulator.refresh_manager.AllBank(),
    row_policy=ramulator.row_policy.Open(),
    addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(),
)
memory_system = ramulator.memory_system.GenericDRAM(
    clock_ratio=1,
    controllers=[controller],
    channel_mapper=ramulator.channel_mapper.CacheLineInterleave(),
)
