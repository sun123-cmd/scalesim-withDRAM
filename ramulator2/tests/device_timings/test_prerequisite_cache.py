"""State-changing commands must invalidate cached access prerequisites."""
import pytest
import ramulator
from tests.device_timings.harness import DeviceUnderTest


@pytest.fixture(params=["HBM3", "LPDDR5"])
def device(request):
    if request.param == "HBM3":
        dram = ramulator.dram.HBM3(org_preset="HBM3_16Gb_8hi", timing_preset="HBM3_6400Mbps")
    else:
        dram = ramulator.dram.LPDDR5(org_preset="LPDDR5_8Gb_x16", timing_preset="LPDDR5_6400")
    return DeviceUnderTest(dram)


def issue_later(device, command, address, clock):
    clock += 10000
    assert device.probe(command, address, clock).ready
    device.issue(command, address, clock)
    return clock


def open_bank(device, address, clock):
    for _ in range(4):
        prerequisite = device.probe("RD", address, clock).preq
        if prerequisite == "RD":
            return clock
        clock = issue_later(device, prerequisite, address, clock)
    raise AssertionError("Could not activate bank")


def test_access_prerequisites_follow_all_bank_close_and_refresh(device):
    addresses = [device.addr_vec(Bank=bank, Row=bank + 3) for bank in range(2)]
    clock = 0
    for address in addresses:
        clock = open_bank(device, address, clock)
    for address in addresses:
        assert device.probe("RD", address, clock + 10000).preq == "RD"
    wildcard = device.addr_vec(**{name: -1 for name in ("Sid", "BankGroup", "Bank", "Row", "Column")
                                  if name in device.level_names})
    clock = issue_later(device, "PREab", wildcard, clock)
    for address in addresses:
        assert device.probe("RD", address, clock).preq in ("ACT", "ACT1")
    clock = issue_later(device, "REFab", wildcard, clock)
    for address in addresses:
        clock = open_bank(device, address, clock)
        assert device.probe("RD", address, clock + 10000).preq == "RD"


@pytest.mark.parametrize("access,auto_precharge", [("RD", "RDA"), ("WR", "WRA")])
def test_access_prerequisites_follow_auto_precharge(device, access, auto_precharge):
    address = device.addr_vec(Row=7)
    clock = open_bank(device, address, 0)
    assert device.probe(access, address, clock + 10000).preq == access
    clock = issue_later(device, auto_precharge, address, clock)
    assert device.probe(access, address, clock + 10000).preq in ("ACT", "ACT1")


def test_access_prerequisite_identity_includes_row_and_command(device):
    address = device.addr_vec(Row=3)
    conflict = device.addr_vec(Row=9)
    clock = open_bank(device, address, 0)
    for later in (clock + 10000, clock + 20000):
        for command in ("RD", "WR", "RDA", "WRA"):
            assert device.probe(command, address, later).preq == command
            assert device.probe(command, conflict, later).preq == "PREpb"
            assert device.probe(command, address, later).preq == command


def test_split_activation_invalidates_each_transition():
    device = DeviceUnderTest(ramulator.dram.LPDDR5(
        org_preset="LPDDR5_8Gb_x16", timing_preset="LPDDR5_6400"))
    address = device.addr_vec(Row=3)
    assert device.probe("RD", address, 0).preq == "ACT1"
    clock = issue_later(device, "ACT1", address, 0)
    assert device.probe("RD", address, clock).preq == "ACT2"
    clock = issue_later(device, "ACT2", address, clock)
    assert device.probe("RD", address, clock + 10000).preq == "RD"


def test_ready_clock_cache_is_not_a_cached_ready_boolean(device):
    address = device.addr_vec(Row=3)
    clock = open_bank(device, address, 0)
    clock = issue_later(device, "RD", address, clock)
    # A future lookahead must not make the same command ready now. The RD has
    # no state action, but does update column timing and must invalidate it.
    assert device.probe("RD", address, clock + 10000).ready
    assert not device.probe("RD", address, clock).timing_OK
    assert device.probe("RD", address, clock + 10000).ready
