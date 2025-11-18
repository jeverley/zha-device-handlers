"""Tests for Tuya quirks."""

import pytest
from zigpy.zcl import foundation

import zhaquirks
import zhaquirks.tuya

zhaquirks.setup()


@pytest.mark.parametrize(
    "msg,attr_suffix,expected_power,expected_current,expected_volt",
    [
        (b"\t0\x02\x00\xd3\x06\x00\x00\x08\tn\x00\nA\x00\x01\xa6", "", 422, 2625, 2414),
        (b"\t2\x02\x00Z\x06\x00\x00\x08\ts\x00\n9\x00\x01\xa5", "", 421, 2617, 2419),
        (b"\t2\x02\x00Z\x06\x00\x00\x08\ts\x00\n9\x00\x91\xa5", "", -2037, 2617, 2419),
        (
            b"\t0\x02\x00\xd3\x07\x00\x00\x08\tn\x00\nA\x00\x01\xa6",
            "_ph_b",
            422,
            2625,
            2414,
        ),
        (
            b"\t2\x02\x00Z\x07\x00\x00\x08\ts\x00\n9\x00\x01\xa5",
            "_ph_b",
            421,
            2617,
            2419,
        ),
        (
            b"\t0\x02\x00\xd3\x08\x00\x00\x08\tn\x00\nA\x00\x01\xa6",
            "_ph_c",
            422,
            2625,
            2414,
        ),
        (
            b"\t2\x02\x00Z\x08\x00\x00\x08\ts\x00\n9\x00\x01\xa5",
            "_ph_c",
            421,
            2617,
            2419,
        ),
    ],
)
async def test_ts0601_electrical_measurement_multi_dp_converter(
    zigpy_device_from_v2_quirk,
    msg,
    attr_suffix,
    expected_power,
    expected_current,
    expected_volt,
):
    """Test converter for multiple electrical attributes mapped to the same tuya datapoint."""

    quirked = zigpy_device_from_v2_quirk("_TZE200_nslr42tt", "TS0601")
    ep = quirked.endpoints[1]

    tuya_manufacturer = ep.tuya_manufacturer
    hdr, data = tuya_manufacturer.deserialize(msg)
    status = tuya_manufacturer.handle_get_data(data.data)
    assert status == foundation.Status.SUCCESS

    electrical_meas_cluster = ep.electrical_measurement
    assert electrical_meas_cluster.get("active_power" + attr_suffix) == expected_power
    assert electrical_meas_cluster.get("rms_current" + attr_suffix) == expected_current
    assert electrical_meas_cluster.get("rms_voltage" + attr_suffix) == expected_volt


@pytest.mark.parametrize(
    "msg,expected_power",
    [
        (b"\x19\x8a\x02\x00\x0f\t\x02\x00\x04\x00\x00\x00\x80", 128),
        (b"\x19\x8a\x02\x00\x0f\t\x02\x00\x04\x19\x99\x99\x00", -156),
    ],
)
async def test_ts0601_power_converter(zigpy_device_from_v2_quirk, msg, expected_power):
    """Test converter for power."""

    quirked = zigpy_device_from_v2_quirk("_TZE200_nslr42tt", "TS0601")
    ep = quirked.endpoints[1]

    tuya_manufacturer = ep.tuya_manufacturer
    hdr, data = tuya_manufacturer.deserialize(msg)
    status = tuya_manufacturer.handle_get_data(data.data)
    assert status == foundation.Status.SUCCESS

    assert tuya_manufacturer.get("power") == expected_power


@pytest.mark.parametrize(
    "power_a_msg,flow_a_msg,power_b_msg,flow_b_msg,expected_power_a,expected_power_b,expected_total",
    [
        # Forward flow for both CTs
        (
            b"\x09\x1f\x02\x00\x04\x65\x02\x00\x04\x00\x00\x03\xe8",  # DP 101: power_a = 1000
            b"\x09\x11\x02\x00\x87\x66\x04\x00\x01\x00",  # DP 102: energy_flow_a = 0 (Forward)
            b"\x09\x1f\x02\x00\x04\x69\x02\x00\x04\x00\x00\x01\xf4",  # DP 105: power_b = 500
            b"\x09\x11\x02\x00\x87\x68\x04\x00\x01\x00",  # DP 104: energy_flow_b = 0 (Forward)
            1000,  # Expected power A (positive for forward)
            500,  # Expected power B (positive for forward)
            1500,  # Expected total
        ),
        # Reverse flow for both CTs
        (
            b"\x09\x1f\x02\x00\x04\x65\x02\x00\x04\x00\x00\x03\xe8",  # DP 101: power_a = 1000
            b"\x09\x0a\x02\x00\x80\x66\x04\x00\x01\x01",  # DP 102: energy_flow_a = 1 (Reverse)
            b"\x09\x1f\x02\x00\x04\x69\x02\x00\x04\x00\x00\x01\xf4",  # DP 105: power_b = 500
            b"\x09\x0a\x02\x00\x80\x68\x04\x00\x01\x01",  # DP 104: energy_flow_b = 1 (Reverse)
            -1000,  # Expected power A (negative for reverse)
            -500,  # Expected power B (negative for reverse)
            -1500,  # Expected total
        ),
        # Mixed flow directions
        (
            b"\x09\x1f\x02\x00\x04\x65\x02\x00\x04\x00\x00\x04\x00",  # DP 101: power_a = 1024
            b"\x09\x11\x02\x00\x87\x66\x04\x00\x01\x00",  # DP 102: energy_flow_a = 0 (Forward)
            b"\x09\x0a\x02\x00\x04\x69\x02\x00\x04\x00\x00\x02\x00",  # DP 105: power_b = 512
            b"\x09\x0a\x02\x00\x80\x68\x04\x00\x01\x01",  # DP 104: energy_flow_b = 1 (Reverse)
            1024,  # Expected power A (positive for forward)
            -512,  # Expected power B (negative for reverse)
            512,  # Expected total (1024 - 512)
        ),
    ],
)
async def test_matseeplus_power_reporting(
    zigpy_device_from_v2_quirk,
    power_a_msg,
    flow_a_msg,
    power_b_msg,
    flow_b_msg,
    expected_power_a,
    expected_power_b,
    expected_total,
):
    """Test power reporting using Tuya DP messages."""
    quirked = zigpy_device_from_v2_quirk("_TZE204_81yrt3lo", "TS0601")
    ep = quirked.endpoints[1]

    tuya_manufacturer = ep.tuya_manufacturer

    # Send messages in order: flow first, then power (so flow is available for alignment)
    hdr, data = tuya_manufacturer.deserialize(flow_a_msg)
    status = tuya_manufacturer.handle_get_data(data.data)
    assert status == foundation.Status.SUCCESS

    hdr, data = tuya_manufacturer.deserialize(power_a_msg)
    status = tuya_manufacturer.handle_get_data(data.data)
    assert status == foundation.Status.SUCCESS

    hdr, data = tuya_manufacturer.deserialize(flow_b_msg)
    status = tuya_manufacturer.handle_get_data(data.data)
    assert status == foundation.Status.SUCCESS

    hdr, data = tuya_manufacturer.deserialize(power_b_msg)
    status = tuya_manufacturer.handle_get_data(data.data)
    assert status == foundation.Status.SUCCESS

    # Check power values on electrical measurement clusters
    ep1_electrical = quirked.endpoints[1].electrical_measurement
    ep2_electrical = quirked.endpoints[2].electrical_measurement
    ep3_electrical = quirked.endpoints[3].electrical_measurement

    assert ep1_electrical.get("active_power") == expected_power_a
    assert ep2_electrical.get("active_power") == expected_power_b
    assert ep3_electrical.get("active_power") == expected_total


@pytest.mark.parametrize(
    "msg,endpoint_id,cluster_name,attr_name,expected_value",
    [
        # Metering DP messages
        (
            b"\x09\x1f\x02\x00\x04\x6a\x02\x00\x04\x00\x00\x30\x39",
            1,
            "smartenergy_metering",
            "current_summ_delivered",
            12345,
        ),  # DP 106: current_summ_delivered CT A
        (
            b"\x09\x1f\x02\x00\x04\x6b\x02\x00\x04\x00\x00\x1a\x85",
            1,
            "smartenergy_metering",
            "current_summ_received",
            6789,
        ),  # DP 107: current_summ_received CT A
        (
            b"\x09\x1f\x02\x00\x04\x6c\x02\x00\x04\x00\x00\xd4\x31",
            2,
            "smartenergy_metering",
            "current_summ_delivered",
            54321,
        ),  # DP 108: current_summ_delivered CT B
        (
            b"\x09\x1f\x02\x00\x04\x6d\x02\x00\x04\x00\x00\x26\x94",
            2,
            "smartenergy_metering",
            "current_summ_received",
            9876,
        ),  # DP 109: current_summ_received CT B
        # Electrical measurement DP messages
        (
            b"\x09\x1f\x02\x00\x04\x6e\x02\x00\x04\x00\x00\x03\xe8",
            1,
            "electrical_measurement",
            "power_factor",
            1000,
        ),  # DP 110: power_factor CT A
        (
            b"\x09\x1f\x02\x00\x04\x71\x02\x00\x04\x00\x00\x03\xe8",
            1,
            "electrical_measurement",
            "rms_current",
            1000,
        ),  # DP 113: rms_current CT A
        (
            b"\x09\x1f\x02\x00\x04\x72\x02\x00\x04\x00\x00\x07\xd0",
            2,
            "electrical_measurement",
            "rms_current",
            2000,
        ),  # DP 114: rms_current CT B
        (
            b"\x09\x1f\x02\x00\x04\x70\x02\x00\x04\x00\x00\x08\xfc",
            3,
            "electrical_measurement",
            "rms_voltage",
            2300,
        ),  # DP 112: rms_voltage (total)
        (
            b"\x09\x1f\x02\x00\x04\x6f\x02\x00\x04\x00\x00\x13\x88",
            3,
            "electrical_measurement",
            "ac_frequency",
            5000,
        ),  # DP 111: ac_frequency (total)
        (
            b"\x09\x1f\x02\x00\x04\x79\x02\x00\x04\x00\x00\x03\xe8",
            2,
            "electrical_measurement",
            "power_factor",
            1000,
        ),  # DP 121: power_factor CT B
    ],
)
async def test_matseeplus_electrical_and_metering(
    zigpy_device_from_v2_quirk,
    msg,
    endpoint_id,
    cluster_name,
    attr_name,
    expected_value,
):
    """Test electrical measurement and metering attributes."""
    quirked = zigpy_device_from_v2_quirk("_TZE204_81yrt3lo", "TS0601")
    ep = quirked.endpoints[1]

    tuya_manufacturer = ep.tuya_manufacturer
    hdr, data = tuya_manufacturer.deserialize(msg)
    status = tuya_manufacturer.handle_get_data(data.data)
    assert status == foundation.Status.SUCCESS

    cluster = getattr(quirked.endpoints[endpoint_id], cluster_name)
    assert cluster.get(attr_name) == expected_value


async def test_matseeplus_late_flow_mitigation(zigpy_device_from_v2_quirk):
    """Test late energy flow mitigation feature when enabled."""
    quirked = zigpy_device_from_v2_quirk("_TZE204_81yrt3lo", "TS0601")
    ep = quirked.endpoints[1]

    # Enable late energy flow mitigation
    local_config = ep.local_config
    await local_config.write_attributes(
        {"late_energy_flow_a": True, "late_energy_flow_b": True}
    )

    tuya_manufacturer = ep.tuya_manufacturer
    ep1_electrical = quirked.endpoints[1].electrical_measurement
    ep2_electrical = quirked.endpoints[2].electrical_measurement
    ep3_electrical = quirked.endpoints[3].electrical_measurement

    # Send power messages first - should be held until flow messages arrive
    hdr, data = tuya_manufacturer.deserialize(
        b"\x09\x1f\x02\x00\x04\x65\x02\x00\x04\x00\x00\x03\x20"
    )  # DP 101: power_a = 800
    status = tuya_manufacturer.handle_get_data(data.data)
    assert status == foundation.Status.SUCCESS

    hdr, data = tuya_manufacturer.deserialize(
        b"\x09\x1f\x02\x00\x04\x69\x02\x00\x04\x00\x00\x02\x58"
    )  # DP 105: power_b = 600
    status = tuya_manufacturer.handle_get_data(data.data)
    assert status == foundation.Status.SUCCESS

    # Power should be None (held) until flow messages arrive
    assert ep1_electrical.get("active_power") is None
    assert ep2_electrical.get("active_power") is None

    # Send flow messages - should release the held power values
    hdr, data = tuya_manufacturer.deserialize(
        b"\x09\x11\x02\x00\x87\x66\x04\x00\x01\x00"
    )  # DP 102: energy_flow_a = 0 (Forward)
    status = tuya_manufacturer.handle_get_data(data.data)
    assert status == foundation.Status.SUCCESS

    # Power A should now be available (positive for forward flow)
    assert ep1_electrical.get("active_power") == 800

    hdr, data = tuya_manufacturer.deserialize(
        b"\x09\x0a\x02\x00\x80\x68\x04\x00\x01\x01"
    )  # DP 104: energy_flow_b = 1 (Reverse)
    status = tuya_manufacturer.handle_get_data(data.data)
    assert status == foundation.Status.SUCCESS

    # Power B should now be available (negative for reverse flow) and total calculated
    assert ep2_electrical.get("active_power") == -600
    assert ep3_electrical.get("active_power") == 200  # 800 + (-600)

    # Test non-power attribute delay
    hdr, data = tuya_manufacturer.deserialize(
        b"\x09\x1f\x02\x00\x04\x71\x02\x00\x04\x00\x00\x03\xe8"
    )  # DP 113: rms_current = 1000
    status = tuya_manufacturer.handle_get_data(data.data)
    assert status == foundation.Status.SUCCESS

    # Current should be held
    assert ep1_electrical.get("rms_current") is None

    # Send another current message - should release the previous one
    hdr, data = tuya_manufacturer.deserialize(
        b"\x09\x1f\x02\x00\x04\x71\x02\x00\x04\x00\x00\x07\xd0"
    )  # DP 113: rms_current = 2000
    status = tuya_manufacturer.handle_get_data(data.data)
    assert status == foundation.Status.SUCCESS

    # Previous current (1000) should now be available
    assert ep1_electrical.get("rms_current") == 1000

    # Test zero power - should be reported immediately even in late flow mode
    hdr, data = tuya_manufacturer.deserialize(
        b"\x09\x1f\x02\x00\x04\x65\x02\x00\x04\x00\x00\x00\x00"
    )  # DP 101: power_a = 0
    status = tuya_manufacturer.handle_get_data(data.data)
    assert status == foundation.Status.SUCCESS

    # Zero power should be immediately available
    assert ep1_electrical.get("active_power") == 0
