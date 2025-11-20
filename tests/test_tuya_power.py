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
    "manufacturer,power_a_msg,flow_a_msg,power_b_msg,flow_b_msg,expected_power_a,expected_power_b,expected_total",
    [
        # Forward flow for both CTs - _TZE204 model
        (
            "_TZE204_81yrt3lo",
            b"\x09\x1f\x02\x00\x04\x65\x02\x00\x04\x00\x00\x03\xe8",  # DP 101: power_a = 1000
            b"\x09\x11\x02\x00\x87\x66\x04\x00\x01\x00",  # DP 102: energy_flow_a = 0 (Forward)
            b"\x09\x1f\x02\x00\x04\x69\x02\x00\x04\x00\x00\x01\xf4",  # DP 105: power_b = 500
            b"\x09\x11\x02\x00\x87\x68\x04\x00\x01\x00",  # DP 104: energy_flow_b = 0 (Forward)
            1000,  # Expected power A (positive for forward)
            500,  # Expected power B (positive for forward)
            1500,  # Expected total
        ),
        # Reverse flow for both CTs - _TZE204 model
        (
            "_TZE204_81yrt3lo",
            b"\x09\x1f\x02\x00\x04\x65\x02\x00\x04\x00\x00\x03\xe8",  # DP 101: power_a = 1000
            b"\x09\x0a\x02\x00\x80\x66\x04\x00\x01\x01",  # DP 102: energy_flow_a = 1 (Reverse)
            b"\x09\x1f\x02\x00\x04\x69\x02\x00\x04\x00\x00\x01\xf4",  # DP 105: power_b = 500
            b"\x09\x0a\x02\x00\x80\x68\x04\x00\x01\x01",  # DP 104: energy_flow_b = 1 (Reverse)
            -1000,  # Expected power A (negative for reverse)
            -500,  # Expected power B (negative for reverse)
            -1500,  # Expected total
        ),
        # Mixed flow directions - _TZE204 model
        (
            "_TZE204_81yrt3lo",
            b"\x09\x1f\x02\x00\x04\x65\x02\x00\x04\x00\x00\x04\x00",  # DP 101: power_a = 1024
            b"\x09\x11\x02\x00\x87\x66\x04\x00\x01\x00",  # DP 102: energy_flow_a = 0 (Forward)
            b"\x09\x0a\x02\x00\x04\x69\x02\x00\x04\x00\x00\x02\x00",  # DP 105: power_b = 512
            b"\x09\x0a\x02\x00\x80\x68\x04\x00\x01\x01",  # DP 104: energy_flow_b = 1 (Reverse)
            1024,  # Expected power A (positive for forward)
            -512,  # Expected power B (negative for reverse)
            512,  # Expected total (1024 - 512)
        ),
        # Forward flow for both CTs - _TZE284 model
        (
            "_TZE284_81yrt3lo",
            b"\x09\x1f\x02\x00\x04\x65\x02\x00\x04\x00\x00\x03\xe8",  # DP 101: power_a = 1000
            b"\x09\x11\x02\x00\x87\x66\x04\x00\x01\x00",  # DP 102: energy_flow_a = 0 (Forward)
            b"\x09\x1f\x02\x00\x04\x69\x02\x00\x04\x00\x00\x01\xf4",  # DP 105: power_b = 500
            b"\x09\x11\x02\x00\x87\x68\x04\x00\x01\x00",  # DP 104: energy_flow_b = 0 (Forward)
            1000,  # Expected power A (positive for forward)
            500,  # Expected power B (positive for forward)
            1500,  # Expected total
        ),
    ],
)
async def test_matseeplus_power_reporting(
    zigpy_device_from_v2_quirk,
    manufacturer,
    power_a_msg,
    flow_a_msg,
    power_b_msg,
    flow_b_msg,
    expected_power_a,
    expected_power_b,
    expected_total,
):
    """Test power reporting using Tuya DP messages with default settings (late flow mitigation disabled)."""
    quirked = zigpy_device_from_v2_quirk(manufacturer, "TS0601")
    ep = quirked.endpoints[1]

    tuya_manufacturer = ep.tuya_manufacturer

    def send_dp_message(msg):
        """Send and verify a DP message."""
        hdr, data = tuya_manufacturer.deserialize(msg)
        status = tuya_manufacturer.handle_get_data(data.data)
        assert status == foundation.Status.SUCCESS

    # Send messages in order: flow first, then power (flow is sent first for correct sign application)
    send_dp_message(flow_a_msg)
    send_dp_message(power_a_msg)
    send_dp_message(flow_b_msg)
    send_dp_message(power_b_msg)

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


@pytest.mark.parametrize(
    "late_flow_a,late_flow_b",
    [
        (False, False),  # Both disabled
        (True, False),  # Only A enabled
        (False, True),  # Only B enabled
        (True, True),  # Both enabled
    ],
)
async def test_matseeplus_late_flow_power_reporting(
    zigpy_device_from_v2_quirk, late_flow_a, late_flow_b
):
    """Test basic power reporting with and without late flow mitigation."""
    quirked = zigpy_device_from_v2_quirk("_TZE204_81yrt3lo", "TS0601")
    ep = quirked.endpoints[1]

    # Set mitigation settings
    local_config = ep.local_config
    await local_config.write_attributes(
        {"late_energy_flow_a": late_flow_a, "late_energy_flow_b": late_flow_b}
    )

    tuya_manufacturer = ep.tuya_manufacturer
    ep1_electrical = quirked.endpoints[1].electrical_measurement
    ep2_electrical = quirked.endpoints[2].electrical_measurement
    ep3_electrical = quirked.endpoints[3].electrical_measurement

    def send_dp_message(msg):
        """Send and verify a DP message."""
        hdr, data = tuya_manufacturer.deserialize(msg)
        status = tuya_manufacturer.handle_get_data(data.data)
        assert status == foundation.Status.SUCCESS

    # Send power messages first
    send_dp_message(
        b"\x09\x1f\x02\x00\x04\x65\x02\x00\x04\x00\x00\x03\x20"
    )  # DP 101: power_a = 800

    send_dp_message(
        b"\x09\x1f\x02\x00\x04\x69\x02\x00\x04\x00\x00\x02\x58"
    )  # DP 105: power_b = 600

    # Check if power is held or available based on mitigation settings
    if late_flow_a:
        assert ep1_electrical.get("active_power") is None
    else:
        # Without mitigation, power is available immediately (unsigned)
        assert ep1_electrical.get("active_power") == 800

    if late_flow_b:
        assert ep2_electrical.get("active_power") is None
    else:
        # Without mitigation, power is available immediately (unsigned)
        assert ep2_electrical.get("active_power") == 600

    # Send flow messages
    send_dp_message(
        b"\x09\x11\x02\x00\x87\x66\x04\x00\x01\x00"
    )  # DP 102: energy_flow_a = 0 (Forward)

    # Power A is now available (positive for forward flow)
    assert ep1_electrical.get("active_power") == 800

    send_dp_message(
        b"\x09\x0a\x02\x00\x80\x68\x04\x00\x01\x01"
    )  # DP 104: energy_flow_b = 1 (Reverse)

    # Check power B and total based on mitigation setting
    if late_flow_b:
        # With mitigation, power B is updated with correct sign (negative for reverse)
        assert ep2_electrical.get("active_power") == -600
        # Channel B processing calls _maybe_report_total_power()
        # Both report intervals now equal _interval (1), so total is calculated
        assert ep3_electrical.get("active_power") == 200  # 800 + (-600)
    else:
        # Without mitigation, power B was already reported as unsigned (600)
        # flow_b doesn't update the power value, but updates _report_interval_b to match _interval
        assert ep2_electrical.get("active_power") == 600
        # Channel B processing calls _maybe_report_total_power()
        # Both report intervals now equal _interval (1), so total is calculated
        assert ep3_electrical.get("active_power") == 1400  # 800 + 600


@pytest.mark.parametrize("late_flow_enabled", [True, False])
async def test_matseeplus_late_flow_non_power_attribute_delay(
    zigpy_device_from_v2_quirk, late_flow_enabled
):
    """Test that non-power attributes are delayed when late flow mitigation is enabled."""
    quirked = zigpy_device_from_v2_quirk("_TZE204_81yrt3lo", "TS0601")
    ep = quirked.endpoints[1]

    # Set mitigation settings
    local_config = ep.local_config
    await local_config.write_attributes(
        {"late_energy_flow_a": late_flow_enabled, "late_energy_flow_b": False}
    )

    tuya_manufacturer = ep.tuya_manufacturer
    ep1_electrical = quirked.endpoints[1].electrical_measurement

    def send_dp_message(msg):
        """Send and verify a DP message."""
        hdr, data = tuya_manufacturer.deserialize(msg)
        status = tuya_manufacturer.handle_get_data(data.data)
        assert status == foundation.Status.SUCCESS

    # Send initial power and flow to establish baseline
    send_dp_message(
        b"\x09\x1f\x02\x00\x04\x65\x02\x00\x04\x00\x00\x03\x20"
    )  # DP 101: power_a = 800
    send_dp_message(
        b"\x09\x11\x02\x00\x87\x66\x04\x00\x01\x00"
    )  # DP 102: energy_flow_a = 0 (Forward)

    # Send current measurement
    send_dp_message(
        b"\x09\x1f\x02\x00\x04\x71\x02\x00\x04\x00\x00\x03\xe8"
    )  # DP 113: rms_current = 1000

    if late_flow_enabled:
        # Current is held
        assert ep1_electrical.get("rms_current") is None

        # Send another current message - releases the previous one
        send_dp_message(
            b"\x09\x1f\x02\x00\x04\x71\x02\x00\x04\x00\x00\x07\xd0"
        )  # DP 113: rms_current = 2000

        # Previous current (1000) is now available
        assert ep1_electrical.get("rms_current") == 1000
    else:
        # Without mitigation, current is available immediately
        assert ep1_electrical.get("rms_current") == 1000


@pytest.mark.parametrize(
    "late_flow_a,late_flow_b",
    [
        (False, False),  # Both disabled
        (True, False),  # Only A enabled
        (False, True),  # Only B enabled
        (True, True),  # Both enabled
    ],
)
async def test_matseeplus_late_flow_zero_power_deferral(
    zigpy_device_from_v2_quirk, late_flow_a, late_flow_b
):
    """Test zero power deferral and cross-channel release with all configuration combinations."""
    quirked = zigpy_device_from_v2_quirk("_TZE204_81yrt3lo", "TS0601")
    ep = quirked.endpoints[1]

    # Set mitigation settings
    local_config = ep.local_config
    await local_config.write_attributes(
        {"late_energy_flow_a": late_flow_a, "late_energy_flow_b": late_flow_b}
    )

    tuya_manufacturer = ep.tuya_manufacturer
    ep1_electrical = quirked.endpoints[1].electrical_measurement
    ep2_electrical = quirked.endpoints[2].electrical_measurement
    ep3_electrical = quirked.endpoints[3].electrical_measurement

    def send_dp_message(msg):
        """Send and verify a DP message."""
        hdr, data = tuya_manufacturer.deserialize(msg)
        status = tuya_manufacturer.handle_get_data(data.data)
        assert status == foundation.Status.SUCCESS

    # Establish baseline with both channels
    send_dp_message(
        b"\x09\x1f\x02\x00\x04\x65\x02\x00\x04\x00\x00\x03\x20"
    )  # DP 101: power_a = 800
    send_dp_message(
        b"\x09\x1f\x02\x00\x04\x69\x02\x00\x04\x00\x00\x02\x58"
    )  # DP 105: power_b = 600
    send_dp_message(
        b"\x09\x11\x02\x00\x87\x66\x04\x00\x01\x00"
    )  # DP 102: energy_flow_a = 0 (Forward)
    send_dp_message(
        b"\x09\x0a\x02\x00\x80\x68\x04\x00\x01\x01"
    )  # DP 104: energy_flow_b = 1 (Reverse)

    assert ep1_electrical.get("active_power") == 800
    if late_flow_b:
        # Channel B flow releases the deferred value with correct sign
        assert ep2_electrical.get("active_power") == -600
        # Channel B processing calls _maybe_report_total_power() with both intervals matching
        assert ep3_electrical.get("active_power") == 200  # 800 + (-600)
    else:
        # Without mitigation for B, power was already reported unsigned as 600
        assert ep2_electrical.get("active_power") == 600
        # Channel B processing updates _report_interval_b and calls _maybe_report_total_power()
        assert ep3_electrical.get("active_power") == 1400  # 800 + 600

    # Test channel A zero power deferral - device omits flow DP when power is 0
    send_dp_message(
        b"\x09\x1f\x02\x00\x04\x65\x02\x00\x04\x00\x00\x00\x00"
    )  # DP 101: power_a = 0 (no flow DP sent)

    if late_flow_a:
        # Zero power is deferred (not reported yet)
        assert ep1_electrical.get("active_power") == 800  # Still showing previous
    else:
        # Without mitigation, zero is reported immediately (unsigned)
        assert ep1_electrical.get("active_power") == 0

    # Test channel B zero power deferral
    send_dp_message(
        b"\x09\x1f\x02\x00\x04\x69\x02\x00\x04\x00\x00\x00\x00"
    )  # DP 105: power_b = 0 (no flow DP sent)

    if late_flow_b:
        # Zero power is deferred (report_interval_b set to _interval + 1)
        # Previous value still showing
        assert ep2_electrical.get("active_power") == -600
    else:
        # Without mitigation, zero is reported immediately
        assert ep2_electrical.get("active_power") == 0
        # Channel B update recalculates total based on current interval matching
        if late_flow_a:
            # A is deferred to interval 2, B is at interval 1, intervals don't match
            # Total stays at previous value (1400 or 200 depending on late_flow_b initial state)
            # Since late_flow_b=False here, baseline total was 1400
            assert ep3_electrical.get("active_power") == 1400
        else:
            # A already reported 0 at interval 1, B now reports 0 at interval 1, both match
            assert ep3_electrical.get("active_power") == 0

    # Next interval: Channel A flow arrives first - tests cross-channel release
    send_dp_message(
        b"\x09\x11\x02\x00\x87\x66\x04\x00\x01\x00"
    )  # DP 102: energy_flow_a = 0 (releases deferred zeros)

    if late_flow_a:
        # Flow message releases the deferred zero for A
        assert ep1_electrical.get("active_power") == 0
    else:
        # Without mitigation, zero was already reported
        assert ep1_electrical.get("active_power") == 0

    if late_flow_b:
        # Cross-channel: A's flow increments interval to 2, releases deferred zero for B (report_interval_b was 2)
        assert ep2_electrical.get("active_power") == 0
        # After flow_a releases both deferred values, _report_interval_a=2, _report_interval_b=2, _interval=2
        assert (
            ep3_electrical.get("active_power") == 0
        )  # 0 + 0 (recalculated by flow_a release)
    else:
        # Without mitigation for B, zero was already reported at interval 1
        # After flow_a: _interval=2, _report_interval_a=2, _report_interval_b=1, no total update
        assert ep2_electrical.get("active_power") == 0
        assert (
            ep3_electrical.get("active_power") == 0
        )  # Unchanged from previous (0 + 0)

    # Then new power A arrives
    send_dp_message(
        b"\x09\x1f\x02\x00\x04\x65\x02\x00\x04\x00\x00\x03\x84"
    )  # DP 101: power_a = 900

    if late_flow_a:
        # Power A is deferred to interval 3, not released yet
        assert ep1_electrical.get("active_power") == 0  # Previous A
        # Total not recalculated (_report_interval_a=3, intervals don't match)
        assert ep3_electrical.get("active_power") == 0  # 0 + 0, unchanged
    else:
        # Without mitigation, power is reported immediately at interval 2
        assert ep1_electrical.get("active_power") == 900
        # Total calculation depends on whether _report_interval_b also equals 2
        if late_flow_b:
            # B was deferred to interval 2, both at interval 2, total calculated
            assert ep3_electrical.get("active_power") == 900  # 900 + 0
        else:
            # B is at interval 1, A is at interval 2, intervals don't match, no total update
            assert ep3_electrical.get("active_power") == 0  # Unchanged from previous

    # Send a flow message for B to trigger total update
    send_dp_message(
        b"\x09\x0a\x02\x00\x80\x68\x04\x00\x01\x00"
    )  # DP 104: energy_flow_b = 0 (Forward)

    # flow_b processing updates _report_interval_b to current interval and calls _maybe_report_total_power()
    if late_flow_a:
        # A is deferred to interval 3, B now at interval 2, intervals don't match, no total update
        assert ep3_electrical.get("active_power") == 0  # 0 + 0, unchanged
    else:
        if late_flow_b:
            # A at interval 2, B at interval 2 (after release by flow_a earlier), both match
            # flow_b processes but doesn't change power_b (still 0), updates _report_interval_b to 3
            # Now intervals don't match (A=2, B=3), no total update
            assert ep3_electrical.get("active_power") == 900  # Unchanged from previous
        else:
            # A at interval 2, B now updated to interval 2, both match, total recalculated
            assert ep3_electrical.get("active_power") == 900  # 900 + 0

    # Test simultaneous zeros on both channels
    # Reset baseline to different values
    send_dp_message(
        b"\x09\x1f\x02\x00\x04\x65\x02\x00\x04\x00\x00\x03\x84"
    )  # DP 101: power_a = 900
    send_dp_message(
        b"\x09\x11\x02\x00\x87\x66\x04\x00\x01\x00"
    )  # DP 102: energy_flow_a = 0 (releases previous deferred if any)
    send_dp_message(
        b"\x09\x1f\x02\x00\x04\x69\x02\x00\x04\x00\x00\x02\x58"
    )  # DP 105: power_b = 600
    send_dp_message(
        b"\x09\x0a\x02\x00\x80\x68\x04\x00\x01\x00"
    )  # DP 104: energy_flow_b = 0 (Forward)

    assert ep1_electrical.get("active_power") == 900
    assert ep2_electrical.get("active_power") == 600
    # Total recalculates when both report intervals match current interval
    if late_flow_a:
        # flow_a released deferred power_a(900), both A and B at same interval, total=1500
        assert ep3_electrical.get("active_power") == 1500
    else:
        if late_flow_b:
            # Total remains at 600 from earlier due to interval desynchronization
            assert ep3_electrical.get("active_power") == 600
        else:
            # power_a(900) and power_b(600) both reported, flow_b updated intervals to match, total=1500
            assert ep3_electrical.get("active_power") == 1500

    # Send zero for both channels simultaneously (neither sends flow DP)
    send_dp_message(
        b"\x09\x1f\x02\x00\x04\x65\x02\x00\x04\x00\x00\x00\x00"
    )  # DP 101: power_a = 0
    send_dp_message(
        b"\x09\x1f\x02\x00\x04\x69\x02\x00\x04\x00\x00\x00\x00"
    )  # DP 105: power_b = 0

    # Check based on mitigation settings
    if late_flow_a:
        assert ep1_electrical.get("active_power") == 900  # Deferred, previous A
    else:
        assert ep1_electrical.get("active_power") == 0  # Reported immediately

    if late_flow_b:
        assert ep2_electrical.get("active_power") == 600  # Deferred, previous B
    else:
        assert ep2_electrical.get("active_power") == 0  # Reported immediately

    # Total recalculation depends on whether report intervals match current interval
    if late_flow_a and late_flow_b:
        # Both deferred, intervals don't match, total unchanged
        assert ep3_electrical.get("active_power") == 1500  # Unchanged
    elif late_flow_a:
        # A deferred, B reported 0 immediately, intervals don't match, total unchanged
        assert ep3_electrical.get("active_power") == 1500  # Unchanged
    elif late_flow_b:
        # A reported 0 immediately, B deferred, intervals don't match, total unchanged
        assert ep3_electrical.get("active_power") == 1500  # Unchanged
    else:
        # Both reported immediately, intervals match, total recalculated
        assert ep3_electrical.get("active_power") == 0  # 0 + 0

    # Send next interval update for A to release deferred zeros
    send_dp_message(
        b"\x09\x1f\x02\x00\x04\x65\x02\x00\x04\x00\x00\x00\x64"
    )  # DP 101: power_a = 100 (deferred if late_flow_a enabled)

    # power_a=100 arrives - deferred zeros NOT released yet (only flow_a releases)
    if late_flow_a and late_flow_b:
        # A's new value deferred, both still showing previous, deferred zeros not yet released
        assert (
            ep1_electrical.get("active_power") == 900
        )  # Previous A (deferred 0 not released)
        assert (
            ep2_electrical.get("active_power") == 600
        )  # Previous B (deferred 0 not released)
        assert ep3_electrical.get("active_power") == 1500  # Unchanged
    elif late_flow_a:
        # A's new value deferred, B already showing 0, A's deferred 0 not yet released
        assert (
            ep1_electrical.get("active_power") == 900
        )  # Previous A (deferred 0 not released)
        assert ep2_electrical.get("active_power") == 0  # Already 0
        assert ep3_electrical.get("active_power") == 1500  # Unchanged
    elif late_flow_b:
        # A reports 100 immediately, B's deferred 0 not yet released (only flow_a releases cross-channel)
        assert ep1_electrical.get("active_power") == 100  # New value reported
        assert (
            ep2_electrical.get("active_power") == 600
        )  # Previous B (deferred 0 not released)
        # Total stays at 600 because intervals don't match (A at new interval, B deferred to different interval)
        assert (
            ep3_electrical.get("active_power") == 600
        )  # Unchanged (intervals don't match)
    else:
        # A reports 100 immediately, B already 0, intervals synchronized
        assert ep1_electrical.get("active_power") == 100  # New value
        assert ep2_electrical.get("active_power") == 0  # Already 0
        # Intervals match, total recalculated
        assert ep3_electrical.get("active_power") == 100  # 100 + 0

    # Send flow_a to release deferred values and process new power value
    send_dp_message(
        b"\x09\x11\x02\x00\x87\x66\x04\x00\x01\x00"
    )  # DP 102: energy_flow_a = 0 (Forward)

    # flow_a increments interval, releases any deferred values, then processes the new power_a(100)
    if late_flow_a and late_flow_b:
        # Both deferred zeros released, then power_a(100) processed and reported
        assert (
            ep1_electrical.get("active_power") == 100
        )  # 100 released with correct sign
        assert ep2_electrical.get("active_power") == 0  # Deferred 0 released by flow_a
        assert ep3_electrical.get("active_power") == 100  # 100 + 0
    elif late_flow_a:
        # Deferred A zero released, then power_a(100) processed, but B interval doesn't match
        assert (
            ep1_electrical.get("active_power") == 100
        )  # 100 released with correct sign
        assert ep2_electrical.get("active_power") == 0  # Already 0
        # Total not recalculated (intervals don't match: A at M+2, B at M+1)
        assert ep3_electrical.get("active_power") == 1500  # Unchanged
    elif late_flow_b:
        # A(100) already reported at M+1, B's deferred 0 (at M+2) released by flow_a cross-channel
        # flow_a increments to M+2, releases B's deferred 0, then processes A updating _report_interval_a to M+2
        assert ep1_electrical.get("active_power") == 100  # Already 100
        assert ep2_electrical.get("active_power") == 0  # Deferred 0 released
        # After flow_a processing, both intervals at M+2, total recalculated
        assert ep3_electrical.get("active_power") == 100  # 100 + 0 (recalculated)
    else:
        # Neither deferred, both at same interval, flow_a updates A's interval to M+2
        assert ep1_electrical.get("active_power") == 100  # Already 100
        assert ep2_electrical.get("active_power") == 0  # Already 0
        # A at M+2, B at M+1, intervals don't match, total unchanged
        assert ep3_electrical.get("active_power") == 100  # Unchanged

    # Send flow_b to update channel B intervals
    send_dp_message(
        b"\x09\x0a\x02\x00\x80\x68\x04\x00\x01\x00"
    )  # DP 104: energy_flow_b = 0

    # flow_b processing doesn't change power values but may update total if intervals sync
    assert ep1_electrical.get("active_power") == 100
    assert ep2_electrical.get("active_power") == 0
    assert ep3_electrical.get("active_power") == 100  # 100 + 0
