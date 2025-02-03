"""Tests for Tuya quirks."""

import pytest
from zigpy.zcl.clusters.homeautomation import ElectricalMeasurement
from zigpy.zcl.clusters.smartenergy import Metering

from tests.common import ClusterListener
import zhaquirks
from zhaquirks import LocalDataCluster
import zhaquirks.tuya
from zhaquirks.tuya.mcu import TuyaMCUCluster

zhaquirks.setup()


@pytest.mark.parametrize(
    "model,manuf,channels,bidirectional",
    [
        (
            "_TZE204_cjbofhxw",
            "TS0601",
            {1},
            False,
        ),
        ("_TZE204_ac0fhfiq", "TS0601", {1}, True),
        ("_TZE200_rks0sgb7", "TS0601", {1, 2, 11}, True),
        ("_TZE204_81yrt3lo", "TS0601", {1, 2, 11}, True),
    ],
)
async def test_tuya_energy_meter_quirk_energy_direction_align(
    zigpy_device_from_v2_quirk,
    model: str,
    manuf: str,
    channels,
    bidirectional: bool,
):
    """Test Tuya Energy Meter Quirk energy direction align in ElectricalMeasurement and Metering clusters."""
    quirked_device = zigpy_device_from_v2_quirk(model, manuf)

    ENERGY_DIRECTION_ATTR = "energy_direction"
    ENERGY_DIRECTION_ATTR_B = "energy_direction_ch_b"
    FORWARD = 0
    REVERSE = 1

    CH_A = 1
    CH_B = 2
    CH_AB = 11

    UNSIGNED_ATTR_SUFFIX = "_attr_unsigned"

    CURRENT = 5
    POWER = 100
    VOLTAGE = 230
    SUMM_RECEIVED = 15000
    DIRECTION_A = REVERSE
    DIRECTION_B = FORWARD

    ep = quirked_device.endpoints[1]

    assert ep.tuya_manufacturer is not None
    assert isinstance(ep.tuya_manufacturer, TuyaMCUCluster)
    mcu_listener = ClusterListener(ep.tuya_manufacturer)

    listeners = {}
    for channel in channels:
        channel_ep = quirked_device.endpoints.get(channel, None)
        assert channel_ep is not None

        assert channel_ep.electrical_measurement is not None
        assert isinstance(channel_ep.electrical_measurement, ElectricalMeasurement)

        assert channel_ep.smartenergy_metering is not None
        assert isinstance(channel_ep.smartenergy_metering, Metering)

        listeners[channel] = {
            "metering": ClusterListener(channel_ep.smartenergy_metering),
            "electrical_measurement": ClusterListener(
                channel_ep.electrical_measurement
            ),
        }

    if bidirectional:
        # verify the direction attribute is present
        attr = getattr(ep.tuya_manufacturer.AttributeDefs, ENERGY_DIRECTION_ATTR, None)
        assert attr is not None

        # set the initial direction
        ep.tuya_manufacturer.update_attribute(ENERGY_DIRECTION_ATTR, DIRECTION_A)
        assert len(mcu_listener.attribute_updates) == 1
        assert mcu_listener.attribute_updates[0][0] == attr.id
        assert mcu_listener.attribute_updates[0][1] == DIRECTION_A
    else:
        # verify the direction & direction B attributes are not present
        attr = getattr(ep.tuya_manufacturer.AttributeDefs, ENERGY_DIRECTION_ATTR, None)
        assert attr is None
        attr = getattr(
            ep.tuya_manufacturer.AttributeDefs,
            ENERGY_DIRECTION_ATTR_B,
            None,
        )
        assert attr is None

    if bidirectional and CH_B in channels:
        # verify the direction B attribute is present
        attr = getattr(
            ep.tuya_manufacturer.AttributeDefs,
            ENERGY_DIRECTION_ATTR_B,
            None,
        )
        assert attr is not None

        # set the initial direction
        ep.tuya_manufacturer.update_attribute(ENERGY_DIRECTION_ATTR_B, DIRECTION_B)
        assert len(mcu_listener.attribute_updates) == 2
        assert mcu_listener.attribute_updates[1][0] == attr.id
        assert mcu_listener.attribute_updates[1][1] == DIRECTION_B

    if CH_AB in channels:
        # verify the config cluster is present
        channel_ep = quirked_device.endpoints[1]
        assert channel_ep.energy_meter_config is not None
        assert isinstance(channel_ep.energy_meter_config, LocalDataCluster)

        config_listener = ClusterListener(ep.energy_meter_config)

        # set the initial virtual channel calculation method (sum A and B)
        channel_ep.energy_meter_config.update_attribute(
            channel_ep.energy_meter_config.AttributeDefs.virtual_channel_config.id,
            channel_ep.energy_meter_config.VirtualChannelConfig.A_plus_B,
        )
        assert len(config_listener.attribute_updates) == 1
        assert (
            config_listener.attribute_updates[0][0]
            == channel_ep.energy_meter_config.AttributeDefs.virtual_channel_config.id
        )
        assert (
            config_listener.attribute_updates[0][1]
            == channel_ep.energy_meter_config.VirtualChannelConfig.A_plus_B
        )

    for channel in channels:
        if channel == CH_A:
            direction = DIRECTION_A
        elif channel == CH_B:
            direction = DIRECTION_B
        elif channel == CH_AB:
            # updates to channel AB will occur as a result of the device updates to channels A & B
            continue
        assert direction is not None

        channel_ep = quirked_device.endpoints[channel]

        # update ElectricalMeasurement attributes
        channel_ep.electrical_measurement.update_attribute(
            ElectricalMeasurement.AttributeDefs.rms_current.name, CURRENT
        )
        channel_ep.electrical_measurement.update_attribute(
            ElectricalMeasurement.AttributeDefs.rms_voltage.name, VOLTAGE
        )
        channel_ep.electrical_measurement.update_attribute(
            # The UNSIGNED_ATTR_SUFFIX applies energy direction on bidirectional devices
            ElectricalMeasurement.AttributeDefs.active_power.name
            + UNSIGNED_ATTR_SUFFIX,
            POWER,
        )

        # verify the ElectricalMeasurement attributes were updated correctly
        assert len(listeners[channel]["electrical_measurement"].attribute_updates) == 4
        assert (
            listeners[channel]["electrical_measurement"].attribute_updates[0][0]
            == ElectricalMeasurement.AttributeDefs.rms_current.id
        )
        assert (
            listeners[channel]["electrical_measurement"].attribute_updates[0][1]
            == CURRENT
        )
        assert (
            listeners[channel]["electrical_measurement"].attribute_updates[1][0]
            == ElectricalMeasurement.AttributeDefs.rms_voltage.id
        )
        assert (
            listeners[channel]["electrical_measurement"].attribute_updates[1][1]
            == VOLTAGE
        )
        assert (
            listeners[channel]["electrical_measurement"].attribute_updates[2][0]
            == ElectricalMeasurement.AttributeDefs.active_power.id
        )
        assert (
            listeners[channel]["electrical_measurement"].attribute_updates[2][1]
            == POWER
            if not bidirectional or direction == FORWARD
            else -POWER
        )
        assert (
            listeners[channel]["electrical_measurement"].attribute_updates[3][0]
            == ElectricalMeasurement.AttributeDefs.measurement_type.id
        )
        assert (
            listeners[channel]["electrical_measurement"].attribute_updates[3][1]
            == ElectricalMeasurement.MeasurementType.Active_measurement_AC
            | ElectricalMeasurement.MeasurementType.Phase_A_measurement  # updated by the _update_measurement_type function
        )

        # update Metering attributes
        channel_ep.smartenergy_metering.update_attribute(
            Metering.AttributeDefs.instantaneous_demand.name + UNSIGNED_ATTR_SUFFIX,
            POWER,
        )
        channel_ep.smartenergy_metering.update_attribute(
            # The UNSIGNED_ATTR_SUFFIX applies energy direction on bidirectional devices
            Metering.AttributeDefs.current_summ_received.name,
            SUMM_RECEIVED,
        )

        # verify the Metering attributes were updated correctly
        assert len(listeners[channel]["metering"].attribute_updates) == 2
        assert (
            listeners[channel]["metering"].attribute_updates[0][0]
            == Metering.AttributeDefs.instantaneous_demand.id
        )
        assert (
            listeners[channel]["metering"].attribute_updates[0][1] == POWER
            if not bidirectional or direction == FORWARD
            else -POWER
        )
        assert (
            listeners[channel]["metering"].attribute_updates[1][0]
            == Metering.AttributeDefs.current_summ_received.id
        )
        assert listeners[channel]["metering"].attribute_updates[1][1] == SUMM_RECEIVED

    if CH_AB in channels:
        # verify the ElectricalMeasurement attributes were updated correctly
        assert len(listeners[CH_AB]["electrical_measurement"].attribute_updates) == 3
        assert (
            listeners[CH_AB]["electrical_measurement"].attribute_updates[0][0]
            == ElectricalMeasurement.AttributeDefs.rms_current.id
        )
        assert (
            listeners[CH_AB]["electrical_measurement"].attribute_updates[0][1]
            == -CURRENT + CURRENT  # -CURRENT + CURRENT = 0
        )
        assert (
            listeners[CH_AB]["electrical_measurement"].attribute_updates[1][0]
            == ElectricalMeasurement.AttributeDefs.active_power.id
        )
        assert (
            listeners[CH_AB]["electrical_measurement"].attribute_updates[1][1] == 0
        )  # -POWER + POWER = 0
        assert (
            listeners[CH_AB]["electrical_measurement"].attribute_updates[2][0]
            == ElectricalMeasurement.AttributeDefs.measurement_type.id
        )
        assert (
            listeners[CH_AB]["electrical_measurement"].attribute_updates[2][1]
            == ElectricalMeasurement.MeasurementType.Active_measurement_AC
            | ElectricalMeasurement.MeasurementType.Phase_A_measurement  # updated by the _update_measurement_type function
        )

        # verify the Metering attributes were updated correctly
        assert len(listeners[CH_AB]["metering"].attribute_updates) == 1
        assert (
            listeners[CH_AB]["metering"].attribute_updates[0][0]
            == Metering.AttributeDefs.instantaneous_demand.id
        )
        assert (
            listeners[CH_AB]["metering"].attribute_updates[0][1] == 0
        )  # -POWER + POWER = 0
