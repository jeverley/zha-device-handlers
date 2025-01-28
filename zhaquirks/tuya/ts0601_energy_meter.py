"""Tuya Energy Meter."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final

from zigpy.quirks.v2.homeassistant import EntityType, UnitOfTime
import zigpy.types as t
from zigpy.zcl import Cluster
from zigpy.zcl.clusters.homeautomation import MeasurementType
from zigpy.zcl.foundation import BaseAttributeDefs, ZCLAttributeDef

from zhaquirks import LocalDataCluster
from zhaquirks.tuya import (
    TuyaLocalCluster,
    TuyaZBElectricalMeasurement,
    TuyaZBMeteringClusterWithUnit,
)
from zhaquirks.tuya.builder import TuyaQuirkBuilder
from zhaquirks.tuya.mcu import TuyaMCUCluster

POWER_FLOW: Final = "power_flow"


class Channel(t.enum8):
    """Enum for meter channel endpoint_id."""

    A = 1
    B = 2
    C = 3
    AB = 11

    @classmethod
    def attr_suffix(cls, channel: Channel | None) -> str:
        """Return the attribute suffix for a channel."""
        return cls.__ATTRIBUTE_SUFFIX.get(channel, "")

    @classmethod
    def virtual_channels(cls) -> set[Channel]:
        """Return set of virtual channels."""
        return cls.__VIRTUAL_CHANNELS

    __ATTRIBUTE_SUFFIX: dict[Channel, str] = {
        B: "_ch_b",
        C: "_ch_c",
    }
    __VIRTUAL_CHANNELS: set[Channel] = {AB}


class TuyaPowerFlow(t.enum1):
    """Power Flow attribute type."""

    Forward = 0x0
    Reverse = 0x1

    @classmethod
    def align_value(
        cls, value: int | None, power_flow: TuyaPowerFlow | None
    ) -> int | None:
        """Align the value with power_flow direction."""
        if value and (
            power_flow == cls.Reverse
            and value > 0
            or power_flow == cls.Forward
            and value < 0
        ):
            value = -value
        return value


class TuyaPowerPhase:
    """Methods for extracting values from a Tuya power phase datapoints."""

    @staticmethod
    def variant_1(value) -> tuple[t.uint_t, t.uint_t]:
        """Variant 1 of power phase Data Point."""
        voltage = value[14] | value[13] << 8
        current = value[12] | value[11] << 8
        return voltage, current

    @staticmethod
    def variant_2(value) -> tuple[t.uint_t, t.uint_t, int]:
        """Variant 2 of power phase Data Point."""
        voltage = value[1] | value[0] << 8
        current = value[4] | value[3] << 8
        power = value[7] | value[6] << 8
        return voltage, current, power * 10

    @staticmethod
    def variant_3(value) -> tuple[t.uint_t, t.uint_t, int]:
        """Variant 3 of power phase Data Point."""
        voltage = (value[0] << 8) | value[1]
        current = (value[2] << 16) | (value[3] << 8) | value[4]
        power = (value[5] << 16) | (value[6] << 8) | value[7]
        return voltage, current, power * 10


class PowerFlowMitigation(t.enum8):
    """Enum type for power flow mitigation attribute."""

    Automatic = 0
    Disabled = 1
    Enabled = 2


class VirtualChannelConfig(t.enum8):
    """Enum type for virtual channel config attribute."""

    none = 0
    A_plus_B = 1
    A_minus_B = 2
    B_minus_A = 3


class EnergyMeterConfiguration(LocalDataCluster):
    """Local cluster for storing meter configuration."""

    cluster_id: Final[t.uint16_t] = 0xFC00
    name: Final = "Energy Meter Config"
    ep_attribute: Final = "energy_meter_config"

    VirtualChannelConfig: Final = VirtualChannelConfig
    PowerFlowMitigation: Final = PowerFlowMitigation

    _ATTRIBUTE_DEFAULTS: tuple[str, Any] = {
        "virtual_channel_config": VirtualChannelConfig.none,
        "power_flow_mitigation": PowerFlowMitigation.Automatic,
    }

    class AttributeDefs(BaseAttributeDefs):
        """Configuration attributes."""

        virtual_channel_config = ZCLAttributeDef(
            id=0x5000,
            type=VirtualChannelConfig,
            access="rw",
            is_manufacturer_specific=True,
        )
        power_flow_mitigation = ZCLAttributeDef(
            id=0x5010,
            type=PowerFlowMitigation,
            access="rw",
            is_manufacturer_specific=True,
        )

    def get(self, key: int | str, default: Any | None = None) -> Any:
        """Attributes are updated with their default value on first access."""
        value = super().get(key, None)
        if value is not None:
            return value
        attr_def = self.find_attribute(key)
        attr_default = self._ATTRIBUTE_DEFAULTS.get(attr_def.name, None)
        if attr_default is None:
            return default
        self.update_attribute(attr_def.id, attr_default)
        return attr_default


class MeterClusterHelper:
    """Common methods for energy meter clusters."""

    _EXTENSIVE_ATTRIBUTES: tuple[str] = ()

    @property
    def channel(self) -> Channel | None:
        """Return the cluster channel."""
        try:
            return Channel(self.endpoint.endpoint_id)
        except ValueError:
            return None

    def get_cluster(
        self,
        endpoint_id: int,
        ep_attribute: str | None = None,
    ) -> Cluster:
        """Return the cluster for the given endpoint, default to current cluster type."""
        return getattr(
            self.endpoint.device.endpoints[endpoint_id],
            ep_attribute or self.ep_attribute,
        )

    def get_config(self, attr_name: str, default: Any = None) -> Any:
        """Return the config attribute's value."""
        cluster = getattr(
            self.endpoint.device.endpoints[1],
            EnergyMeterConfiguration.ep_attribute,
            None,
        )
        if not cluster:
            return None
        return cluster.get(attr_name, default)

    @property
    def mcu_cluster(self) -> TuyaMCUCluster | None:
        """Return the MCU cluster."""
        return getattr(
            self.endpoint.device.endpoints[1], TuyaMCUCluster.ep_attribute, None
        )


class PowerFlowHelper(MeterClusterHelper):
    """Apply Tuya power_flow to ZCL power attributes."""

    UNSIGNED_ATTR_SUFFIX: Final = "_attr_unsigned"

    @property
    def power_flow(self) -> TuyaPowerFlow | None:
        """Return the channel power flow direction."""
        if not self.mcu_cluster:
            return None
        try:
            return self.mcu_cluster.get(POWER_FLOW + Channel.attr_suffix(self.channel))
        except KeyError:
            return None

    @power_flow.setter
    def power_flow(self, value: TuyaPowerFlow):
        """Update the channel power flow direction."""
        if not self.mcu_cluster:
            return
        self.mcu_cluster.update_attribute(
            POWER_FLOW + Channel.attr_suffix(self.channel)
        )

    def power_flow_handler(self, attr_name: str, value) -> tuple[str, Any]:
        """Unsigned attributes are aligned with power flow direction."""
        if attr_name.endswith(self.UNSIGNED_ATTR_SUFFIX):
            attr_name = attr_name.removesuffix(self.UNSIGNED_ATTR_SUFFIX)
            value = TuyaPowerFlow.align_value(value, self.power_flow)
        return attr_name, value


class PowerFlowMitigationHelper(PowerFlowHelper, MeterClusterHelper):
    """Logic compensating for delayed power flow direction reporting.

    _TZE204_81yrt3lo (app_version: 74, hw_version: 1 and stack_version: 0) has a bug
    which results in it reporting power_flow after its power data points.
    This means a change in direction would only be reported after the subsequent DP report,
    resulting in incorrect attribute signing in the ZCL clusters.

    This mitigation holds attribute update values until the subsequent power_flow report,
    resulting in correct values, but a delay in attribute update equal to the update interval.
    """

    HOLD = "hold"
    RELEASE = "release"

    """Devices requiring power flow mitigation."""
    _POWER_FLOW_MITIGATION: tuple[dict] = (
        {
            "manufacturer": "_TZE204_81yrt3lo",
            "model": "TS0601",
            "basic_cluster": {
                "app_version": 74,
                "hw_version": 1,
                "stack_version": 0,
            },
        },
    )

    def __init__(self, *args, **kwargs):
        """Init."""
        self._held_values: dict[str, Any] = {}
        self._mitigation_required: bool | None = None
        super().__init__(*args, **kwargs)

    @property
    def power_flow_mitigation(self) -> bool:
        """Return the mitigation configuration."""
        return self.get_config(
            EnergyMeterConfiguration.AttributeDefs.power_flow_mitigation.name
        )

    @property
    def power_flow_mitigation_required(self) -> bool:
        """Return True if the device requires Power Flow mitigations."""
        if self._mitigation_required is None:
            self._mitigation_required = self._evaluate_device_mitigation()
        return self._mitigation_required

    def power_flow_mitigation_handler(self, attr_name: str, value) -> str | None:
        """Compensate for delay in reported power flow direction."""
        if (
            attr_name.removesuffix(self.UNSIGNED_ATTR_SUFFIX)
            not in self._EXTENSIVE_ATTRIBUTES
            or self.power_flow_mitigation
            not in (
                PowerFlowMitigation.Automatic,
                PowerFlowMitigation.Enabled,
            )
            or self.power_flow_mitigation == PowerFlowMitigation.Automatic
            and not self.power_flow_mitigation_required
        ):
            return None

        return self.RELEASE
        # action = self._mitigation_action(attr_name, value, trigger_channel)
        # if action != self.RELEASE:
        #    self._store_value(attr_name, value)
        # if action != self.PREEMPT:
        #    return action
        # self._release_held_values(attr_name, source_channels, trigger_channel)
        # return action

    def _mitigation_action(
        self, attr_name: str, value: int, trigger_channel: Channel
    ) -> str:
        """Return the action for the power flow mitigation handler."""
        return self.RELEASE

    def _get_held_value(self, attr_name: str) -> int | None:
        """Retrieve the held attribute value."""
        return self._held_values.get(attr_name, None)

    def _store_value(self, attr_name: str, value: int | None):
        """Store the update value."""
        self._held_values[attr_name] = value

    def _release_held_values(
        self, attr_name: str, source_channels: tuple[Channel], trigger_channel: Channel
    ):
        """Release held values to update the cluster attributes."""
        for channel in source_channels:
            cluster = self.get_cluster(channel)
            if channel != trigger_channel:
                value = cluster._get_held_value(attr_name)
                if value is not None:
                    cluster.update_attribute(attr_name, value)
            cluster._store_value(attr_name, None)

    def _evaluate_device_mitigation(self) -> bool:
        """Return True if the device requires Power Flow mitigation."""
        basic_cluster = self.endpoint.device.endpoints[1].basic
        return {
            "manufacturer": self.endpoint.device.manufacturer,
            "model": self.endpoint.device.model,
            "basic_cluster": {
                "app_version": basic_cluster.get(
                    basic_cluster.AttributeDefs.app_version.name
                ),
                "hw_version": basic_cluster.get(
                    basic_cluster.AttributeDefs.hw_version.name
                ),
                "stack_version": basic_cluster.get(
                    basic_cluster.AttributeDefs.stack_version.name
                ),
            },
        } in self._POWER_FLOW_MITIGATION


class VirtualChannelHelper(PowerFlowHelper, MeterClusterHelper):
    """Methods for calculating virtual energy meter channel attributes."""

    @property
    def virtual(self) -> bool:
        """Return True if the cluster channel is virtual."""
        return self.channel in Channel.virtual_channels()

    @property
    def virtual_channel_config(self) -> VirtualChannelConfig | None:
        """Return the virtual channel configuration."""
        return self.get_config(
            EnergyMeterConfiguration.AttributeDefs.virtual_channel_config.name
        )

    def virtual_channel_handler(self, attr_name: str):
        """Handle updates to virtual energy meter channels."""

        if self.virtual or attr_name not in self._EXTENSIVE_ATTRIBUTES:
            return
        for channel in self._device_virtual_channels:
            trigger_channel, method = self._VIRTUAL_CHANNEL_CONFIGURATION.get(
                (channel, self.virtual_channel_config), None
            )
            if self.channel != trigger_channel:
                continue
            value = method(self, attr_name) if method else None
            virtual_cluster = self.get_cluster(channel)
            virtual_cluster.update_attribute(attr_name, value)

    def _is_attr_uint(self, attr_name: str) -> bool:
        """Return True if the attribute type is an unsigned integer."""
        return issubclass(getattr(self.AttributeDefs, attr_name).type, t.uint_t)

    def _retrieve_source_values(
        self, attr_name: str, channels: tuple[Channel]
    ) -> tuple:
        """Retrieve source values from channel clusters."""
        return tuple(
            TuyaPowerFlow.align_value(cluster.get(attr_name), cluster.power_flow)
            if attr_name in self._EXTENSIVE_ATTRIBUTES and self._is_attr_uint(attr_name)
            else cluster.get(attr_name)
            for channel in channels
            for cluster in [self.get_cluster(channel)]
        )

    @property
    def _device_virtual_channels(self) -> set[Channel]:
        """Virtual channels present on the device."""
        return Channel.virtual_channels().intersection(
            self.endpoint.device.endpoints.keys()
        )

    def _virtual_a_plus_b(self, attr_name: str) -> int | None:
        """Calculate virtual channel value for A_plus_B configuration."""
        value_a, value_b = self._retrieve_source_values(
            attr_name, (Channel.A, Channel.B)
        )
        if None in (value_a, value_b):
            return None
        return value_a + value_b

    def _virtual_a_minus_b(self, attr_name: str) -> int | None:
        """Calculate virtual channel value for A_minus_B configuration."""
        value_a, value_b = self._retrieve_source_values(
            attr_name, (Channel.A, Channel.B)
        )
        if None in (value_a, value_b):
            return None
        return value_a - value_b

    def _virtual_b_minus_a(self, attr_name: str) -> int | None:
        """Calculate virtual channel value for A_minus_B configuration."""
        value_a, value_b = self._retrieve_source_values(
            attr_name, (Channel.A, Channel.B)
        )
        if None in (value_a, value_b):
            return None
        return value_b - value_a

    """Map of virtual channels to their trigger channel and calculation method."""
    _VIRTUAL_CHANNEL_CONFIGURATION: dict[
        tuple[Channel, VirtualChannelConfig | None],
        tuple[Channel, Callable | None],
    ] = {
        (Channel.AB, VirtualChannelConfig.A_plus_B): (
            Channel.B,
            _virtual_a_plus_b,
        ),
        (Channel.AB, VirtualChannelConfig.A_minus_B): (
            Channel.B,
            _virtual_a_minus_b,
        ),
        (Channel.AB, VirtualChannelConfig.B_minus_A): (
            Channel.B,
            _virtual_b_minus_a,
        ),
        (Channel.AB, VirtualChannelConfig.none): (Channel.B, None),
    }


class TuyaElectricalMeasurement(
    VirtualChannelHelper,
    PowerFlowMitigationHelper,
    PowerFlowHelper,
    MeterClusterHelper,
    TuyaLocalCluster,
    TuyaZBElectricalMeasurement,
):
    """ElectricalMeasurement cluster for Tuya energy meter devices."""

    _CONSTANT_ATTRIBUTES: dict[int, Any] = {
        **TuyaZBElectricalMeasurement._CONSTANT_ATTRIBUTES,
        TuyaZBElectricalMeasurement.AttributeDefs.ac_frequency_divisor.id: 100,
        TuyaZBElectricalMeasurement.AttributeDefs.ac_frequency_multiplier.id: 1,
        TuyaZBElectricalMeasurement.AttributeDefs.ac_power_divisor.id: 10,
        TuyaZBElectricalMeasurement.AttributeDefs.ac_power_multiplier.id: 1,
        TuyaZBElectricalMeasurement.AttributeDefs.ac_voltage_divisor.id: 10,
        TuyaZBElectricalMeasurement.AttributeDefs.ac_voltage_multiplier.id: 1,
    }

    _ATTRIBUTE_MEASUREMENT_TYPES: dict[str, MeasurementType] = {
        TuyaZBElectricalMeasurement.AttributeDefs.active_power.name: MeasurementType.Active_measurement_AC
        | MeasurementType.Phase_A_measurement,
        TuyaZBElectricalMeasurement.AttributeDefs.active_power_ph_b.name: MeasurementType.Active_measurement_AC
        | MeasurementType.Phase_B_measurement,
        TuyaZBElectricalMeasurement.AttributeDefs.active_power_ph_c.name: MeasurementType.Active_measurement_AC
        | MeasurementType.Phase_C_measurement,
        TuyaZBElectricalMeasurement.AttributeDefs.reactive_power.name: MeasurementType.Reactive_measurement_AC
        | MeasurementType.Phase_A_measurement,
        TuyaZBElectricalMeasurement.AttributeDefs.reactive_power_ph_b.name: MeasurementType.Reactive_measurement_AC
        | MeasurementType.Phase_B_measurement,
        TuyaZBElectricalMeasurement.AttributeDefs.reactive_power_ph_c.name: MeasurementType.Reactive_measurement_AC
        | MeasurementType.Phase_C_measurement,
        TuyaZBElectricalMeasurement.AttributeDefs.apparent_power.name: MeasurementType.Apparent_measurement_AC
        | MeasurementType.Phase_A_measurement,
        TuyaZBElectricalMeasurement.AttributeDefs.apparent_power_ph_b.name: MeasurementType.Apparent_measurement_AC
        | MeasurementType.Phase_B_measurement,
        TuyaZBElectricalMeasurement.AttributeDefs.apparent_power_ph_c.name: MeasurementType.Apparent_measurement_AC
        | MeasurementType.Phase_C_measurement,
    }

    _EXTENSIVE_ATTRIBUTES: tuple[str] = (
        TuyaZBElectricalMeasurement.AttributeDefs.active_power.name,
        TuyaZBElectricalMeasurement.AttributeDefs.active_power_ph_b.name,
        TuyaZBElectricalMeasurement.AttributeDefs.active_power_ph_c.name,
        TuyaZBElectricalMeasurement.AttributeDefs.apparent_power.name,
        TuyaZBElectricalMeasurement.AttributeDefs.apparent_power_ph_b.name,
        TuyaZBElectricalMeasurement.AttributeDefs.apparent_power_ph_c.name,
        TuyaZBElectricalMeasurement.AttributeDefs.reactive_power.name,
        TuyaZBElectricalMeasurement.AttributeDefs.reactive_power_ph_b.name,
        TuyaZBElectricalMeasurement.AttributeDefs.reactive_power_ph_c.name,
        TuyaZBElectricalMeasurement.AttributeDefs.rms_current.name,
        TuyaZBElectricalMeasurement.AttributeDefs.rms_current_ph_b.name,
        TuyaZBElectricalMeasurement.AttributeDefs.rms_current_ph_c.name,
    )

    def update_attribute(self, attr_name: str, value):
        """Update the cluster attribute."""
        if (
            self.power_flow_mitigation_handler(attr_name, value)
            == PowerFlowMitigationHelper.HOLD
        ):
            return
        attr_name, value = self.power_flow_handler(attr_name, value)
        super().update_attribute(attr_name, value)
        self._update_measurement_type(attr_name)
        self.virtual_channel_handler(attr_name)

    def _update_measurement_type(self, attr_name: str):
        """Derive the measurement_type from reported attributes."""
        if attr_name not in self._ATTRIBUTE_MEASUREMENT_TYPES:
            return
        measurement_type = 0
        for measurement, mask in self._ATTRIBUTE_MEASUREMENT_TYPES.items():
            if measurement == attr_name or self.get(measurement) is not None:
                measurement_type |= mask
        super().update_attribute(
            self.AttributeDefs.measurement_type.name, measurement_type
        )


class TuyaMetering(
    VirtualChannelHelper,
    PowerFlowMitigationHelper,
    PowerFlowHelper,
    MeterClusterHelper,
    TuyaLocalCluster,
    TuyaZBMeteringClusterWithUnit,
):
    """Metering cluster for Tuya energy meter devices."""

    @staticmethod
    def format(
        whole_digits: int, dec_digits: int, suppress_leading_zeros: bool = True
    ) -> int:
        """Return the formatter value for summation and demand Metering attributes."""
        assert 0 <= whole_digits <= 7, "must be within range of 0 to 7."
        assert 0 <= dec_digits <= 7, "must be within range of 0 to 7."
        return (suppress_leading_zeros << 6) | (whole_digits << 3) | dec_digits

    _CONSTANT_ATTRIBUTES: dict[int, Any] = {
        **TuyaZBMeteringClusterWithUnit._CONSTANT_ATTRIBUTES,
        TuyaZBMeteringClusterWithUnit.AttributeDefs.status.id: 0x00,
        TuyaZBMeteringClusterWithUnit.AttributeDefs.multiplier.id: 1,
        TuyaZBMeteringClusterWithUnit.AttributeDefs.divisor.id: 10000,  # 1 decimal place after conversion from kW to W
        TuyaZBMeteringClusterWithUnit.AttributeDefs.summation_formatting.id: format(
            whole_digits=7, dec_digits=2
        ),
        TuyaZBMeteringClusterWithUnit.AttributeDefs.demand_formatting.id: format(
            whole_digits=7, dec_digits=1
        ),
    }

    _EXTENSIVE_ATTRIBUTES: tuple[str] = (
        TuyaZBMeteringClusterWithUnit.AttributeDefs.instantaneous_demand.name,
    )

    def update_attribute(self, attr_name: str, value):
        """Update the cluster attribute."""
        if (
            self.power_flow_mitigation_handler(attr_name, value)
            == PowerFlowMitigationHelper.HOLD
        ):
            return
        attr_name, value = self.power_flow_handler(attr_name, value)
        super().update_attribute(attr_name, value)
        self.virtual_channel_handler(attr_name)


(
    ### Tuya PJ-MGW1203 1 channel energy meter.
    TuyaQuirkBuilder("_TZE204_cjbofhxw", "TS0601")
    # .tuya_enchantment()
    .adds(EnergyMeterConfiguration)
    .adds(TuyaElectricalMeasurement)
    .adds(TuyaMetering)
    .tuya_dp(
        dp_id=101,
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_delivered.name,
        converter=lambda x: x * 10,
    )
    .tuya_dp(
        dp_id=19,
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.active_power.name,
    )
    .tuya_dp(
        dp_id=18,
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.rms_current.name,
        endpoint_id=Channel.B,
    )
    .tuya_dp(
        dp_id=20,
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.rms_voltage.name,
    )
    .add_to_registry()
)


(
    ### Tuya bidirectional 1 channel energy meter with Zigbee Green Power.
    TuyaQuirkBuilder("_TZE204_ac0fhfiq", "TS0601")
    # .tuya_enchantment()
    .adds(EnergyMeterConfiguration)
    .adds(TuyaElectricalMeasurement)
    .adds(TuyaMetering)
    .tuya_dp_attribute(
        dp_id=102,
        attribute_name=POWER_FLOW,
        type=TuyaPowerFlow,
        converter=lambda x: TuyaPowerFlow(x),
    )
    .tuya_dp(
        dp_id=101,
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_delivered.name,
        converter=lambda x: x * 100,
    )
    .tuya_dp(
        dp_id=102,
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_received.name,
        converter=lambda x: x * 100,
    )
    .tuya_dp(
        dp_id=108,
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.instantaneous_demand.name
        + PowerFlowHelper.UNSIGNED_ATTR_SUFFIX,
        converter=lambda x: x * 10,
    )
    .tuya_dp(
        dp_id=6,
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=(
            TuyaElectricalMeasurement.AttributeDefs.rms_voltage.name,
            TuyaElectricalMeasurement.AttributeDefs.rms_current.name,
            TuyaElectricalMeasurement.AttributeDefs.active_power.name
            + PowerFlowHelper.UNSIGNED_ATTR_SUFFIX,
        ),
        converter=lambda x: TuyaPowerPhase.variant_3(x),
    )
    .add_to_registry()
)


(
    ### EARU Tuya 2 channel bidirectional energy meter manufacturer cluster.
    TuyaQuirkBuilder("_TZE200_rks0sgb7", "TS0601")
    # .tuya_enchantment()
    .adds_endpoint(Channel.B)
    .adds_endpoint(Channel.AB)
    .adds(EnergyMeterConfiguration)
    .adds(TuyaElectricalMeasurement)
    .adds(TuyaElectricalMeasurement, endpoint_id=Channel.B)
    .adds(TuyaElectricalMeasurement, endpoint_id=Channel.AB)
    .adds(TuyaMetering)
    .adds(TuyaMetering, endpoint_id=Channel.B)
    .adds(TuyaMetering, endpoint_id=Channel.AB)
    .enum(
        EnergyMeterConfiguration.AttributeDefs.virtual_channel_config.name,
        VirtualChannelConfig,
        EnergyMeterConfiguration.cluster_id,
        entity_type=EntityType.CONFIG,
        translation_key="virtual_channel_config",
        fallback_name="Virtual Channel",
    )
    .tuya_dp_attribute(
        dp_id=114,
        attribute_name=POWER_FLOW,
        type=TuyaPowerFlow,
        converter=lambda x: TuyaPowerFlow(x),
    )
    .tuya_dp_attribute(
        dp_id=115,
        attribute_name=POWER_FLOW + Channel.attr_suffix(Channel.B),
        type=TuyaPowerFlow,
        converter=lambda x: TuyaPowerFlow(x),
    )
    .tuya_dp(
        dp_id=113,
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.ac_frequency.name,
    )
    .tuya_dp(
        dp_id=101,
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_delivered.name,
        converter=lambda x: x * 100,
    )
    .tuya_dp(
        dp_id=103,
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_delivered.name,
        converter=lambda x: x * 100,
        endpoint_id=Channel.B,
    )
    .tuya_dp(
        dp_id=102,
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_received.name,
        converter=lambda x: x * 100,
    )
    .tuya_dp(
        dp_id=104,
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_received.name,
        converter=lambda x: x * 100,
        endpoint_id=Channel.B,
    )
    .tuya_dp(
        dp_id=108,
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.instantaneous_demand.name,
    )
    .tuya_dp(
        dp_id=111,
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.instantaneous_demand.name,
        endpoint_id=Channel.B,
    )
    .tuya_dp(
        dp_id=109,
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.power_factor.name,
    )
    .tuya_dp(
        dp_id=112,
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.power_factor.name,
        endpoint_id=Channel.B,
    )
    .tuya_dp(
        dp_id=107,
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.rms_current.name,
    )
    .tuya_dp(
        dp_id=110,
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.rms_current.name,
        endpoint_id=Channel.B,
    )
    .tuya_dp(
        dp_id=106,
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.rms_voltage.name,
    )
    .tuya_number(
        dp_id=116,
        attribute_name="update_interval",
        type=t.uint32_t_be,
        unit=UnitOfTime.SECONDS,
        min_value=5,
        max_value=60,
        step=1,
        translation_key="update_interval",
        fallback_name="Update Interval",
        entity_type=EntityType.CONFIG,
    )
    .add_to_registry()
)


(
    ### MatSee Plus Tuya PJ-1203A 2 channel bidirectional energy meter with Zigbee Green Power.
    TuyaQuirkBuilder("_TZE204_81yrt3lo", "TS0601")
    # .tuya_enchantment()
    .adds_endpoint(Channel.B)
    .adds_endpoint(Channel.AB)
    .adds(EnergyMeterConfiguration)
    .adds(TuyaElectricalMeasurement)
    .adds(TuyaElectricalMeasurement, endpoint_id=Channel.B)
    .adds(TuyaElectricalMeasurement, endpoint_id=Channel.AB)
    .adds(TuyaMetering)
    .adds(TuyaMetering, endpoint_id=Channel.B)
    .adds(TuyaMetering, endpoint_id=Channel.AB)
    .enum(
        EnergyMeterConfiguration.AttributeDefs.virtual_channel_config.name,
        VirtualChannelConfig,
        EnergyMeterConfiguration.cluster_id,
        entity_type=EntityType.CONFIG,
        translation_key="virtual_channel_config",
        fallback_name="Virtual Channel",
    )
    .enum(
        EnergyMeterConfiguration.AttributeDefs.power_flow_mitigation.name,
        PowerFlowMitigation,
        EnergyMeterConfiguration.cluster_id,
        entity_type=EntityType.CONFIG,
        translation_key="power_flow_mitigation",
        fallback_name="Power Flow delay mitigation",
    )
    .tuya_dp_attribute(
        dp_id=102,
        attribute_name=POWER_FLOW,
        type=TuyaPowerFlow,
        converter=lambda x: TuyaPowerFlow(x),
    )
    .tuya_dp_attribute(
        dp_id=104,
        attribute_name=POWER_FLOW + Channel.attr_suffix(Channel.B),
        type=TuyaPowerFlow,
        converter=lambda x: TuyaPowerFlow(x),
    )
    .tuya_dp(
        dp_id=111,
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.ac_frequency.name,
    )
    .tuya_dp(
        dp_id=106,
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_delivered.name,
        converter=lambda x: x * 100,
    )
    .tuya_dp(
        dp_id=108,
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_delivered.name,
        converter=lambda x: x * 100,
        endpoint_id=Channel.B,
    )
    .tuya_dp(
        dp_id=107,
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_received.name,
        converter=lambda x: x * 100,
    )
    .tuya_dp(
        dp_id=109,
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_received.name,
        converter=lambda x: x * 100,
        endpoint_id=Channel.B,
    )
    .tuya_dp(
        dp_id=101,
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.instantaneous_demand.name
        + PowerFlowHelper.UNSIGNED_ATTR_SUFFIX,
    )
    .tuya_dp(
        dp_id=105,
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.instantaneous_demand.name
        + PowerFlowHelper.UNSIGNED_ATTR_SUFFIX,
        endpoint_id=Channel.B,
    )
    .tuya_dp(
        dp_id=110,
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.power_factor.name,
    )
    .tuya_dp(
        dp_id=121,
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.power_factor.name,
        endpoint_id=Channel.B,
    )
    .tuya_dp(
        dp_id=113,
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.rms_current.name,
    )
    .tuya_dp(
        dp_id=114,
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.rms_current.name,
        endpoint_id=Channel.B,
    )
    .tuya_dp(
        dp_id=112,
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.rms_voltage.name,
    )
    .tuya_number(
        dp_id=129,
        attribute_name="update_interval",
        type=t.uint32_t_be,
        unit=UnitOfTime.SECONDS,
        min_value=5,
        max_value=60,
        step=1,
        translation_key="update_interval",
        fallback_name="Update Interval",
        entity_type=EntityType.CONFIG,
    )
    # .tuya_number(
    #    dp_id=122,
    #    attribute_name="ac_frequency_coefficient",
    #    type=t.uint32_t_be,
    #    # unit=PERCENTAGE,
    #    min_value=500,
    #    max_value=1500,
    #    step=1,
    #    multiplier=0.1,
    #    translation_key="ac_frequency_calibration",
    #    fallback_name="AC Frequency Calibration",
    #    entity_type=EntityType.CONFIG,
    # )
    # .tuya_number(
    #    dp_id=119,
    #    attribute_name="current_summ_delivered_coefficient",
    #    type=t.uint32_t_be,
    #    # unit=PERCENTAGE,
    #    min_value=500,
    #    max_value=1500,
    #    step=1,
    #    multiplier=0.1,
    #    translation_key="summ_delivered_calibration",
    #    fallback_name="Summation Delivered Calibration",
    #    entity_type=EntityType.CONFIG,
    # )
    # .tuya_number(
    #    dp_id=125,
    #    attribute_name="current_summ_delivered_coefficient_ch_b",
    #    type=t.uint32_t_be,
    #    # unit=PERCENTAGE,
    #    min_value=500,
    #    max_value=1500,
    #    step=1,
    #    multiplier=0.1,
    #    translation_key="summ_delivered_calibration_b",
    #    fallback_name="Summation Delivered Calibration B",
    #    entity_type=EntityType.CONFIG,
    # )
    # .tuya_number(
    #    dp_id=127,
    #    attribute_name="current_summ_received_coefficient",
    #    type=t.uint32_t_be,
    #    # unit=PERCENTAGE,
    #    min_value=500,
    #    max_value=1500,
    #    step=1,
    #    multiplier=0.1,
    #    translation_key="summ_received_calibration",
    #    fallback_name="Summation Received Calibration",
    #    entity_type=EntityType.CONFIG,
    # )
    # .tuya_number(
    #    dp_id=128,
    #    attribute_name="current_summ_received_coefficient_ch_b",
    #    type=t.uint32_t_be,
    #    # unit=PERCENTAGE,
    #    min_value=500,
    #    max_value=1500,
    #    step=1,
    #    multiplier=0.1,
    #    translation_key="summ_received_calibration_b",
    #    fallback_name="Summation Received Calibration B",
    #    entity_type=EntityType.CONFIG,
    # )
    # .tuya_number(
    #    dp_id=118,
    #    attribute_name="instantaneous_demand_coefficient",
    #    type=t.uint32_t_be,
    #    # unit=PERCENTAGE,
    #    min_value=500,
    #    max_value=1500,
    #    step=1,
    #    multiplier=0.1,
    #    translation_key="instantaneous_demand_calibration",
    #    fallback_name="Instantaneous Demand Calibration",
    #    entity_type=EntityType.CONFIG,
    # )
    # .tuya_number(
    #    dp_id=124,
    #    attribute_name="instantaneous_demand_coefficient_ch_b",
    #    type=t.uint32_t_be,
    #    # unit=PERCENTAGE,
    #    min_value=500,
    #    max_value=1500,
    #    step=1,
    #    multiplier=0.1,
    #    translation_key="instantaneous_demand_calibration_b",
    #    fallback_name="Instantaneous Demand Calibration B",
    #    entity_type=EntityType.CONFIG,
    # )
    # .tuya_number(
    #    dp_id=117,
    #    attribute_name="rms_current_coefficient",
    #    type=t.uint32_t_be,
    #    # unit=PERCENTAGE,
    #    min_value=500,
    #    max_value=1500,
    #    step=1,
    #    multiplier=0.1,
    #    translation_key="rms_current_calibration",
    #    fallback_name="RMS Current Calibration",
    #    entity_type=EntityType.CONFIG,
    # )
    # .tuya_number(
    #    dp_id=123,
    #    attribute_name="rms_current_coefficient_ch_b",
    #    type=t.uint32_t_be,
    #    # unit=PERCENTAGE,
    #    min_value=500,
    #    max_value=1500,
    #    step=1,
    #    multiplier=0.1,
    #    translation_key="rms_current_calibration_b",
    #    fallback_name="RMS Current Calibration B",
    #    entity_type=EntityType.CONFIG,
    # )
    # .tuya_number(
    #    dp_id=116,
    #    attribute_name="rms_voltage_coefficient",
    #    type=t.uint32_t_be,
    #    # unit=PERCENTAGE,
    #    min_value=500,
    #    max_value=1500,
    #    step=1,
    #    multiplier=0.1,
    #    translation_key="rms_voltage_calibration",
    #    fallback_name="RMS Voltage Calibration",
    #    entity_type=EntityType.CONFIG,
    # )
    .add_to_registry()
)
