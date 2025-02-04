"""Tuya Energy Meter."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final

from zigpy.quirks.v2.homeassistant import PERCENTAGE, EntityType, UnitOfTime
import zigpy.types as t
from zigpy.zcl import Cluster
from zigpy.zcl.clusters.homeautomation import MeasurementType
from zigpy.zcl.foundation import BaseAttributeDefs, ZCLAttributeDef

from zhaquirks import LocalDataCluster
from zhaquirks.tuya import (
    DPToAttributeMapping,
    PowerPhaseVariant2,
    PowerPhaseVariant3,
    TuyaLocalCluster,
    TuyaZBElectricalMeasurement,
    TuyaZBMeteringClusterWithUnit,
)
from zhaquirks.tuya.builder import TuyaQuirkBuilder
from zhaquirks.tuya.mcu import TuyaMCUCluster

ENERGY_DIRECTION: Final = "energy_direction"


class Channel(t.enum8):
    """Enum for meter channel endpoint_id."""

    A = 1
    B = 2
    C = 3

    Total = 10
    AB = 11
    ABC = 12

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
    __VIRTUAL_CHANNELS: set[Channel] = {AB, ABC}


class TuyaEnergyDirection(t.enum1):
    """Energy direction attribute type."""

    Forward = 0x0
    Reverse = 0x1


class EnergyDirectionMitigation(t.enum8):
    """Enum type for energy direction mitigation attribute."""

    Automatic = 0
    Disabled = 1
    Enabled = 2


class VirtualChannelConfig(t.enum8):
    """Enum type for virtual channel config attribute."""

    none = 0
    Total = 1
    A_minus_B = 2
    B_minus_A = 3


class EnergyMeterConfiguration(LocalDataCluster):
    """Local cluster for storing meter configuration."""

    cluster_id: Final[t.uint16_t] = 0xFC00
    name: Final = "Energy Meter Config"
    ep_attribute: Final = "energy_meter_config"

    VirtualChannelConfig: Final = VirtualChannelConfig
    EnergyDirectionMitigation: Final = EnergyDirectionMitigation

    _ATTRIBUTE_DEFAULTS: tuple[str, Any] = {
        "virtual_channel_config": VirtualChannelConfig.Total,
        "energy_direction_mitigation": EnergyDirectionMitigation.Automatic,
    }

    class AttributeDefs(BaseAttributeDefs):
        """Configuration attributes."""

        virtual_channel_config = ZCLAttributeDef(
            id=0x5000,
            type=VirtualChannelConfig,
            access="rw",
            is_manufacturer_specific=True,
        )
        energy_direction_mitigation = ZCLAttributeDef(
            id=0x5010,
            type=EnergyDirectionMitigation,
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
            None,
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

    def is_attr_type(self, attr_name: str, compare_type: type) -> bool:
        """Return True if the attribute type is a subclass of the compare type."""
        return issubclass(getattr(self.AttributeDefs, attr_name).type, compare_type)

    @property
    def mcu_cluster(self) -> TuyaMCUCluster | None:
        """Return the MCU cluster."""
        return getattr(
            self.endpoint.device.endpoints[1], TuyaMCUCluster.ep_attribute, None
        )

    @property
    def virtual(self) -> bool:
        """Return True if the cluster channel is virtual."""
        return self.channel in Channel.virtual_channels()


class EnergyDirectionHelper(MeterClusterHelper):
    """Apply Tuya EnergyDirection to ZCL power attributes."""

    UNSIGNED_ATTR_SUFFIX: Final = "_attr_unsigned"

    def __init__(self, *args, **kwargs):
        """Init."""
        self._energy_direction: TuyaEnergyDirection | None = None
        super().__init__(*args, **kwargs)

    def align_with_energy_direction(self, value: int | None) -> int | None:
        """Align the value with current energy_direction."""
        if value and (
            self.energy_direction == TuyaEnergyDirection.Reverse
            and value > 0
            or self.energy_direction == TuyaEnergyDirection.Forward
            and value < 0
        ):
            value = -value
        return value

    @property
    def energy_direction(self) -> TuyaEnergyDirection | None:
        """Return the channel energy direction."""
        if not self.mcu_cluster:
            return self._energy_direction
        try:
            return self.mcu_cluster.get(
                ENERGY_DIRECTION + Channel.attr_suffix(self.channel)
            )
        except KeyError:
            return self._energy_direction

    def energy_direction_handler(self, attr_name: str, value) -> tuple[str, Any]:
        """Unsigned device values are aligned with the energy direction."""
        if attr_name not in self._EXTENSIVE_ATTRIBUTES or not self.is_attr_type(
            attr_name, t.int_t
        ):
            return attr_name, value
        if attr_name.endswith(self.UNSIGNED_ATTR_SUFFIX):
            attr_name = attr_name.removesuffix(self.UNSIGNED_ATTR_SUFFIX)
            value = self.align_with_energy_direction(value)

        if value is not None:
            # _energy_direction used for virtual calculations on devices with native signed values
            self._energy_direction = (
                TuyaEnergyDirection.Reverse
                if value < 0
                else TuyaEnergyDirection.Forward
            )
        return attr_name, value


class EnergyDirectionMitigationHelper(EnergyDirectionHelper, MeterClusterHelper):
    """Logic compensating for delayed energy direction reporting.

    _TZE204_81yrt3lo (app_version: 74, hw_version: 1 and stack_version: 0) has a bug
    which results in it reporting energy_direction after its power data points.
    This means a change in direction would only be reported after the subsequent DP report,
    resulting in incorrect attribute signing in the ZCL clusters.

    This mitigation holds attribute update values until the subsequent energy_direction report,
    resulting in correct values, but a delay in attribute update equal to the update interval.
    """

    """Devices requiring energy direction mitigation."""
    _ENERGY_DIRECTION_MITIGATION_MATCHES: tuple[dict] = (
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
    def energy_direction_mitigation(self) -> bool:
        """Return the mitigation configuration."""
        return self.get_config(
            EnergyMeterConfiguration.AttributeDefs.energy_direction_mitigation.name
        )

    @property
    def energy_direction_mitigation_required(self) -> bool:
        """Return True if the device requires Energy direction mitigations."""
        if self._mitigation_required is None:
            self._mitigation_required = self._evaluate_device_mitigation()
        return self._mitigation_required

    def energy_direction_mitigation_handler(self, attr_name: str, value: Any) -> Any:
        """Hold the attribute value until the next update is received from the device."""
        if self.virtual or (
            self.energy_direction_mitigation
            not in (
                EnergyDirectionMitigation.Automatic,
                EnergyDirectionMitigation.Enabled,
            )
            or self.energy_direction_mitigation == EnergyDirectionMitigation.Automatic
            and not self.energy_direction_mitigation_required
        ):
            if attr_name in self._held_values:
                self._held_values.remove(attr_name)
            return value

        held_value = self._held_values.get(attr_name, None)
        self._held_values[attr_name] = value
        return held_value

    def _evaluate_device_mitigation(self) -> bool:
        """Return True if the device requires energy direction mitigation."""
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
        } in self._ENERGY_DIRECTION_MITIGATION_MATCHES


class VirtualChannelHelper(EnergyDirectionHelper, MeterClusterHelper):
    """Methods for calculating virtual energy meter channel attributes."""

    """Map of virtual channels to their trigger channel and calculation method."""
    _VIRTUAL_CHANNEL_CALCULATIONS: dict[
        tuple[Channel, VirtualChannelConfig],
        tuple[tuple[Channel], Callable, Channel | None],
    ] = {
        (Channel.AB, VirtualChannelConfig.Total): (
            (Channel.A, Channel.B),
            lambda a, b: a + b,
            Channel.B,
        ),
        (Channel.ABC, VirtualChannelConfig.Total): (
            (Channel.A, Channel.B, Channel.C),
            lambda a, b, c: a + b + c,
            Channel.C,
        ),
        (Channel.AB, VirtualChannelConfig.A_minus_B): (
            (Channel.A, Channel.B),
            lambda a, b: a - b,
            Channel.B,
        ),
        (Channel.AB, VirtualChannelConfig.B_minus_A): (
            (Channel.A, Channel.B),
            lambda a, b: b - a,
            Channel.B,
        ),
    }

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
            virtual_cluster = self.get_cluster(channel)
            if not virtual_cluster:
                continue
            source_channels, method, trigger_channel = (
                self._VIRTUAL_CHANNEL_CALCULATIONS.get(
                    (channel, self.virtual_channel_config), (None, None, None)
                )
            )
            if trigger_channel is not None and self.channel != trigger_channel:
                continue
            value = self._calculate_virtual_value(attr_name, source_channels, method)
            virtual_cluster.update_attribute(attr_name, value)

    def _calculate_virtual_value(
        self,
        attr_name: str,
        source_channels: tuple[Channel] | None,
        method: Callable | None,
    ) -> int | None:
        """Calculate virtual channel value from source channels."""
        if source_channels is None or method is None:
            return None
        source_values = self._get_source_values(attr_name, source_channels)
        if None in source_values:
            return None
        return method(*source_values)

    @property
    def _device_virtual_channels(self) -> set[Channel]:
        """Virtual channels present on the device."""
        return Channel.virtual_channels().intersection(
            self.endpoint.device.endpoints.keys()
        )

    def _get_source_values(
        self,
        attr_name: str,
        channels: tuple[Channel],
        align_uint_with_energy_direction: bool = True,
    ) -> tuple:
        """Get source values from channel clusters."""
        return tuple(
            cluster.align_with_energy_direction(cluster.get(attr_name))
            if align_uint_with_energy_direction
            and self.is_attr_type(attr_name, t.uint_t)
            else cluster.get(attr_name)
            for channel in channels
            for cluster in [self.get_cluster(channel)]
        )


class TuyaElectricalMeasurement(
    VirtualChannelHelper,
    EnergyDirectionMitigationHelper,
    EnergyDirectionHelper,
    MeterClusterHelper,
    TuyaLocalCluster,
    TuyaZBElectricalMeasurement,
):
    """ElectricalMeasurement cluster for Tuya energy meter devices.

    Attribute units prior to cluster formatting:
        Current: A * 1000 (milliampres)
        Frequency: Hz * 100
        Power: W * 10 (deciwatt)
        Voltage: V * 10 (decivolt)
    """

    AMPERE_MULTIPLIER = 1000
    HZ_MULTIPLIER = 100
    VOLT_MULTIPLIER = 10
    WATT_MULTIPLIER = 10

    _CONSTANT_ATTRIBUTES: dict[int, Any] = {
        **TuyaZBElectricalMeasurement._CONSTANT_ATTRIBUTES,  # imports current divisor of 1000
        TuyaZBElectricalMeasurement.AttributeDefs.ac_frequency_divisor.id: 100,  # 2 decimals
        TuyaZBElectricalMeasurement.AttributeDefs.ac_frequency_multiplier.id: 1,
        TuyaZBElectricalMeasurement.AttributeDefs.ac_power_divisor.id: 10,  # 1 decimal
        TuyaZBElectricalMeasurement.AttributeDefs.ac_power_multiplier.id: 1,
        TuyaZBElectricalMeasurement.AttributeDefs.ac_voltage_divisor.id: 10,  # 1 decimal
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
        value = self.energy_direction_mitigation_handler(attr_name, value)
        attr_name, value = self.energy_direction_handler(attr_name, value)
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
    EnergyDirectionMitigationHelper,
    EnergyDirectionHelper,
    MeterClusterHelper,
    TuyaLocalCluster,
    TuyaZBMeteringClusterWithUnit,
):
    """Metering cluster for Tuya energy meter devices."""

    WATT_MULTIPLIER = 10
    WATT_HOUR_MULTIPLIER = 1000
    DECIWATT_HOUR_MULTIPLIER = 100

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
        TuyaZBMeteringClusterWithUnit.AttributeDefs.divisor.id: 10000,  # 2 decimals for summation attributes
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
        value = self.energy_direction_mitigation_handler(attr_name, value)
        attr_name, value = self.energy_direction_handler(attr_name, value)
        super().update_attribute(attr_name, value)
        self.virtual_channel_handler(attr_name)


(
    ### Tuya PJ-MGW1203 1 channel energy meter.
    TuyaQuirkBuilder("_TZE204_cjbofhxw", "TS0601")
    .applies_to("_TZE284_cjbofhxw", "TS0601")
    .tuya_enchantment()
    .adds(EnergyMeterConfiguration)
    .adds(TuyaElectricalMeasurement)
    .adds(TuyaMetering)
    .tuya_dp(
        dp_id=101,  # Wh (watt/hour)
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_delivered.name,
        converter=lambda x: x * TuyaMetering.WATT_HOUR_MULTIPLIER,
    )
    .tuya_dp(
        dp_id=19,  # W * 10 (deciwatt)
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.instantaneous_demand.name,
    )
    .tuya_dp(
        dp_id=18,  # A * 1000 (milliampre)
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.rms_current.name,
        endpoint_id=Channel.B,
    )
    .tuya_dp(
        dp_id=20,  # V * 10 (decivolt)
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.rms_voltage.name,
    )
    .add_to_registry()
)


(
    ### Tuya bidirectional 1 channel energy meter.
    TuyaQuirkBuilder("_TZE204_ac0fhfiq", "TS0601")
    .tuya_enchantment()
    .adds(EnergyMeterConfiguration)
    .adds(TuyaElectricalMeasurement)
    .adds(TuyaMetering)
    .tuya_dp(
        dp_id=1,  # Wh * 10 (deciwatt/hour)
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_delivered.name,
        converter=lambda x: x * TuyaMetering.DECIWATT_HOUR_MULTIPLIER,
    )
    .tuya_dp(
        dp_id=2,  # Wh * 10 (deciwatt/hour)
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_received.name,
        converter=lambda x: x * TuyaMetering.DECIWATT_HOUR_MULTIPLIER,
    )
    .tuya_dp(
        dp_id=101,  # deciwatt?
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.instantaneous_demand.name
        + EnergyDirectionHelper.UNSIGNED_ATTR_SUFFIX,
    )
    .tuya_dp_multi(
        dp_id=6,
        attribute_mapping=[
            DPToAttributeMapping(
                ep_attribute=TuyaElectricalMeasurement.ep_attribute,
                attribute_name=TuyaElectricalMeasurement.AttributeDefs.rms_voltage.name,
                converter=PowerPhaseVariant3.voltage_dV,  # V * 10 (decivolt)
            ),
            DPToAttributeMapping(
                ep_attribute=TuyaElectricalMeasurement.ep_attribute,
                attribute_name=TuyaElectricalMeasurement.AttributeDefs.rms_current.name,
                converter=PowerPhaseVariant3.current_mA,  # A * 1000 (milliampre)
            ),
            DPToAttributeMapping(
                ep_attribute=TuyaElectricalMeasurement.ep_attribute,
                attribute_name=TuyaElectricalMeasurement.AttributeDefs.active_power.name
                + EnergyDirectionHelper.UNSIGNED_ATTR_SUFFIX,
                converter=lambda x: PowerPhaseVariant3.power_W(x)
                * TuyaElectricalMeasurement.WATT_MULTIPLIER,  # W (watt)
            ),
        ],
    )
    .tuya_dp_attribute(
        dp_id=102,  # 0=Forward/1=Reverse
        attribute_name=ENERGY_DIRECTION,
        type=TuyaEnergyDirection,
        converter=lambda x: TuyaEnergyDirection(x),
    )
    .add_to_registry()
)


(
    ### EARU Tuya 2 channel bidirectional energy meter.
    TuyaQuirkBuilder("_TZE200_rks0sgb7", "TS0601")
    .tuya_enchantment()
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
        fallback_name="Virtual channel",
    )
    .tuya_dp(
        dp_id=113,  # Hz
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.ac_frequency.name,
        converter=lambda x: x * TuyaElectricalMeasurement.HZ_MULTIPLIER,
    )
    .tuya_dp(
        dp_id=101,  # Wh * 10 (deciwatt/hour)
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_delivered.name,
        converter=lambda x: x * TuyaMetering.DECIWATT_HOUR_MULTIPLIER,
    )
    .tuya_dp(
        dp_id=103,  # Wh * 10 (deciwatt/hour)
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_delivered.name,
        converter=lambda x: x * TuyaMetering.DECIWATT_HOUR_MULTIPLIER,
        endpoint_id=Channel.B,
    )
    .tuya_dp(
        dp_id=102,  # Wh * 10 (deciwatt/hour)
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_received.name,
        converter=lambda x: x * TuyaMetering.DECIWATT_HOUR_MULTIPLIER,
    )
    .tuya_dp(
        dp_id=104,  # Wh * 10 (deciwatt/hour)
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_received.name,
        converter=lambda x: x * TuyaMetering.DECIWATT_HOUR_MULTIPLIER,
        endpoint_id=Channel.B,
    )
    .tuya_dp(
        dp_id=108,  # W (watt)
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.instantaneous_demand.name,
        converter=lambda x: x * TuyaMetering.WATT_MULTIPLIER,
    )
    .tuya_dp(
        dp_id=111,  # W (watt)
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.instantaneous_demand.name,
        converter=lambda x: x * TuyaMetering.WATT_MULTIPLIER,
        endpoint_id=Channel.B,
    )
    .tuya_dp(
        dp_id=109,  # % (power factor)
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.power_factor.name,
    )
    .tuya_dp(
        dp_id=112,  # % (power factor)
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.power_factor.name,
        endpoint_id=Channel.B,
    )
    .tuya_dp(
        dp_id=107,  # A * 1000 (milliampre)
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.rms_current.name,
    )
    .tuya_dp(
        dp_id=110,  # A * 1000 (milliampre)
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.rms_current.name,
        endpoint_id=Channel.B,
    )
    .tuya_dp(
        dp_id=106,  # V * 10 (decivolt)
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.rms_voltage.name,
    )
    .tuya_dp_attribute(
        dp_id=114,  # 0=Forward/1=Reverse
        attribute_name=ENERGY_DIRECTION,
        type=TuyaEnergyDirection,
        converter=lambda x: TuyaEnergyDirection(x),
    )
    .tuya_dp_attribute(
        dp_id=115,  # 0=Forward/1=Reverse
        attribute_name=ENERGY_DIRECTION + Channel.attr_suffix(Channel.B),
        type=TuyaEnergyDirection,
        converter=lambda x: TuyaEnergyDirection(x),
    )
    .tuya_number(
        dp_id=116,  # seconds
        attribute_name="reporting_interval",
        type=t.uint32_t_be,
        unit=UnitOfTime.SECONDS,
        min_value=5,
        max_value=60,
        step=1,
        translation_key="reporting_interval",
        fallback_name="Reporting interval",
        entity_type=EntityType.CONFIG,
    )
    .add_to_registry()
)


(
    ### MatSee Plus Tuya PJ-1203A 2 channel bidirectional energy meter.
    TuyaQuirkBuilder("_TZE204_81yrt3lo", "TS0601")
    .applies_to("_TZE284_81yrt3lo", "TS0601")
    .tuya_enchantment()
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
        fallback_name="Virtual channel",
    )
    .enum(
        EnergyMeterConfiguration.AttributeDefs.energy_direction_mitigation.name,
        EnergyDirectionMitigation,
        EnergyMeterConfiguration.cluster_id,
        entity_type=EntityType.CONFIG,
        translation_key="energy_direction_delay_mitigation",
        fallback_name="Energy direction delay mitigation",
    )
    .tuya_dp(
        dp_id=111,  # Hz * 100
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.ac_frequency.name,
    )
    .tuya_dp(
        dp_id=106,  # Wh * 10 (deciwatt/hour)
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_delivered.name,
        converter=lambda x: x * TuyaMetering.DECIWATT_HOUR_MULTIPLIER,
    )
    .tuya_dp(
        dp_id=108,  # Wh * 10 (deciwatt/hour)
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_delivered.name,
        converter=lambda x: x * TuyaMetering.DECIWATT_HOUR_MULTIPLIER,
        endpoint_id=Channel.B,
    )
    .tuya_dp(
        dp_id=107,  # Wh * 10 (deciwatt/hour)
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_received.name,
        converter=lambda x: x * TuyaMetering.DECIWATT_HOUR_MULTIPLIER,
    )
    .tuya_dp(
        dp_id=109,  # Wh * 10 (deciwatt/hour)
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_received.name,
        converter=lambda x: x * TuyaMetering.DECIWATT_HOUR_MULTIPLIER,
        endpoint_id=Channel.B,
    )
    .tuya_dp(
        dp_id=101,  # W * 10 (deciwatt)
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.instantaneous_demand.name
        + EnergyDirectionHelper.UNSIGNED_ATTR_SUFFIX,
    )
    .tuya_dp(
        dp_id=105,  # W * 10 (deciwatt)
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.instantaneous_demand.name
        + EnergyDirectionHelper.UNSIGNED_ATTR_SUFFIX,
        endpoint_id=Channel.B,
    )
    .tuya_dp(
        dp_id=110,  # % (power factor)
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.power_factor.name,
    )
    .tuya_dp(
        dp_id=121,  # % (power factor)
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.power_factor.name,
        endpoint_id=Channel.B,
    )
    .tuya_dp(
        dp_id=113,  # A * 1000 (milliampre)
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.rms_current.name,
    )
    .tuya_dp(
        dp_id=114,  # A * 1000 (milliampre)
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.rms_current.name,
        endpoint_id=Channel.B,
    )
    .tuya_dp(
        dp_id=112,  # V * 10 (decivolt)
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.rms_voltage.name,
    )
    .tuya_dp_attribute(
        dp_id=102,  # 0=Forward/1=Reverse
        attribute_name=ENERGY_DIRECTION,
        type=TuyaEnergyDirection,
        converter=lambda x: TuyaEnergyDirection(x),
    )
    .tuya_dp_attribute(
        dp_id=104,  # 0=Forward/1=Reverse
        attribute_name=ENERGY_DIRECTION + Channel.attr_suffix(Channel.B),
        type=TuyaEnergyDirection,
        converter=lambda x: TuyaEnergyDirection(x),
    )
    .tuya_number(
        dp_id=129,  # seconds
        attribute_name="reporting_interval",
        type=t.uint32_t_be,
        unit=UnitOfTime.SECONDS,
        min_value=5,
        max_value=60,
        step=1,
        translation_key="reporting_interval",
        fallback_name="Reporting interval",
        entity_type=EntityType.CONFIG,
    )
    .tuya_number(
        dp_id=122,  # % * 10 (1 decimal precision)
        attribute_name="ac_frequency_coefficient",
        type=t.uint32_t_be,
        unit=PERCENTAGE,
        min_value=0,
        max_value=2000,
        step=0.1,
        multiplier=0.1,
        translation_key="calibrate_ac_frequency",
        fallback_name="Calibrate AC frequency",
        entity_type=EntityType.CONFIG,
        initially_disabled=True,
    )
    .tuya_number(
        dp_id=119,  # % * 10 (1 decimal precision)
        attribute_name="current_summ_delivered_coefficient",
        type=t.uint32_t_be,
        unit=PERCENTAGE,
        min_value=0,
        max_value=2000,
        step=0.1,
        multiplier=0.1,
        translation_key="calibrate_summ_delivered",
        fallback_name="Calibrate summation delivered",
        entity_type=EntityType.CONFIG,
        initially_disabled=True,
    )
    .tuya_number(
        dp_id=125,  # % * 10 (1 decimal precision)
        attribute_name="current_summ_delivered_coefficient"
        + Channel.attr_suffix(Channel.B),
        type=t.uint32_t_be,
        unit=PERCENTAGE,
        min_value=0,
        max_value=2000,
        step=0.1,
        multiplier=0.1,
        translation_key="calibrate_summ_delivered_b",
        fallback_name="Calibrate summation delivered B",
        entity_type=EntityType.CONFIG,
        initially_disabled=True,
    )
    .tuya_number(
        dp_id=127,  # % * 10 (1 decimal precision)
        attribute_name="current_summ_received_coefficient",
        type=t.uint32_t_be,
        unit=PERCENTAGE,
        min_value=0,
        max_value=2000,
        step=0.1,
        multiplier=0.1,
        translation_key="calibrate_summ_received",
        fallback_name="Calibrate summation received",
        entity_type=EntityType.CONFIG,
        initially_disabled=True,
    )
    .tuya_number(
        dp_id=128,  # % * 10 (1 decimal precision)
        attribute_name="current_summ_received_coefficient"
        + Channel.attr_suffix(Channel.B),
        type=t.uint32_t_be,
        unit=PERCENTAGE,
        min_value=0,
        max_value=2000,
        step=0.1,
        multiplier=0.1,
        translation_key="calibrate_summ_received_b",
        fallback_name="Calibrate summation received B",
        entity_type=EntityType.CONFIG,
        initially_disabled=True,
    )
    .tuya_number(
        dp_id=118,  # % * 10 (1 decimal precision)
        attribute_name="instantaneous_demand_coefficient",
        type=t.uint32_t_be,
        unit=PERCENTAGE,
        min_value=0,
        max_value=2000,
        step=0.1,
        multiplier=0.1,
        translation_key="calibrate_instantaneous_demand",
        fallback_name="Calibrate instantaneous demand",
        entity_type=EntityType.CONFIG,
        initially_disabled=True,
    )
    .tuya_number(
        dp_id=124,  # % * 10 (1 decimal precision)
        attribute_name="instantaneous_demand_coefficient"
        + Channel.attr_suffix(Channel.B),
        type=t.uint32_t_be,
        unit=PERCENTAGE,
        min_value=0,
        max_value=2000,
        step=0.1,
        multiplier=0.1,
        translation_key="calibrate_instantaneous_demand_b",
        fallback_name="Calibrate instantaneous demand B",
        entity_type=EntityType.CONFIG,
        initially_disabled=True,
    )
    .tuya_number(
        dp_id=117,  # % * 10 (1 decimal precision)
        attribute_name="rms_current_coefficient",
        type=t.uint32_t_be,
        unit=PERCENTAGE,
        min_value=0,
        max_value=2000,
        step=0.1,
        multiplier=0.1,
        translation_key="calibrate_current",
        fallback_name="Calibrate current",
        entity_type=EntityType.CONFIG,
        initially_disabled=True,
    )
    .tuya_number(
        dp_id=123,  # % * 10 (1 decimal precision)
        attribute_name="rms_current_coefficient" + Channel.attr_suffix(Channel.B),
        type=t.uint32_t_be,
        unit=PERCENTAGE,
        min_value=0,
        max_value=2000,
        step=0.1,
        multiplier=0.1,
        translation_key="calibrate_current_b",
        fallback_name="Calibrate current B",
        entity_type=EntityType.CONFIG,
        initially_disabled=True,
    )
    .tuya_number(
        dp_id=116,  # % * 10 (1 decimal precision)
        attribute_name="rms_voltage_coefficient",
        type=t.uint32_t_be,
        unit=PERCENTAGE,
        min_value=0,
        max_value=2000,
        step=0.1,
        multiplier=0.1,
        translation_key="calibrate_voltage",
        fallback_name="Calibrate voltage",
        entity_type=EntityType.CONFIG,
        initially_disabled=True,
    )
    .add_to_registry()
)


(
    ### Tuya PC321-Z-TY 3 phase energy meter.
    TuyaQuirkBuilder("_TZE200_nslr42tt", "TS0601")
    .tuya_enchantment()
    .adds_endpoint(Channel.B)
    .adds_endpoint(Channel.C)
    .adds_endpoint(Channel.Total)
    .adds(EnergyMeterConfiguration)
    .adds(TuyaElectricalMeasurement)
    .adds(TuyaElectricalMeasurement, endpoint_id=Channel.Total)
    .adds(TuyaMetering)
    .adds(TuyaMetering, endpoint_id=Channel.B)
    .adds(TuyaMetering, endpoint_id=Channel.C)
    .adds(TuyaMetering, endpoint_id=Channel.Total)
    .tuya_temperature(dp_id=133, scale=10)
    .tuya_sensor(
        dp_id=134,
        attribute_name="device_status",
        type=t.int32s,
        translation_key="device_status",
        fallback_name="Device status",
    )
    .tuya_dp(
        dp_id=101,  # Wh (watt/hour)
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_delivered.name,
        converter=lambda x: x * TuyaMetering.WATT_HOUR_MULTIPLIER,
    )
    .tuya_dp(
        dp_id=111,  # Wh (watt/hour)
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_delivered.name,
        converter=lambda x: x * TuyaMetering.WATT_HOUR_MULTIPLIER,
        endpoint_id=Channel.B,
    )
    .tuya_dp(
        dp_id=121,  # Wh (watt/hour)
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_delivered.name,
        converter=lambda x: x * TuyaMetering.WATT_HOUR_MULTIPLIER,
        endpoint_id=Channel.C,
    )
    .tuya_dp(
        dp_id=1,  # Wh * 10 (deciwatt/hour)
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_delivered.name,
        converter=lambda x: x * TuyaMetering.DECIWATT_HOUR_MULTIPLIER,
        endpoint_id=Channel.Total,
    )
    .tuya_dp(
        dp_id=9,  # W (watt)
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.total_active_power.name,
        converter=lambda x: x * TuyaElectricalMeasurement.WATT_MULTIPLIER,
    )
    .tuya_dp(
        dp_id=131,  # A * 1000 (milliampre)
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.rms_current.name,
        endpoint_id=Channel.Total,
    )
    .tuya_dp_multi(
        dp_id=6,
        attribute_mapping=[
            DPToAttributeMapping(
                ep_attribute=TuyaElectricalMeasurement.ep_attribute,
                attribute_name=TuyaElectricalMeasurement.AttributeDefs.active_power.name,
                converter=lambda x: PowerPhaseVariant3.power_W(x)
                * TuyaElectricalMeasurement.WATT_MULTIPLIER,  # W (watt)
            ),
            DPToAttributeMapping(
                ep_attribute=TuyaElectricalMeasurement.ep_attribute,
                attribute_name=TuyaElectricalMeasurement.AttributeDefs.rms_voltage.name,
                converter=PowerPhaseVariant2.voltage_dV,  # V * 10 (decivolt)
            ),
            DPToAttributeMapping(
                ep_attribute=TuyaElectricalMeasurement.ep_attribute,
                attribute_name=TuyaElectricalMeasurement.AttributeDefs.rms_current.name,
                converter=PowerPhaseVariant2.current_mA,  # A * 1000 (milliampre)
            ),
        ],
    )
    .tuya_dp_multi(
        dp_id=7,
        attribute_mapping=[
            DPToAttributeMapping(
                ep_attribute=TuyaElectricalMeasurement.ep_attribute,
                attribute_name=TuyaElectricalMeasurement.AttributeDefs.active_power_ph_b.name,
                converter=lambda x: PowerPhaseVariant3.power_W(x)
                * TuyaElectricalMeasurement.WATT_MULTIPLIER,  # W (watt)
            ),
            DPToAttributeMapping(
                ep_attribute=TuyaElectricalMeasurement.ep_attribute,
                attribute_name=TuyaElectricalMeasurement.AttributeDefs.rms_voltage_ph_b.name,
                converter=PowerPhaseVariant2.voltage_dV,  # V * 10 (decivolt)
            ),
            DPToAttributeMapping(
                ep_attribute=TuyaElectricalMeasurement.ep_attribute,
                attribute_name=TuyaElectricalMeasurement.AttributeDefs.rms_current_ph_b.name,
                converter=PowerPhaseVariant2.current_mA,  # A * 1000 (milliampre)
            ),
        ],
    )
    .tuya_dp_multi(
        dp_id=7,
        attribute_mapping=[
            DPToAttributeMapping(
                ep_attribute=TuyaElectricalMeasurement.ep_attribute,
                attribute_name=TuyaElectricalMeasurement.AttributeDefs.active_power_ph_c.name,
                converter=lambda x: PowerPhaseVariant3.power_W(x)
                * TuyaElectricalMeasurement.WATT_MULTIPLIER,  # W (watt)
            ),
            DPToAttributeMapping(
                ep_attribute=TuyaElectricalMeasurement.ep_attribute,
                attribute_name=TuyaElectricalMeasurement.AttributeDefs.rms_voltage_ph_c.name,
                converter=PowerPhaseVariant2.voltage_dV,  # V * 10 (decivolt)
            ),
            DPToAttributeMapping(
                ep_attribute=TuyaElectricalMeasurement.ep_attribute,
                attribute_name=TuyaElectricalMeasurement.AttributeDefs.rms_current_ph_c.name,
                converter=PowerPhaseVariant2.current_mA,  # A * 1000 (milliampre)
            ),
        ],
    )
    .tuya_dp(
        dp_id=102,  # % (power factor)
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.power_factor.name,
    )
    .tuya_dp(
        dp_id=112,  # % (power factor)
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.power_factor_ph_b.name,
    )
    .tuya_dp(
        dp_id=122,  # % (power factor)
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.power_factor_ph_c.name,
    )
    .skip_configuration()
    .add_to_registry()
)


(
    ### Tuya PC321-Z-TY 3 phase energy meter.
    TuyaQuirkBuilder("_TZE204_v9hkz2yn", "TS0601")
    .applies_to("_TZE200_v9hkz2yn", "TS0601")
    .tuya_enchantment()
    .adds(EnergyMeterConfiguration)
    .adds(TuyaElectricalMeasurement)
    .adds(TuyaMetering)
    .tuya_dp(
        dp_id=1,  # Wh * 10 (deciwatt/hour)
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_delivered.name,
        converter=lambda x: x * TuyaMetering.DECIWATT_HOUR_MULTIPLIER,
    )
    .tuya_dp(
        dp_id=2,  # Wh * 10 (deciwatt/hour)
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_received.name,
        converter=lambda x: x * TuyaMetering.DECIWATT_HOUR_MULTIPLIER,
    )
    .tuya_dp(
        dp_id=15,  # % (power factor)
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.power_factor.name,
    )
    .tuya_dp(
        dp_id=101,  # Hz * 100
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.ac_frequency.name,
    )
    .tuya_dp(
        dp_id=102,  # V * 10 (decivolt)
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.ac_frequency.name,
    )
    .tuya_dp(
        dp_id=103,  # A * 1000 (milliampre)
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.rms_current.name,
    )
    .tuya_dp(
        dp_id=104,  # W (watt)
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.active_power.name,
        converter=lambda x: x * TuyaElectricalMeasurement.WATT_MULTIPLIER,
    )
    .tuya_dp(
        dp_id=105,  # V * 10 (decivolt)
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.ac_frequency_ph_b.name,
    )
    .tuya_dp(
        dp_id=106,  # A * 1000 (milliampre)
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.rms_current_ph_b.name,
    )
    .tuya_dp(
        dp_id=107,  # W (watt)
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.active_power_ph_b.name,
        converter=lambda x: x * TuyaElectricalMeasurement.WATT_MULTIPLIER,
    )
    .tuya_dp(
        dp_id=108,  # V * 10 (decivolt)
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.ac_frequency_ph_c.name,
    )
    .tuya_dp(
        dp_id=109,  # A * 1000 (milliampre)
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.rms_current_ph_c.name,
        endpoint_id=Channel.Total,
    )
    .tuya_dp(
        dp_id=110,  # W (watt)
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.active_power_ph_c.name,
        converter=lambda x: x * TuyaElectricalMeasurement.WATT_MULTIPLIER,
    )
    .tuya_dp(
        dp_id=111,  # W (watt)
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.total_active_power.name,
        converter=lambda x: x * TuyaElectricalMeasurement.WATT_MULTIPLIER,
    )
    .skip_configuration()
    .add_to_registry()
)
