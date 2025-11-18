"""Tuya MatSeePlus 2 CT Bidirectional Energy Meter."""

from __future__ import annotations

from typing import Any, Final

from zigpy.quirks.v2.homeassistant import PERCENTAGE, EntityType, UnitOfTime
import zigpy.types as t
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

ENDPOINT_ID_CT_A = 1
ENDPOINT_ID_CT_B = 2
ENDPOINT_ID_TOTAL = 3


class TuyaEnergyFlow(t.enum8):
    """Energy flow direction attribute values."""

    Forward = 0x0
    Reverse = 0x1


class MatSeePlusLocalConfig(LocalDataCluster):
    """Cluster for storing local configuration.

    Allows control over the delayed energy flow bug mitigation.
    """

    cluster_id: Final[t.uint16_t] = 0xFC00
    name: Final = "Local Configuration"
    ep_attribute: Final = "local_config"

    class AttributeDefs(BaseAttributeDefs):
        """Configuration attributes."""

        late_energy_flow_a = ZCLAttributeDef(
            id=0x5010,
            type=t.Bool,
            access="rw",
            is_manufacturer_specific=True,
        )
        late_energy_flow_b = ZCLAttributeDef(
            id=0x5011,
            type=t.Bool,
            access="rw",
            is_manufacturer_specific=True,
        )


class MatSeePlusElectricalMeasurement(TuyaZBElectricalMeasurement, TuyaLocalCluster):
    """ElectricalMeasurement cluster for MatSeePlus CT energy meter with flow delay mitigation.

    _TZE204_81yrt3lo (app_version: 74, hw_version: 1 and stack_version: 0) has a bug
    where the current energy flow values are incorrectly emitted during the next reporting interval.
    This means a change in direction result in incorrect power values.

    The bug has remained unfixed for multiple years, with no provided firmware updates from the manufacturer.
    When enabled this mitigation holds non-power attribute values until the subsequent interval's attribute report.
    This ensures correct values, but introduces a delay in entity updates.

    This is optional and defaults to off because some use cases only have energy flowing in a single direction.
    """

    # Maps endpoint IDs to their corresponding late energy flow configuration attribute names
    _EP_MITIGATION_CONFIG_ATTR: dict[int, str] = {
        ENDPOINT_ID_CT_A: MatSeePlusLocalConfig.AttributeDefs.late_energy_flow_a.name,
        ENDPOINT_ID_CT_B: MatSeePlusLocalConfig.AttributeDefs.late_energy_flow_b.name,
    }

    _CONSTANT_ATTRIBUTES: dict[int, Any] = {
        **TuyaZBElectricalMeasurement._CONSTANT_ATTRIBUTES,
        TuyaZBElectricalMeasurement.AttributeDefs.ac_frequency_divisor.id: 100,
        TuyaZBElectricalMeasurement.AttributeDefs.ac_frequency_multiplier.id: 1,
        TuyaZBElectricalMeasurement.AttributeDefs.ac_power_divisor.id: 10,
        TuyaZBElectricalMeasurement.AttributeDefs.ac_power_multiplier.id: 1,
        TuyaZBElectricalMeasurement.AttributeDefs.ac_voltage_divisor.id: 10,
        TuyaZBElectricalMeasurement.AttributeDefs.ac_voltage_multiplier.id: 1,
        TuyaZBElectricalMeasurement.AttributeDefs.measurement_type.id: MeasurementType.Active_measurement_AC
        | MeasurementType.Phase_A_measurement,
    }

    _VALID_ATTRIBUTES: set[int] = {
        TuyaZBElectricalMeasurement.AttributeDefs.active_power.id,
        TuyaZBElectricalMeasurement.AttributeDefs.power_factor.id,
        TuyaZBElectricalMeasurement.AttributeDefs.rms_current.id,
    }

    def __init__(self, *args, **kwargs):
        """Init."""
        self._held_values: dict[str, Any] = {}
        super().__init__(*args, **kwargs)

    @property
    def _late_energy_flow(self) -> bool:
        """Return the config value for the channel endpoint."""
        config_attr = self._EP_MITIGATION_CONFIG_ATTR.get(self.endpoint.endpoint_id)
        if not config_attr:
            return False
        return bool(self.endpoint.device.endpoints[1].local_config.get(config_attr))

    def _late_energy_flow_handler(self, attr_name: str, value: Any) -> Any:
        """Hold non-power attribute values until the next update is received from the device."""
        if attr_name == TuyaZBElectricalMeasurement.AttributeDefs.active_power.name:
            return value

        held_value = self._held_values.pop(attr_name, None)
        if not self._late_energy_flow:
            return value

        self._held_values[attr_name] = value
        return held_value

    def update_attribute(self, attr_name: str, value):
        """Update the cluster attribute."""
        value = self._late_energy_flow_handler(attr_name, value)
        super().update_attribute(attr_name, value)


class MatSeePlusElectricalMeasurementTotal(MatSeePlusElectricalMeasurement):
    """ElectricalMeasurement cluster for MatSeePlus CT Energy Meter common measurements and total power."""

    _VALID_ATTRIBUTES: set[int] = {
        TuyaZBElectricalMeasurement.AttributeDefs.active_power.id,
        TuyaZBElectricalMeasurement.AttributeDefs.ac_frequency.id,
        TuyaZBElectricalMeasurement.AttributeDefs.rms_voltage.id,
    }


class MatSeePlusMetering(TuyaZBMeteringClusterWithUnit, TuyaLocalCluster):
    """Metering cluster for MatSeePlus CT energy meter."""

    _VALID_ATTRIBUTES: set[int] = {
        TuyaZBMeteringClusterWithUnit.AttributeDefs.current_summ_delivered.id,
        TuyaZBMeteringClusterWithUnit.AttributeDefs.current_summ_received.id,
    }


class TuyaMatSeePlusManufCluster(TuyaMCUCluster):
    """Handle MatSeePlus power datapoint logic, addressing known firmware issues.

    - Sequence power DP value signing and reporting to Electrical Measurement clusters respective of reporting sequence and config.
    - Recalculate the AB total power because the reported value on DP 115 is inaccurate due to the flow delay bug.
    """

    ENERGY_FLOW_A: Final = "energy_flow_a"
    ENERGY_FLOW_B: Final = "energy_flow_b"
    POWER_A: Final = "power_a"
    POWER_B: Final = "power_b"

    def __init__(self, *args, **kwargs):
        """Init."""
        self._power_signed_a: int | None = None
        self._power_signed_b: int | None = None
        self._deferred_a: bool = False
        self._deferred_b: bool = False
        super().__init__(*args, **kwargs)

    @staticmethod
    def _align_value_with_energy_flow(
        value: int | None, direction: TuyaEnergyFlow | None
    ) -> int | None:
        """Align the input value with specified energy flow direction."""
        if value and value > 0 and direction == 1:
            value = -value
        return value

    def _compute_signed_power(
        self,
        attr_name: str,
        value: int,
        power_attr: str,
        energy_flow_attr: str,
        late_energy_flow: bool,
    ) -> tuple[int | None, bool]:
        """Compute signed power based on energy configuration and DP reporting order.

        The flow DP value the previous interval is reported prior to current power,
        the device omits the flow DP in intervals with 0 power.
        """
        signed_value = None
        deferred = False

        if late_energy_flow:
            if attr_name == energy_flow_attr:
                signed_value = self._align_value_with_energy_flow(
                    self.get(power_attr), value
                )
            elif attr_name == power_attr and value == 0:
                # Hold zero power until next interval because the flow DP is not reported with 0 power
                signed_value = 0
                deferred = True
        elif attr_name == power_attr:
            signed_value = self._align_value_with_energy_flow(
                value, self.get(energy_flow_attr)
            )
        return signed_value, deferred

    def _report_power_value(self, value: int, endpoint_id: int):
        """Report the power value to the specified ElectricalMeasurement endpoint cluster."""
        self.endpoint.device.endpoints[
            endpoint_id
        ].electrical_measurement.update_attribute(
            MatSeePlusElectricalMeasurement.AttributeDefs.active_power.name,
            value,
        )

    def _maybe_report_total_power(self):
        """Calculate and report total power if both channels are ready."""
        if (
            not self._deferred_a
            and not self._deferred_b
            and self._power_signed_a is not None
            and self._power_signed_b is not None
        ):  # Normal case: both channels ready
            self._report_power_value(
                self._power_signed_a + self._power_signed_b, ENDPOINT_ID_TOTAL
            )
        elif (
            not self._deferred_a
            and self._power_signed_a is not None
            and self._deferred_b
            and self._power_signed_b == 0
        ):  # Special case: B is deferred with zero power
            self._report_power_value(self._power_signed_a, ENDPOINT_ID_TOTAL)

    def _process_power_and_energy_flow(
        self,
        attr_name: str,
        value: int,
        power_attr: str,
        energy_flow_attr: str,
        late_energy_flow: bool,
        endpoint_id: int,
        deferred_flag_name: str,
        stored_power_name: str,
    ):
        """Process power and flow updates for a single channel."""
        # Get current deferred state and stored power
        is_deferred = getattr(self, deferred_flag_name)
        stored_power = getattr(self, stored_power_name)

        # Release previously deferred value
        if is_deferred and stored_power is not None:
            self._report_power_value(stored_power, endpoint_id)
            setattr(self, deferred_flag_name, False)
            # Check if total can be reported after releasing deferred value
            self._maybe_report_total_power()

        # Compute the new signed power value
        power_signed, deferred = self._compute_signed_power(
            attr_name,
            value,
            power_attr=power_attr,
            energy_flow_attr=energy_flow_attr,
            late_energy_flow=late_energy_flow,
        )

        # Store signed value and deferred state
        if power_signed is not None:
            setattr(self, stored_power_name, power_signed)
        setattr(self, deferred_flag_name, deferred)

        # Report the signed value to the cluster (unless deferred)
        if not deferred and power_signed is not None:
            self._report_power_value(power_signed, endpoint_id)
            # Check if total can be reported using the new value
            self._maybe_report_total_power()

    def update_attribute(self, attr_name: str, value):
        """Handle reports to Electrical Measurement power attributes after aligning with energy flow."""
        super().update_attribute(attr_name, value)

        if attr_name in (self.POWER_A, self.ENERGY_FLOW_A):
            self._process_power_and_energy_flow(
                attr_name,
                value,
                power_attr=self.POWER_A,
                energy_flow_attr=self.ENERGY_FLOW_A,
                late_energy_flow=self.endpoint.local_config.get(
                    MatSeePlusLocalConfig.AttributeDefs.late_energy_flow_a.name
                ),
                endpoint_id=ENDPOINT_ID_CT_A,
                deferred_flag_name="_deferred_a",
                stored_power_name="_power_signed_a",
            )

        elif attr_name in (self.POWER_B, self.ENERGY_FLOW_B):
            self._process_power_and_energy_flow(
                attr_name,
                value,
                power_attr=self.POWER_B,
                energy_flow_attr=self.ENERGY_FLOW_B,
                late_energy_flow=self.endpoint.local_config.get(
                    MatSeePlusLocalConfig.AttributeDefs.late_energy_flow_b.name
                ),
                endpoint_id=ENDPOINT_ID_CT_B,
                deferred_flag_name="_deferred_b",
                stored_power_name="_power_signed_b",
            )


(
    ### MatSee Plus Tuya PJ-1203A 2 CT Bidirectional Energy Meter
    TuyaQuirkBuilder("_TZE204_81yrt3lo", "TS0601")
    .also_applies_to("_TZE284_81yrt3lo", "TS0601")
    .tuya_enchantment()
    .adds_endpoint(ENDPOINT_ID_CT_B)
    .adds_endpoint(ENDPOINT_ID_TOTAL)
    .adds(MatSeePlusElectricalMeasurement)
    .adds(MatSeePlusElectricalMeasurement, endpoint_id=ENDPOINT_ID_CT_B)
    .adds(MatSeePlusElectricalMeasurement, endpoint_id=ENDPOINT_ID_TOTAL)
    .adds(MatSeePlusMetering)
    .adds(MatSeePlusMetering, endpoint_id=ENDPOINT_ID_CT_B)
    .adds(MatSeePlusLocalConfig)
    # Metering attributes
    .tuya_dp(
        dp_id=106,
        ep_attribute=MatSeePlusMetering.ep_attribute,
        attribute_name=MatSeePlusMetering.AttributeDefs.current_summ_delivered.name,
    )
    .tuya_dp(
        dp_id=108,
        ep_attribute=MatSeePlusMetering.ep_attribute,
        attribute_name=MatSeePlusMetering.AttributeDefs.current_summ_delivered.name,
        endpoint_id=ENDPOINT_ID_CT_B,
    )
    .tuya_dp(
        dp_id=107,
        ep_attribute=MatSeePlusMetering.ep_attribute,
        attribute_name=MatSeePlusMetering.AttributeDefs.current_summ_received.name,
    )
    .tuya_dp(
        dp_id=109,
        ep_attribute=MatSeePlusMetering.ep_attribute,
        attribute_name=MatSeePlusMetering.AttributeDefs.current_summ_received.name,
        endpoint_id=ENDPOINT_ID_CT_B,
    )
    # Power attributes handled within manufacturer cluster
    .tuya_dp_attribute(
        dp_id=101,
        attribute_name=TuyaMatSeePlusManufCluster.POWER_A,
        type=t.uint32_t_be,
    )
    .tuya_dp_attribute(
        dp_id=105,
        attribute_name=TuyaMatSeePlusManufCluster.POWER_B,
        type=t.uint32_t_be,
    )
    .tuya_dp_attribute(
        dp_id=102,
        attribute_name=TuyaMatSeePlusManufCluster.ENERGY_FLOW_A,
        type=TuyaEnergyFlow,
    )
    .tuya_dp_attribute(
        dp_id=104,
        attribute_name=TuyaMatSeePlusManufCluster.ENERGY_FLOW_B,
        type=TuyaEnergyFlow,
    )
    # Electrical measurement attributes
    .tuya_dp(
        dp_id=110,
        ep_attribute=MatSeePlusElectricalMeasurement.ep_attribute,
        attribute_name=MatSeePlusElectricalMeasurement.AttributeDefs.power_factor.name,
    )
    .tuya_dp(
        dp_id=121,
        ep_attribute=MatSeePlusElectricalMeasurement.ep_attribute,
        attribute_name=MatSeePlusElectricalMeasurement.AttributeDefs.power_factor.name,
        endpoint_id=ENDPOINT_ID_CT_B,
    )
    .tuya_dp(
        dp_id=113,
        ep_attribute=MatSeePlusElectricalMeasurement.ep_attribute,
        attribute_name=MatSeePlusElectricalMeasurement.AttributeDefs.rms_current.name,
    )
    .tuya_dp(
        dp_id=114,
        ep_attribute=MatSeePlusElectricalMeasurement.ep_attribute,
        attribute_name=MatSeePlusElectricalMeasurement.AttributeDefs.rms_current.name,
        endpoint_id=ENDPOINT_ID_CT_B,
    )
    .tuya_dp(
        dp_id=112,
        ep_attribute=MatSeePlusElectricalMeasurementTotal.ep_attribute,
        attribute_name=MatSeePlusElectricalMeasurementTotal.AttributeDefs.rms_voltage.name,
        endpoint_id=ENDPOINT_ID_TOTAL,
    )
    .tuya_dp(
        dp_id=111,
        ep_attribute=MatSeePlusElectricalMeasurementTotal.ep_attribute,
        attribute_name=MatSeePlusElectricalMeasurementTotal.AttributeDefs.ac_frequency.name,
        endpoint_id=ENDPOINT_ID_TOTAL,
    )
    # Local configuration attributes
    .switch(
        MatSeePlusLocalConfig.AttributeDefs.late_energy_flow_a.name,
        MatSeePlusLocalConfig.cluster_id,
        entity_type=EntityType.CONFIG,
        translation_key="mitigate_flow_a_delay",
        fallback_name="Mitigate flow A delay",
        initially_disabled=False,
    )
    .switch(
        MatSeePlusLocalConfig.AttributeDefs.late_energy_flow_b.name,
        MatSeePlusLocalConfig.cluster_id,
        entity_type=EntityType.CONFIG,
        translation_key="mitigate_flow_b_delay",
        fallback_name="Mitigate flow B delay",
        initially_disabled=False,
    )
    # Device configuration attributes
    .tuya_number(
        dp_id=129,
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
        dp_id=122,
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
        dp_id=116,
        attribute_name="voltage_coefficient",
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
    .tuya_number(
        dp_id=119,
        attribute_name="current_summ_delivered_coefficient_a",
        type=t.uint32_t_be,
        unit=PERCENTAGE,
        min_value=0,
        max_value=2000,
        step=0.1,
        multiplier=0.1,
        translation_key="calibrate_summ_delivered_a",
        fallback_name="Calibrate summation delivered A",
        entity_type=EntityType.CONFIG,
        initially_disabled=True,
    )
    .tuya_number(
        dp_id=125,
        attribute_name="current_summ_delivered_coefficient_b",
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
        dp_id=127,
        attribute_name="current_summ_received_coefficient_a",
        type=t.uint32_t_be,
        unit=PERCENTAGE,
        min_value=0,
        max_value=2000,
        step=0.1,
        multiplier=0.1,
        translation_key="calibrate_summ_received_a",
        fallback_name="Calibrate summation received A",
        entity_type=EntityType.CONFIG,
        initially_disabled=True,
    )
    .tuya_number(
        dp_id=128,
        attribute_name="current_summ_received_coefficient_b",
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
        dp_id=118,
        attribute_name="power_coefficient_a",
        type=t.uint32_t_be,
        unit=PERCENTAGE,
        min_value=0,
        max_value=2000,
        step=0.1,
        multiplier=0.1,
        translation_key="calibrate_power_a",
        fallback_name="Calibrate power A",
        entity_type=EntityType.CONFIG,
        initially_disabled=True,
    )
    .tuya_number(
        dp_id=124,
        attribute_name="power_coefficient_b",
        type=t.uint32_t_be,
        unit=PERCENTAGE,
        min_value=0,
        max_value=2000,
        step=0.1,
        multiplier=0.1,
        translation_key="calibrate_power_b",
        fallback_name="Calibrate power B",
        entity_type=EntityType.CONFIG,
        initially_disabled=True,
    )
    .tuya_number(
        dp_id=117,
        attribute_name="current_coefficient_a",
        type=t.uint32_t_be,
        unit=PERCENTAGE,
        min_value=0,
        max_value=2000,
        step=0.1,
        multiplier=0.1,
        translation_key="calibrate_current_a",
        fallback_name="Calibrate current A",
        entity_type=EntityType.CONFIG,
        initially_disabled=True,
    )
    .tuya_number(
        dp_id=123,
        attribute_name="current_coefficient_b",
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
    .add_to_registry(replacement_cluster=TuyaMatSeePlusManufCluster)
)
