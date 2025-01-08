"""Aqara Roller Shade Driver E1 device."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final

from zigpy import types as t
from zigpy.quirks.v2 import QuirkBuilder
from zigpy.quirks.v2.homeassistant.binary_sensor import BinarySensorDeviceClass
from zigpy.zcl import Cluster, foundation
from zigpy.zcl.clusters.closures import WindowCovering
from zigpy.zcl.clusters.general import AnalogOutput, MultistateOutput, OnOff
from zigpy.zcl.foundation import BaseAttributeDefs, DataTypeId, ZCLAttributeDef

from zhaquirks import CustomCluster
from zhaquirks.xiaomi import (
    LUMI,
    BasicCluster,
    XiaomiAqaraE1Cluster,
    XiaomiPowerConfigurationPercent,
)


class AqaraRollerDriverCharging(t.enum8):
    """Aqara roller driver charging status attribute values."""

    true = 0x01
    false = 0x02


class AqaraRollerDriverSpeed(t.enum8):
    """Aqara roller driver speed attribute values."""

    Low = 0x00
    Medium = 0x01
    High = 0x02


class AqaraRollerState(t.enum8):
    """Aqara roller state values."""

    Closing = 0x00
    Opening = 0x01
    Stopped = 0x02
    Blocked = 0x03


class RedirectAttributes:
    """Methods for redirecting attribute reads to another cluster."""

    _REDIRECT_ATTRIBUTES: (
        dict[ZCLAttributeDef, tuple[ZCLAttributeDef, Cluster, Callable | None]] | None
    ) = None

    async def read_attributes(
        self,
        attributes: list[int | str],
        allow_cache: bool = False,
        only_cache: bool = False,
        manufacturer: int | t.uint16_t | None = None,
    ):
        """Redirect attribute reads to another cluster."""

        successful, failed = {}, {}
        remaining_attributes = attributes.copy()
        redirect_attributes = self._REDIRECT_ATTRIBUTES or {}

        for attr in redirect_attributes:
            if attr.id not in attributes and attr.name not in attributes:
                continue
            if attr.id in attributes:
                remaining_attributes.remove(attr.id)
            if attr.name in attributes:
                remaining_attributes.remove(attr.name)

            target_attr, target_cluster, format_func = redirect_attributes[attr]
            result_s, result_f = await getattr(
                self.endpoint, target_cluster.ep_attribute
            ).read_attributes(
                [target_attr.id],
                allow_cache,
                only_cache,
                manufacturer,
            )

            if target_attr.id in result_s:
                value = result_s[target_attr.id]
                successful[attr.id] = format_func(value) if format_func else value
            if target_attr.id in result_f:
                failed[attr.id] = result_f[target_attr.id]

        if remaining_attributes:
            result_s, result_f = await super().read_attributes(
                remaining_attributes, allow_cache, only_cache, manufacturer
            )
            successful.update(result_s)
            failed.update(result_f)

        return successful, failed


class WriteAwareUpdateAttribute:
    """Methods providing 'is_write' arg to _update_attribute."""

    async def write_attributes_raw(
        self,
        attrs: list[foundation.Attribute],
        manufacturer: int | None = None,
        **kwargs,
    ) -> list:
        """Provide the is_write=True flag when calling _update_attribute."""

        result = await self._write_attributes(
            attrs, manufacturer=manufacturer, **kwargs
        )
        if not isinstance(result[0], list):
            return result

        records = result[0]
        if len(records) == 1 and records[0].status == foundation.Status.SUCCESS:
            for attr_rec in attrs:
                self._update_attribute(
                    attr_rec.attrid, attr_rec.value.value, is_write=True
                )
        else:
            failed = [rec.attrid for rec in records]
            for attr_rec in attrs:
                if attr_rec.attrid not in failed:
                    self._update_attribute(
                        attr_rec.attrid, attr_rec.value.value, is_write=True
                    )

        return result

    def _update_attribute(
        self, attrid: int, value: Any, is_write: bool | None = None
    ) -> None:
        super()._update_attribute(attrid, value)


class XiaomiAqaraRollerE1(XiaomiAqaraE1Cluster):
    """Aqara manufacturer cluster for the Roller Driver E1."""

    class AttributeDefs(BaseAttributeDefs):
        """Manufacturer specific attributes."""

        reverse_direction = ZCLAttributeDef(
            id=0x0400,
            type=t.Bool,
            access="rwp",
            is_manufacturer_specific=True,
        )

        positions_stored = ZCLAttributeDef(
            id=0x0402,
            type=t.Bool,
            access="rwp",
            is_manufacturer_specific=True,
        )

        store_position = ZCLAttributeDef(
            id=0x0407,
            type=t.uint8_t,
            access="rwp",
            is_manufacturer_specific=True,
        )

        speed = ZCLAttributeDef(
            id=0x0408,
            type=AqaraRollerDriverSpeed,
            zcl_type=DataTypeId.uint8,
            access="rwp",
            is_manufacturer_specific=True,
        )

        charging = ZCLAttributeDef(
            id=0x0409,
            type=AqaraRollerDriverCharging,
            zcl_type=DataTypeId.uint8,
            access="rp",
            is_manufacturer_specific=True,
        )

        aqara_attributes = ZCLAttributeDef(
            id=0x00F7,
            type=t.LVBytes,
            is_manufacturer_specific=True,
        )

    @staticmethod
    def _enum_value_to_bool(enum_type, value: Any) -> t.Bool | None:
        """Return the t.Bool equivalent of enum mapped true/false values."""

        if value == enum_type.true:
            return t.Bool.true
        if value == enum_type.false:
            return t.Bool.false
        return None

    def _update_attribute(self, attrid: int, value: Any) -> None:
        """Convert Aqara charging status values to binary_sensor charging true/false."""

        if attrid == self.AttributeDefs.charging.id:
            value = self._enum_value_to_bool(AqaraRollerDriverCharging, value)
        super()._update_attribute(attrid, value)


class AnalogOutputRollerE1(WriteAwareUpdateAttribute, CustomCluster, AnalogOutput):
    """Analog output cluster, only used to relay present_value to WindowCovering current_position_lift_percentage."""

    _CONSTANT_ATTRIBUTES = {
        AnalogOutput.AttributeDefs.description.id: "Current position",
        AnalogOutput.AttributeDefs.max_present_value.id: 100.0,
        AnalogOutput.AttributeDefs.min_present_value.id: 0.0,
        AnalogOutput.AttributeDefs.out_of_service.id: 0,
        AnalogOutput.AttributeDefs.resolution.id: 1.0,
        AnalogOutput.AttributeDefs.status_flags.id: 0x00,
    }

    def _update_attribute(
        self, attrid: int, value: Any, is_write: bool | None = None
    ) -> None:
        """Non-write 'present_value' updates should update the WindowCovering position."""

        super()._update_attribute(attrid, value)
        if attrid == self.AttributeDefs.present_value.id and not is_write:
            self.endpoint.window_covering.update_attribute(
                WindowCovering.AttributeDefs.current_position_lift_percentage.id,
                (100 - value),
            )


class WindowCoveringRollerE1(RedirectAttributes, CustomCluster, WindowCovering):
    """Window covering cluster for handling motor commands."""

    _CONSTANT_ATTRIBUTES = {
        WindowCovering.AttributeDefs.window_covering_type.id: WindowCovering.WindowCoveringType.Rollershade,
    }

    # This is used to redirect 'current_position_lift_percentage' reads to AnalogOutput 'present_value'
    _REDIRECT_ATTRIBUTES = {
        WindowCovering.AttributeDefs.current_position_lift_percentage: (
            AnalogOutput.AttributeDefs.present_value,
            AnalogOutput,
            lambda x: 100 - x,
        ),
    }

    async def command(
        self,
        command_id: foundation.GeneralCommand | int | t.uint8_t,
        *args: Any,
        manufacturer: int | t.uint16_t | None = None,
        expect_reply: bool = True,
        tsn: int | t.uint8_t | None = None,
        **kwargs: Any,
    ) -> Any:
        """Overwrite the commands to make it work for both firmware 1425 and 1427.

        We either overwrite analog_output's current_value or multistate_output's current
        value to make the roller work.
        """
        if command_id == WindowCovering.ServerCommandDefs.up_open.id:
            (res,) = await self.endpoint.multistate_output.write_attributes(
                {
                    MultistateOutput.AttributeDefs.present_value.name: AqaraRollerState.Opening
                }
            )
            return foundation.GENERAL_COMMANDS[
                foundation.GeneralCommand.Default_Response
            ].schema(command_id=command_id, status=res[0].status)

        if command_id == WindowCovering.ServerCommandDefs.down_close.id:
            (res,) = await self.endpoint.multistate_output.write_attributes(
                {
                    MultistateOutput.AttributeDefs.present_value.name: AqaraRollerState.Closing
                }
            )
            return foundation.GENERAL_COMMANDS[
                foundation.GeneralCommand.Default_Response
            ].schema(command_id=command_id, status=res[0].status)

        if command_id == WindowCovering.ServerCommandDefs.go_to_lift_percentage.id:
            (res,) = await self.endpoint.analog_output.write_attributes(
                {AnalogOutput.AttributeDefs.present_value.name: (100 - args[0])}
            )
            return foundation.GENERAL_COMMANDS[
                foundation.GeneralCommand.Default_Response
            ].schema(command_id=command_id, status=res[0].status)

        if command_id == WindowCovering.ServerCommandDefs.stop.id:
            (res,) = await self.endpoint.multistate_output.write_attributes(
                {
                    MultistateOutput.AttributeDefs.present_value.name: AqaraRollerState.Stopped
                }
            )
            # Current position is read because it is not consistently reported
            await self.read_attributes(
                [self.AttributeDefs.current_position_lift_percentage.id]
            )
            return foundation.GENERAL_COMMANDS[
                foundation.GeneralCommand.Default_Response
            ].schema(command_id=command_id, status=res[0].status)

        return None


class MultistateOutputRollerE1(CustomCluster, MultistateOutput):
    """Multistate Output cluster which overwrites present_value attribute type.

    The device responds with an error when using the standard t.Single type.
    """

    class AttributeDefs(MultistateOutput.AttributeDefs):
        """Aqara attribute definition overrides."""

        present_value: Final = ZCLAttributeDef(
            id=0x0055, type=t.uint16_t, access="r*w", mandatory=True
        )


(
    QuirkBuilder(LUMI, "lumi.curtain.acn002")
    # temporarily commented out due to potentially breaking existing blueprints
    #    .friendly_name(
    #        manufacturer="Aqara", model="Roller Shade Driver E1"
    #    )
    .removes(OnOff.cluster_id)
    .replaces(AnalogOutputRollerE1)
    .replaces(BasicCluster)
    .replaces(MultistateOutputRollerE1)
    .replaces(XiaomiPowerConfigurationPercent)
    .replaces(WindowCoveringRollerE1)
    .replaces(XiaomiAqaraRollerE1)
    .enum(
        XiaomiAqaraRollerE1.AttributeDefs.speed.name,
        AqaraRollerDriverSpeed,
        XiaomiAqaraRollerE1.cluster_id,
        translation_key="speed",
        fallback_name="Speed",
    )
    .binary_sensor(
        XiaomiAqaraRollerE1.AttributeDefs.charging.name,
        XiaomiAqaraRollerE1.cluster_id,
        device_class=BinarySensorDeviceClass.BATTERY_CHARGING,
        translation_key="charging",
        fallback_name="Charging",
    )
    .binary_sensor(
        XiaomiAqaraRollerE1.AttributeDefs.positions_stored.name,
        XiaomiAqaraRollerE1.cluster_id,
        translation_key="calibrated",
        fallback_name="Calibrated",
    )
    .add_to_registry()
)
