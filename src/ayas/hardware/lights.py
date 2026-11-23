"""Switch the white LEDs, one at a time.

LEDs are switched fully on or off. Do not dim with PWM: software PWM beats
against the rolling shutter and paints bands across the image. Set brightness
with the LED resistor/driver and the camera exposure instead.
"""

from __future__ import annotations


class Lights:
    def on(self, name: str) -> None:
        raise NotImplementedError

    def off(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        self.off()


class GpioLights(Lights):
    """One logic-level MOSFET per LED, gate driven from a GPIO pin."""

    def __init__(self, pins: dict[str, int], active_high: bool = True):
        from gpiozero import DigitalOutputDevice

        self._outputs = {
            name: DigitalOutputDevice(pin, active_high=active_high, initial_value=False) for name, pin in pins.items()
        }

    def on(self, name: str) -> None:
        self.off()
        self._outputs[name].on()

    def off(self) -> None:
        for out in self._outputs.values():
            out.off()

    def close(self) -> None:
        self.off()
        for out in self._outputs.values():
            out.close()
