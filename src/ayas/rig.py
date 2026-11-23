"""Build the rig (real hardware or simulator) from the config."""

from __future__ import annotations

from .config import VIEWS, Config


class HardwareRig:
    def __init__(self, cfg: Config):
        from .hardware.cameras import PiCamera, UsbCamera
        from .hardware.lights import GpioLights

        pins = {name: spec.pin for name, spec in cfg.lights.items()}
        missing = [name for name, pin in pins.items() if pin is None]
        if missing:
            raise ValueError(f"lights without a GPIO pin: {missing}")
        self.lights = GpioLights(pins, active_high=cfg.lights_active_high)
        self.cameras = {}
        for view in VIEWS:
            cam = cfg.cameras[view]
            driver = cam.get("driver", {})
            if cam["backend"] == "picamera2":
                self.cameras[view] = PiCamera(**driver)
            elif cam["backend"] == "usb":
                self.cameras[view] = UsbCamera(**driver)
            else:
                raise ValueError(f"camera {view}: unknown backend {cam['backend']!r}")

    def place_part(self, defects=None, prompt: str = "Place the part against the jig stops and press Enter... "):
        if defects is not None:
            raise ValueError("defects can only be chosen on the simulator")
        input(prompt)
        return []

    def place_developed(self, indications=None):
        """Penetrant step: the plate has been through clean -> penetrant -> wipe -> developer."""
        if indications is not None:
            raise ValueError("indications can only be chosen on the simulator")
        input("Place the DEVELOPED plate (penetrant process done, developer dry) in the jig and press Enter... ")
        return []

    def close(self) -> None:
        self.lights.close()
        for cam in self.cameras.values():
            cam.close()


def build_rig(cfg: Config, seed: int | None = None):
    if cfg.is_sim:
        from .sim import SimRig

        return SimRig(cfg, seed=seed)
    return HardwareRig(cfg)
