"""
Type B uncertainty: uncertainty evaluated by means other than repeated
observation (GUM Section 4.3) -- here, instrument resolution and, where
known, manufacturer accuracy.

PROVENANCE OF THE DEFAULT NUMBERS BELOW (read before trusting them):

  - `current`, `voltage`, `frequency` resolutions come directly from this
    project's own Delta VFD Modbus register scaling, documented in the
    project report (Table 4.2 / Appendix A "Consolidated calibration
    reference") and mirrored in control_panel.py's register reads:
        current  = register / 100  -> 0.01 A step
        voltage  = register / 10   -> 0.1 V step
        frequency= register / 100  -> 0.01 Hz step
    These ARE sourced from the codebase/report.

  - `vibration` resolution comes from the MPU-6050 scale factor documented in
    the same Appendix A: 4096 LSB per g (+/-8g range) -> 1 LSB = 1/4096 g.
    This IS sourced from the report. Note control_panel.py does not currently
    read the MPU-6050 at all (see the codebase-understanding note given at
    the start of this task); this spec is ready for when that sensor is wired
    in, or for use against a vibration RMS derived from an incoming window.

  - `temperature` resolution (0.0625 C) is the DS18B20's default 12-bit
    conversion resolution. This number is NOT stated anywhere in the
    uploaded project files -- the report only gives the DS18B20's valid
    reading range (10-125 C) and its CRC-error sentinel value (85 C), not its
    resolution. 0.0625 C is the well-documented default resolution from the
    DS18B20 datasheet referenced in the report's bibliography ([17]), but
    since it isn't itself quoted in the repository, treat it as a
    configurable placeholder to confirm against your actual sensor
    configuration (some DS18B20 firmwares are configured for 9/10/11-bit
    resolution instead of the 12-bit default, which changes this number).

  - No manufacturer "accuracy" (e.g. +/- % of reading) figures for the
    MPU-6050, DS18B20, or the VFD's internal ADC are given anywhere in the
    uploaded materials. Rather than invent numbers, every channel below
    falls back to a resolution-only estimate:

        u_B = resolution / sqrt(12)

    which is the standard GUM approach (Section 4.3.7) for a quantity known
    only to lie within a resolution step, assumed rectangularly distributed.
    Fill in `manufacturer_accuracy` for any channel once you have a real
    datasheet or calibration-certificate figure -- see `load_channel_specs()`
    for how to override these without touching this file.
"""
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    resolution: float
    manufacturer_accuracy: Optional[float] = None  # absolute +/- tolerance, if known
    distribution: str = "rectangular"               # "rectangular" or "normal"
    accuracy_coverage_k: float = 2.0                # if manufacturer_accuracy is a stated 95%-ish CI half-width

    def standard_uncertainty(self) -> float:
        if self.manufacturer_accuracy is not None:
            if self.distribution == "normal":
                return self.manufacturer_accuracy / self.accuracy_coverage_k
            return self.manufacturer_accuracy / math.sqrt(3)  # rectangular assumption for a stated tolerance
        return self.resolution / math.sqrt(12)  # resolution-only rectangular assumption


DEFAULT_CHANNEL_SPECS: Dict[str, ChannelSpec] = {
    "current":     ChannelSpec("current",     resolution=0.01),
    "voltage":     ChannelSpec("voltage",     resolution=0.1),
    "frequency":   ChannelSpec("frequency",   resolution=0.01),
    "temperature": ChannelSpec("temperature", resolution=0.0625),
    "vibration":   ChannelSpec("vibration",   resolution=1.0 / 4096.0),
}


def load_channel_specs(path: Optional[str] = None) -> Dict[str, ChannelSpec]:
    """
    Load channel specs from a JSON file (env var UNCERTAINTY_CONFIG_PATH, or
    an explicit `path`) and overlay them on the defaults above, so specs can
    be corrected/extended without a code change as real datasheet or
    calibration numbers become available. Missing file -> defaults only.

    Expected JSON shape:
        {
          "temperature": {"resolution": 0.0625, "manufacturer_accuracy": 0.5},
          "new_sensor_channel": {"resolution": 0.001}
        }
    """
    path = path or os.environ.get("UNCERTAINTY_CONFIG_PATH")
    specs = dict(DEFAULT_CHANNEL_SPECS)
    if not path or not os.path.exists(path):
        return specs
    with open(path, "r") as f:
        raw = json.load(f)
    for name, values in raw.items():
        specs[name] = ChannelSpec(name=name, **values)
    return specs
