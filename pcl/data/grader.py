"""Strict, versioned final-answer grammar with dimensional unit conversion.

The prompt declares the grammar, so this checks instruction following too.
Explanations may precede a single FINAL: line. Duplicate keys/lines fail.
"""

import math
import re
from dataclasses import dataclass

VERSION = "1"
# unit -> (dimension, multiplicative conversion to SI)
UNITS = {
    "1": ("ratio", 1.0), "": ("ratio", 1.0), "dimensionless": ("ratio", 1.0),
    "Pa": ("pressure", 1.0), "kPa": ("pressure", 1000.0),
    "MPa": ("pressure", 1e6), "bar": ("pressure", 1e5),
    "N": ("force", 1.0), "kN": ("force", 1000.0), "lbf": ("force", 4.4482216152605),
    "m/s": ("speed", 1.0), "km/h": ("speed", 1 / 3.6), "ft/s": ("speed", 0.3048),
    "K": ("temperature", 1.0), "kg/m^3": ("density", 1.0), "m^2": ("area", 1.0),
}
NUMBER = re.compile(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*?)\s*\Z")


@dataclass(frozen=True)
class Grade:
    correct: bool
    format_ok: bool
    reason: str


def format_answer(reference: dict) -> str:
    fields = []
    for key, spec in reference.items():
        if isinstance(spec, str):
            fields.append(f"{key}={spec}")
        else:
            fields.append(f"{key}={spec['value']:.7g} {spec['unit']}")  # same digits as the worked lines
    return "FINAL: " + "; ".join(fields)


def grade_answer(text: str, reference: dict, *, rtol: float = 0.001) -> Grade:
    if not math.isfinite(rtol) or rtol < 0:
        raise ValueError("rtol must be finite and nonnegative")
    if not reference:
        raise ValueError("an answer must have at least one required field")
    for spec in reference.values():
        if not isinstance(spec, str) and (spec['unit'] not in UNITS or
                not math.isfinite(spec['value']) or not math.isfinite(spec['atol']) or spec['atol'] < 0):
            raise ValueError('reference quantities require valid units and finite values/tolerances')
    lines = text.strip().splitlines()
    finals = [line for line in lines if line.strip().startswith("FINAL:")]
    if len(finals) != 1 or finals[0] != lines[-1] or text.count("FINAL:") != 1:
        return Grade(False, False, "expected exactly one terminal FINAL: line")
    fields = {}
    for field in finals[0].strip()[6:].split(";"):
        if field.count("=") != 1:
            return Grade(False, False, "malformed field")
        key, value = (part.strip() for part in field.split("="))
        if not key or not value or key in fields:
            return Grade(False, False, "empty or duplicate field")
        fields[key] = value
    if fields.keys() != reference.keys():
        return Grade(False, False, "missing or unexpected quantities")
    failures = []
    for key, spec in reference.items():
        if isinstance(spec, str):
            if fields[key] != spec:
                failures.append(key)
            continue
        match = NUMBER.fullmatch(fields[key])
        if not match or match[2] not in UNITS:
            return Grade(False, False, f"invalid number or unit: {key}")
        dimension, scale = UNITS[match[2]]
        target_dimension, target_scale = UNITS[spec["unit"]]
        value = float(match[1]) * scale
        if not math.isfinite(value):
            return Grade(False, False, f"nonfinite quantity: {key}")
        if dimension != target_dimension:
            failures.append(key)
            continue
        expected = spec["value"] * target_scale
        tolerance = spec['atol'] * target_scale + rtol * abs(expected)
        if abs(value - expected) > tolerance:
            failures.append(key)
    return Grade(not failures, True, "ok" if not failures else "incorrect: " + ", ".join(failures))
