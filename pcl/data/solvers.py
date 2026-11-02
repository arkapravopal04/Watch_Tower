"""Calorically perfect gas and low-speed relations, in SI units.

References: NASA Glenn normal-shock and isentropic-flow equations.
All pressures and temperatures are absolute; coefficients must be supplied.
"""

import math


def _positive(**values: float) -> None:
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")


def _nonnegative(**values: float) -> None:
    for name, value in values.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")


def _gas(gamma: float) -> None:
    if not math.isfinite(gamma) or gamma <= 1:
        raise ValueError("gamma must be finite and greater than one")


def speed_of_sound(temperature: float, gamma: float = 1.4, gas_constant: float = 287.05) -> float:
    _gas(gamma)
    _positive(temperature=temperature, gas_constant=gas_constant)
    return math.sqrt(gamma * gas_constant * temperature)


def mach_number(velocity: float, temperature: float, gamma: float = 1.4,
                gas_constant: float = 287.05) -> float:
    _nonnegative(velocity=velocity)
    return velocity / speed_of_sound(temperature, gamma, gas_constant)


def isentropic(mach: float, gamma: float = 1.4) -> dict[str, float]:
    """Return stagnation/static ratios for isentropic deceleration."""
    _gas(gamma)
    _nonnegative(mach=mach)
    factor = 1 + (gamma - 1) * mach**2 / 2
    return {"T0_T": factor, "p0_p": factor ** (gamma / (gamma - 1))}


def mach_from_pressure_ratio(p0_p: float, gamma: float = 1.4) -> float:
    _gas(gamma)
    if not math.isfinite(p0_p) or p0_p < 1:
        raise ValueError("stagnation/static pressure ratio must be at least one")
    return math.sqrt(2 * (p0_p ** ((gamma - 1) / gamma) - 1) / (gamma - 1))


def normal_shock(mach: float, gamma: float = 1.4) -> dict[str, float]:
    _gas(gamma)
    if not math.isfinite(mach) or mach <= 1:
        raise ValueError("a normal shock requires supersonic upstream Mach")
    m2 = mach * mach
    return {"M2": math.sqrt((1 + (gamma - 1) * m2 / 2) / (gamma * m2 - (gamma - 1) / 2)),
            "p2_p1": (2 * gamma * m2 - (gamma - 1)) / (gamma + 1)}


def dynamic_pressure(density: float, velocity: float) -> float:
    _positive(density=density)
    _nonnegative(velocity=velocity)
    return 0.5 * density * velocity**2


def aerodynamic_force(density: float, velocity: float, area: float, coefficient: float) -> float:
    _positive(area=area)
    if not math.isfinite(coefficient):
        raise ValueError("coefficient must be finite")
    return dynamic_pressure(density, velocity) * area * coefficient


def velocity_from_pressure(pressure: float, density: float) -> float:
    _nonnegative(pressure=pressure)
    _positive(density=density)
    return math.sqrt(2 * pressure / density)
