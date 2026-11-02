"""Deterministic development data. Locked data requires an explicit opt-in.

Token lengths stay null until measured with the selected chat tokenizer.
"""

import math
import random

from . import solvers
from .grader import format_answer
from .splits import allocate_groups, problem_key

FAMILIES = {
    "A": ["mach", "isentropic", "pressure_inverse", "shock", "applicability"],
    "B": ["dynamic_pressure", "lift", "drag", "speed_inverse", "missing_input"],
    "general": ["arithmetic", "instruction", "reasoning"],
}
PREFIXES = ["Solve this stated scenario. ", "Using only the supplied inputs, answer the following. "]


def quantity(value: float, unit: str = "1") -> dict:
    return {"value": value, "unit": unit, "atol": 1e-6 if unit == "1" else 1e-3}


def n(value: float) -> str:
    """Intermediate values in worked answers: 7 significant digits keep the final within tolerance."""
    return f"{value:.7g}"


def _agrees(shown: float, reference: dict) -> None:
    # The worked calculation is separate arithmetic; it must land on the solver's answer.
    if not math.isclose(shown, reference['value'], rel_tol=2e-6, abs_tol=1e-9):
        raise AssertionError(f"worked answer {shown} disagrees with the solver's {reference['value']}")


def _scenario(skill: str, family: str, rng: random.Random, hard: bool) -> tuple:
    def sample(lo, hi):
        return round(rng.uniform(lo, hi), 6)

    assumptions = ["Use only given inputs; do not infer missing coefficients."]
    if skill == "A":
        gamma = rng.choice([1.3, 1.4, 1.67]) if hard else 1.4
        assumptions = [f"Calorically perfect gas, constant gamma={gamma}; absolute pressures and temperatures."]
        mach = sample(1.01, 4.0) if hard else sample(1.2, 3.0)
        inputs = {"gamma": gamma, "mach": mach}
        if family == "mach":
            inputs = {"gamma": gamma, "velocity": sample(100, 1000), "temperature": sample(220, 340), "R": 287.05}
            prompt = f"Find Mach M for V={inputs['velocity']} m/s, T={inputs['temperature']} K, R=287.05 J/(kg K)."
            ref = {"M": quantity(solvers.mach_number(inputs['velocity'], inputs['temperature'], gamma))}
            product = gamma * 287.05 * inputs['temperature']
            sound = float(n(math.sqrt(product)))
            explanation = (f"a = sqrt(gamma*R*T) = sqrt({gamma}*287.05*{inputs['temperature']}) = sqrt({n(product)}) "
                           f"= {n(sound)} m/s\nM = V/a = {inputs['velocity']}/{n(sound)} = {n(inputs['velocity'] / sound)}")
            _agrees(inputs['velocity'] / sound, ref['M'])
        elif family == "isentropic":
            prompt = f"At Mach {mach}, find stagnation/static ratios T0_T and p0_p for isentropic deceleration."
            values = solvers.isentropic(mach, gamma)
            ref = {key: quantity(values[key]) for key in ("T0_T", "p0_p")}
            square, exponent = mach * mach, gamma / (gamma - 1)
            explanation = (f"T0/T = 1 + (gamma-1)/2*M^2 = 1 + {n((gamma - 1) / 2)}*{n(square)} = {n(values['T0_T'])}\n"
                           f"p0/p = (T0/T)^(gamma/(gamma-1)) = {n(values['T0_T'])}^{n(exponent)} = {n(values['p0_p'])}")
        elif family == "pressure_inverse":
            ratio = round(solvers.isentropic(mach, gamma)["p0_p"], 9)
            inputs = {"gamma": gamma, "p0_p": ratio}
            prompt = f"For isentropic deceleration, p0/p={ratio}. Find nonnegative Mach M."
            ref = {"M": quantity(solvers.mach_from_pressure_ratio(ratio, gamma))}
            exponent = (gamma - 1) / gamma
            temperature_ratio = float(n(ratio ** exponent))
            shown = math.sqrt(2 * (temperature_ratio - 1) / (gamma - 1))
            explanation = (f"T0/T = (p0/p)^((gamma-1)/gamma) = {ratio}^{n(exponent)} = {n(temperature_ratio)}\n"
                           f"M = sqrt(2*(T0/T - 1)/(gamma-1)) = sqrt(2*{n(temperature_ratio - 1)}/{n(gamma - 1)}) "
                           f"= {n(shown)}")
            _agrees(shown, ref['M'])
        elif family == "shock":
            prompt = f"A one-dimensional adiabatic normal shock has upstream Mach {mach}. Find M2 and p2_p1."
            values = solvers.normal_shock(mach, gamma)
            ref = {key: quantity(values[key]) for key in ("M2", "p2_p1")}
            square = mach * mach
            top, bottom = 1 + (gamma - 1) / 2 * square, gamma * square - (gamma - 1) / 2
            explanation = (f"M1^2 = {n(square)}\n"
                           f"M2^2 = (1 + (gamma-1)/2*M1^2)/(gamma*M1^2 - (gamma-1)/2) = {n(top)}/{n(bottom)} "
                           f"= {n(top / bottom)}, so M2 = {n(values['M2'])}\n"
                           f"p2/p1 = (2*gamma*M1^2 - (gamma-1))/(gamma+1) = {n(2 * gamma * square - (gamma - 1))}/"
                           f"{n(gamma + 1)} = {n(values['p2_p1'])}")
        else:
            mach = sample(0.05, 0.99) if rng.random() < 0.5 else mach
            inputs["mach"] = mach
            prompt = f"At upstream Mach {mach}, is a stationary compressive normal shock applicable? Use status=applicable or status=not_applicable."
            ref = {"status": "applicable" if mach > 1 else "not_applicable"}
            explanation = (f"A compressive normal shock needs supersonic upstream flow; M1 = {mach} is "
                           f"{'above' if mach > 1 else 'not above'} 1.")
    elif skill == "B":
        rho, velocity, area = sample(0.8, 1.4), sample(5, 80), sample(0.5, 30)
        inputs = {"rho": rho, "V": velocity}
        if family == "dynamic_pressure":
            shown_v = round(velocity * 3.6, 7) if hard else velocity
            if hard:
                # Reference uses exactly the rounded input displayed to the model.
                inputs["V"] = shown_v / 3.6
            prompt = f"Find dynamic pressure q at density {rho} kg/m^3 and speed {shown_v} {'km/h' if hard else 'm/s'}."
            ref = {"q": quantity(solvers.dynamic_pressure(rho, inputs['V']), "Pa")}
            speed = float(n(inputs['V'])) if hard else velocity
            explanation = f"V = {shown_v}/3.6 = {n(speed)} m/s\n" if hard else ""
            explanation += (f"q = rho*V^2/2 = {rho}*{n(speed)}^2/2 = {rho}*{n(speed * speed)}/2 = "
                            f"{n(rho * speed * speed / 2)} Pa")
            _agrees(rho * speed * speed / 2, ref['q'])
        elif family in ("lift", "drag"):
            coef = sample(0.2, 1.2) if family == "lift" else sample(0.01, 0.15)
            symbol, name = ("L", "CL") if family == "lift" else ("D", "CD")
            inputs.update(S=area, coefficient=coef)
            prompt = f"At rho={rho} kg/m^3, V={velocity} m/s, S={area} m^2 and given {name}={coef}, find {symbol}."
            ref = {symbol: quantity(solvers.aerodynamic_force(rho, velocity, area, coef), "N")}
            pressure = float(n(rho * velocity * velocity / 2))
            explanation = (f"q = rho*V^2/2 = {rho}*{velocity}^2/2 = {rho}*{n(velocity * velocity)}/2 = {n(pressure)} Pa\n"
                           f"{symbol} = q*S*{name} = {n(pressure)}*{area}*{coef} = {n(pressure * area * coef)} N")
            _agrees(pressure * area * coef, ref[symbol])
        elif family == "speed_inverse":
            q = round(solvers.dynamic_pressure(rho, velocity), 7)
            inputs = {"rho": rho, "q": q}
            prompt = f"Dynamic pressure is {q} Pa at density {rho} kg/m^3. Find speed V."
            ref = {"V": quantity(solvers.velocity_from_pressure(q, rho), "m/s")}
            explanation = f"V = sqrt(2*q/rho) = sqrt(2*{q}/{rho}) = sqrt({n(2 * q / rho)}) = {n(ref['V']['value'])} m/s"
        else:
            inputs["S"] = area
            prompt = (f"Given rho={rho} kg/m^3, V={velocity} m/s and S={area} m^2, can lift be determined? "
                      "Use status=determined or status=missing_input and name the missing coefficient in field missing.")
            ref = {"status": "missing_input", "missing": "CL"}
            explanation = "Lift = q*S*CL needs the lift coefficient CL, which is not given."
    else:
        a, b, c = (rng.randrange(10, 10000) for _ in range(3))
        inputs = {"a": a, "b": b, "c": c}
        if family == "arithmetic":
            prompt = f"Compute result=({a}+{b})-{c}."
            ref = {"result": quantity(a + b - c)}
            explanation = f"({a}+{b})-{c} = {a + b}-{c} = {a + b - c}"
        elif family == "instruction":
            prompt = f"From the codes {a}, {b}, {c}, copy only the second code into field code."
            ref = {"code": str(b)}
            explanation = f"The second code is {b}."
        else:
            prompt = f"A container starts with {a} objects, receives {b}, then receives {c}. How many objects total? Use field total."
            ref = {"total": quantity(a + b + c)}
            explanation = f"{a}+{b}+{c} = {a + b}+{c} = {a + b + c}"
    grammar = (" Show a short calculation, then end with exactly one final line in the form "
               "FINAL: name=number unit; name=number unit, for example FINAL: x=12.3456 m/s; y=0.987654 1. "
               "Use only the requested field names. Write 1 as the unit of a dimensionless number. "
               "Status, missing and code fields are plain words or digits with no unit. "
               "Keep at least six significant digits.")
    return inputs, " ".join(assumptions) + " " + prompt + grammar, ref, explanation


def generate(skill: str, sizes: dict[str, int], seed: int = 0, *, allow_test: bool = False) -> list[dict]:
    if skill not in FAMILIES:
        raise ValueError(f"unknown skill: {skill}")
    if any(key not in {"train", "dev", "test", "test_hard"} for key in sizes):
        raise ValueError("unknown split")
    if not allow_test and any(sizes.get(key, 0) for key in ("test", "test_hard")):
        raise ValueError("locked test generation requires an explicit opt-in after pilot revisions")
    allocations = allocate_groups(FAMILIES[skill], sizes, seed, variants=len(PREFIXES))
    rng = random.Random(seed)
    records, seen = [], set()
    for group, family, split, count in allocations:
        for _ in range(1000):
            inputs, prompt, ref, explanation = _scenario(skill, family, rng, split == "test_hard")
            key = problem_key({"skill": skill, "family": family, "inputs": inputs})
            if key not in seen:
                seen.add(key)
                break
        else:
            raise ValueError("exhausted unique scenarios")
        for variant in range(count):
            records.append({"id": f"{skill}-{group}-{variant}", "skill": skill, "family": family,
                            "group": f"{skill}-{group}", "split": split,
                            "prompt": PREFIXES[variant] + prompt, "reference": ref,
                            "response": explanation + "\n" + format_answer(ref), "token_length": None})
    # Sealing a later test bank must not reorder already-generated train/dev files.
    ordered = []
    for split in sizes:
        subset = [record for record in records if record['split'] == split]
        random.Random(f'{seed}:{skill}:{split}').shuffle(subset)
        ordered.extend(subset)
    # Groups carry their split and scenarios are deduplicated above, so splits cannot leak.
    return ordered
