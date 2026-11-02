import math

import pytest

from pcl.data.solvers import (aerodynamic_force, dynamic_pressure, isentropic,
                              mach_from_pressure_ratio, normal_shock, speed_of_sound,
                              velocity_from_pressure)


def test_known_shock_and_conservation():
    # Textbook values; conservation is checked independently in pcl.data.audit.
    result = normal_shock(2)
    assert result['M2'] == pytest.approx(1 / math.sqrt(3))
    assert result['p2_p1'] == pytest.approx(4.5)


def test_known_low_speed_values():
    assert dynamic_pressure(1.225, 50) == pytest.approx(1531.25)
    assert aerodynamic_force(1.225, 50, 10, 0.5) == pytest.approx(7656.25)
    assert velocity_from_pressure(1531.25, 1.225) == pytest.approx(50)
    assert speed_of_sound(288.15) == pytest.approx(340.2922869)


def test_isentropic_identity():
    result = isentropic(2)
    assert result['T0_T'] == pytest.approx(1.8)
    assert result['p0_p'] == pytest.approx(7.8244490669)
    assert mach_from_pressure_ratio(result['p0_p']) == pytest.approx(2)
    assert isentropic(0)['p0_p'] == 1


@pytest.mark.parametrize('mach', [0, 1, -1, float('nan'), float('inf')])
def test_invalid_shock(mach):
    with pytest.raises(ValueError):
        normal_shock(mach)


@pytest.mark.parametrize('gamma', [1, 0, float('nan')])
def test_invalid_gas(gamma):
    with pytest.raises(ValueError):
        isentropic(2, gamma)
