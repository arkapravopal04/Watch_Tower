import pytest

from pcl.data.grader import grade_answer

REF = {'L': {'value': 7656.25, 'unit': 'N', 'atol': 0.001}}


@pytest.mark.parametrize('answer', ['FINAL: L=7656.25 N', 'Explanation\nFINAL: L=7.65625 kN',
                                    'FINAL: L=7.65625e3 N'])
def test_equivalent_units(answer):
    assert grade_answer(answer, REF).correct


@pytest.mark.parametrize('answer', ['FINAL: L=7656.25 Pa', 'FINAL: L=NaN N', 'FINAL: L=inf N',
                                    'FINAL: L=1e999 N', 'FINAL: L=7656.25', 'FINAL: L=7656.25 N; L=1 N',
                                    'FINAL: L=7656.25 N\nFINAL: L=7656.25 N',
                                    'FINAL: L=7656.25 N\nmore', 'L=7656.25 N',
                                    'FINAL: L=7656.25 N; extra=1', 'FINAL: L=7000 N'])
def test_bad_answers_fail(answer):
    assert not grade_answer(answer, REF).correct


def test_dimensionless_and_status():
    assert grade_answer('FINAL: M=2', {'M': {'value': 2, 'unit': '1', 'atol': 1e-6}}).correct
    assert not grade_answer('FINAL: M=2 m/s', {'M': {'value': 2, 'unit': '1', 'atol': 1e-6}}).correct
    assert grade_answer('FINAL: status=missing_input; missing=CL', {'status': 'missing_input', 'missing': 'CL'}).correct


def test_nonfinite_reference_cannot_silently_pass():
    with pytest.raises(ValueError, match='reference'):
        grade_answer('FINAL: L=1 N', {'L': {'value': float('nan'), 'unit': 'N', 'atol': 0.001}})
