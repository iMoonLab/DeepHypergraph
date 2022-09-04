from cmath import pi
import math
from math import pi
import pytest

import dhg.visualization.structure.geometry as g

def test_radian_from_atan():
    assert g.radian_from_atan(1, 0) == 0
    assert g.radian_from_atan(1, 1) == pytest.approx(pi / 4)
    assert g.radian_from_atan(0, 1) == pytest.approx(pi / 2)
    assert g.radian_from_atan(-1, 1) == pytest.approx(3 * pi / 4)
    assert g.radian_from_atan(-1, 0) == pytest.approx(pi)
    assert g.radian_from_atan(-1, -1) == pytest.approx(5 * pi / 4)
    assert g.radian_from_atan(0, -1) == pytest.approx(3 * pi / 2)
    assert g.radian_from_atan(1, -1) == pytest.approx(7 * pi / 4)
