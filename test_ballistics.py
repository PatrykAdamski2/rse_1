"""Unit tests for the ballistics package."""

import math
import pytest
import pint
import ballistics

ureg = pint.UnitRegistry()


# ── max_range ────────────────────────────────────────────────────────────────

def test_max_range_analytical():
    """Range at 45° matches analytical formula v²/g."""
    # arrange
    v0 = ureg.Quantity(100, "m/s")
    angle = ureg.Quantity(45, "degree")
    expected = v0.magnitude ** 2 / 9.81   # m

    # act
    result = ballistics.max_range(angle, v0)

    # assert
    assert abs(result.to("m").magnitude - expected) < 0.5


def test_max_range_symmetry():
    """Range is symmetric: angle θ gives same range as 90°-θ."""
    # arrange
    v0 = ureg.Quantity(50, "m/s")
    a1 = ureg.Quantity(30, "degree")
    a2 = ureg.Quantity(60, "degree")

    # act
    r1 = ballistics.max_range(a1, v0).to("m").magnitude
    r2 = ballistics.max_range(a2, v0).to("m").magnitude

    # assert
    assert abs(r1 - r2) < 0.01


def test_max_range_unit_conversion():
    """Result is the same whether v0 is given in m/s or km/h."""
    # arrange
    v0_ms = ureg.Quantity(72, "m/s")
    v0_kmh = ureg.Quantity(259.2, "km/h")   # 72 m/s exactly
    angle = ureg.Quantity(30, "degree")

    # act
    r_ms = ballistics.max_range(angle, v0_ms).to("m").magnitude
    r_kmh = ballistics.max_range(angle, v0_kmh).to("m").magnitude

    # assert
    assert abs(r_ms - r_kmh) < 0.01


# ── max_height ───────────────────────────────────────────────────────────────

def test_max_height_vertical():
    """Straight-up shot: height = v²/(2g)."""
    # arrange
    v0 = ureg.Quantity(20, "m/s")
    angle = ureg.Quantity(90, "degree")
    expected = v0.magnitude ** 2 / (2 * 9.81)

    # act
    result = ballistics.max_height(angle, v0)

    # assert
    assert abs(result.to("m").magnitude - expected) < 0.01


def test_max_height_zero_at_horizontal():
    """Horizontal shot (0°) has zero height."""
    # arrange
    v0 = ureg.Quantity(50, "m/s")
    angle = ureg.Quantity(0, "degree")

    # act
    h = ballistics.max_height(angle, v0).to("m").magnitude

    # assert
    assert abs(h) < 1e-9


# ── trajectory ───────────────────────────────────────────────────────────────

def test_trajectory_starts_at_origin():
    """Trajectory begins at (0, 0)."""
    # arrange
    v0 = ureg.Quantity(30, "m/s")
    angle = ureg.Quantity(45, "degree")

    # act
    t, x, y = ballistics.trajectory(angle, v0)

    # assert
    assert x[0].to("m").magnitude == pytest.approx(0.0)
    assert y[0].to("m").magnitude == pytest.approx(0.0)


def test_trajectory_stays_non_negative():
    """All y-values in trajectory are >= 0."""
    # arrange
    v0 = ureg.Quantity(40, "m/s")
    angle = ureg.Quantity(60, "degree")

    # act
    _, _, y = ballistics.trajectory(angle, v0)

    # assert
    assert all(yi.magnitude >= -1e-6 for yi in y)


def test_trajectory_x_range_matches_analytical():
    """Final x of trajectory agrees with analytical range within 1%."""
    # arrange
    v0 = ureg.Quantity(80, "m/s")
    angle = ureg.Quantity(45, "degree")

    # act
    _, x, _ = ballistics.trajectory(angle, v0)
    simulated_range = x[-1].to("m").magnitude
    analytical_range = ballistics.max_range(angle, v0).to("m").magnitude

    # assert – 1% tolerance due to discrete time step
    assert abs(simulated_range - analytical_range) / analytical_range < 0.01


def test_trajectory_has_unit_quantities():
    """Trajectory outputs are Pint Quantity objects."""
    # arrange / act
    t, x, y = ballistics.trajectory(
        ureg.Quantity(45, "degree"),
        ureg.Quantity(10, "m/s"),
    )

    # assert
    assert isinstance(t, pint.UnitRegistry.Quantity if False else pint.facets.plain.quantity.PlainQuantity)
    assert str(t.units) == "second"
    assert str(x.units) == "meter"
    assert str(y.units) == "meter"
