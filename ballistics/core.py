"""Core projectile motion calculations.

Uses Numba for JIT-compiled trajectory integration and
Pint for unit-aware public API.

Example
-------
>>> import pint
>>> ureg = pint.UnitRegistry()
>>> t, x, y = trajectory(ureg.Quantity(45, "degree"), ureg.Quantity(100, "m/s"))
>>> float(x[-1].to("m").magnitude) > 900
True
"""

import math
import numpy as np
import numba
import pint

ureg = pint.UnitRegistry()


@numba.njit
def _integrate(v0: float, angle_rad: float, g: float, dt: float):
    """JIT-compiled Euler integration of projectile motion (SI units).

    Parameters
    ----------
    v0 : float
        Initial speed in m/s.
    angle_rad : float
        Launch angle in radians.
    g : float
        Gravitational acceleration in m/s².
    dt : float
        Time step in seconds.

    Returns
    -------
    t_arr, x_arr, y_arr : numpy arrays
        Time, horizontal and vertical position arrays (SI).
    """
    vx = v0 * math.cos(angle_rad)
    vy = v0 * math.sin(angle_rad)
    x, y = 0.0, 0.0
    t = 0.0

    t_list = [t]
    x_list = [x]
    y_list = [y]

    while y >= 0.0 or t == 0.0:
        x += vx * dt
        vy -= g * dt
        y += vy * dt
        t += dt
        if y < 0.0:
            break
        t_list.append(t)
        x_list.append(x)
        y_list.append(y)

    return (
        np.array(t_list),
        np.array(x_list),
        np.array(y_list),
    )


def trajectory(
    angle: "pint.Quantity",
    v0: "pint.Quantity",
    g: "pint.Quantity" = ureg.Quantity(9.81, "m/s**2"),
    dt: "pint.Quantity" = ureg.Quantity(0.01, "s"),
):
    """Compute the full trajectory of a projectile.

    Parameters
    ----------
    angle : pint.Quantity
        Launch angle (any angular unit, e.g. degrees or radians).
    v0 : pint.Quantity
        Initial speed (any speed unit, e.g. m/s or km/h).
    g : pint.Quantity, optional
        Gravitational acceleration. Defaults to 9.81 m/s².
    dt : pint.Quantity, optional
        Integration time step. Defaults to 0.01 s.

    Returns
    -------
    t : pint.Quantity
        Time array.
    x : pint.Quantity
        Horizontal position array.
    y : pint.Quantity
        Vertical position array.

    Examples
    --------
    >>> import pint
    >>> ureg = pint.UnitRegistry()
    >>> t, x, y = trajectory(ureg.Quantity(45, "degree"), ureg.Quantity(50, "m/s"))
    >>> y.units
    <Unit('meter')>
    """
    angle_rad = angle.to("radian").magnitude
    v0_si = v0.to("m/s").magnitude
    g_si = g.to("m/s**2").magnitude
    dt_si = dt.to("s").magnitude

    t_arr, x_arr, y_arr = _integrate(v0_si, angle_rad, g_si, dt_si)

    return (
        ureg.Quantity(t_arr, "s"),
        ureg.Quantity(x_arr, "m"),
        ureg.Quantity(y_arr, "m"),
    )


def max_range(
    angle: "pint.Quantity",
    v0: "pint.Quantity",
    g: "pint.Quantity" = ureg.Quantity(9.81, "m/s**2"),
) -> "pint.Quantity":
    """Return the horizontal range of the projectile.

    Parameters
    ----------
    angle : pint.Quantity
        Launch angle.
    v0 : pint.Quantity
        Initial speed.
    g : pint.Quantity, optional
        Gravitational acceleration. Defaults to 9.81 m/s².

    Returns
    -------
    pint.Quantity
        Horizontal range in metres.

    Examples
    --------
    >>> import pint
    >>> ureg = pint.UnitRegistry()
    >>> r = max_range(ureg.Quantity(45, "degree"), ureg.Quantity(100, "m/s"))
    >>> abs(float(r.to("m").magnitude) - 1019.4) < 1.0
    True
    """
    angle_rad = angle.to("radian").magnitude
    v0_si = v0.to("m/s").magnitude
    g_si = g.to("m/s**2").magnitude
    r = v0_si**2 * math.sin(2 * angle_rad) / g_si
    return ureg.Quantity(r, "m")


def max_height(
    angle: "pint.Quantity",
    v0: "pint.Quantity",
    g: "pint.Quantity" = ureg.Quantity(9.81, "m/s**2"),
) -> "pint.Quantity":
    """Return the maximum height reached by the projectile.

    Parameters
    ----------
    angle : pint.Quantity
        Launch angle.
    v0 : pint.Quantity
        Initial speed.
    g : pint.Quantity, optional
        Gravitational acceleration. Defaults to 9.81 m/s².

    Returns
    -------
    pint.Quantity
        Maximum height in metres.

    Examples
    --------
    >>> import pint
    >>> ureg = pint.UnitRegistry()
    >>> h = max_height(ureg.Quantity(90, "degree"), ureg.Quantity(10, "m/s"))
    >>> abs(float(h.to("m").magnitude) - 5.1) < 0.1
    True
    """
    angle_rad = angle.to("radian").magnitude
    v0_si = v0.to("m/s").magnitude
    g_si = g.to("m/s**2").magnitude
    h = (v0_si * math.sin(angle_rad)) ** 2 / (2 * g_si)
    return ureg.Quantity(h, "m")
