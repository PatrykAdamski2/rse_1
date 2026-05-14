"""Ballistics: projectile motion calculations with unit support.

Combines Numba JIT-compilation for fast trajectory integration
and Pint for unit-aware inputs and outputs.
"""

from .core import trajectory, max_range, max_height

__all__ = ["trajectory", "max_range", "max_height"]
