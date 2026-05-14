# ballistics2

> Projectile motion calculations with **Numba** JIT-compilation and **Pint** unit support.

[![pytest](https://github.com/PatrykAdamski2/rse_1/actions/workflows/pytest.yml/badge.svg)](https://github.com/PatrykAdamski2/rse_1/actions/workflows/pytest.yml)
[![docs](https://github.com/PatrykAdamski2/rse_1/actions/workflows/pdoc.yml/badge.svg)](https://PatrykAdamski2.github.io/rse_1)
[![PyPI](https://img.shields.io/badge/test.pypi-ballistics2-blue)](https://test.pypi.org/project/ballistics2/)

## Install

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple ballistics2
```

## Quick start

```python
import pint
import ballistics

ureg = pint.UnitRegistry()

t, x, y = ballistics.trajectory(
    angle=ureg.Quantity(45, "degree"),
    v0=ureg.Quantity(100, "m/s"),
)
print(f"Range: {x[-1]:.1f}")
print(f"Max height: {ballistics.max_height(ureg.Quantity(45,'degree'), ureg.Quantity(100,'m/s')):.1f}")
```

## API

| Function | Description |
|---|---|
| `trajectory(angle, v0)` | Full (t, x, y) arrays |
| `max_range(angle, v0)` | Horizontal range |
| `max_height(angle, v0)` | Peak altitude |

## Demo notebook

Open in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PatrykAdamski2/rse_1/blob/main/demo.ipynb)

## License

MIT