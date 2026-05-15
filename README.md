# primewords

![Primewords logo](assets/logo.png)

`primewords` renders prime numbers as dot graphs and visualises their structure
in 2D grids, interactive 3D cubes, side snapshots, and straight-line prime-run
analysis. Numbers are laid out row by row; prime numbers become bright dots, and
non-primes remain dark.

## Example

Generate a small prime graph image:

```bash
uv run python Examples/Primes/generate_img.py
```

That writes `Examples/Primes/primes.png`.

## Project Layout

- `src/primewords/primes.py` contains the reusable package code for rendering
  prime-dot PNGs, generating interactive 3D prime cubes, and ranking
  straight-line prime runs.
- `Examples/Primes/generate_img.py` creates a small prime graph image.
- `Examples/Primes/generate_3d_cube.py` creates a Plotly-powered 3D prime cube.
- `Examples/Primes/generate_3d_cube_side_snapshots.py` renders orthographic PNG
  snapshots of prime cubes.
- `Examples/Primes/analyse_line_length.py` ranks graph widths by the longest
  contiguous prime lines.
- `Examples/Primes/search_zero_white_box.py` searches cube dimensions for
  projections with no empty cells.

## Setup With uv

This project is configured as a `uv` package project and pins local development
to Python 3.13 with `.python-version`. From the repository root:

```bash
uv sync --python 3.13
```

That creates a Python 3.13 virtual environment and installs the dependencies
declared in `pyproject.toml`:

- `numpy`
- `pandas`
- `pillow`

Run a quick package smoke test:

```bash
uv run python -c "from primewords import estimate_png_dimensions; print(estimate_png_dimensions(30, 100))"
```

Generate the small example prime graph:

```bash
uv run python Examples/Primes/generate_img.py
```

Generate an interactive 3D prime cube:

```bash
uv run python Examples/Primes/generate_3d_cube.py
```

That writes `Examples/Primes/primes_3d_cube.html`, a standalone Plotly scene
with controls for prime opacity, non-prime opacity, marker size, labels, grid
visibility, and camera reset. The default cube is `3 x 3 x 3`: `1` sits above
`10` and `19`, `2` sits above `11` and `20`, and `9` sits above `18` and `27`.
The HTML loads Plotly.js from the CDN, so no extra Python dependency is needed.

Use `--square-dots` when you want multi-pixel prime cells rendered as filled
squares instead of round dots.

## API Example

```python
from primewords import (
  generate_prime_cube_plot_html,
  generate_prime_dot_png,
  rank_widths_by_prime_lines,
)

metadata = generate_prime_dot_png(
    width=30,
    max_number=100,
    output_path="Examples/Primes/primes.png",
    cell_size=1,
)
print(metadata)

best_lines = rank_widths_by_prime_lines(
    range(2, 100),
    max_number=100_000,
    min_length=5,
    workers=1,
)
print(best_lines[:5])

cube = generate_prime_cube_plot_html(
  output_path="Examples/Primes/primes_3d_cube.html",
  plane_width=3,
  plane_height=3,
  layers=3,
)
print(cube)
```
