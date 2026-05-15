"""Render prime numbers as dot grids and stream the result to PNG.

The graph layout is row-based: with ``width=10``, numbers 1-10 are in
the first row, 11-20 in the second row, and so on. Prime numbers are
drawn as bright dots, and non-primes remain dark.

The renderer is built for large upper bounds. It uses a segmented sieve
and writes PNG scanlines directly instead of building a full image in
memory. A 1,000,000,000-number plot with ``width=50`` is still a very
large image, but memory usage stays bounded by the segment size rather
than the final pixel count.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from html import escape
import json
from math import isqrt
import os
from pathlib import Path
import struct
from typing import Any, BinaryIO, Callable, Iterable, Sequence, TypedDict
import zlib

PNG_MAX_DIMENSION = 2_147_483_647
DEFAULT_SEGMENT_SIZE = 8_000_000
DEFAULT_IDAT_CHUNK_SIZE = 1_048_576
DEFAULT_PLOTLY_CDN_URL = "https://cdn.plot.ly/plotly-2.35.2.min.js"

ProgressCallback = Callable[[int, int, int], None]
LineProgressCallback = Callable[[int, int, int], None]


class _PrimeCubeCellData(TypedDict):
    x: list[float]
    y: list[float]
    z: list[float]
    text: list[str]
    hovertext: list[str]


class _PrimeCubeLabelData(TypedDict):
    x: list[float]
    y: list[float]
    z: list[float]
    text: list[str]


class _PrimeCubeCellGroups(TypedDict):
    primes: _PrimeCubeCellData
    composites: _PrimeCubeCellData
    labels: _PrimeCubeLabelData


class _PrimeCubeMeshData(TypedDict):
    x: list[float]
    y: list[float]
    z: list[float]
    i: list[int]
    j: list[int]
    k: list[int]
    text: list[str]
    hovertext: list[str]


LINE_DIRECTIONS = (
    "horizontal",
    "vertical",
    "diagonal_down_right",
    "diagonal_down_left",
)


@dataclass(frozen=True)
class PrimeDotImage:
    """Metadata for a generated prime-dot image."""

    output_path: Path
    number_width: int
    max_number: int
    number_rows: int
    image_width: int
    image_height: int
    cell_size: int
    primes_plotted: int


@dataclass(frozen=True)
class PrimeCubeHtml:
    """Metadata for a generated interactive prime-cube HTML file."""

    output_path: Path
    plane_width: int
    plane_height: int
    layers: int
    max_number: int
    primes_plotted: int
    composites_plotted: int
    longest_prime_line: tuple[int, ...]
    longest_prime_line_direction: tuple[int, int, int]


@dataclass(frozen=True)
class PrimeLine:
    """A contiguous visual line of primes in a row-based prime grid."""

    width: int
    max_number: int
    direction: str
    step: int
    length: int
    start_number: int
    end_number: int
    start_row: int
    start_column: int
    end_row: int
    end_column: int

    def numbers(self) -> range:
        """Return the arithmetic progression represented by this line."""

        return range(self.start_number, self.end_number + self.step, self.step)


@dataclass(frozen=True)
class PrimeLineWorkload:
    """Estimated work for scanning prime-line lengths across widths."""

    width_count: int
    min_width: int | None
    max_width: int | None
    max_number: int
    min_length: int
    directions: tuple[str, ...]
    segment_size: int
    total_number_slots: int
    total_direction_checks: int
    total_rows: int
    total_segment_sieves: int
    estimated_peak_bytes_per_worker: int
    suggested_workers: int

    def as_dict(self) -> dict[str, int | tuple[str, ...] | None]:
        """Return a flat dictionary suitable for printing or DataFrames."""

        return {
            "width_count": self.width_count,
            "min_width": self.min_width,
            "max_width": self.max_width,
            "max_number": self.max_number,
            "min_length": self.min_length,
            "directions": self.directions,
            "segment_size": self.segment_size,
            "total_number_slots": self.total_number_slots,
            "total_direction_checks": self.total_direction_checks,
            "total_rows": self.total_rows,
            "total_segment_sieves": self.total_segment_sieves,
            "estimated_peak_bytes_per_worker": self.estimated_peak_bytes_per_worker,
            "suggested_workers": self.suggested_workers,
        }


def generate_prime_dot_png(
    width: int,
    max_number: int,
    output_path: str | Path,
    *,
    cell_size: int = 1,
    dot_radius: float | None = None,
    prime_value: int = 255,
    background_value: int = 0,
    segment_size: int = DEFAULT_SEGMENT_SIZE,
    compression: int = 6,
    progress: ProgressCallback | None = None,
) -> PrimeDotImage:
    """Generate a PNG showing primes up to ``max_number``.

    Args:
            width: How many integers appear in each row of the graph.
            max_number: Highest integer included in the graph.
            output_path: Destination PNG file.
            cell_size: Pixel size of each integer cell. Use 1 for huge plots.
            dot_radius: Dot radius within each cell when ``cell_size`` is above 1.
            prime_value: 8-bit grayscale value for prime dots.
            background_value: 8-bit grayscale value for non-prime cells.
            segment_size: Number of integers sieved at once.
            compression: zlib compression level, from 0 to 9.
            progress: Optional callback receiving
                    ``(rows_done, total_rows, primes_plotted)`` after each segment.

    Returns:
            Metadata describing the generated image.
    """

    _validate_positive_integer(width, "width")
    _validate_positive_integer(max_number, "max_number")
    _validate_positive_integer(cell_size, "cell_size")
    _validate_positive_integer(segment_size, "segment_size")
    _validate_grayscale_value(prime_value, "prime_value")
    _validate_grayscale_value(background_value, "background_value")

    if not 0 <= compression <= 9:
        raise ValueError("compression must be between 0 and 9")
    if dot_radius is not None and dot_radius <= 0:
        raise ValueError("dot_radius must be positive when provided")

    number_rows = (max_number + width - 1) // width
    image_width, image_height = estimate_png_dimensions(
        width=width,
        max_number=max_number,
        cell_size=cell_size,
    )

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    base_primes = _base_primes_upto(isqrt(max_number))
    rows_per_segment = max(1, segment_size // width)
    translation_table = _flag_to_grayscale_table(
        prime_value=prime_value,
        background_value=background_value,
    )
    dot_patterns = _dot_patterns(
        cell_size=cell_size,
        dot_radius=dot_radius,
        prime_value=prime_value,
        background_value=background_value,
    )

    primes_plotted = 0

    with output.open("wb") as file:
        writer = _StreamingPngWriter(
            file=file,
            width=image_width,
            height=image_height,
            compression=compression,
        )

        for row_start in range(0, number_rows, rows_per_segment):
            row_stop = min(number_rows, row_start + rows_per_segment)
            first_number = row_start * width + 1
            last_number = min(row_stop * width, max_number)
            flags = _prime_flags_for_range(
                start=first_number,
                stop=last_number + 1,
                base_primes=base_primes,
            )
            primes_plotted += flags.count(1)

            for row_index in range(row_start, row_stop):
                number_at_row_start = row_index * width + 1
                offset = number_at_row_start - first_number
                row_flags = flags[offset : offset + width]

                if len(row_flags) < width:
                    row_flags.extend(b"\x00" * (width - len(row_flags)))

                _write_prime_row(
                    writer=writer,
                    row_flags=row_flags,
                    cell_size=cell_size,
                    translation_table=translation_table,
                    dot_patterns=dot_patterns,
                )

            if progress is not None:
                progress(row_stop, number_rows, primes_plotted)

        writer.close()

    return PrimeDotImage(
        output_path=output,
        number_width=width,
        max_number=max_number,
        number_rows=number_rows,
        image_width=image_width,
        image_height=image_height,
        cell_size=cell_size,
        primes_plotted=primes_plotted,
    )


def generate_graph_dots(
    width: int,
    max_number: int,
    output_path: str | Path,
    **kwargs: Any,
) -> PrimeDotImage:
    """Compatibility wrapper around :func:`generate_prime_dot_png`."""

    return generate_prime_dot_png(
        width=width,
        max_number=max_number,
        output_path=output_path,
        **kwargs,
    )


def generate_prime_cube_plot_html(
    output_path: str | Path,
    *,
    plane_width: int = 3,
    plane_height: int = 3,
    layers: int | None = 3,
    max_number: int | None = None,
    title: str | None = None,
    plotly_cdn_url: str = DEFAULT_PLOTLY_CDN_URL,
) -> PrimeCubeHtml:
    """Generate an interactive Plotly HTML view of primes in a 3D grid.

    Numbers fill each horizontal plane row by row, then continue downward to
    the next plane. With the default ``3 x 3 x 3`` layout, ``1`` is directly
    above ``10`` and ``19``, ``2`` is directly above ``11`` and ``20``, and
    ``9`` is directly above ``18`` and ``27``.
    """

    _validate_positive_integer(plane_width, "plane_width")
    _validate_positive_integer(plane_height, "plane_height")

    plane_size = plane_width * plane_height
    if max_number is None:
        if layers is None:
            raise ValueError("layers is required when max_number is omitted")
        _validate_positive_integer(layers, "layers")
        max_number = plane_size * layers
    else:
        _validate_positive_integer(max_number, "max_number")
        required_layers = (max_number + plane_size - 1) // plane_size
        if layers is None:
            layers = required_layers
        else:
            _validate_positive_integer(layers, "layers")
            if max_number > plane_size * layers:
                raise ValueError(
                    "max_number does not fit inside the requested 3D grid; "
                    "increase layers or pass layers=None"
                )

    assert layers is not None
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    base_primes = _base_primes_upto(isqrt(max_number))
    prime_flags = _prime_flags_for_range(
        start=1,
        stop=max_number + 1,
        base_primes=base_primes,
    )
    cells = _prime_cube_cells(
        prime_flags=prime_flags,
        plane_width=plane_width,
        plane_height=plane_height,
        layers=layers,
    )
    longest_line_numbers, longest_line_direction = _longest_prime_cube_line(
        prime_flags=prime_flags,
        plane_width=plane_width,
        plane_height=plane_height,
        layers=layers,
    )
    grid = _prime_cube_grid_lines(
        plane_width=plane_width,
        plane_height=plane_height,
        layers=layers,
    )
    document = _prime_cube_html_document(
        title=title or f"Prime Cube {plane_width}x{plane_height}x{layers}",
        plane_width=plane_width,
        plane_height=plane_height,
        layers=layers,
        max_number=max_number,
        cells=cells,
        longest_line_numbers=longest_line_numbers,
        longest_line_direction=longest_line_direction,
        grid=grid,
        plotly_cdn_url=plotly_cdn_url,
    )
    output.write_text(document, encoding="utf-8")

    return PrimeCubeHtml(
        output_path=output,
        plane_width=plane_width,
        plane_height=plane_height,
        layers=layers,
        max_number=max_number,
        primes_plotted=len(cells["primes"]["x"]),
        composites_plotted=len(cells["composites"]["x"]),
        longest_prime_line=longest_line_numbers,
        longest_prime_line_direction=longest_line_direction,
    )


def estimate_png_dimensions(
    width: int,
    max_number: int,
    *,
    cell_size: int = 1,
) -> tuple[int, int]:
    """Return ``(pixel_width, pixel_height)`` for a prime-dot graph."""

    _validate_positive_integer(width, "width")
    _validate_positive_integer(max_number, "max_number")
    _validate_positive_integer(cell_size, "cell_size")

    number_rows = (max_number + width - 1) // width
    image_width = width * cell_size
    image_height = number_rows * cell_size

    if image_width > PNG_MAX_DIMENSION or image_height > PNG_MAX_DIMENSION:
        raise ValueError(
            "PNG dimensions are too large. Reduce max_number/cell_size, "
            "increase width, or generate tiled images."
        )

    return image_width, image_height


def analyze_prime_lines(
    width: int,
    max_number: int,
    *,
    directions: Sequence[str] = LINE_DIRECTIONS,
    min_length: int = 2,
    segment_size: int = DEFAULT_SEGMENT_SIZE,
    progress: LineProgressCallback | None = None,
) -> tuple[PrimeLine, ...]:
    """Measure the longest contiguous prime lines for a graph width.

    Visual directions map to arithmetic progressions:
    ``horizontal`` has step 1, ``vertical`` has step ``width``,
    ``diagonal_down_right`` has step ``width + 1``, and
    ``diagonal_down_left`` has step ``width - 1``.

    This is exact for the requested ``max_number`` and streams prime rows
    through a segmented sieve. Runtime is proportional to ``max_number`` for
    each width; memory is proportional to ``width`` plus ``segment_size``.
    """

    _validate_positive_integer(width, "width")
    _validate_positive_integer(max_number, "max_number")
    _validate_positive_integer(min_length, "min_length")
    _validate_positive_integer(segment_size, "segment_size")

    requested_directions = _normalize_line_directions(directions, width=width)
    if not requested_directions:
        return ()

    number_rows = (max_number + width - 1) // width
    rows_per_segment = max(1, segment_size // width)
    base_primes = _base_primes_upto(isqrt(max_number))
    best_by_direction: dict[str, PrimeLine] = {}

    vertical_runs = [0] * width if "vertical" in requested_directions else []
    previous_down_right = (
        [0] * width if "diagonal_down_right" in requested_directions else []
    )
    previous_down_left = (
        [0] * width if "diagonal_down_left" in requested_directions else []
    )

    for row_start in range(0, number_rows, rows_per_segment):
        row_stop = min(number_rows, row_start + rows_per_segment)
        first_number = row_start * width + 1
        last_number = min(row_stop * width, max_number)
        flags = _prime_flags_for_range(
            start=first_number,
            stop=last_number + 1,
            base_primes=base_primes,
        )

        for row_index in range(row_start, row_stop):
            number_at_row_start = row_index * width + 1
            offset = number_at_row_start - first_number
            row_flags = flags[offset : offset + width]

            if len(row_flags) < width:
                row_flags.extend(b"\x00" * (width - len(row_flags)))

            if "horizontal" in requested_directions:
                _measure_horizontal_row(
                    best_by_direction=best_by_direction,
                    row_flags=row_flags,
                    row_index=row_index,
                    width=width,
                    max_number=max_number,
                    min_length=min_length,
                )

            if "vertical" in requested_directions:
                _measure_vertical_row(
                    best_by_direction=best_by_direction,
                    row_flags=row_flags,
                    row_index=row_index,
                    width=width,
                    max_number=max_number,
                    min_length=min_length,
                    vertical_runs=vertical_runs,
                )

            if "diagonal_down_right" in requested_directions:
                current_down_right = [0] * width
                _measure_diagonal_row(
                    best_by_direction=best_by_direction,
                    row_flags=row_flags,
                    row_index=row_index,
                    width=width,
                    max_number=max_number,
                    min_length=min_length,
                    direction="diagonal_down_right",
                    dx=1,
                    previous_runs=previous_down_right,
                    current_runs=current_down_right,
                )
                previous_down_right = current_down_right

            if "diagonal_down_left" in requested_directions:
                current_down_left = [0] * width
                _measure_diagonal_row(
                    best_by_direction=best_by_direction,
                    row_flags=row_flags,
                    row_index=row_index,
                    width=width,
                    max_number=max_number,
                    min_length=min_length,
                    direction="diagonal_down_left",
                    dx=-1,
                    previous_runs=previous_down_left,
                    current_runs=current_down_left,
                )
                previous_down_left = current_down_left

        if progress is not None:
            progress(row_stop, number_rows, len(best_by_direction))

    return tuple(
        sorted(
            best_by_direction.values(),
            key=lambda line: (-line.length, line.direction, line.start_number),
        )
    )


def rank_widths_by_prime_lines(
    widths: Iterable[int],
    max_number: int,
    *,
    directions: Sequence[str] = LINE_DIRECTIONS,
    min_length: int = 2,
    segment_size: int = DEFAULT_SEGMENT_SIZE,
    top_n: int | None = 20,
    workers: int | None = 1,
    chunk_size: int = 1,
) -> tuple[PrimeLine, ...]:
    """Rank widths by their best exact prime-line length.

    Set ``workers`` above 1 to parallelize across widths. Pass
    ``workers=None`` to let :class:`ProcessPoolExecutor` choose the worker
    count, usually one per available CPU.
    """

    _validate_positive_integer(max_number, "max_number")
    _validate_positive_integer(min_length, "min_length")
    _validate_positive_integer(segment_size, "segment_size")
    _validate_positive_integer(chunk_size, "chunk_size")
    if top_n is not None and top_n < 1:
        raise ValueError("top_n must be positive or None")
    if workers is not None and workers < 1:
        raise ValueError("workers must be positive or None")

    width_list = tuple(widths)
    best_lines: list[PrimeLine] = []

    if workers == 1 or len(width_list) < 2:
        for width in width_list:
            lines = analyze_prime_lines(
                width=width,
                max_number=max_number,
                directions=directions,
                min_length=min_length,
                segment_size=segment_size,
            )
            if lines:
                best_lines.append(lines[0])
    else:
        worker_args = (
            (width, max_number, tuple(directions), min_length, segment_size)
            for width in width_list
        )
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for line in executor.map(
                _best_prime_line_for_width,
                worker_args,
                chunksize=chunk_size,
            ):
                if line is not None:
                    best_lines.append(line)

    ranked = sorted(
        best_lines,
        key=lambda line: (-line.length, line.width, line.direction, line.start_number),
    )
    if top_n is None:
        return tuple(ranked)
    return tuple(ranked[:top_n])


def estimate_prime_line_workload(
    widths: Iterable[int],
    max_number: int,
    *,
    directions: Sequence[str] = LINE_DIRECTIONS,
    min_length: int = 2,
    segment_size: int = DEFAULT_SEGMENT_SIZE,
    workers: int | None = None,
) -> PrimeLineWorkload:
    """Estimate exact line-scan work before running width rankings.

    ``total_number_slots`` is the count of integer positions scanned by the
    segmented sieve across all widths. ``total_direction_checks`` is the count
    of visual line cells examined after applying direction aliases for each
    width. These are estimates of work size, not runtime guarantees.
    """

    _validate_positive_integer(max_number, "max_number")
    _validate_positive_integer(min_length, "min_length")
    _validate_positive_integer(segment_size, "segment_size")
    if workers is not None and workers < 1:
        raise ValueError("workers must be positive or None")

    width_list = tuple(widths)
    for width in width_list:
        _validate_positive_integer(width, "width")

    if not width_list:
        suggested_workers = 0
        return PrimeLineWorkload(
            width_count=0,
            min_width=None,
            max_width=None,
            max_number=max_number,
            min_length=min_length,
            directions=tuple(directions),
            segment_size=segment_size,
            total_number_slots=0,
            total_direction_checks=0,
            total_rows=0,
            total_segment_sieves=0,
            estimated_peak_bytes_per_worker=0,
            suggested_workers=suggested_workers,
        )

    total_rows = 0
    total_segment_sieves = 0
    total_direction_checks = 0

    for width in width_list:
        requested_directions = _normalize_line_directions(directions, width=width)
        rows = (max_number + width - 1) // width
        rows_per_segment = max(1, segment_size // width)
        total_rows += rows
        total_segment_sieves += (rows + rows_per_segment - 1) // rows_per_segment
        total_direction_checks += max_number * len(requested_directions)

    max_width = max(width_list)
    direction_slots = len(_normalize_line_directions(directions, width=max_width))
    run_arrays = max(1, direction_slots - 1)
    estimated_peak_bytes_per_worker = segment_size + max_width * run_arrays * 8
    available_cpus = os.cpu_count() or 1
    suggested_workers = min(len(width_list), workers or available_cpus)

    return PrimeLineWorkload(
        width_count=len(width_list),
        min_width=min(width_list),
        max_width=max_width,
        max_number=max_number,
        min_length=min_length,
        directions=tuple(directions),
        segment_size=segment_size,
        total_number_slots=len(width_list) * max_number,
        total_direction_checks=total_direction_checks,
        total_rows=total_rows,
        total_segment_sieves=total_segment_sieves,
        estimated_peak_bytes_per_worker=estimated_peak_bytes_per_worker,
        suggested_workers=suggested_workers,
    )


def _best_prime_line_for_width(
    args: tuple[int, int, tuple[str, ...], int, int],
) -> PrimeLine | None:
    width, max_number, directions, min_length, segment_size = args
    lines = analyze_prime_lines(
        width=width,
        max_number=max_number,
        directions=directions,
        min_length=min_length,
        segment_size=segment_size,
    )
    return lines[0] if lines else None


def _normalize_line_directions(
    directions: Sequence[str],
    *,
    width: int,
) -> tuple[str, ...]:
    aliases = {
        "h": "horizontal",
        "row": "horizontal",
        "rows": "horizontal",
        "horizontal": "horizontal",
        "v": "vertical",
        "column": "vertical",
        "columns": "vertical",
        "vertical": "vertical",
        "dr": "diagonal_down_right",
        "down_right": "diagonal_down_right",
        "diagonal_right": "diagonal_down_right",
        "diagonal_down_right": "diagonal_down_right",
        "dl": "diagonal_down_left",
        "down_left": "diagonal_down_left",
        "diagonal_left": "diagonal_down_left",
        "diagonal_down_left": "diagonal_down_left",
    }
    normalized: list[str] = []

    for direction in directions:
        if direction == "diagonal":
            candidates = ("diagonal_down_right", "diagonal_down_left")
        else:
            canonical = aliases.get(direction)
            if canonical is None:
                valid = ", ".join(LINE_DIRECTIONS)
                raise ValueError(f"unknown direction {direction!r}; expected {valid}")
            candidates = (canonical,)

        for candidate in candidates:
            if candidate == "diagonal_down_left" and width == 1:
                continue
            if candidate not in normalized:
                normalized.append(candidate)

    return tuple(normalized)


def _measure_horizontal_row(
    *,
    best_by_direction: dict[str, PrimeLine],
    row_flags: bytearray,
    row_index: int,
    width: int,
    max_number: int,
    min_length: int,
) -> None:
    run_length = 0

    for column, is_prime in enumerate(row_flags):
        if is_prime:
            run_length += 1
            _store_best_line(
                best_by_direction=best_by_direction,
                width=width,
                max_number=max_number,
                direction="horizontal",
                dx=1,
                dy=0,
                end_row=row_index,
                end_column=column,
                length=run_length,
                min_length=min_length,
            )
        else:
            run_length = 0


def _measure_vertical_row(
    *,
    best_by_direction: dict[str, PrimeLine],
    row_flags: bytearray,
    row_index: int,
    width: int,
    max_number: int,
    min_length: int,
    vertical_runs: list[int],
) -> None:
    for column, is_prime in enumerate(row_flags):
        if is_prime:
            vertical_runs[column] += 1
            _store_best_line(
                best_by_direction=best_by_direction,
                width=width,
                max_number=max_number,
                direction="vertical",
                dx=0,
                dy=1,
                end_row=row_index,
                end_column=column,
                length=vertical_runs[column],
                min_length=min_length,
            )
        else:
            vertical_runs[column] = 0


def _measure_diagonal_row(
    *,
    best_by_direction: dict[str, PrimeLine],
    row_flags: bytearray,
    row_index: int,
    width: int,
    max_number: int,
    min_length: int,
    direction: str,
    dx: int,
    previous_runs: list[int],
    current_runs: list[int],
) -> None:
    for column, is_prime in enumerate(row_flags):
        if not is_prime:
            continue

        previous_column = column - dx
        previous_length = (
            previous_runs[previous_column] if 0 <= previous_column < width else 0
        )
        current_runs[column] = previous_length + 1
        _store_best_line(
            best_by_direction=best_by_direction,
            width=width,
            max_number=max_number,
            direction=direction,
            dx=dx,
            dy=1,
            end_row=row_index,
            end_column=column,
            length=current_runs[column],
            min_length=min_length,
        )


def _store_best_line(
    *,
    best_by_direction: dict[str, PrimeLine],
    width: int,
    max_number: int,
    direction: str,
    dx: int,
    dy: int,
    end_row: int,
    end_column: int,
    length: int,
    min_length: int,
) -> None:
    if length < min_length:
        return

    start_row = end_row - dy * (length - 1)
    start_column = end_column - dx * (length - 1)
    if start_column < 0 or start_column >= width:
        return

    start_number = start_row * width + start_column + 1
    end_number = end_row * width + end_column + 1
    step = dy * width + dx
    candidate = PrimeLine(
        width=width,
        max_number=max_number,
        direction=direction,
        step=step,
        length=length,
        start_number=start_number,
        end_number=end_number,
        start_row=start_row,
        start_column=start_column,
        end_row=end_row,
        end_column=end_column,
    )
    current = best_by_direction.get(direction)

    if current is None or _line_is_better(candidate, current):
        best_by_direction[direction] = candidate


def _line_is_better(candidate: PrimeLine, current: PrimeLine) -> bool:
    return (candidate.length, -candidate.start_number) > (
        current.length,
        -current.start_number,
    )


class _StreamingPngWriter:
    def __init__(
        self,
        *,
        file: BinaryIO,
        width: int,
        height: int,
        compression: int,
        idat_chunk_size: int = DEFAULT_IDAT_CHUNK_SIZE,
    ) -> None:
        self._file = file
        self._width = width
        self._height = height
        self._rows_written = 0
        self._idat_chunk_size = idat_chunk_size
        self._idat_buffer = bytearray()
        self._compressor = zlib.compressobj(compression)
        self._closed = False

        self._file.write(b"\x89PNG\r\n\x1a\n")
        self._write_chunk(
            b"IHDR",
            struct.pack(
                ">IIBBBBB",
                width,
                height,
                8,
                0,
                0,
                0,
                0,
            ),
        )

    def write_row(self, row: bytes | bytearray) -> None:
        if self._closed:
            raise RuntimeError("cannot write to a closed PNG writer")
        if len(row) != self._width:
            raise ValueError(f"row must be exactly {self._width} bytes")
        if self._rows_written >= self._height:
            raise RuntimeError("more rows were written than the PNG height allows")

        compressed = self._compressor.compress(b"\x00" + row)
        self._append_idat(compressed)
        self._rows_written += 1

    def close(self) -> None:
        if self._closed:
            return
        if self._rows_written != self._height:
            raise RuntimeError(
                f"PNG expected {self._height} rows, got {self._rows_written}"
            )

        self._append_idat(self._compressor.flush())
        self._flush_idat()
        self._write_chunk(b"IEND", b"")
        self._closed = True

    def _append_idat(self, data: bytes) -> None:
        if not data:
            return

        self._idat_buffer.extend(data)
        while len(self._idat_buffer) >= self._idat_chunk_size:
            chunk = bytes(self._idat_buffer[: self._idat_chunk_size])
            del self._idat_buffer[: self._idat_chunk_size]
            self._write_chunk(b"IDAT", chunk)

    def _flush_idat(self) -> None:
        if self._idat_buffer:
            self._write_chunk(b"IDAT", bytes(self._idat_buffer))
            self._idat_buffer.clear()

    def _write_chunk(self, chunk_type: bytes, data: bytes) -> None:
        self._file.write(struct.pack(">I", len(data)))
        self._file.write(chunk_type)
        self._file.write(data)
        checksum = zlib.crc32(chunk_type)
        checksum = zlib.crc32(data, checksum)
        self._file.write(struct.pack(">I", checksum & 0xFFFFFFFF))


def _write_prime_row(
    *,
    writer: _StreamingPngWriter,
    row_flags: bytearray,
    cell_size: int,
    translation_table: bytes,
    dot_patterns: list[tuple[bytes, bytes]],
) -> None:
    if cell_size == 1:
        writer.write_row(row_flags.translate(translation_table))
        return

    for y in range(cell_size):
        empty_cell, prime_cell = dot_patterns[y]
        output_row = bytearray(len(row_flags) * cell_size)
        position = 0

        for is_prime in row_flags:
            cell = prime_cell if is_prime else empty_cell
            output_row[position : position + cell_size] = cell
            position += cell_size

        writer.write_row(output_row)


def _prime_flags_for_range(
    *,
    start: int,
    stop: int,
    base_primes: list[int],
) -> bytearray:
    if stop <= start:
        return bytearray()

    length = stop - start
    flags = bytearray(b"\x01") * length

    if start == 0:
        flags[0] = 0
        if length > 1:
            flags[1] = 0
    elif start == 1:
        flags[0] = 0

    first_even = start if start % 2 == 0 else start + 1
    if first_even < stop:
        offset = first_even - start
        count = _slice_count(length=length, offset=offset, step=2)
        flags[offset::2] = b"\x00" * count

    if start <= 2 < stop:
        flags[2 - start] = 1

    for prime in base_primes:
        if prime == 2:
            continue
        square = prime * prime
        if square >= stop:
            break

        first_multiple = max(square, ((start + prime - 1) // prime) * prime)
        if first_multiple % 2 == 0:
            first_multiple += prime
        if first_multiple >= stop:
            continue

        offset = first_multiple - start
        step = prime * 2
        count = _slice_count(length=length, offset=offset, step=step)
        flags[offset::step] = b"\x00" * count

    return flags


def _prime_cube_cells(
    *,
    prime_flags: bytearray,
    plane_width: int,
    plane_height: int,
    layers: int,
) -> _PrimeCubeCellGroups:
    plane_size = plane_width * plane_height
    primes: _PrimeCubeCellData = {
        "x": [],
        "y": [],
        "z": [],
        "text": [],
        "hovertext": [],
    }
    composites: _PrimeCubeCellData = {
        "x": [],
        "y": [],
        "z": [],
        "text": [],
        "hovertext": [],
    }
    labels: _PrimeCubeLabelData = {
        "x": [],
        "y": [],
        "z": [],
        "text": [],
    }

    for number, is_prime in enumerate(prime_flags, start=1):
        layer_index = (number - 1) // plane_size
        position_in_layer = (number - 1) % plane_size
        row = position_in_layer // plane_width
        column = position_in_layer % plane_width
        x = column + 0.5
        y = plane_height - row - 0.5
        z = layers - layer_index - 0.5
        target = primes if is_prime else composites
        label = str(number)
        hover_kind = "Prime" if is_prime else "Composite / non-prime"
        hovertext = (
            f"Number {number}<br>{hover_kind}<br>"
            f"Layer {layer_index + 1}, row {row + 1}, column {column + 1}"
        )

        target["x"].append(x)
        target["y"].append(y)
        target["z"].append(z)
        target["text"].append(label)
        target["hovertext"].append(hovertext)
        labels["x"].append(x)
        labels["y"].append(y)
        labels["z"].append(z)
        labels["text"].append(label)

    return {"primes": primes, "composites": composites, "labels": labels}


def _longest_prime_cube_line(
    *,
    prime_flags: bytearray,
    plane_width: int,
    plane_height: int,
    layers: int,
) -> tuple[tuple[int, ...], tuple[int, int, int]]:
    plane_size = plane_width * plane_height
    prime_coordinates: set[tuple[int, int, int]] = set()

    for number, is_prime in enumerate(prime_flags, start=1):
        if not is_prime:
            continue
        layer = (number - 1) // plane_size
        position_in_layer = (number - 1) % plane_size
        row = position_in_layer // plane_width
        column = position_in_layer % plane_width
        prime_coordinates.add((column, row, layer))

    best_numbers: tuple[int, ...] = ()
    best_direction = (0, 0, 0)

    for direction in _prime_cube_line_directions():
        dx, dy, dz = direction
        for coordinate in prime_coordinates:
            previous = (coordinate[0] - dx, coordinate[1] - dy, coordinate[2] - dz)
            if previous in prime_coordinates:
                continue

            run_coordinates: list[tuple[int, int, int]] = []
            current = coordinate
            while current in prime_coordinates:
                run_coordinates.append(current)
                current = (current[0] + dx, current[1] + dy, current[2] + dz)

            numbers = tuple(
                coordinate[2] * plane_size
                + coordinate[1] * plane_width
                + coordinate[0]
                + 1
                for coordinate in run_coordinates
            )
            if _prime_cube_line_is_better(numbers, best_numbers):
                best_numbers = numbers
                best_direction = direction

    return best_numbers, best_direction


def _prime_cube_line_directions() -> tuple[tuple[int, int, int], ...]:
    directions: list[tuple[int, int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                direction = (dx, dy, dz)
                if direction == (0, 0, 0):
                    continue
                if next(value for value in direction if value != 0) < 0:
                    continue
                directions.append(direction)
    return tuple(directions)


def _prime_cube_line_is_better(
    candidate: tuple[int, ...],
    current: tuple[int, ...],
) -> bool:
    if len(candidate) != len(current):
        return len(candidate) > len(current)
    if not current:
        return True
    return min(candidate) < min(current)


def _prime_cube_direction_label(direction: tuple[int, int, int]) -> str:
    dx, dy, dz = direction
    return f"column {dx:+d}, row {dy:+d}, layer {dz:+d}"


def _prime_cube_line_cells(
    *,
    cells: _PrimeCubeCellData,
    numbers: tuple[int, ...],
    direction: tuple[int, int, int],
) -> _PrimeCubeCellData:
    selected = {str(number) for number in numbers}
    result: _PrimeCubeCellData = {
        "x": [],
        "y": [],
        "z": [],
        "text": [],
        "hovertext": [],
    }
    line_summary = (
        f"Longest prime line<br>Length {len(numbers)}<br>"
        f"Direction {_prime_cube_direction_label(direction)}"
    )

    for x, y, z, text, hovertext in zip(
        cells["x"],
        cells["y"],
        cells["z"],
        cells["text"],
        cells["hovertext"],
    ):
        if str(text) not in selected:
            continue
        result["x"].append(float(x))
        result["y"].append(float(y))
        result["z"].append(float(z))
        result["text"].append(str(text))
        result["hovertext"].append(f"{hovertext}<br>{line_summary}")

    return result


def _prime_cube_grid_lines(
    *,
    plane_width: int,
    plane_height: int,
    layers: int,
) -> dict[str, list[int | None]]:
    grid: dict[str, list[int | None]] = {"x": [], "y": [], "z": []}

    def add_segment(
        start: tuple[int, int, int],
        end: tuple[int, int, int],
    ) -> None:
        grid["x"].extend((start[0], end[0], None))
        grid["y"].extend((start[1], end[1], None))
        grid["z"].extend((start[2], end[2], None))

    for z in range(layers + 1):
        for y in range(plane_height + 1):
            add_segment((0, y, z), (plane_width, y, z))
        for x in range(plane_width + 1):
            add_segment((x, 0, z), (x, plane_height, z))

    for y in range(plane_height + 1):
        for x in range(plane_width + 1):
            add_segment((x, y, 0), (x, y, layers))

    return grid


def _prime_cube_html_document(
    *,
    title: str,
    plane_width: int,
    plane_height: int,
    layers: int,
    max_number: int,
    cells: _PrimeCubeCellGroups,
    longest_line_numbers: tuple[int, ...],
    longest_line_direction: tuple[int, int, int],
    grid: dict[str, list[int | None]],
    plotly_cdn_url: str,
) -> str:
    prime_count = len(cells["primes"]["x"])
    composite_count = len(cells["composites"]["x"])
    scene_range_padding = 0.45
    longest_line_cells = _prime_cube_line_cells(
        cells=cells["primes"],
        numbers=longest_line_numbers,
        direction=longest_line_direction,
    )
    prime_mesh = _prime_cube_mesh_trace(
        name="Primes",
        cells=cells["primes"],
        color="#e11d48",
        opacity=0.94,
        scale=0.5,
    )
    composite_mesh = _prime_cube_mesh_trace(
        name="Composites",
        cells=cells["composites"],
        color="#64748b",
        opacity=0,
        scale=0.5,
    )
    longest_line_mesh = _prime_cube_mesh_trace(
        name="Longest Line",
        cells=longest_line_cells,
        color="#050505",
        opacity=1,
        scale=0.54,
    )
    longest_line_mesh["visible"] = True
    layout = {
        "autosize": True,
        "paper_bgcolor": "#f7f9fc",
        "plot_bgcolor": "#f7f9fc",
        "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
        "showlegend": False,
        "scene": {
            "aspectmode": "cube",
            "bgcolor": "#ffffff",
            "camera": {
                "eye": {"x": 1.45, "y": -1.6, "z": 1.25},
                "up": {"x": 0, "y": 0, "z": 1},
            },
            "xaxis": _prime_cube_axis("column", -scene_range_padding, plane_width),
            "yaxis": _prime_cube_axis("row", -scene_range_padding, plane_height),
            "zaxis": _prime_cube_axis("layer", -scene_range_padding, layers),
        },
    }
    traces = [
        {
            "type": "scatter3d",
            "mode": "lines",
            "name": "Grid",
            "x": grid["x"],
            "y": grid["y"],
            "z": grid["z"],
            "line": {"color": "#7c8797", "width": 3},
            "opacity": 0,
            "hoverinfo": "skip",
        },
        composite_mesh,
        prime_mesh,
        longest_line_mesh,
        {
            "type": "scatter3d",
            "mode": "text",
            "name": "Labels",
            "x": cells["labels"]["x"],
            "y": cells["labels"]["y"],
            "z": cells["labels"]["z"],
            "text": cells["labels"]["text"],
            "textfont": {"color": "#111827", "size": 12},
            "textposition": "middle center",
            "visible": False,
            "hoverinfo": "skip",
        },
    ]
    replacements = {
        "__TITLE__": escape(title),
        "__PLOTLY_CDN_URL__": escape(plotly_cdn_url, quote=True),
        "__SUMMARY__": escape(
            f"{plane_width}x{plane_height}x{layers} | "
            f"1-{max_number} | {prime_count} primes | {composite_count} non-primes"
        ),
        "__TRACES_JSON__": json.dumps(traces, separators=(",", ":")),
        "__LAYOUT_JSON__": json.dumps(layout, separators=(",", ":")),
        "__PRIME_CELLS_JSON__": json.dumps(cells["primes"], separators=(",", ":")),
        "__COMPOSITE_CELLS_JSON__": json.dumps(
            cells["composites"],
            separators=(",", ":"),
        ),
        "__LONGEST_LINE_CELLS_JSON__": json.dumps(
            longest_line_cells,
            separators=(",", ":"),
        ),
        "__LONGEST_LINE_TITLE__": escape(
            f"Length {len(longest_line_numbers)}: "
            f"{', '.join(str(number) for number in longest_line_numbers)} | "
            f"{_prime_cube_direction_label(longest_line_direction)}",
            quote=True,
        ),
    }
    document = _PRIME_CUBE_HTML_TEMPLATE
    for placeholder, value in replacements.items():
        document = document.replace(placeholder, value)
    return document


def _prime_cube_axis(title: str, padding: float, size: int) -> dict[str, object]:
    return {
        "title": title,
        "range": [padding, size - padding],
        "tickmode": "linear",
        "dtick": 1,
        "showbackground": True,
        "backgroundcolor": "#ffffff",
        "gridcolor": "#dde3ea",
        "zerolinecolor": "#b6c0cc",
    }


def _prime_cube_mesh_trace(
    *,
    name: str,
    cells: _PrimeCubeCellData,
    color: str,
    opacity: float,
    scale: float,
) -> dict[str, object]:
    mesh = _prime_cube_mesh(cells=cells, scale=scale)
    return {
        "type": "mesh3d",
        "name": name,
        "x": mesh["x"],
        "y": mesh["y"],
        "z": mesh["z"],
        "i": mesh["i"],
        "j": mesh["j"],
        "k": mesh["k"],
        "text": mesh["text"],
        "hovertext": mesh["hovertext"],
        "hovertemplate": "%{hovertext}<extra></extra>",
        "color": color,
        "flatshading": True,
        "lighting": {"ambient": 0.48, "diffuse": 0.82, "roughness": 0.72},
        "lightposition": {"x": 100, "y": -120, "z": 180},
        "opacity": opacity,
    }


def _prime_cube_mesh(
    *,
    cells: _PrimeCubeCellData,
    scale: float,
) -> _PrimeCubeMeshData:
    x_values = cells["x"]
    y_values = cells["y"]
    z_values = cells["z"]
    text_values = cells["text"]
    hover_values = cells["hovertext"]
    half_size = scale / 2
    mesh: _PrimeCubeMeshData = {
        "x": [],
        "y": [],
        "z": [],
        "i": [],
        "j": [],
        "k": [],
        "text": [],
        "hovertext": [],
    }
    vertex_offsets = (
        (-half_size, -half_size, -half_size),
        (half_size, -half_size, -half_size),
        (half_size, half_size, -half_size),
        (-half_size, half_size, -half_size),
        (-half_size, -half_size, half_size),
        (half_size, -half_size, half_size),
        (half_size, half_size, half_size),
        (-half_size, half_size, half_size),
    )
    faces = (
        (0, 1, 2),
        (0, 2, 3),
        (4, 6, 5),
        (4, 7, 6),
        (0, 4, 5),
        (0, 5, 1),
        (1, 5, 6),
        (1, 6, 2),
        (2, 6, 7),
        (2, 7, 3),
        (3, 7, 4),
        (3, 4, 0),
    )

    for x, y, z, text, hovertext in zip(
        x_values,
        y_values,
        z_values,
        text_values,
        hover_values,
    ):
        vertex_start = len(mesh["x"])
        for dx, dy, dz in vertex_offsets:
            mesh["x"].append(float(x) + dx)
            mesh["y"].append(float(y) + dy)
            mesh["z"].append(float(z) + dz)
            mesh["text"].append(str(text))
            mesh["hovertext"].append(str(hovertext))

        for i, j, k in faces:
            mesh["i"].append(vertex_start + i)
            mesh["j"].append(vertex_start + j)
            mesh["k"].append(vertex_start + k)

    return mesh


def _base_primes_upto(limit: int) -> list[int]:
    if limit < 2:
        return []

    sieve = bytearray(b"\x01") * (limit // 2 + 1)
    sieve[0] = 0

    for number in range(3, isqrt(limit) + 1, 2):
        if not sieve[number // 2]:
            continue

        start = number * number // 2
        count = _slice_count(length=len(sieve), offset=start, step=number)
        sieve[start::number] = b"\x00" * count

    primes = [2]
    primes.extend(2 * index + 1 for index in range(1, len(sieve)) if sieve[index])
    return primes


def _dot_patterns(
    *,
    cell_size: int,
    dot_radius: float | None,
    prime_value: int,
    background_value: int,
) -> list[tuple[bytes, bytes]]:
    radius = dot_radius if dot_radius is not None else max(0.75, cell_size * 0.42)
    center = (cell_size - 1) / 2
    radius_squared = radius * radius
    empty_cell = bytes([background_value]) * cell_size
    patterns: list[tuple[bytes, bytes]] = []

    for y in range(cell_size):
        prime_cell = bytearray([background_value] * cell_size)
        for x in range(cell_size):
            x_distance = x - center
            y_distance = y - center
            if x_distance * x_distance + y_distance * y_distance <= radius_squared:
                prime_cell[x] = prime_value
        patterns.append((empty_cell, bytes(prime_cell)))

    return patterns


def _flag_to_grayscale_table(*, prime_value: int, background_value: int) -> bytes:
    return bytes(
        background_value if value == 0 else prime_value for value in range(256)
    )


def _slice_count(*, length: int, offset: int, step: int) -> int:
    if offset >= length:
        return 0
    return (length - 1 - offset) // step + 1


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_grayscale_value(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
        raise ValueError(f"{name} must be an integer from 0 to 255")


_PRIME_CUBE_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="__PLOTLY_CDN_URL__"></script>
<style>
:root {
    color-scheme: light;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #f7f9fc;
    color: #141922;
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    min-height: 100vh;
    background: #f7f9fc;
}

.shell {
    display: grid;
    grid-template-rows: auto 1fr;
    min-height: 100vh;
}

.toolbar {
    display: grid;
    grid-template-columns: minmax(220px, 1fr) repeat(4, minmax(150px, 190px)) auto;
    gap: 12px;
    align-items: end;
    padding: 16px;
    border-bottom: 1px solid #d9e0e8;
    background: #ffffff;
}

.brand h1 {
    margin: 0;
    font-size: 20px;
    line-height: 1.15;
    font-weight: 720;
}

.brand span {
    display: block;
    margin-top: 5px;
    color: #5f6b7a;
    font-size: 13px;
}

.control {
    display: grid;
    gap: 6px;
    color: #303846;
    font-size: 12px;
    font-weight: 650;
}

.control-row {
    display: flex;
    justify-content: space-between;
    gap: 8px;
}

output {
    color: #5f6b7a;
    font-variant-numeric: tabular-nums;
}

input[type="range"] {
    width: 100%;
    accent-color: #e11d48;
}

.toggles {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: flex-end;
}

.toggle,
button {
    min-height: 34px;
    border: 1px solid #cdd6e1;
    border-radius: 8px;
    background: #ffffff;
    color: #202734;
    font: inherit;
    font-size: 13px;
    font-weight: 650;
}

.toggle {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 7px 10px;
    white-space: nowrap;
}

.toggle input {
    width: 14px;
    height: 14px;
    accent-color: #0f766e;
}

button {
    padding: 7px 12px;
    cursor: pointer;
}

button:hover,
.toggle:hover {
    border-color: #9aa8b8;
    background: #f8fafc;
}

.plot-wrap {
    min-height: 0;
    padding: 0;
}

#primeCube {
    width: 100%;
    height: calc(100vh - 94px);
    min-height: 520px;
}

@media (max-width: 980px) {
    .toolbar {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .brand,
    .toggles {
        grid-column: 1 / -1;
    }

    .toggles {
        justify-content: flex-start;
    }

    #primeCube {
        height: calc(100vh - 238px);
        min-height: 460px;
    }
}

@media (max-width: 560px) {
    .toolbar {
        grid-template-columns: 1fr;
    }

    .brand,
    .toggles {
        grid-column: auto;
    }

    #primeCube {
        height: 62vh;
        min-height: 390px;
    }
}
</style>
</head>
<body>
<main class="shell">
    <section class="toolbar" aria-label="Prime cube controls">
        <div class="brand">
            <h1>__TITLE__</h1>
            <span>__SUMMARY__</span>
        </div>
        <label class="control" for="primeOpacity">
            <span class="control-row"><span>Prime opacity</span><output id="primeOpacityValue">94%</output></span>
            <input id="primeOpacity" type="range" min="0.05" max="1" step="0.01" value="0.94">
        </label>
        <label class="control" for="compositeOpacity">
            <span class="control-row"><span>Non-prime opacity</span><output id="compositeOpacityValue">0%</output></span>
            <input id="compositeOpacity" type="range" min="0" max="1" step="0.01" value="0">
        </label>
        <label class="control" for="cubeScale">
            <span class="control-row"><span>Cube scale</span><output id="cubeScaleValue">50%</output></span>
            <input id="cubeScale" type="range" min="0.25" max="0.98" step="0.01" value="0.5">
        </label>
        <label class="control" for="gridOpacity">
            <span class="control-row"><span>Grid opacity</span><output id="gridOpacityValue">0%</output></span>
            <input id="gridOpacity" type="range" min="0" max="1" step="0.01" value="0">
        </label>
        <div class="toggles">
            <label class="toggle"><input id="toggleComposites" type="checkbox" checked>Non-primes</label>
            <label class="toggle" title="__LONGEST_LINE_TITLE__"><input id="toggleLongestLine" type="checkbox" checked>Longest line</label>
            <label class="toggle"><input id="toggleLabels" type="checkbox">Labels</label>
            <label class="toggle"><input id="toggleGrid" type="checkbox" checked>Grid</label>
            <button id="resetCamera" type="button">Reset view</button>
        </div>
    </section>
    <section class="plot-wrap" aria-label="3D prime cube plot">
        <div id="primeCube"></div>
    </section>
</main>
<script>
const traces = __TRACES_JSON__;
const layout = __LAYOUT_JSON__;
const primeCells = __PRIME_CELLS_JSON__;
const compositeCells = __COMPOSITE_CELLS_JSON__;
const longestLineCells = __LONGEST_LINE_CELLS_JSON__;
const config = {
    responsive: true,
    scrollZoom: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d"]
};
const chart = document.getElementById("primeCube");
const traceIndex = { grid: 0, composites: 1, primes: 2, longestLine: 3, labels: 4 };
const initialCamera = JSON.parse(JSON.stringify(layout.scene.camera));

Plotly.newPlot(chart, traces, layout, config);

function percent(value) {
    return `${Math.round(Number(value) * 100)}%`;
}

function bindRange(inputId, outputId, formatter, callback) {
    const input = document.getElementById(inputId);
    const output = document.getElementById(outputId);
    input.addEventListener("input", () => {
        output.value = formatter(input.value);
        callback(Number(input.value));
    });
}

bindRange("primeOpacity", "primeOpacityValue", percent, value => {
    Plotly.restyle(chart, { opacity: value }, [traceIndex.primes]);
});

bindRange("compositeOpacity", "compositeOpacityValue", percent, value => {
    Plotly.restyle(chart, { opacity: value }, [traceIndex.composites]);
});

function cubeMesh(cells, scale) {
    const halfSize = scale / 2;
    const offsets = [
        [-halfSize, -halfSize, -halfSize],
        [halfSize, -halfSize, -halfSize],
        [halfSize, halfSize, -halfSize],
        [-halfSize, halfSize, -halfSize],
        [-halfSize, -halfSize, halfSize],
        [halfSize, -halfSize, halfSize],
        [halfSize, halfSize, halfSize],
        [-halfSize, halfSize, halfSize]
    ];
    const faces = [
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1],
        [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3],
        [3, 7, 4], [3, 4, 0]
    ];
    const mesh = { x: [], y: [], z: [], i: [], j: [], k: [], text: [], hovertext: [] };

    cells.x.forEach((x, cellIndex) => {
        const vertexStart = mesh.x.length;
        offsets.forEach(offset => {
            mesh.x.push(x + offset[0]);
            mesh.y.push(cells.y[cellIndex] + offset[1]);
            mesh.z.push(cells.z[cellIndex] + offset[2]);
            mesh.text.push(cells.text[cellIndex]);
            mesh.hovertext.push(cells.hovertext[cellIndex]);
        });
        faces.forEach(face => {
            mesh.i.push(vertexStart + face[0]);
            mesh.j.push(vertexStart + face[1]);
            mesh.k.push(vertexStart + face[2]);
        });
    });

    return mesh;
}

function restyleMesh(trace, mesh) {
    Plotly.restyle(chart, {
        x: [mesh.x],
        y: [mesh.y],
        z: [mesh.z],
        i: [mesh.i],
        j: [mesh.j],
        k: [mesh.k],
        text: [mesh.text],
        hovertext: [mesh.hovertext]
    }, [trace]);
}

bindRange("cubeScale", "cubeScaleValue", percent, value => {
    restyleMesh(traceIndex.composites, cubeMesh(compositeCells, value));
    restyleMesh(traceIndex.primes, cubeMesh(primeCells, value));
    restyleMesh(traceIndex.longestLine, cubeMesh(longestLineCells, Math.min(1, value + 0.04)));
});

bindRange("gridOpacity", "gridOpacityValue", percent, value => {
    Plotly.restyle(chart, { opacity: value }, [traceIndex.grid]);
});

document.getElementById("toggleComposites").addEventListener("change", event => {
    Plotly.restyle(chart, { visible: event.target.checked }, [traceIndex.composites]);
});

document.getElementById("toggleLongestLine").addEventListener("change", event => {
    Plotly.restyle(chart, { visible: event.target.checked }, [traceIndex.longestLine]);
});

document.getElementById("toggleLabels").addEventListener("change", event => {
    Plotly.restyle(chart, { visible: event.target.checked }, [traceIndex.labels]);
});

document.getElementById("toggleGrid").addEventListener("change", event => {
    Plotly.restyle(chart, { visible: event.target.checked }, [traceIndex.grid]);
});

document.getElementById("resetCamera").addEventListener("click", () => {
    Plotly.relayout(chart, { "scene.camera": initialCamera });
});
</script>
</body>
</html>
"""


__all__ = [
    "DEFAULT_PLOTLY_CDN_URL",
    "LINE_DIRECTIONS",
    "PrimeCubeHtml",
    "PrimeDotImage",
    "PrimeLine",
    "PrimeLineWorkload",
    "analyze_prime_lines",
    "estimate_png_dimensions",
    "estimate_prime_line_workload",
    "generate_graph_dots",
    "generate_prime_cube_plot_html",
    "generate_prime_dot_png",
    "rank_widths_by_prime_lines",
]
