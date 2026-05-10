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
from math import isqrt
import os
from pathlib import Path
import struct
from typing import Any, BinaryIO, Callable, Iterable, Sequence
import zlib

PNG_MAX_DIMENSION = 2_147_483_647
DEFAULT_SEGMENT_SIZE = 8_000_000
DEFAULT_IDAT_CHUNK_SIZE = 1_048_576

ProgressCallback = Callable[[int, int, int], None]
LineProgressCallback = Callable[[int, int, int], None]

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


__all__ = [
    "LINE_DIRECTIONS",
    "PrimeDotImage",
    "PrimeLine",
    "PrimeLineWorkload",
    "analyze_prime_lines",
    "estimate_png_dimensions",
    "estimate_prime_line_workload",
    "generate_graph_dots",
    "generate_prime_dot_png",
    "rank_widths_by_prime_lines",
]
