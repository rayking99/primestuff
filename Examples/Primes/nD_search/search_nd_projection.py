from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import isqrt
from time import monotonic
from typing import Literal

import numpy as np

from primestuff.primes import _base_primes_upto, _prime_flags_for_range

SearchMode = Literal["any", "all"]
SearchMethod = Literal["auto", "segment", "line"]

MILLER_RABIN_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


@dataclass(frozen=True)
class SideCheck:
    side: int
    dimensions: int
    max_number: int
    success: bool
    full_axes: tuple[int, ...]
    method: str
    covered_counts: tuple[int, ...] | None = None
    first_empty_line: tuple[int, int] | None = None


@dataclass(frozen=True)
class SearchResult:
    dimensions: int
    side: int
    max_number: int
    largest_prime: int
    full_axes: tuple[int, ...]
    method: str
    checked_sides: int
    elapsed_seconds: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Search n-dimensional side lengths whose orthographic prime "
            "projection has no empty cells. The 3D --mode any baseline is 67."
        )
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=[4],
        help="Dimension counts to search, for example: --dimensions 4 5 6",
    )
    parser.add_argument(
        "--start-side",
        type=int,
        default=3,
        help="First hypercube side length to test. Use 3 to ignore the trivial 2.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        required=True,
        help="Largest hypercube side length to test for each dimension.",
    )
    parser.add_argument(
        "--mode",
        choices=("any", "all"),
        default="any",
        help=(
            "any: stop when one projection axis is full, matching the 3D 67 result. "
            "all: require every projection axis to be full."
        ),
    )
    parser.add_argument(
        "--method",
        choices=("auto", "segment", "line"),
        default="auto",
        help=(
            "segment streams prime flags and marks covered projection cells; line "
            "checks each projected line with Miller-Rabin; auto chooses segment "
            "while the coverage arrays are modest."
        ),
    )
    parser.add_argument(
        "--segment-size",
        type=int,
        default=8_000_000,
        help="Numbers to sieve at once when using the segment method.",
    )
    parser.add_argument(
        "--max-covered-cells",
        type=int,
        default=100_000_000,
        help=(
            "Maximum total projection cells that auto/segment may keep in memory. "
            "The coverage arrays use roughly one byte per cell."
        ),
    )
    parser.add_argument(
        "--trial-prime-limit",
        type=int,
        default=1_000,
        help="Small-prime trial division limit for the line method.",
    )
    parser.add_argument(
        "--progress",
        type=int,
        default=10,
        help="Print progress every N side lengths. Use 0 to silence progress.",
    )
    args = parser.parse_args()

    _validate_args(args)

    for dimensions in args.dimensions:
        print(
            f"searching {dimensions}D: sides {args.start_side}..{args.max_side}, "
            f"mode={args.mode}, method={args.method}"
        )
        result = search_dimension(
            dimensions=dimensions,
            start_side=args.start_side,
            max_side=args.max_side,
            mode=args.mode,
            method=args.method,
            segment_size=args.segment_size,
            max_covered_cells=args.max_covered_cells,
            trial_prime_limit=args.trial_prime_limit,
            progress=args.progress,
        )
        if result is None:
            print(f"  no result found for {dimensions}D through side {args.max_side}")
            continue

        axes = ", ".join(f"axis {axis}" for axis in result.full_axes)
        print(
            f"  found {dimensions}D side {result.side} using {result.method}: "
            f"full projection axis/axes: {axes}"
        )
        print(f"  max number in hypercube: {result.max_number:,}")
        print(f"  largest prime in hypercube: {result.largest_prime:,}")
        print(
            f"  checked {result.checked_sides} side lengths in "
            f"{result.elapsed_seconds:.2f}s"
        )


def search_dimension(
    *,
    dimensions: int,
    start_side: int,
    max_side: int,
    mode: SearchMode,
    method: SearchMethod,
    segment_size: int,
    max_covered_cells: int,
    trial_prime_limit: int,
    progress: int,
) -> SearchResult | None:
    started = monotonic()
    trial_primes = _base_primes_upto(trial_prime_limit)

    for checked_sides, side in enumerate(range(start_side, max_side + 1), start=1):
        if progress and checked_sides > 1 and (checked_sides - 1) % progress == 0:
            print(f"  checked through side {side - 1}")

        check = check_side(
            dimensions=dimensions,
            side=side,
            mode=mode,
            method=method,
            segment_size=segment_size,
            max_covered_cells=max_covered_cells,
            trial_primes=trial_primes,
        )
        if not check.success:
            continue

        largest_prime = previous_prime(check.max_number, trial_primes)
        return SearchResult(
            dimensions=dimensions,
            side=side,
            max_number=check.max_number,
            largest_prime=largest_prime,
            full_axes=check.full_axes,
            method=check.method,
            checked_sides=checked_sides,
            elapsed_seconds=monotonic() - started,
        )

    return None


def check_side(
    *,
    dimensions: int,
    side: int,
    mode: SearchMode,
    method: SearchMethod,
    segment_size: int,
    max_covered_cells: int,
    trial_primes: list[int],
) -> SideCheck:
    projection_cells = side ** (dimensions - 1)
    total_covered_cells = projection_cells * dimensions
    selected_method = method
    if selected_method == "auto":
        selected_method = (
            "segment" if total_covered_cells <= max_covered_cells else "line"
        )

    if selected_method == "segment":
        if total_covered_cells > max_covered_cells:
            raise ValueError(
                "segment method would allocate too many coverage cells "
                f"({total_covered_cells:,} > {max_covered_cells:,}); use "
                "--method line or raise --max-covered-cells"
            )
        return check_side_segmented(
            dimensions=dimensions,
            side=side,
            mode=mode,
            segment_size=segment_size,
        )

    return check_side_by_lines(
        dimensions=dimensions,
        side=side,
        mode=mode,
        trial_primes=trial_primes,
    )


def check_side_segmented(
    *,
    dimensions: int,
    side: int,
    mode: SearchMode,
    segment_size: int,
) -> SideCheck:
    max_number = side**dimensions
    projection_cells = side ** (dimensions - 1)
    powers = tuple(side**power for power in range(dimensions))
    axis_strides = tuple(powers[dimensions - axis - 1] for axis in range(dimensions))
    base_primes = _base_primes_upto(isqrt(max_number))
    covered = [np.zeros(projection_cells, dtype=np.bool_) for _ in range(dimensions)]
    covered_counts = [0] * dimensions

    for start in range(1, max_number + 1, segment_size):
        stop = min(max_number + 1, start + segment_size)
        prime_flags = np.frombuffer(
            _prime_flags_for_range(start=start, stop=stop, base_primes=base_primes),
            dtype=np.uint8,
        )
        prime_offsets = np.flatnonzero(prime_flags)
        if len(prime_offsets) == 0:
            continue

        flat_indices = prime_offsets + start - 1
        for axis, stride in enumerate(axis_strides):
            projection_indices = _projection_indices(flat_indices, side, stride)
            unique_indices = np.unique(projection_indices)
            new_indices = unique_indices[~covered[axis][unique_indices]]
            if len(new_indices) == 0:
                continue

            covered[axis][new_indices] = True
            covered_counts[axis] += len(new_indices)

        full_axes = tuple(
            axis
            for axis, count in enumerate(covered_counts)
            if count == projection_cells
        )
        if _mode_satisfied(full_axes, dimensions, mode):
            return SideCheck(
                side=side,
                dimensions=dimensions,
                max_number=max_number,
                success=True,
                full_axes=full_axes,
                method="segment",
                covered_counts=tuple(covered_counts),
            )

    full_axes = tuple(
        axis for axis, count in enumerate(covered_counts) if count == projection_cells
    )
    return SideCheck(
        side=side,
        dimensions=dimensions,
        max_number=max_number,
        success=_mode_satisfied(full_axes, dimensions, mode),
        full_axes=full_axes,
        method="segment",
        covered_counts=tuple(covered_counts),
    )


def check_side_by_lines(
    *,
    dimensions: int,
    side: int,
    mode: SearchMode,
    trial_primes: list[int],
) -> SideCheck:
    max_number = side**dimensions
    projection_cells = side ** (dimensions - 1)
    full_axes: list[int] = []
    first_empty_line: tuple[int, int] | None = None

    for axis in range(dimensions):
        axis_is_full = True
        stride = side ** (dimensions - axis - 1)

        for projection_index in range(projection_cells):
            line_start = _line_start_index(projection_index, side, stride)
            if _line_has_prime(line_start, stride, side, trial_primes):
                continue

            axis_is_full = False
            if first_empty_line is None:
                first_empty_line = (axis, projection_index)
            break

        if axis_is_full:
            full_axes.append(axis)
            if mode == "any":
                return SideCheck(
                    side=side,
                    dimensions=dimensions,
                    max_number=max_number,
                    success=True,
                    full_axes=tuple(full_axes),
                    method="line",
                )
        elif mode == "all":
            return SideCheck(
                side=side,
                dimensions=dimensions,
                max_number=max_number,
                success=False,
                full_axes=tuple(full_axes),
                method="line",
                first_empty_line=first_empty_line,
            )

    return SideCheck(
        side=side,
        dimensions=dimensions,
        max_number=max_number,
        success=_mode_satisfied(tuple(full_axes), dimensions, mode),
        full_axes=tuple(full_axes),
        method="line",
        first_empty_line=first_empty_line,
    )


def _projection_indices(flat_indices: np.ndarray, side: int, stride: int) -> np.ndarray:
    higher_digits = flat_indices // (stride * side)
    lower_digits = flat_indices % stride
    return higher_digits * stride + lower_digits


def _line_start_index(projection_index: int, side: int, stride: int) -> int:
    higher_digits = projection_index // stride
    lower_digits = projection_index % stride
    return higher_digits * stride * side + lower_digits


def _line_has_prime(
    line_start_index: int,
    stride: int,
    side: int,
    trial_primes: list[int],
) -> bool:
    for offset in range(side):
        number = line_start_index + offset * stride + 1
        if is_prime(number, trial_primes):
            return True
    return False


def previous_prime(number: int, trial_primes: list[int]) -> int:
    if number < 2:
        raise ValueError("there is no prime below 2")
    candidate = number
    if candidate > 2 and candidate % 2 == 0:
        candidate -= 1
    while candidate >= 2:
        if is_prime(candidate, trial_primes):
            return candidate
        candidate -= 1 if candidate == 3 else 2
    raise ValueError("there is no prime below 2")


def is_prime(number: int, trial_primes: list[int]) -> bool:
    if number < 2:
        return False

    for prime in trial_primes:
        if prime * prime > number:
            return True
        if number % prime == 0:
            return number == prime

    return _is_strong_probable_prime(number)


def _is_strong_probable_prime(number: int) -> bool:
    if number % 2 == 0:
        return number == 2

    odd_part = number - 1
    shift_count = 0
    while odd_part % 2 == 0:
        odd_part //= 2
        shift_count += 1

    for base in MILLER_RABIN_BASES:
        if base >= number:
            continue
        witness = pow(base, odd_part, number)
        if witness == 1 or witness == number - 1:
            continue
        for _ in range(shift_count - 1):
            witness = pow(witness, 2, number)
            if witness == number - 1:
                break
        else:
            return False

    return True


def _mode_satisfied(
    full_axes: tuple[int, ...], dimensions: int, mode: SearchMode
) -> bool:
    if mode == "any":
        return bool(full_axes)
    return len(full_axes) == dimensions


def _validate_args(args: argparse.Namespace) -> None:
    if not args.dimensions:
        raise ValueError("--dimensions must include at least one dimension")
    if any(dimension < 2 for dimension in args.dimensions):
        raise ValueError("all --dimensions values must be at least 2")
    if args.start_side < 1:
        raise ValueError("--start-side must be positive")
    if args.max_side < args.start_side:
        raise ValueError("--max-side must be greater than or equal to --start-side")
    if args.segment_size < 1:
        raise ValueError("--segment-size must be positive")
    if args.max_covered_cells < 1:
        raise ValueError("--max-covered-cells must be positive")
    if args.trial_prime_limit < 2:
        raise ValueError("--trial-prime-limit must be at least 2")
    if args.progress < 0:
        raise ValueError("--progress must be non-negative")


if __name__ == "__main__":
    main()
