from __future__ import annotations

import argparse
from functools import lru_cache
from math import isqrt

import numpy as np

from primewords.primes import _base_primes_upto, _prime_flags_for_range


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find the smallest width x height x layers box where any orthographic projection has no white cells."
    )
    parser.add_argument("--max-volume", type=int, default=1_000_000)
    parser.add_argument("--min-dimension", type=int, default=2)
    parser.add_argument("--progress", type=int, default=50_000)
    args = parser.parse_args()

    if args.max_volume < 1:
        raise ValueError("--max-volume must be positive")
    if args.min_dimension < 1:
        raise ValueError("--min-dimension must be positive")

    prime_flags = np.frombuffer(
        _prime_flags_for_range(
            start=1,
            stop=args.max_volume + 1,
            base_primes=_base_primes_upto(isqrt(args.max_volume)),
        ),
        dtype=np.uint8,
    ).astype(bool)
    divisors = _divisor_function(args.max_volume)
    checked = 0

    for volume in range(1, args.max_volume + 1):
        if args.progress and volume % args.progress == 0:
            print(f"checked volumes through {volume}; tested {checked} boxes")

        for width in divisors(volume):
            if width < args.min_dimension:
                continue
            rows_across_box = volume // width

            for height in divisors(rows_across_box):
                layers = rows_across_box // height
                if height < args.min_dimension or layers < args.min_dimension:
                    continue

                checked += 1
                white_counts = _white_counts(
                    prime_flags=prime_flags,
                    width=width,
                    height=height,
                    layers=layers,
                )
                zero_views = tuple(
                    view for view, count in white_counts.items() if count == 0
                )
                if zero_views:
                    print("found")
                    print(f"volume: {volume}")
                    print(f"dimensions: {width}x{height}x{layers}")
                    print(
                        "white counts: "
                        f"front {white_counts['front']}, "
                        f"side {white_counts['side']}, "
                        f"top {white_counts['top']}"
                    )
                    print(f"zero-white views: {', '.join(zero_views)}")
                    print(f"tested boxes: {checked}")
                    return

    print(f"no zero-white box found up to volume {args.max_volume}")
    print(f"tested boxes: {checked}")


def _white_counts(
    *,
    prime_flags: np.ndarray,
    width: int,
    height: int,
    layers: int,
) -> dict[str, int]:
    box = prime_flags[: width * height * layers].reshape((layers, height, width))
    return {
        "front": int((~box.any(axis=0)).sum()),
        "side": int((~box.any(axis=2)).sum()),
        "top": int((~box.any(axis=1)).sum()),
    }


def _divisor_function(limit: int):
    smallest_prime_factor = list(range(limit + 1))
    for number in range(2, isqrt(limit) + 1):
        if smallest_prime_factor[number] != number:
            continue
        for multiple in range(number * number, limit + 1, number):
            if smallest_prime_factor[multiple] == multiple:
                smallest_prime_factor[multiple] = number

    @lru_cache(maxsize=None)
    def divisors(number: int) -> tuple[int, ...]:
        if number == 1:
            return (1,)

        factor_counts: list[tuple[int, int]] = []
        remaining = number
        while remaining > 1:
            factor = smallest_prime_factor[remaining]
            count = 0
            while remaining % factor == 0:
                remaining //= factor
                count += 1
            factor_counts.append((factor, count))

        values = [1]
        for factor, count in factor_counts:
            powers = [factor**power for power in range(1, count + 1)]
            values = values + [value * power for value in values for power in powers]
        return tuple(sorted(values))

    return divisors


if __name__ == "__main__":
    main()
