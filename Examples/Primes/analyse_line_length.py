from primewords.primes import estimate_prime_line_workload, rank_widths_by_prime_lines
import pandas as pd


def main() -> None:
    widths = range(2, 1000)
    max_number = 100_000_000
    min_length = 5

    workload = estimate_prime_line_workload(
        widths,
        max_number=max_number,
        min_length=min_length,
    )
    print(pd.Series(workload.as_dict()))

    input("Press Enter to continue to line analysis...")

    results = []

    for line in rank_widths_by_prime_lines(
        widths,
        max_number=max_number,
        min_length=min_length,
        workers=16,
        chunk_size=4,
    ):
        results.append(
            {
                "width": line.width,
                "direction": line.direction,
                "length": line.length,
                "step": line.step,
                "start_number": line.start_number,
                "end_number": line.end_number,
            }
        )

    df = pd.DataFrame(results)
    print(df)


if __name__ == "__main__":
    main()
