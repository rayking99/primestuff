from __future__ import annotations

from dataclasses import dataclass
from math import isqrt
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from primewords.primes import (
    _base_primes_upto,
    _longest_prime_cube_line,
    _prime_flags_for_range,
)

OUTPUT_DIR = Path("Examples/Primes/cube_side_snapshots")
SIZE_RANGE = range(2, 5)
CELL_SIZE = 22
LABEL_SQUARES = True
SQUARE_LABEL_FONT_SIZE = 7

BACKGROUND = "#f7f9fc"
PANEL = "#ffffff"
GRID = "#d8e0ea"
TEXT = "#1f2937"
MUTED = "#5f6b7a"
PRIME = "#e11d48"
LONGEST_LINE = "#050505"


@dataclass(frozen=True)
class SideView:
    name: str
    x_axis: str
    y_axis: str
    depth_axis: str
    reverse_x: bool = False
    reverse_y: bool = False
    reverse_depth: bool = False


@dataclass(frozen=True)
class VisibleCell:
    state: str
    label: int | None = None


SIDE_VIEWS = (
    SideView("front", "column", "row", "layer"),
    SideView("side", "layer", "row", "column"),
    SideView("top", "column", "layer", "row"),
)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT_DIR.glob("*.png"):
        path.unlink()

    written = 0

    for size in SIZE_RANGE:
        max_number = size**3
        flags = _prime_flags_for_range(
            start=1,
            stop=max_number + 1,
            base_primes=_base_primes_upto(isqrt(max_number)),
        )
        longest_line, direction = _longest_prime_cube_line(
            prime_flags=flags,
            plane_width=size,
            plane_height=size,
            layers=size,
        )

        white_counts: list[str] = []
        for view in SIDE_VIEWS:
            output_path = OUTPUT_DIR / f"{size}-{view.name}.png"
            white_count = render_side_snapshot(
                size=size,
                view=view,
                prime_flags=flags,
                longest_line=set(longest_line),
                longest_line_direction=direction,
                output_path=output_path,
            )
            white_counts.append(f"{view.name} {white_count} white squares")
            written += 1

        print(
            f"size {size}: wrote {len(SIDE_VIEWS)} snapshots; "
            f"longest line length {len(longest_line)}; "
            f"{', '.join(white_counts)}"
        )

    print(f"wrote {written} PNGs to {OUTPUT_DIR}")


def render_side_snapshot(
    *,
    size: int,
    view: SideView,
    prime_flags: bytearray,
    longest_line: set[int],
    longest_line_direction: tuple[int, int, int],
    output_path: Path,
) -> int:
    title_font = _font(16)
    label_font = _font(11)
    square_label_font = _font(SQUARE_LABEL_FONT_SIZE)
    small_font = _font(10)
    axis_font = _font(12)

    cell_size = CELL_SIZE
    grid_size = size * cell_size
    left_margin = 74
    right_margin = 54
    top_margin = 120
    bottom_margin = 78
    image_width = left_margin + grid_size + right_margin
    image_height = top_margin + grid_size + bottom_margin
    grid_left = left_margin
    grid_top = top_margin
    grid_right = grid_left + grid_size
    grid_bottom = grid_top + grid_size

    image = Image.new("RGB", (image_width, image_height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image_width, image_height), fill=BACKGROUND)
    draw.rectangle(
        (grid_left - 1, grid_top - 1, grid_right + 1, grid_bottom + 1),
        fill=PANEL,
        outline=GRID,
    )

    x_values = _axis_values(size, reverse=view.reverse_x)
    y_values = _axis_values(size, reverse=view.reverse_y)
    depth_values = _axis_values(size, reverse=view.reverse_depth)
    state_grid = _visible_2d_state_grid(
        size=size,
        view=view,
        x_values=x_values,
        y_values=y_values,
        depth_values=depth_values,
        prime_flags=prime_flags,
        longest_line=longest_line,
    )
    white_count = sum(cell.state == "empty" for row in state_grid for cell in row)

    for y_index, row in enumerate(state_grid):
        for x_index, cell in enumerate(row):
            if cell.state == "empty":
                continue

            x0 = grid_left + x_index * cell_size + 1
            y0 = grid_top + y_index * cell_size + 1
            x1 = x0 + cell_size - 2
            y1 = y0 + cell_size - 2
            draw.rectangle(
                (x0, y0, x1, y1), fill=LONGEST_LINE if cell.state == "line" else PRIME
            )

    for index in range(size + 1):
        position = grid_left + index * cell_size
        draw.line((position, grid_top, position, grid_bottom), fill=GRID)
        position = grid_top + index * cell_size
        draw.line((grid_left, position, grid_right, position), fill=GRID)

    if LABEL_SQUARES:
        _draw_square_labels(
            draw=draw,
            state_grid=state_grid,
            grid_left=grid_left,
            grid_top=grid_top,
            cell_size=cell_size,
            label_font=square_label_font,
        )

    _draw_axis_labels(
        draw=draw,
        x_values=x_values,
        y_values=y_values,
        grid_left=grid_left,
        grid_top=grid_top,
        cell_size=cell_size,
        grid_right=grid_right,
        grid_bottom=grid_bottom,
        label_font=label_font,
    )
    _draw_centered_text(
        draw,
        f"{size}-{view.name}",
        (image_width / 2, 20),
        title_font,
        TEXT,
    )
    _draw_centered_text(
        draw,
        f"{view.x_axis} x {view.y_axis}; looking through {view.depth_axis}",
        (image_width / 2, 43),
        axis_font,
        MUTED,
    )
    _draw_centered_text(
        draw,
        f"2D white squares: {white_count}",
        (image_width / 2, 67),
        small_font,
        MUTED,
    )
    _draw_centered_text(
        draw,
        view.x_axis,
        (image_width / 2, grid_bottom + 50),
        axis_font,
        TEXT,
    )
    draw.text((12, grid_top - 26), view.y_axis, font=axis_font, fill=TEXT)

    image.save(output_path)
    return white_count


def _visible_2d_state_grid(
    *,
    size: int,
    view: SideView,
    x_values: tuple[int, ...],
    y_values: tuple[int, ...],
    depth_values: tuple[int, ...],
    prime_flags: bytearray,
    longest_line: set[int],
) -> tuple[tuple[VisibleCell, ...], ...]:
    rows: list[tuple[VisibleCell, ...]] = []
    for y_value in y_values:
        row: list[VisibleCell] = []
        for x_value in x_values:
            row.append(
                _visible_2d_cell_state(
                    size=size,
                    view=view,
                    x_value=x_value,
                    y_value=y_value,
                    depth_values=depth_values,
                    prime_flags=prime_flags,
                    longest_line=longest_line,
                )
            )
        rows.append(tuple(row))
    return tuple(rows)


def _visible_2d_cell_state(
    *,
    size: int,
    view: SideView,
    x_value: int,
    y_value: int,
    depth_values: tuple[int, ...],
    prime_flags: bytearray,
    longest_line: set[int],
) -> VisibleCell:
    for depth_value in depth_values:
        coordinate = {
            view.x_axis: x_value,
            view.y_axis: y_value,
            view.depth_axis: depth_value,
        }
        number = _cube_number(
            column=coordinate["column"],
            row=coordinate["row"],
            layer=coordinate["layer"],
            size=size,
        )
        if prime_flags[number - 1]:
            return VisibleCell(
                state="line" if number in longest_line else "prime",
                label=number,
            )
    return VisibleCell(state="empty")


def _cube_number(*, column: int, row: int, layer: int, size: int) -> int:
    return (layer - 1) * size * size + (row - 1) * size + column


def _axis_values(size: int, *, reverse: bool = False) -> tuple[int, ...]:
    values = range(size, 0, -1) if reverse else range(1, size + 1)
    return tuple(values)


def _draw_square_labels(
    *,
    draw: ImageDraw.ImageDraw,
    state_grid: tuple[tuple[VisibleCell, ...], ...],
    grid_left: int,
    grid_top: int,
    cell_size: int,
    label_font: ImageFont.ImageFont,
) -> None:
    for y_index, row in enumerate(state_grid):
        for x_index, cell in enumerate(row):
            if cell.label is None:
                continue

            label = str(cell.label)
            x = grid_left + x_index * cell_size + cell_size / 2
            y = grid_top + y_index * cell_size + cell_size / 2
            _draw_centered_text(draw, label, (x, y), label_font, PANEL)


def _draw_axis_labels(
    *,
    draw: ImageDraw.ImageDraw,
    x_values: tuple[int, ...],
    y_values: tuple[int, ...],
    grid_left: int,
    grid_top: int,
    cell_size: int,
    grid_right: int,
    grid_bottom: int,
    label_font: ImageFont.ImageFont,
) -> None:
    for index, value in enumerate(x_values):
        x = grid_left + index * cell_size + cell_size / 2
        _draw_centered_text(draw, str(value), (x, grid_top - 15), label_font, MUTED)
        _draw_centered_text(draw, str(value), (x, grid_bottom + 15), label_font, MUTED)

    for index, value in enumerate(y_values):
        y = grid_top + index * cell_size + cell_size / 2
        _draw_centered_text(draw, str(value), (grid_left - 20, y), label_font, MUTED)
        _draw_centered_text(draw, str(value), (grid_right + 20, y), label_font, MUTED)


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    center: tuple[float, float],
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.text(
        (center[0] - width / 2, center[1] - height / 2), text, font=font, fill=fill
    )


def _font(size: int) -> Any:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _direction_label(direction: tuple[int, int, int]) -> str:
    dx, dy, dz = direction
    return f"column {dx:+d}, row {dy:+d}, layer {dz:+d}"


if __name__ == "__main__":
    main()
