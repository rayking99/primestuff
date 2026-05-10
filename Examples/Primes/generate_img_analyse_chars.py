"""Experiment with OCR and word-template search on prime-dot images.

MNIST is not a good fit here: it recognizes isolated handwritten digits.
For generated prime-dot images, two more useful tools are:

1. OCR, if the image already contains readable text-like shapes.
2. Template matching, if you want to ask whether a known word-shaped pattern
   appears somewhere in the dot field.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
from typing import TextIO

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from primewords.primes import generate_prime_dot_png

DEFAULT_IMAGE_PATH = Path("Examples/Primes/primes.png")
DEFAULT_OCR_IMAGE_PATH = Path("Examples/Primes/primes_ocr_ready.png")
DEFAULT_TEMPLATE_OVERLAY_PATH = Path("Examples/Primes/primes_template_matches.png")
DEFAULT_SCAN_CSV_PATH = Path("Examples/Primes/prime_ocr_width_scan.csv")
DEFAULT_SCAN_JSONL_PATH = Path("Examples/Primes/prime_ocr_width_scan.jsonl")
DEFAULT_REVIEW_CACHE_DIR = Path("Examples/Primes/word_search_cache")
DEFAULT_FOUND_WORD_DIR = DEFAULT_REVIEW_CACHE_DIR / "found"
DEFAULT_REVIEW_CACHE_SIZE = 5
DEFAULT_CHART_HEIGHT = 6
DEFAULT_OCR_BATCH_HEIGHT = 6
DEFAULT_OCR_BATCH_OVERLAP = 2
DEFAULT_WORKERS = 10
DEFAULT_MIN_WORD_HEIGHT = 10
DICTIONARY_CANDIDATES = (
    Path("/usr/share/dict/words"),
    Path("/usr/dict/words"),
)
FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
)
WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


@dataclass(frozen=True)
class OcrWord:
    text: str
    confidence: float
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class TemplateMatch:
    word: str
    score: float
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class OcrWidthJob:
    index: int
    total: int
    width: int
    max_number: int
    chart_height: int
    cell_size: int
    square_dots: bool
    scale: int
    dilation: int
    close: int
    min_confidence: float
    psm: int
    word: str
    dictionary_source: str
    min_word_length: int
    min_word_height: int
    ocr_batch_height: int
    ocr_batch_overlap: int
    image_dir: Path
    keep_images: bool
    review_cache_dir: Path | None
    found_word_dir: Path | None
    review_cache_size: int


_WORKER_DICTIONARY_WORDS: set[str] = set()


def generate_prime_image(
    *,
    output_path: Path,
    width: int,
    max_number: int,
    cell_size: int,
    square_dots: bool = False,
) -> None:
    meta = generate_prime_dot_png(
        width=width,
        max_number=max_number,
        output_path=output_path,
        cell_size=cell_size,
        dot_radius=cell_size if square_dots else None,
    )
    print(meta)


def preprocess_for_ocr(
    image_path: Path,
    output_path: Path,
    *,
    scale: int = 4,
    dilation: int = 1,
    close: int = 1,
    border: int = 24,
) -> Path:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    ocr_ready = preprocess_array_for_ocr(
        image,
        scale=scale,
        dilation=dilation,
        close=close,
        border=border,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), ocr_ready)
    return output_path


def preprocess_array_for_ocr(
    image: np.ndarray,
    *,
    scale: int = 4,
    dilation: int = 1,
    close: int = 1,
    border: int = 24,
) -> np.ndarray:
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    if scale > 1:
        binary = cv2.resize(
            binary,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )

    if dilation > 0:
        kernel_size = max(1, dilation * 2 + 1)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        binary = cv2.dilate(binary, kernel, iterations=1)

    if close > 0:
        kernel_size = max(1, close * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    ocr_ready = 255 - binary
    if border > 0:
        ocr_ready = cv2.copyMakeBorder(
            ocr_ready,
            border,
            border,
            border,
            border,
            cv2.BORDER_CONSTANT,
            value=255,
        )

    return ocr_ready


def run_tesseract_words(
    image_path: Path,
    *,
    min_confidence: float = 25.0,
    psm: int = 11,
) -> list[OcrWord]:
    tesseract = shutil.which("tesseract")
    if tesseract is None:
        raise RuntimeError(
            "Tesseract is not installed. On macOS: brew install tesseract"
        )

    command = [
        tesseract,
        str(image_path),
        "stdout",
        "--oem",
        "1",
        "--psm",
        str(psm),
        "-l",
        "eng",
        "tsv",
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )

    words: list[OcrWord] = []
    reader = csv.DictReader(result.stdout.splitlines(), delimiter="\t")
    for row in reader:
        text = (row.get("text") or "").strip()
        if not text:
            continue

        try:
            confidence = float(row.get("conf") or -1)
        except ValueError:
            continue

        if confidence < min_confidence:
            continue

        words.append(
            OcrWord(
                text=text,
                confidence=confidence,
                x=int(row.get("left") or 0),
                y=int(row.get("top") or 0),
                width=int(row.get("width") or 0),
                height=int(row.get("height") or 0),
            )
        )

    return words


def run_tesseract_words_in_batches(
    image_path: Path,
    *,
    batch_dir: Path,
    min_confidence: float = 25.0,
    psm: int = 11,
    scale: int = 4,
    dilation: int = 1,
    close: int = 1,
    batch_height: int = DEFAULT_OCR_BATCH_HEIGHT,
    batch_overlap: int = DEFAULT_OCR_BATCH_OVERLAP,
    min_word_height: int = DEFAULT_MIN_WORD_HEIGHT,
    border: int = 24,
) -> tuple[list[OcrWord], int, list[str]]:
    if batch_height < 1:
        raise ValueError("batch_height must be at least 1")
    if batch_overlap < 0:
        raise ValueError("batch_overlap must be zero or greater")
    if batch_overlap >= batch_height:
        raise ValueError("batch_overlap must be smaller than batch_height")
    if min_word_height < 1:
        raise ValueError("min_word_height must be at least 1")

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    batch_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, batch_height - batch_overlap)
    height = image.shape[0]
    words: list[OcrWord] = []
    errors: list[str] = []
    batch_count = 0

    for batch_count, y_start in enumerate(range(0, height, step), start=1):
        y_stop = min(height, y_start + batch_height)
        if y_stop <= y_start:
            continue

        batch = image[y_start:y_stop, :]
        batch_ocr = preprocess_array_for_ocr(
            batch,
            scale=scale,
            dilation=dilation,
            close=close,
            border=border,
        )
        batch_path = batch_dir / f"batch_{batch_count:06d}_{y_start}_{y_stop}.png"
        cv2.imwrite(str(batch_path), batch_ocr)

        try:
            batch_words = run_tesseract_words(
                batch_path,
                min_confidence=min_confidence,
                psm=psm,
            )
        except subprocess.CalledProcessError as error:
            errors.append(f"batch {batch_count} y={y_start}:{y_stop}: {error}")
            continue

        words.extend(
            offset_batch_words(
                batch_words,
                y_start=y_start,
                scale=scale,
                border=border,
            )
        )

        if y_stop == height:
            break

    return (
        dedupe_ocr_words(
            filter_ocr_words_by_height(words, min_word_height=min_word_height)
        ),
        batch_count,
        errors,
    )


def filter_ocr_words_by_height(
    words: list[OcrWord],
    *,
    min_word_height: int,
) -> list[OcrWord]:
    if min_word_height < 1:
        raise ValueError("min_word_height must be at least 1")
    return [word for word in words if word.height >= min_word_height]


def offset_batch_words(
    words: list[OcrWord],
    *,
    y_start: int,
    scale: int,
    border: int,
) -> list[OcrWord]:
    adjusted: list[OcrWord] = []
    scale = max(1, scale)

    for word in words:
        adjusted.append(
            OcrWord(
                text=word.text,
                confidence=word.confidence,
                x=max(0, (word.x - border) // scale),
                y=y_start + max(0, (word.y - border) // scale),
                width=max(1, word.width // scale),
                height=max(1, word.height // scale),
            )
        )

    return adjusted


def dedupe_ocr_words(words: list[OcrWord], *, tolerance: int = 2) -> list[OcrWord]:
    deduped: list[OcrWord] = []

    for word in sorted(words, key=lambda item: item.confidence, reverse=True):
        normalized = normalize_dictionary_word(word.text)
        duplicate = False
        for existing in deduped:
            if normalize_dictionary_word(existing.text) != normalized:
                continue
            if (
                abs(existing.x - word.x) <= tolerance
                and abs(existing.y - word.y) <= tolerance
            ):
                duplicate = True
                break
        if not duplicate:
            deduped.append(word)

    return sorted(deduped, key=lambda item: (item.y, item.x, item.text))


def find_ocr_word(words: list[OcrWord], target: str) -> list[OcrWord]:
    target_folded = target.casefold()
    return [word for word in words if target_folded in word.text.casefold()]


def default_dictionary_path() -> Path | None:
    for path in DICTIONARY_CANDIDATES:
        if path.exists():
            return path
    return None


def load_word_dictionary(
    dictionary_path: Path | None,
    *,
    min_word_length: int,
) -> set[str]:
    if dictionary_path is None:
        return set()

    if not dictionary_path.exists():
        raise FileNotFoundError(f"Dictionary not found: {dictionary_path}")

    words: set[str] = set()
    with dictionary_path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            normalized = normalize_dictionary_word(line.strip())
            if len(normalized) >= min_word_length:
                words.add(normalized)
    return words


def normalize_dictionary_word(word: str) -> str:
    return "".join(character for character in word.casefold() if character.isalpha())


def normalized_ocr_tokens(text: str, *, min_word_length: int) -> list[str]:
    tokens: list[str] = []
    for match in WORD_PATTERN.findall(text):
        normalized = normalize_dictionary_word(match)
        if len(normalized) >= min_word_length:
            tokens.append(normalized)
    return tokens


def valid_ocr_word_boxes(
    words: list[OcrWord],
    *,
    dictionary_words: set[str],
    min_word_length: int,
    min_word_height: int,
) -> list[tuple[OcrWord, str]]:
    valid_words: list[tuple[OcrWord, str]] = []

    for word in words:
        if word.height < min_word_height:
            continue
        normalized = normalize_dictionary_word(word.text)
        if len(normalized) < min_word_length:
            continue
        if normalized in dictionary_words:
            valid_words.append((word, normalized))

    return valid_words


def find_word_template_matches(
    image_path: Path,
    word: str,
    *,
    top_n: int = 5,
    font_size: int = 28,
    image_scale: int = 4,
    dilation: int = 1,
) -> list[TemplateMatch]:
    if not word:
        return []

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if image_scale > 1:
        binary = cv2.resize(
            binary,
            None,
            fx=image_scale,
            fy=image_scale,
            interpolation=cv2.INTER_NEAREST,
        )

    if dilation > 0:
        kernel_size = max(1, dilation * 2 + 1)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        binary = cv2.dilate(binary, kernel, iterations=1)

    template = render_word_template(word, font_size=font_size)
    if template.shape[0] > binary.shape[0] or template.shape[1] > binary.shape[1]:
        return []

    source = binary.astype(np.float32) / 255.0
    target = template.astype(np.float32) / 255.0
    scores = cv2.matchTemplate(source, target, cv2.TM_CCOEFF_NORMED)

    matches: list[TemplateMatch] = []
    for _ in range(top_n):
        _, max_value, _, max_location = cv2.minMaxLoc(scores)
        if not np.isfinite(max_value):
            break

        x, y = max_location
        matches.append(
            TemplateMatch(
                word=word,
                score=float(max_value),
                x=x,
                y=y,
                width=template.shape[1],
                height=template.shape[0],
            )
        )

        left = max(0, x - template.shape[1] // 2)
        right = min(scores.shape[1], x + template.shape[1] // 2)
        top = max(0, y - template.shape[0] // 2)
        bottom = min(scores.shape[0], y + template.shape[0] // 2)
        scores[top:bottom, left:right] = -1

    return matches


def render_word_template(word: str, *, font_size: int) -> np.ndarray:
    font = load_font(font_size)
    probe = Image.new("L", (1, 1), 0)
    draw = ImageDraw.Draw(probe)
    bbox = draw.textbbox((0, 0), word, font=font)
    text_width = int(bbox[2] - bbox[0])
    text_height = int(bbox[3] - bbox[1])
    padding = max(4, font_size // 5)

    image = Image.new(
        "L",
        (text_width + padding * 2, text_height + padding * 2),
        0,
    )
    draw = ImageDraw.Draw(image)
    draw.text((padding - bbox[0], padding - bbox[1]), word, fill=255, font=font)
    array = np.array(image)
    _, binary = cv2.threshold(array, 1, 255, cv2.THRESH_BINARY)
    return binary


def load_font(font_size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for font_path in FONT_CANDIDATES:
        path = Path(font_path)
        if path.exists():
            return ImageFont.truetype(str(path), font_size)
    return ImageFont.load_default()


def save_template_overlay(
    image_path: Path,
    output_path: Path,
    matches: list[TemplateMatch],
    *,
    image_scale: int,
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if image_scale > 1:
        image = cv2.resize(
            image,
            None,
            fx=image_scale,
            fy=image_scale,
            interpolation=cv2.INTER_NEAREST,
        )

    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for match in matches:
        cv2.rectangle(
            overlay,
            (match.x, match.y),
            (match.x + match.width, match.y + match.height),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            overlay,
            f"{match.score:.2f}",
            (match.x, max(0, match.y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)


def scan_widths_for_ocr(
    *,
    width_start: int,
    max_number: int,
    chart_height: int = DEFAULT_CHART_HEIGHT,
    cell_size: int,
    square_dots: bool = False,
    scale: int,
    dilation: int,
    close: int,
    min_confidence: float,
    psm: int,
    word: str,
    dictionary_path: Path | None = None,
    min_word_length: int = 3,
    min_word_height: int = DEFAULT_MIN_WORD_HEIGHT,
    ocr_batch_height: int = DEFAULT_OCR_BATCH_HEIGHT,
    ocr_batch_overlap: int = DEFAULT_OCR_BATCH_OVERLAP,
    csv_path: Path | None = DEFAULT_SCAN_CSV_PATH,
    jsonl_path: Path | None = DEFAULT_SCAN_JSONL_PATH,
    review_cache_dir: Path | None = DEFAULT_REVIEW_CACHE_DIR,
    found_word_dir: Path | None = DEFAULT_FOUND_WORD_DIR,
    review_cache_size: int = DEFAULT_REVIEW_CACHE_SIZE,
    image_dir: Path | None = None,
    keep_images: bool = False,
    progress_every: int = 25,
    workers: int = DEFAULT_WORKERS,
) -> pd.DataFrame:
    """Generate prime images for widths and return OCR results as a DataFrame."""

    tesseract = shutil.which("tesseract")
    if tesseract is None:
        raise RuntimeError(
            "Tesseract is not installed. On macOS: brew install tesseract"
        )

    dictionary_words = load_word_dictionary(
        dictionary_path,
        min_word_length=min_word_length,
    )
    dictionary_source = str(dictionary_path) if dictionary_path is not None else ""
    print(
        f"Dictionary words loaded: {len(dictionary_words)} "
        f"from {dictionary_source or 'none'}"
    )

    if image_dir is not None:
        image_dir.mkdir(parents=True, exist_ok=True)
        return _scan_widths_for_ocr_in_dir(
            width_start=width_start,
            max_number=max_number,
            chart_height=chart_height,
            cell_size=cell_size,
            square_dots=square_dots,
            scale=scale,
            dilation=dilation,
            close=close,
            min_confidence=min_confidence,
            psm=psm,
            word=word,
            dictionary_words=dictionary_words,
            dictionary_source=dictionary_source,
            min_word_length=min_word_length,
            min_word_height=min_word_height,
            ocr_batch_height=ocr_batch_height,
            ocr_batch_overlap=ocr_batch_overlap,
            csv_path=csv_path,
            jsonl_path=jsonl_path,
            review_cache_dir=review_cache_dir,
            found_word_dir=found_word_dir,
            review_cache_size=review_cache_size,
            image_dir=image_dir,
            keep_images=keep_images,
            progress_every=progress_every,
            workers=workers,
        )

    with TemporaryDirectory(prefix="prime_ocr_widths_") as temp_dir:
        return _scan_widths_for_ocr_in_dir(
            width_start=width_start,
            max_number=max_number,
            chart_height=chart_height,
            cell_size=cell_size,
            square_dots=square_dots,
            scale=scale,
            dilation=dilation,
            close=close,
            min_confidence=min_confidence,
            psm=psm,
            word=word,
            dictionary_words=dictionary_words,
            dictionary_source=dictionary_source,
            min_word_length=min_word_length,
            min_word_height=min_word_height,
            ocr_batch_height=ocr_batch_height,
            ocr_batch_overlap=ocr_batch_overlap,
            csv_path=csv_path,
            jsonl_path=jsonl_path,
            review_cache_dir=review_cache_dir,
            found_word_dir=found_word_dir,
            review_cache_size=review_cache_size,
            image_dir=Path(temp_dir),
            keep_images=False,
            progress_every=progress_every,
            workers=workers,
        )


def _scan_widths_for_ocr_in_dir(
    *,
    width_start: int,
    max_number: int,
    chart_height: int,
    cell_size: int,
    square_dots: bool,
    scale: int,
    dilation: int,
    close: int,
    min_confidence: float,
    psm: int,
    word: str,
    dictionary_words: set[str],
    dictionary_source: str,
    min_word_length: int,
    min_word_height: int,
    ocr_batch_height: int,
    ocr_batch_overlap: int,
    csv_path: Path | None,
    jsonl_path: Path | None,
    review_cache_dir: Path | None,
    found_word_dir: Path | None,
    review_cache_size: int,
    image_dir: Path,
    keep_images: bool,
    progress_every: int,
    workers: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if width_start < 1:
        raise ValueError("width_start must be at least 1")
    if max_number < 1:
        raise ValueError("max_number must be at least 1")
    if chart_height < 1:
        raise ValueError("chart_height must be at least 1")
    if min_word_height < 1:
        raise ValueError("min_word_height must be at least 1")
    if workers < 1:
        raise ValueError("workers must be at least 1")

    final_width = (max_number + chart_height - 1) // chart_height
    width_values = list(range(width_start, final_width + 1))
    total = len(width_values)
    jobs = [
        OcrWidthJob(
            index=index,
            total=total,
            width=width,
            max_number=max_number,
            chart_height=chart_height,
            cell_size=cell_size,
            square_dots=square_dots,
            scale=scale,
            dilation=dilation,
            close=close,
            min_confidence=min_confidence,
            psm=psm,
            word=word,
            dictionary_source=dictionary_source,
            min_word_length=min_word_length,
            min_word_height=min_word_height,
            ocr_batch_height=ocr_batch_height,
            ocr_batch_overlap=ocr_batch_overlap,
            image_dir=image_dir,
            keep_images=keep_images,
            review_cache_dir=review_cache_dir,
            found_word_dir=found_word_dir,
            review_cache_size=review_cache_size,
        )
        for index, width in enumerate(width_values, start=1)
    ]

    jsonl_file = None
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = jsonl_path.open("w", encoding="utf-8")
        print(f"Streaming OCR width scan JSONL: {jsonl_path}")

    try:
        if workers == 1:
            for completed, job in enumerate(jobs, start=1):
                row = _process_ocr_width_job(job, dictionary_words)
                if row is not None:
                    print_scan_result(row, index=completed, total=total)
                    _write_scan_row(row=row, rows=rows, jsonl_file=jsonl_file)

                if progress_every > 0 and (
                    completed == 1
                    or completed % progress_every == 0
                    or completed == total
                ):
                    print(
                        f"scanned {completed}/{total} widths; latest width={job.width}"
                    )
        else:
            _run_parallel_scan_jobs(
                jobs=jobs,
                dictionary_words=dictionary_words,
                workers=workers,
                progress_every=progress_every,
                rows=rows,
                jsonl_file=jsonl_file,
            )
    finally:
        if jsonl_file is not None:
            jsonl_file.close()

    df = pd.DataFrame(rows)
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved OCR width scan CSV: {csv_path}")

    return df


def _init_ocr_worker(dictionary_words: set[str]) -> None:
    global _WORKER_DICTIONARY_WORDS
    _WORKER_DICTIONARY_WORDS = dictionary_words


def _process_ocr_width_job_in_worker(job: OcrWidthJob) -> dict[str, object] | None:
    return _process_ocr_width_job(job, _WORKER_DICTIONARY_WORDS)


def _process_ocr_width_job(
    job: OcrWidthJob,
    dictionary_words: set[str],
) -> dict[str, object] | None:
    chart_max_number = min(job.max_number, job.width * job.chart_height)
    image_path = job.image_dir / f"prime_width_{job.width}.png"
    ocr_image_path = job.image_dir / f"prime_width_{job.width}_ocr.png"
    batch_dir = job.image_dir / f"prime_width_{job.width}_ocr_batches"
    ocr_error = ""
    batch_errors: list[str] = []
    ocr_batch_count = 0
    words: list[OcrWord] = []

    try:
        meta = generate_prime_dot_png(
            width=job.width,
            max_number=chart_max_number,
            output_path=image_path,
            cell_size=job.cell_size,
            dot_radius=job.cell_size if job.square_dots else None,
        )

        try:
            preprocess_for_ocr(
                image_path,
                ocr_image_path,
                scale=job.scale,
                dilation=job.dilation,
                close=job.close,
            )
            words, ocr_batch_count, batch_errors = run_tesseract_words_in_batches(
                image_path,
                batch_dir=batch_dir,
                min_confidence=job.min_confidence,
                psm=job.psm,
                scale=job.scale,
                dilation=job.dilation,
                close=job.close,
                batch_height=job.ocr_batch_height,
                batch_overlap=job.ocr_batch_overlap,
                min_word_height=job.min_word_height,
            )
        except (
            RuntimeError,
            subprocess.CalledProcessError,
            FileNotFoundError,
            ValueError,
        ) as error:
            ocr_error = str(error)

        if batch_errors:
            preview_errors = "; ".join(batch_errors[:3])
            extra_errors = len(batch_errors) - 3
            if extra_errors > 0:
                preview_errors = f"{preview_errors}; +{extra_errors} more"
            ocr_error = "; ".join(part for part in (ocr_error, preview_errors) if part)

        summary = summarize_ocr_words(
            words,
            target_word=job.word,
            dictionary_words=dictionary_words,
            min_word_length=job.min_word_length,
            min_word_height=job.min_word_height,
        )
        row = {
            "width": job.width,
            "max_number": chart_max_number,
            "scan_max_number": job.max_number,
            "chart_height": job.chart_height,
            "cell_size": job.cell_size,
            "square_dots": job.square_dots,
            "scale": job.scale,
            "dilation": job.dilation,
            "close": job.close,
            "psm": job.psm,
            "min_confidence": job.min_confidence,
            "target_word": job.word,
            "dictionary": job.dictionary_source,
            "dictionary_word_count": len(dictionary_words),
            "min_word_length": job.min_word_length,
            "min_word_height": job.min_word_height,
            "ocr_batch_height": job.ocr_batch_height,
            "ocr_batch_overlap": job.ocr_batch_overlap,
            "ocr_batch_count": ocr_batch_count,
            "ocr_batch_error_count": len(batch_errors),
            "image_width_px": meta.image_width,
            "image_height_px": meta.image_height,
            "primes_plotted": meta.primes_plotted,
            "ocr_error": ocr_error,
            **summary,
        }

        if int(str(row["valid_word_count"])) == 0:
            return None

        if job.found_word_dir is not None:
            found_word_paths = save_found_word_crops(
                image_path=image_path,
                words=words,
                dictionary_words=dictionary_words,
                min_word_length=job.min_word_length,
                min_word_height=job.min_word_height,
                output_dir=job.found_word_dir,
                graph_width=job.width,
                graph_height=job.chart_height,
                scale=job.scale,
                dilation=job.dilation,
                close=job.close,
            )
            row["found_word_images"] = " ".join(str(path) for path in found_word_paths)
        else:
            row["found_word_images"] = ""

        if job.review_cache_dir is not None and job.review_cache_size > 0:
            review_image_path = save_review_cache_image(
                image_path=image_path,
                ocr_image_path=ocr_image_path,
                row=row,
                cache_dir=job.review_cache_dir,
                cache_size=job.review_cache_size,
            )
            row["review_image"] = str(review_image_path)
        else:
            row["review_image"] = ""

        return row
    finally:
        if not job.keep_images:
            image_path.unlink(missing_ok=True)
            ocr_image_path.unlink(missing_ok=True)
        shutil.rmtree(batch_dir, ignore_errors=True)


def _write_scan_row(
    *,
    row: dict[str, object],
    rows: list[dict[str, object]],
    jsonl_file: TextIO | None,
) -> None:
    rows.append(row)
    if jsonl_file is not None:
        jsonl_file.write(json.dumps(row, ensure_ascii=True) + "\n")
        jsonl_file.flush()


def _run_parallel_scan_jobs(
    *,
    jobs: list[OcrWidthJob],
    dictionary_words: set[str],
    workers: int,
    progress_every: int,
    rows: list[dict[str, object]],
    jsonl_file: TextIO | None,
) -> None:
    if not jobs:
        return

    jobs_iter = iter(jobs)
    max_pending = min(len(jobs), max(workers * 2, workers))
    completed = 0

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_ocr_worker,
        initargs=(dictionary_words,),
    ) as executor:
        pending = {}

        def submit_next() -> None:
            try:
                job = next(jobs_iter)
            except StopIteration:
                return
            pending[executor.submit(_process_ocr_width_job_in_worker, job)] = job

        for _ in range(max_pending):
            submit_next()

        while pending:
            for future in as_completed(tuple(pending)):
                job = pending.pop(future)
                completed += 1
                try:
                    row = future.result()
                except Exception as error:
                    print(f"width {job.width} failed: {error}")
                else:
                    if row is not None:
                        print_scan_result(row, index=completed, total=job.total)
                        _write_scan_row(row=row, rows=rows, jsonl_file=jsonl_file)

                if progress_every > 0 and (
                    completed == 1
                    or completed % progress_every == 0
                    or completed == job.total
                ):
                    print(
                        f"scanned {completed}/{job.total} widths; latest width={job.width}"
                    )

                submit_next()
                break


def print_scan_result(row: dict[str, object], *, index: int, total: int) -> None:
    message = {
        "index": index,
        "total": total,
        "width": row["width"],
        "tokens": row["ocr_token_count"],
        "valid_words": row["valid_words"],
        "valid_word_count": row["valid_word_count"],
        "valid_letters": row["valid_letters"],
        "target_found": row["valid_target_found"],
        "max_confidence": row["ocr_max_confidence"],
        "ocr_batches": row["ocr_batch_count"],
        "ocr_batch_errors": row["ocr_batch_error_count"],
        "review_image": row["review_image"],
        "found_word_images": row["found_word_images"],
        "error": row["ocr_error"],
    }
    print(json.dumps(message, ensure_ascii=True))


def save_review_cache_image(
    *,
    image_path: Path,
    ocr_image_path: Path,
    row: dict[str, object],
    cache_dir: Path,
    cache_size: int,
    panel_size: tuple[int, int] = (420, 720),
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    width = int(str(row["width"]))
    cache_path = cache_dir / f"width_{width:04d}.png"

    original = load_review_panel(image_path, panel_size=panel_size, fill=(8, 8, 8))
    ocr_ready = load_review_panel(
        ocr_image_path,
        panel_size=panel_size,
        fill=(255, 255, 255),
    )
    label_height = 92
    gap = 16
    canvas = Image.new(
        "RGB",
        (panel_size[0] * 2 + gap, panel_size[1] + label_height),
        (242, 242, 242),
    )
    canvas.paste(original, (0, label_height))
    canvas.paste(ocr_ready, (panel_size[0] + gap, label_height))

    draw = ImageDraw.Draw(canvas)
    title = (
        f"width={row['width']} max={row['max_number']} "
        f"valid={row['valid_word_count']} raw_tokens={row['ocr_token_count']}"
    )
    valid_words = str(row["valid_words"])[:140] or "no dictionary words"
    draw.text((12, 10), title, fill=(0, 0, 0))
    draw.text((12, 32), f"valid: {valid_words}", fill=(0, 90, 0))
    draw.text((12, 54), "non-dictionary OCR discarded", fill=(80, 80, 80))
    draw.text((12, 76), "left: prime dots    right: OCR-ready", fill=(80, 80, 80))

    canvas.save(cache_path)
    prune_review_cache(cache_dir=cache_dir, cache_size=cache_size)
    return cache_path


def save_found_word_crops(
    *,
    image_path: Path,
    words: list[OcrWord],
    dictionary_words: set[str],
    min_word_length: int,
    min_word_height: int,
    output_dir: Path,
    graph_width: int,
    graph_height: int,
    scale: int,
    dilation: int,
    close: int,
    context_scale: float = 1.5,
) -> list[Path]:
    valid_words = valid_ocr_word_boxes(
        words,
        dictionary_words=dictionary_words,
        min_word_length=min_word_length,
        min_word_height=min_word_height,
    )
    if not valid_words:
        return []

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    ocr_ready = preprocess_array_for_ocr(
        image,
        scale=scale,
        dilation=dilation,
        close=close,
        border=0,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    image_height, image_width = ocr_ready.shape[:2]
    scale = max(1, scale)

    for word, normalized in valid_words:
        box_x_start = word.x * scale
        box_y_start = word.y * scale
        box_x_stop = (word.x + word.width) * scale
        box_y_stop = (word.y + word.height) * scale
        box_width = max(1, box_x_stop - box_x_start)
        box_height = max(1, box_y_stop - box_y_start)
        crop_width = max(box_width, int(round(box_width * context_scale)))
        crop_height = max(box_height, int(round(box_height * context_scale)))
        center_x = (box_x_start + box_x_stop) // 2
        center_y = (box_y_start + box_y_stop) // 2

        x_start = max(0, center_x - crop_width // 2)
        y_start = max(0, center_y - crop_height // 2)
        x_stop = min(image_width, x_start + crop_width)
        y_stop = min(image_height, y_start + crop_height)
        x_start = max(0, x_stop - crop_width)
        y_start = max(0, y_stop - crop_height)
        if x_stop <= x_start or y_stop <= y_start:
            continue

        crop = cv2.cvtColor(
            ocr_ready[y_start:y_stop, x_start:x_stop], cv2.COLOR_GRAY2BGR
        )
        rectangle_start = (max(0, box_x_start - x_start), max(0, box_y_start - y_start))
        rectangle_stop = (
            min(crop.shape[1] - 1, box_x_stop - x_start),
            min(crop.shape[0] - 1, box_y_stop - y_start),
        )
        cv2.rectangle(crop, rectangle_start, rectangle_stop, (255, 220, 120), 2)

        file_name = f"{safe_filename_word(normalized)}-{graph_width}-{graph_height}.png"
        output_path = unique_path(output_dir / file_name)
        cv2.imwrite(str(output_path), crop)
        saved_paths.append(output_path)

    return saved_paths


def safe_filename_word(word: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", word).strip("_").lower()
    return cleaned or "word"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    for index in range(2, 10_000):
        candidate = path.with_name(f"{stem}-{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find unique path for {path}")


def load_review_panel(
    image_path: Path,
    *,
    panel_size: tuple[int, int],
    fill: tuple[int, int, int],
) -> Image.Image:
    panel = Image.new("RGB", panel_size, fill)
    if not image_path.exists():
        return panel

    with Image.open(image_path) as image:
        preview = image.convert("RGB")
        preview.thumbnail(panel_size, Image.Resampling.NEAREST)
        x = (panel_size[0] - preview.width) // 2
        y = (panel_size[1] - preview.height) // 2
        panel.paste(preview, (x, y))
    return panel


def prune_review_cache(*, cache_dir: Path, cache_size: int) -> None:
    cache_files = sorted(
        cache_dir.glob("width_*.png"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for stale_path in cache_files[cache_size:]:
        stale_path.unlink(missing_ok=True)


def summarize_ocr_words(
    words: list[OcrWord],
    *,
    target_word: str,
    dictionary_words: set[str],
    min_word_length: int,
    min_word_height: int,
) -> dict[str, object]:
    words = filter_ocr_words_by_height(words, min_word_height=min_word_height)
    texts = [word.text for word in words]
    confidences = [word.confidence for word in words]
    normalized_tokens = []
    for text in texts:
        normalized = normalize_dictionary_word(text)
        if len(normalized) >= min_word_length:
            normalized_tokens.append(normalized)
    valid_tokens = [token for token in normalized_tokens if token in dictionary_words]
    invalid_tokens = [
        token for token in normalized_tokens if token not in dictionary_words
    ]
    valid_letters = "".join(valid_tokens)
    target_normalized = normalize_dictionary_word(target_word)
    valid_target_matches = [
        token
        for token in valid_tokens
        if target_normalized and target_normalized in token
    ]

    return {
        "ocr_token_count": len(texts),
        "normalized_token_count": len(normalized_tokens),
        "valid_word_count": len(valid_tokens),
        "valid_words": " ".join(valid_tokens),
        "valid_letters": valid_letters,
        "valid_unique_letters": "".join(sorted(set(valid_letters.upper()))),
        "invalid_word_count": len(invalid_tokens),
        "ocr_mean_confidence": float(np.mean(confidences)) if confidences else None,
        "ocr_max_confidence": max(confidences) if confidences else None,
        "valid_target_found": bool(valid_target_matches),
        "valid_target_matches": " ".join(valid_target_matches),
    }


def print_scan_config(args: argparse.Namespace) -> None:
    final_width = (args.max_number + args.chart_height - 1) // args.chart_height
    config = {
        "width_start": args.width_start,
        "derived_final_width": final_width,
        "chart_height": args.chart_height,
        "max_number": args.max_number,
        "cell_size": args.cell_size,
        "square_dots": args.square_dots,
        "scale": args.scale,
        "dilation": args.dilation,
        "close": args.close,
        "psm": args.psm,
        "ocr_batch_height": args.ocr_batch_height,
        "ocr_batch_overlap": args.ocr_batch_overlap,
        "min_confidence": args.min_confidence,
        "target_word": args.word,
        "dictionary": None if args.no_dictionary else args.dictionary,
        "min_word_length": args.min_word_length,
        "min_word_height": args.min_word_height,
        "csv": args.csv,
        "jsonl": args.jsonl,
        "cache_dir": None if args.no_cache else args.cache_dir,
        "found_word_dir": None if args.no_found_crops else args.found_word_dir,
        "cache_size": args.cache_size,
        "keep_images": args.keep_images,
        "scan_dir": args.scan_dir,
        "workers": args.workers,
    }
    print("Scan config:")
    print(pd.Series(config))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--word", default="PRIME")
    parser.add_argument("--width", type=int, default=29)
    parser.add_argument("--width-start", type=int, default=2)
    parser.add_argument("--chart-height", type=int, default=DEFAULT_CHART_HEIGHT)
    parser.add_argument("--max-number", type=int, default=1_000_000)
    parser.add_argument("--cell-size", type=int, default=1)
    parser.add_argument(
        "--square-dots",
        action="store_true",
        help="Render prime cells as filled squares instead of round dots.",
    )
    parser.add_argument("--no-generate", action="store_true")
    parser.add_argument("--ocr-image", type=Path, default=DEFAULT_OCR_IMAGE_PATH)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_TEMPLATE_OVERLAY_PATH)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--dilation", type=int, default=1)
    parser.add_argument("--close", type=int, default=1)
    parser.add_argument("--font-size", type=int, default=28)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--min-confidence", type=float, default=90.0)
    parser.add_argument("--psm", type=int, default=11)
    parser.add_argument(
        "--ocr-batch-height", type=int, default=DEFAULT_OCR_BATCH_HEIGHT
    )
    parser.add_argument(
        "--ocr-batch-overlap", type=int, default=DEFAULT_OCR_BATCH_OVERLAP
    )
    parser.add_argument("--dictionary", type=Path, default=default_dictionary_path())
    parser.add_argument("--no-dictionary", action="store_true")
    parser.add_argument("--min-word-length", type=int, default=3)
    parser.add_argument("--min-word-height", type=int, default=DEFAULT_MIN_WORD_HEIGHT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_SCAN_CSV_PATH)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_SCAN_JSONL_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_REVIEW_CACHE_DIR)
    parser.add_argument("--found-word-dir", type=Path, default=DEFAULT_FOUND_WORD_DIR)
    parser.add_argument("--no-found-crops", action="store_true")
    parser.add_argument("--cache-size", type=int, default=DEFAULT_REVIEW_CACHE_SIZE)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--scan-dir", type=Path)
    parser.add_argument("--keep-images", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.single:
        print_scan_config(args)
        dictionary_path = None if args.no_dictionary else args.dictionary
        review_cache_dir = None if args.no_cache else args.cache_dir
        found_word_dir = None if args.no_found_crops else args.found_word_dir
        df = scan_widths_for_ocr(
            width_start=args.width_start,
            max_number=args.max_number,
            chart_height=args.chart_height,
            cell_size=args.cell_size,
            square_dots=args.square_dots,
            scale=args.scale,
            dilation=args.dilation,
            close=args.close,
            min_confidence=args.min_confidence,
            psm=args.psm,
            word=args.word,
            dictionary_path=dictionary_path,
            min_word_length=args.min_word_length,
            min_word_height=args.min_word_height,
            ocr_batch_height=args.ocr_batch_height,
            ocr_batch_overlap=args.ocr_batch_overlap,
            csv_path=args.csv,
            jsonl_path=args.jsonl,
            review_cache_dir=review_cache_dir,
            found_word_dir=found_word_dir,
            review_cache_size=args.cache_size,
            image_dir=args.scan_dir,
            keep_images=args.keep_images,
            progress_every=args.progress_every,
            workers=args.workers,
        )
        print("OCR scan results:")
        print(df)
        return

    if not args.no_generate:
        generate_prime_image(
            output_path=args.image,
            width=args.width,
            max_number=args.max_number,
            cell_size=args.cell_size,
            square_dots=args.square_dots,
        )

    ocr_image = preprocess_for_ocr(
        args.image,
        args.ocr_image,
        scale=args.scale,
        dilation=args.dilation,
        close=args.close,
    )
    print(f"OCR-ready image: {ocr_image}")

    try:
        with TemporaryDirectory(prefix="prime_single_ocr_batches_") as temp_dir:
            words, batch_count, batch_errors = run_tesseract_words_in_batches(
                args.image,
                batch_dir=Path(temp_dir),
                min_confidence=args.min_confidence,
                psm=args.psm,
                scale=args.scale,
                dilation=args.dilation,
                close=args.close,
                batch_height=args.ocr_batch_height,
                batch_overlap=args.ocr_batch_overlap,
                min_word_height=args.min_word_height,
            )
        print(f"OCR batches: {batch_count}; batch errors: {len(batch_errors)}")
        for error in batch_errors[:3]:
            print(error)
    except (RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        print(error)
        words = []

    if words:
        print("OCR words:")
        for word in words:
            print(word)
    else:
        print("OCR found no confident words.")

    if args.word:
        ocr_matches = find_ocr_word(words, args.word)
        print(f"OCR matches for {args.word!r}: {ocr_matches}")

        template_matches = find_word_template_matches(
            args.image,
            args.word,
            top_n=args.top_n,
            font_size=args.font_size,
            image_scale=args.scale,
            dilation=args.dilation,
        )
        print(f"Template matches for {args.word!r}:")
        for match in template_matches:
            print(match)

        if template_matches:
            save_template_overlay(
                args.image,
                args.overlay,
                template_matches,
                image_scale=args.scale,
            )
            print(f"Template overlay: {args.overlay}")


if __name__ == "__main__":
    main()
