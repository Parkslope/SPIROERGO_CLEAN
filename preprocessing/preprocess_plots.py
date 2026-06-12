#!/usr/bin/env python3
"""
CPET Waterman Plot Preprocessing Script (bounding-box version)

- Reads PDF files from:  D:\\Spiroergo\\OnlyPlots
- Writes 9 cropped plots to: D:\\Spiroergo\\PreprocessedPlots\\<pdf_name>\\plot_01.png ... plot_09.png

Strategy:
1) Render first page of each PDF to an image.
2) Find the tight bounding box around all non-white content.
3) Uniformly split that bounding box into a 3x3 grid.
"""

import argparse
import os
import sys
import io
import logging
from pathlib import Path
from typing import List

# ----------------------------------------------------------------------
# Imports (with optional auto-install)
# ----------------------------------------------------------------------
try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not found. Installing...")
    os.system("pip install PyMuPDF")
    import fitz  # type: ignore

try:
    from PIL import Image
except ImportError:
    print("Pillow not found. Installing...")
    os.system("pip install Pillow")
    from PIL import Image  # type: ignore

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found. Installing...")
    os.system("pip install tqdm")
    from tqdm import tqdm  # type: ignore

try:
    import numpy as np
except ImportError:
    print("numpy not found. Installing...")
    os.system("pip install numpy")
    import numpy as np  # type: ignore

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# PDF -> Image
# ----------------------------------------------------------------------
def load_pdf_as_image(pdf_path: Path, zoom: float = 2.0) -> Image.Image:
    """
    Render the first page of a PDF as a PIL Image (RGB).
    """
    try:
        doc = fitz.open(pdf_path)
        if doc.page_count == 0:
            raise ValueError("PDF has no pages")

        page = doc[0]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        doc.close()
        return img
    except Exception as e:
        logger.error(f"Failed to load PDF '{pdf_path}': {e}")
        raise


# ----------------------------------------------------------------------
# Bounding-box + grid cropping
# ----------------------------------------------------------------------
def find_content_bbox(img: Image.Image,
                      threshold: int = 240,
                      min_frac: float = 0.01):
    """
    Find the bounding box of the 3 plot rows (ignore footer / large margins).

    Strategy:
    - Convert to grayscale and mark "dark" pixels (non-white).
    - For each row and column, compute fraction of dark pixels.
    - Group consecutive rows with enough dark pixels into segments.
    - Take the 3 *largest* row segments (= 3 plot rows).
    - y_min / y_max = min / max of those 3 segments.
    - For columns we just take the full span with enough dark pixels.

    Returns:
        (x_min, y_min, x_max, y_max)
    """
    gray = img.convert("L")
    arr = np.array(gray)
    h, w = arr.shape

    dark = arr < threshold

    # --- rows ---
    row_frac = dark.mean(axis=1)
    content_rows = np.where(row_frac > min_frac)[0]

    if content_rows.size == 0:
        # Fallback: whole image
        logger.warning("No content rows detected, using full height.")
        y_min, y_max = 0, h
    else:
        # group consecutive rows into segments
        segments = []
        current = [content_rows[0]]
        for r in content_rows[1:]:
            if r == current[-1] + 1:
                current.append(r)
            else:
                segments.append(current)
                current = [r]
        segments.append(current)

        # sort segments by size (largest first)
        segments_sorted = sorted(segments, key=len, reverse=True)

        # take up to 3 largest segments (3 plot rows)
        main_segments = segments_sorted[:3]

        y_min = min(seg[0] for seg in main_segments)
        y_max = max(seg[-1] for seg in main_segments) + 1  # +1 to include last row

    # --- columns ---
    col_frac = dark.mean(axis=0)
    content_cols = np.where(col_frac > min_frac)[0]

    if content_cols.size == 0:
        logger.warning("No content cols detected, using full width.")
        x_min, x_max = 0, w
    else:
        x_min = int(content_cols.min())
        x_max = int(content_cols.max()) + 1

    # final safety clamp
    x_min = max(0, min(x_min, w - 1))
    x_max = max(1, min(x_max, w))
    y_min = max(0, min(y_min, h - 1))
    y_max = max(1, min(y_max, h))

    return x_min, y_min, x_max, y_max



def crop_into_9_panels(img: Image.Image) -> List[Image.Image]:
    """
    Find the content bounding box and split it into a 3x3 grid.

    Returns:
        List of 9 PIL Images in row-major order.
    """
    width, height = img.size

    x_min, y_min, x_max, y_max = find_content_bbox(img)
    logger.debug(f"Content bbox: x[{x_min},{x_max}], y[{y_min},{y_max}]")

    # Safety clamp
    x_min = max(0, min(x_min, width - 1))
    x_max = max(1, min(x_max, width))
    y_min = max(0, min(y_min, height - 1))
    y_max = max(1, min(y_max, height))

    rows = 3
    cols = 3
    box_w = x_max - x_min
    box_h = y_max - y_min

    step_x = box_w // cols
    step_y = box_h // rows

    panels: List[Image.Image] = []

    for r in range(rows):
        for c in range(cols):
            left = x_min + c * step_x
            right = x_min + (c + 1) * step_x if c < cols - 1 else x_max
            top = y_min + r * step_y
            bottom = y_min + (r + 1) * step_y if r < rows - 1 else y_max

            panel = img.crop((left, top, right, bottom))
            panels.append(panel)

    if len(panels) != 9:
        raise RuntimeError(f"Expected 9 panels, got {len(panels)}")

    return panels


# ----------------------------------------------------------------------
# Processing
# ----------------------------------------------------------------------
def process_file(pdf_path: Path, output_dir: Path) -> bool:
    """
    Render, crop, and save 9 panels for a single PDF.

    Returns:
        True on success, False otherwise.
    """
    try:
        name = pdf_path.stem
        out_dir = output_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing {name}")

        img = load_pdf_as_image(pdf_path, zoom=2.0)
        logger.info(f"  Rendered image size: {img.size[0]}x{img.size[1]}")

        panels = crop_into_9_panels(img)

        for i, panel in enumerate(panels, start=1):
            out_path = out_dir / f"plot_{i:02d}.png"
            panel.save(out_path, format="PNG", optimize=True)

        logger.info(f"  Saved 9 panels to {out_dir}")
        return True

    except Exception as e:
        logger.error(f"  ERROR processing '{pdf_path}': {e}")
        return False


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CPET Waterman Plot Preprocessing")
    parser.add_argument("--input_dir",  required=True, help="Directory containing PDF files")
    parser.add_argument("--output_dir", required=True, help="Directory to write cropped panel PNGs")
    args = parser.parse_args()

    logger.info("=== CPET Waterman Plot Preprocessing ===")
    logger.info(f"Input directory:  {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    in_dir = Path(args.input_dir)
    out_root = Path(args.output_dir)

    if not in_dir.exists():
        logger.error(f"Input directory does not exist: {in_dir}")
        return

    out_root.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(
        p for p in in_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".pdf"
    )

    if not pdf_files:
        logger.warning("No PDF files found.")
        return

    logger.info(f"Found {len(pdf_files)} PDF files.")

    success = 0
    fail = 0

    for pdf in tqdm(pdf_files, desc="Processing PDFs"):
        if process_file(pdf, out_root):
            success += 1
        else:
            fail += 1

    logger.info("=== Done ===")
    logger.info(f"Successfully processed: {success}")
    logger.info(f"Failed:                {fail}")
    logger.info(f"Total:                 {len(pdf_files)}")


if __name__ == "__main__":
    main()
