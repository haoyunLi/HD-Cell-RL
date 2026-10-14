#!/usr/bin/env python3
"""Build matched high-resolution H&E segmentation crops for the progress deck.

The nuclear and expanded-cell panels use the exact same source-image window.
Label masks are cropped without interpolation; only their boundaries are drawn.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_IMAGE = Path(
    "workspace_outputs/colorectal_nucleus_based/intermediate/"
    "cropped_visium_hd_colorectal_nucleus_based.png"
)
DEFAULT_GROUND_TRUTH_DIR = Path(
    "/projects/illinois/vetmed/cb/kwang222/Haoyun_Li/Bin2Cell_Validation/"
    "outputs/ground_truth/colorectal_nucleus_based/cells"
)
DEFAULT_NUCLEAR_MASK = DEFAULT_GROUND_TRUTH_DIR / "retained_nuclear_labels.npy"
DEFAULT_CELL_MASK = DEFAULT_GROUND_TRUTH_DIR / "cell_labels.npy"
DEFAULT_MICRONS_PER_PIXEL = 0.2737012522439323
# A fibroblast-rich field containing large, collision-aware whole-cell expansions.
DEFAULT_ROI = (5086, 12378, 5786, 13078)
NUCLEAR_BOUNDARY_COLOR = (226, 45, 146)
WHOLE_CELL_BOUNDARY_COLOR = (0, 213, 235)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create matched H&E crops with nuclear and expanded-cell boundaries."
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--nuclear-mask", type=Path, default=DEFAULT_NUCLEAR_MASK)
    parser.add_argument("--cell-mask", type=Path, default=DEFAULT_CELL_MASK)
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        default=DEFAULT_ROI,
        help="Full-resolution source-image crop in x0 y0 x1 y1 order.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--microns-per-pixel", type=float, default=DEFAULT_MICRONS_PER_PIXEL
    )
    parser.add_argument("--scale-bar-um", type=float, default=20.0)
    parser.add_argument("--boundary-width-px", type=int, default=3)
    return parser.parse_args()


def label_boundaries(labels: np.ndarray, width: int) -> np.ndarray:
    """Return visible 4-connected label boundaries without resizing labels."""
    if labels.ndim != 2:
        raise ValueError(f"Expected a 2D label image, got shape {labels.shape}")
    if width < 1:
        raise ValueError("boundary width must be at least 1 pixel")

    center = labels[1:-1, 1:-1]
    foreground = center != 0
    boundary = foreground & (
        (center != labels[:-2, 1:-1])
        | (center != labels[2:, 1:-1])
        | (center != labels[1:-1, :-2])
        | (center != labels[1:-1, 2:])
    )
    if width > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
        boundary = cv2.dilate(boundary.astype(np.uint8), kernel) > 0
    return boundary


def overlay_boundaries(
    image: np.ndarray,
    boundaries: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.92,
) -> np.ndarray:
    """Blend one boundary color over RGB H&E pixels."""
    if image.shape[:2] != boundaries.shape:
        raise ValueError("Image and boundary arrays must have matching dimensions")
    output = image.astype(np.float32, copy=True)
    color_array = np.asarray(color, dtype=np.float32)
    output[boundaries] = (
        (1.0 - alpha) * output[boundaries] + alpha * color_array
    )
    return np.clip(output, 0, 255).astype(np.uint8)


def overlay_region(
    image: np.ndarray,
    region: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    """Add a light, explicitly annotated tint without changing source geometry."""
    if image.shape[:2] != region.shape:
        raise ValueError("Image and region arrays must have matching dimensions")
    output = image.astype(np.float32, copy=True)
    color_array = np.asarray(color, dtype=np.float32)
    output[region] = (1.0 - alpha) * output[region] + alpha * color_array
    return np.clip(output, 0, 255).astype(np.uint8)


def add_scale_bar(
    image: np.ndarray,
    *,
    microns_per_pixel: float,
    scale_bar_um: float,
) -> np.ndarray:
    """Add a calibrated lower-left scale bar to an RGB panel."""
    if microns_per_pixel <= 0 or scale_bar_um <= 0:
        raise ValueError("Scale values must be positive")
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    width, height = pil_image.size
    bar_length = round(scale_bar_um / microns_per_pixel)
    if bar_length >= width - 48:
        raise ValueError("Scale bar is too long for the selected ROI")

    margin = max(22, round(width * 0.025))
    bar_height = max(7, round(height * 0.008))
    x0 = margin
    y1 = height - margin
    x1 = x0 + bar_length
    y0 = y1 - bar_height
    shadow = max(2, bar_height // 3)
    draw.rectangle((x0 - shadow, y0 - shadow, x1 + shadow, y1 + shadow), fill=(0, 0, 0))
    draw.rectangle((x0, y0, x1, y1), fill=(255, 255, 255))

    label = f"{scale_bar_um:g} µm"
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if font_path.exists():
        font = ImageFont.truetype(str(font_path), size=max(16, round(width * 0.022)))
    else:
        label = f"{scale_bar_um:g} um"
        font = ImageFont.load_default()
    label_box = draw.textbbox((0, 0), label, font=font)
    label_height = label_box[3] - label_box[1]
    text_xy = (x0, y0 - label_height - 10)
    draw.text((text_xy[0] + 2, text_xy[1] + 2), label, font=font, fill=(0, 0, 0))
    draw.text(text_xy, label, font=font, fill=(255, 255, 255))
    return np.asarray(pil_image)


def crop_labels(mask_path: Path, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Memory-map a label image and include a one-pixel boundary halo."""
    x0, y0, x1, y1 = roi
    labels = np.load(mask_path, mmap_mode="r")
    height, width = labels.shape
    if not (1 <= x0 < x1 < width and 1 <= y0 < y1 < height):
        raise ValueError(f"ROI {roi} falls outside label image {width}x{height}")
    return np.asarray(labels[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1])


def main() -> None:
    args = parse_args()
    roi = tuple(args.roi)
    x0, y0, x1, y1 = roi

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(args.image) as source:
        if source.size != tuple(reversed(np.load(args.cell_mask, mmap_mode="r").shape)):
            raise ValueError("Source H&E and cell-mask dimensions do not match")
        he_crop = np.asarray(source.crop((x0, y0, x1, y1)).convert("RGB"))

    nuclear_crop = crop_labels(args.nuclear_mask, roi)
    cell_crop = crop_labels(args.cell_mask, roi)
    nuclear_boundaries = label_boundaries(nuclear_crop, args.boundary_width_px)
    cell_boundaries = label_boundaries(cell_crop, args.boundary_width_px)

    nuclear_panel = overlay_boundaries(
        he_crop, nuclear_boundaries, NUCLEAR_BOUNDARY_COLOR
    )
    nuclear_region = nuclear_crop[1:-1, 1:-1] > 0
    cell_region = cell_crop[1:-1, 1:-1] > 0
    expanded_only_region = cell_region & ~nuclear_region
    cell_panel = overlay_region(
        he_crop, expanded_only_region, WHOLE_CELL_BOUNDARY_COLOR, alpha=0.12
    )
    cell_panel = overlay_boundaries(
        cell_panel, nuclear_boundaries, NUCLEAR_BOUNDARY_COLOR, alpha=0.70
    )
    cell_panel = overlay_boundaries(
        cell_panel, cell_boundaries, WHOLE_CELL_BOUNDARY_COLOR, alpha=0.98
    )
    nuclear_panel = add_scale_bar(
        nuclear_panel,
        microns_per_pixel=args.microns_per_pixel,
        scale_bar_um=args.scale_bar_um,
    )
    cell_panel = add_scale_bar(
        cell_panel,
        microns_per_pixel=args.microns_per_pixel,
        scale_bar_um=args.scale_bar_um,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nuclear_path = args.output_dir / "nuclear_segmentation_cell_roi.png"
    cell_path = args.output_dir / "expanded_cell_mask_cell_roi.png"
    Image.fromarray(nuclear_panel).save(nuclear_path, optimize=True)
    Image.fromarray(cell_panel).save(cell_path, optimize=True)

    print(f"ROI: x={x0}:{x1}, y={y0}:{y1}")
    print(
        "Physical field: "
        f"{(x1 - x0) * args.microns_per_pixel:.1f} × "
        f"{(y1 - y0) * args.microns_per_pixel:.1f} µm"
    )
    print(f"Nuclear labels in crop: {np.unique(nuclear_crop[nuclear_crop > 0]).size}")
    print(f"Cell labels in crop: {np.unique(cell_crop[cell_crop > 0]).size}")
    print(nuclear_path)
    print(cell_path)


if __name__ == "__main__":
    main()
