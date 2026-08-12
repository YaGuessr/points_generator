#!/usr/bin/env python3
"""
Script for finding panorama points on a map by overlay color
and converting their pixel coordinates into geographic coordinates.

1. Vectorized (NumPy) computation of each pixel's similarity to the target color (0..100%), producing a mask of pixels that passed the threshold
2. The mask is reduced to one point per panorama marker via a distance-transform + local-maxima + greedy min-distance pass (scipy.ndimage + scipy.spatial.cKDTree)
  which stays correct even when many markers' blobs touch or overlap in dense areas (e.g. a busy downtown), unlike plain connected-component + centroid
3. Marker centers are converted to latitude/longitude via linear interpolation between the map corners, in a single vectorized pass for all points at once
"""

import sys
import time
import logging
import argparse
import numpy as np
from PIL import Image
from typing import Tuple
from scipy import ndimage
from scipy.spatial import cKDTree


def parse_color(color_str: str) -> Tuple[int, int, int, int]:
    """
    Parses a color from a string in RGB/RGBA or hex format. Returns (r,g,b,a)

    color_str: str - a color string in "R,G,B[,A]" or hex ("#RRGGBB[AA]" / "#RGB[A]") format

    Returns Tuple[int, int, int, int] - the color as (r, g, b, a); if alpha is not given, it defaults to 255 (opaque)
    """
    color_str = color_str.strip()
    if color_str.startswith('#'):
        hex_color = color_str[1:]
        if len(hex_color) == 8:      # RRGGBBAA
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4, 6))
        elif len(hex_color) == 6:    # RRGGBB
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)) + (255,)
        elif len(hex_color) == 4:    # short RGBA
            return tuple(int(c * 2, 16) for c in hex_color)
        elif len(hex_color) == 3:    # short RGB
            return tuple(int(c * 2, 16) for c in hex_color) + (255,)
        raise ValueError(f"Invalid hex color length: {color_str}")
    if ',' in color_str:
        parts = [int(p.strip()) for p in color_str.split(',')]
        if len(parts) == 4:
            return tuple(parts)
        elif len(parts) == 3:
            return tuple(parts) + (255,)
    raise ValueError(f"Invalid color format: {color_str}")


def find_color_mask(pixels: np.ndarray, target_color: Tuple[int, int, int, int], similarity_threshold: float) -> np.ndarray:
    """
    Computes each image pixel's similarity to the target color and returns a mask of pixels that passed the similarity threshold

    pixels: np.ndarray - array of image pixels of shape HxWx4 (RGBA)
    target_color: Tuple[int, int, int, int] - the target color as (r, g, b, a); the alpha channel is not used in the comparison
    similarity_threshold: float - color similarity threshold in percent (0-100)

    Returns np.ndarray - boolean mask of shape HxW, True for pixels that passed the similarity threshold

    The threshold is compared against the squared distance, so sqrt() doesn't need to be computed for every one of the millions of pixels.
    """
    target_rgb = np.asarray(target_color[:3], dtype=np.float32)
    rgb = pixels[:, :, :3].astype(np.float32)
    diff = rgb - target_rgb
    sq_distance = np.einsum('ijk,ijk->ij', diff, diff)
    max_distance = np.sqrt(255 ** 2 * 3)
    threshold_distance = (1 - similarity_threshold / 100) * max_distance
    return sq_distance <= threshold_distance ** 2


def cluster_to_points(mask: np.ndarray, min_distance: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduces the mask to one point per panorama marker, keeping distinct markers at least min_distance apart even when their blobs touch or overlap

    mask: np.ndarray - boolean mask of shape HxW with pixels that passed the color similarity threshold
    min_distance: float - minimum distance in pixels between two distinct points (panoramas)

    Returns Tuple[np.ndarray, np.ndarray] - an (N, 2) array of marker centers in (x, y) format, and an (N,) array of marker sizes (number of matched pixels assigned to each)

    A naive "dilate the mask, then take one centroid per connected component" approach breaks down in dense areas: if marker A touches marker B and B touches
    C, connected-component labeling treats A-B-C as a single blob no matter how far apart A and C actually are, so an entire busy district can collapse into
    one giant blob and a single point ("chain-merging"). To avoid this:

    1. A distance transform (scipy.ndimage.distance_transform_cdt) gives each matched pixel its approximate distance to the nearest non-matched pixel —
       i.e. how deep inside a blob it sits. The chamfer ("chessboard") variant is used instead of the exact Euclidean transform because it is int32 and
       noticeably cheaper in both time and memory, which matters at 9999x9999+ resolutions; the approximation is fine since only relative peak locations
       matter here, not exact distances.
    2. Local maxima of that distance map (scipy.ndimage.maximum_filter over a min_distance-sized window) approximate individual marker centers, including
       inside merged/touching blobs — a dense cluster of many small markers produces many separate local peaks rather than one.
    3. maximum_filter doesn't strictly guarantee the resulting peaks end up min_distance apart (ties/plateaus can leave two peaks slightly closer), so a
       final greedy pass with scipy.spatial.cKDTree explicitly enforces it: candidates closer than min_distance to an already-kept point are dropped. This
       only runs over the peak candidates (typically thousands), not the raw matched pixels (can be millions), so it stays fast.

    Marker size is then computed by assigning every original matched pixel to its nearest surviving point (cKDTree query) and counting.
    """
    if not mask.any():
        return np.empty((0, 2)), np.empty((0,), dtype=int)
    distance = ndimage.distance_transform_cdt(mask, metric='chessboard')
    footprint = max(3, int(round(min_distance)))
    local_max = ndimage.maximum_filter(distance, size=footprint)
    peak_mask = (distance == local_max) & mask
    del distance, local_max
    structure = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
    labeled, num_peaks = ndimage.label(peak_mask, structure=structure)
    pys, pxs = np.nonzero(peak_mask)
    labels_at_peaks = labeled[pys, pxs]
    counts = np.bincount(labels_at_peaks, minlength=num_peaks + 1)
    sum_y = np.bincount(labels_at_peaks, weights=pys.astype(np.float64), minlength=num_peaks + 1)
    sum_x = np.bincount(labels_at_peaks, weights=pxs.astype(np.float64), minlength=num_peaks + 1)
    counts = counts[1:num_peaks + 1]
    cy = sum_y[1:num_peaks + 1] / counts
    cx = sum_x[1:num_peaks + 1] / counts
    candidates_xy = np.column_stack([cx, cy])
    del labeled, peak_mask
    tree = cKDTree(candidates_xy)
    pairs = tree.query_pairs(r=min_distance)
    removed = np.zeros(len(candidates_xy), dtype=bool)
    adjacency = [[] for _ in range(len(candidates_xy))]
    for i, j in pairs:
        adjacency[i].append(j)
        adjacency[j].append(i)
    keep = []
    for i in range(len(candidates_xy)):
        if removed[i]:
            continue
        keep.append(i)
        for j in adjacency[i]:
            removed[j] = True
    points_xy = candidates_xy[keep]
    ys, xs = np.nonzero(mask)
    point_tree = cKDTree(points_xy)
    _, nearest_idx = point_tree.query(np.column_stack([xs, ys]))
    sizes = np.bincount(nearest_idx, minlength=len(points_xy))
    return points_xy, sizes


def pixels_to_geo(points_xy: np.ndarray, width: int, height: int, nw_lat: float, nw_lon: float, se_lat: float, se_lon: float) -> np.ndarray:
    """
    Converts pixel coordinates of points into geographic coordinates via linear interpolation between the map corners

    points_xy: np.ndarray - an (N, 2) array of pixel coordinates of points in (x, y) format
    width: int - map width in pixels
    height: int - map height in pixels
    nw_lat: float - latitude of the map's north-west corner
    nw_lon: float - longitude of the map's north-west corner
    se_lat: float - latitude of the map's south-east corner
    se_lon: float - longitude of the map's south-east corner

    Returns np.ndarray - an (N, 2) array of geographic coordinates in (lat, lon) format
    """
    lat_step = (se_lat - nw_lat) / height
    lon_step = (se_lon - nw_lon) / width
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    lat = nw_lat + y * lat_step
    lon = nw_lon + x * lon_step
    out_of_bounds = int(np.sum((x < 0) | (x >= width) | (y < 0) | (y >= height)))
    if out_of_bounds:
        logging.warning(f"[!] {out_of_bounds} points fell outside the map bounds "
              f"(width={width}, height={height})")
    return np.column_stack([lat, lon])


def save_debug_image(pixels: np.ndarray, mask: np.ndarray, fill_color: Tuple[int, int, int, int], output_file: str) -> None:
    """
    Saves a debug image: pixels that matched the color are kept as in the original, the rest are filled with the fill color

    pixels: np.ndarray - array of the source image's pixels of shape HxWx4 (RGBA)
    mask: np.ndarray - boolean mask of shape HxW with pixels that passed the color similarity threshold
    fill_color: Tuple[int, int, int, int] - fill color for non-matched pixels, as (r, g, b, a)
    output_file: str - path to save the resulting image to

    Returns None
    """
    result = np.zeros_like(pixels)
    result[:, :] = fill_color
    result[mask] = pixels[mask]
    Image.fromarray(result).save(output_file, format='PNG', compress_level=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finds pixels of a given color (panoramas) on a map and converts them into geographic coordinates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to the map image"
    )
    parser.add_argument(
        "color",
        type=str,
        help="Target color: '108,118,255[,255]' or '#6C76FF[FF]'"
    )
    parser.add_argument(
        "nw_lat",
        type=float,
        help="Latitude of the map's north-west corner"
    )
    parser.add_argument(
        "nw_lon",
        type=float,
        help="Longitude of the map's north-west corner"
    )
    parser.add_argument(
        "se_lat",
        type=float,
        help="Latitude of the map's south-east corner"
    )
    parser.add_argument(
        "se_lon",
        type=float,
        help="Longitude of the map's south-east corner"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=90.0,
        help="Color similarity threshold in percent (0-100)"
    )
    parser.add_argument(
        "-d", "--min-distance",
        type=float,
        default=5.0,
        help="Min. distance in pixels between two distinct points (closer — treated as one)"
    )
    parser.add_argument(
        "-o", "--output",
        default="geo_report.txt",
        help="Output file for the result (latitude and longitude separated by a space, one pair per line)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Map width in pixels for coordinate conversion (defaults to the image's actual width)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Map height in pixels for coordinate conversion (defaults to the image's actual height)"
    )
    parser.add_argument(
        "--debug-image",
        dest="debug_image",
        action="store_true",
        default=True,
        help="Save a debug image with the matched pixels highlighted (enabled by default)"
    )
    parser.add_argument(
        "--no-debug-image",
        dest="debug_image",
        action="store_false",
        help="Don't save the debug image (faster on large images)"
    )
    parser.add_argument(
        "--fill-color",
        default="0,0,0,255",
        help="Fill color for non-matched pixels in the debug image"
    )
    parser.add_argument(
        "--include-size",
        action="store_true",
        help="Add a third column to the output file — the blob size in pixels"
    )
    parser.add_argument(
        "-v", "--verbose",
        action='store_true',
        help="Print debug logs"
    )
    args = parser.parse_args()
    if not 0 <= args.threshold <= 100:
        parser.error("Similarity threshold must be between 0 and 100")
    if args.min_distance <= 0:
        parser.error("--min-distance must be greater than 0")
    try:
        target_color = parse_color(args.color)
    except ValueError as e:
        parser.error(str(e))
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler("main_log.txt", mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    t0 = time.time()
    Image.MAX_IMAGE_PIXELS = 1_073_741_824
    logging.info("Loading image...")
    img = Image.open(args.image).convert('RGBA')
    pixels = np.array(img)
    height, width = pixels.shape[:2]
    logging.info(f"Loaded image: {width}x{height} ({width * height:,} pixels)")
    map_width = args.width or width
    map_height = args.height or height
    logging.debug(f"Target color (RGB): {target_color[:3]}, similarity threshold: {args.threshold}%")
    mask = find_color_mask(pixels, target_color, args.threshold)
    matched = int(mask.sum())
    logging.info(f"Pixels passing the threshold: {matched:,}")
    if args.debug_image:
        try:
            fill_color = parse_color(args.fill_color)
        except ValueError as e:
            parser.error(f"Fill color error: {e}")
        debug_path = f"debug_image.png"
        logging.debug(f"Saving debug image to {debug_path}...")
        save_debug_image(pixels, mask, fill_color, debug_path)
    logging.debug(f"Clustering pixels into points (min-distance={args.min_distance})...")
    points_xy, sizes = cluster_to_points(mask, args.min_distance)
    logging.info(f"Points found (panoramas): {len(points_xy):,}")
    geo = pixels_to_geo(points_xy, map_width, map_height,
                         args.nw_lat, args.nw_lon, args.se_lat, args.se_lon)
    with open(args.output, "w", encoding="utf-8") as f:
        for i, (lat, lon) in enumerate(geo):
            if args.include_size:
                f.write(f"{lat:.6f} {lon:.6f} {sizes[i]}\n")
            else:
                f.write(f"{lat:.6f} {lon:.6f}\n")
    dt = time.time() - t0
    logging.info(f"[OK] Done in {dt:.1f} sec.")
    logging.info(f"Result: {args.output} ({len(points_xy):,} points)")
    if len(sizes):
        logging.info(f"Blob sizes: min={sizes.min()}, max={sizes.max()}, average={sizes.mean():.1f} px")


if __name__ == "__main__":
    main()
