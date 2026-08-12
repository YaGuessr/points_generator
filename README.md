# YaGuessr points generator

A two-stage pipeline for turning a map image with panorama overlay into a
list of precise, real-world panorama coordinates:

```
map.png --[main.py]--> geo_report.txt --[precise_cords.py]--> cords.txt
 (image)   (color match    (approximate      (headless-browser   (precise
            + clustering)    lat/lon)          lookup on Yandex)   lat/lon)
```

1. **`main.py`** scans a map image for pixels of a given overlay color,
   clusters them into individual markers, and converts each marker's pixel
   position into an approximate latitude/longitude via linear interpolation
   between the map's corners.
2. **`precise_cords.py`** takes those approximate coordinates and, for each
   one, opens Yandex's panorama viewer in a headless browser at that
   location. Yandex snaps to the nearest actual panorama and reports its
   exact recorded coordinates, which is what gets saved. This corrects for
   both the pixel-resolution rounding error from stage 1 and any drift
   between the map raster and Yandex's own panorama positions.

## Quick start

```bash
python -m pip install -r requirements.txt
python -m playwright install chromium
```

`playwright install chromium` is a separate step from `pip install` - it
downloads the actual browser binary Playwright drives, which isn't bundled
with the Python package.

```bash
# Stage 1: image -> approximate coordinates
python main.py map.png "108,118,255" 55.7558 37.6173 55.7500 37.6250

# Stage 2: approximate coordinates -> precise coordinates
python precise_cords.py geo_report.txt -o cords.txt
```

The first command looks for pixels close to RGB `(108,118,255)` in
`map.png`, treats `55.7558, 37.6173` as the latitude/longitude of the
image's **top-left** (north-west) corner and `55.7500, 37.6250` as the
**bottom-right** (south-east) corner, and writes one `lat lon` pair per
detected marker to `geo_report.txt`.

The second command opens each of those points in a headless browser,
reads back Yandex's precise panorama coordinates, and writes the deduplicated
result to `cords.txt`.

> **Compatibility note:** `precise_cords.py` expects a plain two-column
> `lat lon` input file. Don't feed it a `main.py` output generated with
> `--include-size` (three columns) - its line parser expects exactly two
> values per line and will error out on a third column.

---

## Main script `main.py`

Finds pixels of a given overlay color on a map image (e.g. panorama
markers), clusters nearby matches into single points, and converts their
pixel coordinates into real-world latitude/longitude. Designed to handle
very large images (9999x9999 and beyond) quickly.

### Positional arguments

| Argument  | Description |
|-----------|-------------|
| `image`   | Path to the map image (any format Pillow can open) |
| `color`   | Target color to search for: `"R,G,B"`, `"R,G,B,A"`, `"#RRGGBB"`, `"#RRGGBBAA"`, or the short forms `"#RGB"` / `"#RGBA"` |
| `nw_lat`  | Latitude of the map's north-west (top-left) corner |
| `nw_lon`  | Longitude of the map's north-west (top-left) corner |
| `se_lat`  | Latitude of the map's south-east (bottom-right) corner |
| `se_lon`  | Longitude of the map's south-east (bottom-right) corner |

### Optional flags

| Flag | Default | Description |
|------|---------|-------------|
| `-t`, `--threshold` | `90.0` | Color similarity threshold in percent (0-100). `100` = exact match only; lower values tolerate anti-aliasing, compression artifacts, etc. |
| `-d`, `--min-distance` | `5.0` | Minimum distance in pixels between two distinct points. Matched pixels closer than this to each other are merged into a single point (e.g. a 5px-diameter marker circle produces one point, not one per pixel). |
| `-o`, `--output` | `geo_report.txt` | Output file. One line per point: `lat lon` (or `lat lon size` with `--include-size`). |
| `--width` | image width | Map width in pixels used for the geo conversion, if it differs from the image's actual pixel width. |
| `--height` | image height | Same as above, for height. |
| `--debug-image` | on | Save a debug PNG showing matched vs. non-matched pixels. |
| `--no-debug-image` | - | Skip the debug PNG. Recommended for very large images if you don't need it, since it doubles memory use and adds disk I/O. |
| `--fill-color` | `0,0,0,255` | Fill color used for non-matched pixels in the debug PNG. Same format as `color`. |
| `--include-size` | off | Add a third column to the output file with the number of pixels assigned to each detected marker. Useful for tuning `--threshold` / `--min-distance` - **not** compatible with `precise_cords.py` as input, see note above. |
| `-v`, `--verbose` | off | Print `DEBUG`-level logs to the console as well as `INFO`. The log file (`main_log.txt`) always gets full `DEBUG` detail regardless of this flag. |

### Examples

Loosen the color match and require markers to be at least 8px apart:

```bash
python main.py map.png "#6C76FF" 55.7558 37.6173 55.7500 37.6250 -t 85 -d 8
```

Large image, skip the debug PNG for speed, keep marker sizes for inspection:

```bash
python main.py huge_map.png "108,118,255,255" 40.71 -74.02 40.68 -73.97 --no-debug-image --include-size -o points.txt
```

### How it works

The script runs three stages, each fully vectorized (NumPy / SciPy) so it
scales to images with tens of millions of pixels without a Python-level loop
over pixels or points.

#### 1. Color matching (`find_color_mask`)

For every pixel, the Euclidean distance between its RGB value and the target
color is computed, converted to a 0-100% similarity score, and compared
against `--threshold`. The result is a boolean mask of matched pixels.

The alpha channel is ignored for matching - only RGB is compared. Internally
this is done in `float32` and the threshold is checked on **squared**
distance, avoiding a `sqrt()` over every pixel and cutting memory traffic
roughly in half compared to a naive `float64` implementation - this matters
once you're at 9999x9999+ (100M+ pixels).

#### 2. Clustering into points (`cluster_to_points`)

A single marker (e.g. a panorama dot) is drawn as a small blob of several
adjacent matched pixels, not a single pixel - so the raw mask needs to be
reduced to one point per blob. In busy areas markers can also sit close
enough together, or their blobs can touch/overlap outright, that they can't
just be treated as isolated dots.

A naive "dilate the mask, then label connected components, then take one
centroid per component" approach breaks down in dense areas: if marker A's
blob touches marker B's, and B's touches C's, connected-component labeling
merges A-B-C into a *single* component no matter how far apart A and C
actually are - a "chaining" effect. In a busy downtown full of closely-spaced
panoramas this can collapse an entire district into one giant blob and a
single output point, silently losing every marker in it. The current
implementation avoids this:

1. **Distance transform** (`scipy.ndimage.distance_transform_cdt`) gives
  every matched pixel its approximate distance to the nearest non-matched
  pixel - i.e. how deep inside a blob it sits. The chamfer/chessboard
  variant is used instead of an exact Euclidean transform because it's
  `int32` and considerably cheaper in time and memory; only the relative
  location of peaks matters here, not the exact distance value.
2. **Local maxima** of that depth map (`scipy.ndimage.maximum_filter` over a
  `min_distance`-sized window) approximate individual marker centers -
  including inside merged/touching blobs, since a cluster of many small
  markers produces many separate nearby peaks rather than one big flat
  region.
3. `maximum_filter` doesn't strictly *guarantee* the resulting peaks end up
  `min_distance` apart (ties/plateaus near the window edge can leave two
  peaks slightly closer than that), so a final **greedy pass**
  (`scipy.spatial.cKDTree.query_pairs`) explicitly enforces it: any
  candidate closer than `min_distance` to an already-kept point is dropped.
  This only runs over the peak candidates - typically thousands, even on a
  huge image - not the raw matched pixels, which can be in the millions, so
  it stays fast.

The result is one `(x, y)` point per marker. Each point's `size` (pixel
count, used with `--include-size`) is computed by assigning every original
matched pixel to its nearest surviving point and counting - so sizes across
all points still add up to the total number of matched pixels, but are
distributed correctly across individual markers rather than one blob.

Note: this clusters in **pixel space**, not in geographic space. Since
marker size is a fixed number of pixels on the image, and a degree of
longitude doesn't correspond to a fixed real-world distance the way a degree
of latitude roughly does, pixel-space clustering is both simpler and more
correct than clustering after converting to lat/lon.

#### 3. Pixel to geographic conversion (`pixels_to_geo`)

Latitude and longitude are obtained by linear interpolation between the
map's north-west and south-east corners:

```
lat = nw_lat + y * (se_lat - nw_lat) / height
lon = nw_lon + x * (se_lon - nw_lon) / width
```

This assumes the map image is a simple equirectangular (unprojected)
raster where pixel rows/columns map linearly onto latitude/longitude. It
does **not** account for map projections such as Mercator. If your source
map uses a projection, you'll need to reproject the corner coordinates (or
the points) accordingly before/after this step. In practice this is also
exactly what stage 2 (`precise_cords.py`) corrects for, since it reads the
true coordinate back from Yandex rather than trusting the interpolation.

All points are converted in one vectorized pass; any points that land
outside the `[0, width) x [0, height)` pixel range are still converted, with
a single warning logged with the count (not one warning per point).

### Debug image

If `--debug-image` is enabled (default), a PNG named `debug_image.png` is
saved next to the output: matched pixels keep their original color,
everything else is replaced with `--fill-color`. This is purely for visually
verifying that your color/threshold settings are picking up the right
pixels - it has no effect on the geo conversion. Note this file can be
**very large** (uncompressed, same resolution as the source map) - expect
it to be comparable in size to the source image, or larger.

### Output format

Plain text, one point per line, space-separated:

```
55.755234 37.617881
55.755198 37.618344
```

With `--include-size`, a third column is appended:

```
55.755234 37.617881 21
55.755198 37.618344 34
```

`size` is the number of matched pixels assigned to that marker - a rough
proxy for marker confidence, handy for spotting outliers (e.g. a size of `1`
is often a single stray pixel that happened to pass the similarity threshold
rather than a real marker).

### Performance notes

- Runtime is dominated by the color-matching pass and scales roughly
  linearly with pixel count; clustering adds a further pass that also scales
  roughly linearly with pixel count (not with the number of matched pixels
  or markers). A 6000x6000 image (~36M pixels, ~39K matched pixels)
  processes end-to-end in under 10 seconds on a typical machine; a
  9999x9999+ image should take roughly proportionally longer.
- `--no-debug-image` avoids allocating a second full-size image buffer and
  skips a PNG encode/write - worth using on very large images if you don't
  need the visual check.
- Memory usage is roughly `width * height * (4 + a few bytes)` for the
  source pixels plus mask/distance/label buffers; a 9999x9999+ image needs
  on the order of several GB of RAM. The distance-transform step uses
  `int32` specifically to keep this from growing further.

---

## Precise script `precise_cords.py`

Takes the approximate coordinates from stage 1 and refines each one by
actually opening it in Yandex's panorama viewer via a headless Chromium
browser (Playwright), then reading back the exact coordinates Yandex has
recorded for the nearest real panorama at that location.

This matters because a point from `main.py` is only as precise as the map
image's resolution and the corner coordinates you gave it - it's a linear
estimate, not the panorama's actual recorded position. Opening the point in
Yandex directly sidesteps that: whatever panorama Yandex resolves to at that
location, its own reported coordinates are used as ground truth.

### Usage

```bash
python precise_cords.py geo_report.txt -o cords.txt -pl 32
```

### Arguments

| Argument | Default | Description |
|----------|---------|--------------|
| `input_file` | - | Path to a plain-text file of approximate points, one `lat lon` pair per line (i.e. `main.py`'s output *without* `--include-size`) |
| `-o`, `--output` | `cords.txt` | Output file for the precise, deduplicated coordinates (`lat lon` per line) |
| `-pl`, `--parallel_limit` | `32` | Maximum number of browser pages processed concurrently |
| `-v`, `--verbose` | off | Print `DEBUG`-level logs to the console (including per-point results), and save a screenshot to `err_screenshots/` for any point that fails all 3 attempts. The log file (`precise_log.txt`) always gets full `DEBUG` detail regardless of this flag. |

### How it works

For each input point:

1. A Yandex map-widget URL is built (`generate_yandex_panorama_url`)
  pointing the panorama viewer at that coordinate, with the panorama layer
  forced open (`panorama[full]=true`).
2. A new isolated browser context/page opens that URL. Requests to
  `pano.maps.yandex.net` (the actual panorama image tiles) are blocked -
  the script only needs the coordinate metadata, not the rendered imagery,
  so this saves bandwidth and load time.
3. Once the page settles, the script locates the panorama widget's share
  link and parses the precise `lat, lon` out of its URL - this is the
  coordinate Yandex has on record for the panorama it resolved to at that
  location, not the input point itself.
4. If a point fails (no panorama found nearby, network error, page timeout,
  etc.), it's retried up to 3 times total with a randomized 1-2s delay
  between attempts, before being logged as an error and excluded from the
  output.

Points are processed concurrently (bounded by `--parallel_limit` via an
`asyncio.Semaphore`, one browser context per point), with a small randomized
0.2-0.5s delay after each request completes to avoid hammering Yandex.
Progress is printed as `done/total` to the console (unless `--verbose`,
which prints per-point debug logs instead).

Once every point has been attempted, results are **deduplicated** - it's
common for two nearby approximate points to resolve to the same actual
panorama - and only unique coordinates are written to the output file. A
summary is logged at the end:

```
Precised: <successful lookups, including repeats>
Unique:   <distinct panoramas written to the output file>
Repeats:  <successful lookups that matched an already-seen panorama>
Errors:   <points that failed after all retries>
```

### Requirements-specific notes

- Runs one headless Chromium instance with up to `--parallel_limit`
  concurrent pages/contexts - increasing this speeds things up but uses
  more memory/CPU and network bandwidth, and risks tripping Yandex's own
  rate limiting if pushed too high.
- Needs network access to `yandex.ru`; nothing is cached between runs.
- `err_screenshots/` is only created (and only populated) when running with
  `--verbose` and a point fails all 3 attempts - useful for diagnosing why
  a specific coordinate didn't resolve.

---

## Generated files reference

Running both stages in the project directory produces:

| File | Produced by | Contents |
|------|-------------|----------|
| `geo_report.txt` (or `-o` target) | `main.py` | Approximate `lat lon` per detected marker |
| `debug_image.png` | `main.py` (unless `--no-debug-image`) | Visual check of which pixels matched the target color |
| `main_log.txt` | `main.py` | Full debug log of the run |
| `cords.txt` (or `-o` target) | `precise_cords.py` | Precise, deduplicated `lat lon` per panorama |
| `precise_log.txt` | `precise_cords.py` | Full debug log of the run |
| `err_screenshots/` | `precise_cords.py` (`--verbose` only) | Screenshots of points that failed all retry attempts |

## Requirements

- Python 3.9+
- `numpy`
- `pillow`
- `scipy`
- `playwright` (plus `playwright install chromium` to fetch the browser binary)
