#!/usr/bin/env python3
"""
Makes points taken from picture more precise
Simulates headless browser to open pano using known cords and extract its exact cords 
"""

import os
import sys
import random
import asyncio
import logging
import argparse
from urllib.parse import urlparse
from playwright.async_api import async_playwright, Browser


def generate_yandex_panorama_url(
    longitude: float,
    latitude: float,
    azimuth: float = 0.0,
    tilt: float = 0.0,
    zoom: float = 13.5,
    panorama_id: str = None,
    layer: str = "stv,sta"
) -> str:
    base_url = "https://yandex.ru/map-widget/v1/?"
    params = {
        'l': layer.replace(',', '%2C'),
        'll': f"{longitude}%2C{latitude}",
        'z': str(zoom),
        'panorama[point]': f"{longitude}%2C{latitude}",
        'panorama[direction]': f"{azimuth}%2C{tilt}",
        'panorama[full]': 'true',
        'panorama[span]': '96.335379%2C60.000000'
    }
    if panorama_id:
        params['panorama[id]'] = panorama_id
    param_strings = []
    for key, value in params.items():
        encoded_key = key.replace('[', '%5B').replace(']', '%5D')
        param_strings.append(f"{encoded_key}={value}")
    full_url = base_url + "&".join(param_strings)
    return full_url


async def block_domains(route):
    BLACK_LIST = (
        "pano.maps.yandex.net",
    )
    host = urlparse(route.request.url).hostname or ""
    host = host.lower()
    if host in BLACK_LIST:
        await route.abort("blockedbyclient")
    else:
        await route.continue_()


async def parse_url(browser: Browser, point: str, take_screenshot: bool = False) -> tuple:
    """
    Opens point page in browser and tries to parse precise cords

    browser: playwright.async_api.Browser - Browser instance to open page
    point: list[lat, lng] - single point to precise
    take_screenshot: bool - takes screenshot if error occures default False

    Returns tuple on success and str on error
    """
    context = await browser.new_context()
    await context.route("**/*", block_domains)
    page = await context.new_page()
    try:
        await page.goto(
            generate_yandex_panorama_url(point[1], point[0]),
            wait_until="domcontentloaded",
            timeout=30000,
        )
        try:
            await page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            pass
        href = await page.locator('a[role="button"]').get_attribute("href")
        href = str(href).split("&")[2]
        href = href.split("=")[-1]
        lng, lat = href.split("%2C")
        lat = float(lat)
        lng = float(lng)
        return lat, lng
    except Exception as e:
        if take_screenshot:
            os.makedirs("err_screenshots", exist_ok=True)
            await page.screenshot(path=f"err_screenshots/{point}.png")
        return str(e).strip()
    finally:
        await page.close()
        await context.close()
        await asyncio.sleep(random.uniform(0.2, 0.5)) # anti DOS system


async def limited_load(semaphore: asyncio.Semaphore, browser: Browser, point: list[float], verbose: bool = False):
    """
    Limits async sessions with semaphore and retrys 3 times if error occures

    semaphore: asyncio.Semaphore - limits parallel sessions
    browser: playwright.async_api.Browser - Browser instance to open page
    point: list[lat, lng] - single point to precise
    verbose: bool - takes screenshot when error occures 3 times default False

    Returns tuple on success and str on error
    """
    async with semaphore:
        for _ in range(2):
            result = await parse_url(browser, point)
            if isinstance(result, tuple):
                logging.debug(f"{point} -> {result}")
                return result
            logging.debug(f"{point} failed, retrying...")
            await asyncio.sleep(random.uniform(1.0, 2.0))
        result = await parse_url(browser, point, take_screenshot=verbose)
        if isinstance(result, tuple):
            logging.debug(f"{point} -> {result}")
        else:
            logging.warning(f"Error processing {point}: {result}")
        return result


async def launch_browser(PARALLEL_LIMIT: int, points: list[list[float]], output: str, verbose: bool = False) -> None:
    """
    Launch browser and process points

    PARALLEL_LIMIT: int - limit how many parallel sessions will be created
    points: list[list[lat, lng]] - list of cords to precise
    output: str - path to output file
    verbose: bool - prints progress bar to stdout if False default False

    Returns None
    """
    semaphore = asyncio.Semaphore(PARALLEL_LIMIT)
    async with async_playwright() as p:
        logging.info("Launching browser...")
        browser = await p.chromium.launch(headless=True)
        logging.debug("Browser launched")
        tasks = [
            asyncio.create_task(limited_load(semaphore, browser, point, verbose=verbose))
            for point in points
        ]
        logging.info("Processing points...")
        if not verbose:
            logging.debug("Printing progress bar to terminal")
            print(f"0/{len(tasks)}\r", end="", flush=True)
        results = []
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            if not verbose:
                print(f"{len(results)}/{len(tasks)}\r", end="", flush=True)
        if not verbose:
            print()
        logging.debug("All points processed")
        await browser.close()
    new_points = []
    repeats = 0
    for result in results:
        if isinstance(result, tuple) and len(result) == 2:
            if result in new_points:
                repeats += 1
            else:
                new_points.append(result)
    logging.info("Saving to file...")
    with open(output, "w", encoding="utf-8") as f:
        f.write("".join([f"{p[0]} {p[1]}\n" for p in new_points]))
    logging.debug("Saved to file")
    logging.info("")
    logging.info(f"Precised: {len(new_points)+repeats}")
    logging.info(f"Unique: {len(new_points)}")
    logging.info(f"Repeats: {repeats}")
    logging.info(f"Errors: {len(points)-len(new_points)-repeats}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Makes points taken from picture more precise",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the points file"
    )
    parser.add_argument(
        "-o", "--output",
        default="cords.txt",
        help="Output file for the result (latitude and longitude separated by a space, one pair per line)"
    )
    parser.add_argument(
        "-pl", "--parallel_limit",
        default="32",
        help="maximum parallel requests limit"
    )
    parser.add_argument(
        "-v", "--verbose",
        action='store_true',
        help="Print debug logs"
    )
    args = parser.parse_args()
    try:
        PARALLEL_LIMIT = int(args.parallel_limit)
    except ValueError:
        parser.error("Error: parallel limit must be decimal number")
    if PARALLEL_LIMIT < 1:
        parser.error("Error: parallel limit must be 1 or more")
    points = []
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            points = [list(map(float, x.split(maxsplit=1))) for x in f.read().strip().split("\n")]
            if len(points) < 1 or any(len(x) != 2 for x in points):
                raise ValueError("File content is corrupted")
    except Exception as e:
        parser.error(f"Input file error: {e}")
    try:
        output = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output), exist_ok=True)
    except OSError:
        parser.error("Error: output path parent folder is a file")
    if os.path.exists(output) and not os.path.isfile(output):
        parser.error("Error: output path is not a file")
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler("precise_log.txt", mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logging.debug(f"Config: ")
    logging.debug(f"  input_file: {args.input_file}")
    logging.debug(f"  output: {output}")
    logging.debug(f"  parallel_limit: {PARALLEL_LIMIT}")
    logging.debug(f"  verbose: {args.verbose}")
    logging.debug(f"")
    logging.info(f"Parsed {len(points)} points")
    asyncio.run(launch_browser(PARALLEL_LIMIT, points, output, verbose=args.verbose))


if __name__ == "__main__":
    main()
