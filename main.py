from __future__ import annotations

import ctypes
import gc
import json
import math
import os
import random
import struct
import time
from pathlib import Path
from typing import Any, Iterable

from fastquadtree import QuadTree


Point = tuple[float, float]

RUNS = 30
SEED = 0x5EED_5EED_5EED_5EED
KNN_N = 5
RANGE_HALF = 0.05
CAP = 100
DEPTH = 100
SIMRA_DIR = "simra"
RESULT_DIR = "results"


def f32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]


def next_f32(value: float, direction: float) -> float:
    value = f32(value)

    if math.isnan(value):
        return value

    if value == direction:
        return direction

    if value == 0.0:
        bits = 1
        if direction < 0.0:
            bits |= 0x8000_0000
        return struct.unpack("!f", struct.pack("!I", bits))[0]

    bits = struct.unpack("!I", struct.pack("!f", value))[0]

    if (direction > value) == (value > 0.0):
        bits += 1
    else:
        bits -= 1

    return struct.unpack("!f", struct.pack("!I", bits))[0]


def parse(path: str | Path) -> list[Point]:
    path = Path(path)

    if path.is_dir():
        return parse_dir(path)

    return parse_file(path)


def parse_dir(path: Path) -> list[Point]:
    data: list[Point] = []

    for child in path.iterdir():
        if child.is_dir():
            data.extend(parse_dir(child))
        elif child.is_file():
            data.extend(parse_file(child))

    return data


def parse_file(path: Path) -> list[Point]:
    lines = read(path)

    marker = "=" * 10

    try:
        idx = next(i for i, line in enumerate(lines) if marker in line)
    except StopIteration as exc:
        raise ValueError(f"data has no marker: {path}") from exc

    points: list[Point] = []

    for line in lines[idx + 3:]:
        parts = line.split(",")

        if len(parts) < 2:
            continue

        try:
            x = float(parts[0].strip())
            y = float(parts[1].strip())
        except ValueError:
            continue

        if not math.isfinite(x) or not math.isfinite(y):
            continue

        points.append((x, y))

    return points


def read(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as file:
        return file.read().splitlines()


def bounds(points: Iterable[Point]) -> tuple[float, float, float, float]:
    points = list(points)

    if not points:
        raise ValueError("cannot compute bounds for empty data")

    min_x = min(f32(x) for x, _ in points)
    max_x = max(f32(x) for x, _ in points)
    min_y = min(f32(y) for _, y in points)
    max_y = max(f32(y) for _, y in points)

    min_x = next_f32(min_x, -math.inf)
    min_y = next_f32(min_y, -math.inf)
    max_x = next_f32(max_x, math.inf)
    max_y = next_f32(max_y, math.inf)

    return min_x, min_y, max_x, max_y


def rss_bytes() -> int:
    with open("/proc/self/statm", "r") as file:
        resident_pages = int(file.read().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def trim_memory() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def duration(seconds: float) -> dict[str, int]:
    total_nanos = round(seconds * 1_000_000_000)
    secs, nanos = divmod(total_nanos, 1_000_000_000)
    return {"secs": secs, "nanos": nanos}


def datapoint(point_id: int, point: Point) -> dict[str, Any]:
    x, y = point
    return {"id": point_id, "point": {"x": x, "y": y}}


def build_quad(
    points: list[Point], cap: int, depth: int
) -> tuple[QuadTree, float, int]:
    trim_memory()
    rss_before = rss_bytes()

    start = time.perf_counter()

    min_x, min_y, max_x, max_y = bounds(points)
    tree = QuadTree((min_x, min_y, max_x, max_y), capacity=cap, max_depth=depth)
    tree.insert_many(points)

    construction_time = time.perf_counter() - start

    trim_memory()
    rss_after = rss_bytes()

    return tree, construction_time, rss_after - rss_before


def store_report(data: dict[str, Any]) -> Path:
    result_dir = Path(RESULT_DIR)
    result_dir.mkdir(parents=True, exist_ok=True)

    i = 0
    while (result_dir / f"benchmark_fastquadtree_{i:02}.json").exists():
        i += 1

    path = result_dir / f"benchmark_fastquadtree_{i:02}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def bench_build_quad() -> dict[str, Any]:
    full_start = time.perf_counter()

    parse_start = time.perf_counter()
    points = parse(SIMRA_DIR)
    parse_time = time.perf_counter() - parse_start
    print(f"parsed {len(points):,} points in {parse_time * 1_000:.3f}ms")

    candidates = list(range(len(points)))
    if len(candidates) < RUNS:
        raise ValueError(f"expected at least {RUNS} candidates, found {len(candidates)}")

    candidates.sort()

    tree, construction_time, tree_size = build_quad(points, CAP, DEPTH)
    print(
        f"built quadtree in {construction_time * 1_000:.3f}ms, "
        f"size={tree_size / 1e6:.3f}MB"
    )

    rng = random.Random(SEED)
    rng.shuffle(candidates)

    source_points: list[dict[str, Any]] = []
    knn_results: list[dict[str, Any]] = []
    range_results: list[dict[str, Any]] = []

    for point_id in candidates[:RUNS]:
        target = points[point_id]
        source_points.append(datapoint(point_id, target))

        knn_start = time.perf_counter()
        neighbors = tree.nearest_neighbors(target, KNN_N)
        knn_time = time.perf_counter() - knn_start
        if len(neighbors) != KNN_N:
            raise ValueError(f"expected {KNN_N} points, got {len(neighbors)}")

        knn_results.append(
            {
                "pointId": point_id,
                "dur": duration(knn_time),
                "points": [
                    datapoint(neighbor_id, (x, y)) for neighbor_id, x, y in neighbors
                ],
            }
        )

        x, y = target
        rect = (x - RANGE_HALF, y - RANGE_HALF, x + RANGE_HALF, y + RANGE_HALF)
        range_start = time.perf_counter()
        _range_points = tree.query(rect)
        range_time = time.perf_counter() - range_start

        range_results.append(
            {
                "pointId": point_id,
                "dur": duration(range_time),
            }
        )

    full_time = time.perf_counter() - full_start

    return {
        "type": "BuildQuad",
        "engine": "fastquadtree",
        "data": {
            "full_runtime": duration(full_time),
            "cap": CAP,
            "depth": DEPTH,
            "num points": len(points),
            "data parse time": duration(parse_time),
            "quadtree construction time": duration(construction_time),
            "quadtree size": tree_size,
            "source points": source_points,
            "knn results": knn_results,
            "knn n": KNN_N,
            "range results": range_results,
        },
    }


def main() -> None:
    print("fastquadtree bench")

    report = bench_build_quad()
    path = store_report(report)

    print(f"wrote report to {path}")


if __name__ == "__main__":
    main()
