"""Connected forest-guided growth with broad, seeded shape variation.

Preferences are evaluated in local ground kilometres. Only edge-adjacent frontier
cells can join the region; the tree-cover grid supplies geographic evidence,
while the smooth field supplies explicitly synthetic variation in its outline.
"""
import heapq
from math import cos, exp, hypot, log, pi, radians, sin, sqrt

import numpy as np

from .geometry import EARTH_RADIUS_KM


class GrowthField:
    def __init__(self, rng, radius, max_elongation, variation):
        self.elongation = exp(rng.uniform(0, log(max_elongation)))
        self.bearing = rng.uniform(0, pi)
        self.strength = variation * rng.uniform(.65, 1)
        self.radius = radius
        self.stretch = sqrt(self.elongation)
        self.cosine, self.sine = cos(self.bearing), sin(self.bearing)
        # Several broad waves produce coherent lobes, not independent noisy pixels.
        angles = rng.uniform(0, 2*pi, 6)
        frequencies = rng.uniform(2, 6, 6)
        phases = rng.uniform(0, 2*pi, 6)
        self.waves = tuple((float(f*cos(a)), float(f*sin(a)), float(p))
                           for a, f, p in zip(angles, frequencies, phases))

    def coordinates(self, east, north):
        return ((east*self.cosine + north*self.sine) / (self.radius*self.stretch),
                (-east*self.sine + north*self.cosine) * self.stretch / self.radius)

    def priority(self, u, v):
        noise = sum(sin(x*u + y*v + phase) for x, y, phase in self.waves) / sqrt(3)
        return hypot(u, v) * exp(self.strength * max(-1.5, min(1.5, noise)))


def grow(mask, cell_areas, row, col, step, requested, rng, config):
    rows, cols = mask.shape
    dy = EARTH_RADIUS_KM * radians(step)
    dx = dy * cos(radians(90 - (row + .5)*step))
    radius = max(sqrt(requested/pi), dy)
    field = GrowthField(rng, radius, config.max_elongation, config.outline_variation)
    penalty = 3 * config.forest_adherence / (1 - .7*config.forest_adherence)
    pending, seen, selected = [(0.0, row, col)], {(row, col)}, []
    area = 0.0
    while pending and area < requested:
        _, y, x = heapq.heappop(pending)
        selected.append((y, x))
        area += cell_areas[y]
        if len(selected) > 50000:
            raise ValueError("Forest patch exceeds 50,000 sampling cells; reduce area")
        for yy, xx in ((y-1, x), (y+1, x), (y, (x-1) % cols), (y, (x+1) % cols)):
            if not 0 <= yy < rows or (yy, xx) in seen:
                continue
            seen.add((yy, xx))
            east = ((xx-col+cols/2) % cols-cols/2) * dx
            u, v = field.coordinates(east, (row-yy)*dy)
            # Bound every patch even when a connected forest spans a continent.
            if hypot(u, v) > 4:
                continue
            priority = field.priority(u, v) + penalty * (not mask[yy, xx])
            heapq.heappush(pending, (priority, yy, xx))
    if area < requested:
        raise ValueError("Forest growth exhausted its bounded region before reaching the target area")
    return selected, {
        "forest_shape_model": "coherent_growth_v1",
        "growth_elongation": field.elongation,
        "growth_bearing_deg": field.bearing * 180/pi,
        "growth_outline_variation": field.strength,
        "growth_area_km2": float(area),
        "tree_cover_area_fraction": float(sum(cell_areas[y] for y, x in selected if mask[y, x]) / area),
    }


def grow_legacy(mask, cell_areas, row, col, step, requested, rng, compactness):
    """Replay explicit old compactness configurations without changing geometry."""
    rows, cols = mask.shape
    dy = EARTH_RADIUS_KM * radians(step)
    dx = dy * cos(radians(90 - (row+.5)*step))
    radius = max(sqrt(requested/pi), dy)
    pending, seen, selected = [(0.0, row, col)], {(row, col)}, []
    area = 0.0
    while pending and area < requested:
        _, y, x = heapq.heappop(pending)
        selected.append((y, x))
        area += cell_areas[y]
        if len(selected) > 50000:
            raise ValueError("Forest patch exceeds 50,000 sampling cells; reduce area")
        for yy, xx in ((y-1, x), (y+1, x), (y, (x-1) % cols), (y, (x+1) % cols)):
            if not 0 <= yy < rows or (yy, xx) in seen:
                continue
            seen.add((yy, xx))
            east = ((xx-col+cols/2) % cols-cols/2) * dx
            distance = np.hypot(east, (yy-row)*dy) / radius
            penalty = (2 + 8*(1-compactness)) * (not mask[yy, xx])
            heapq.heappush(pending, (distance*(1+3*compactness) + penalty + rng.uniform(0, .3), yy, xx))
    return selected, {"forest_shape_model": "legacy_compactness"}
