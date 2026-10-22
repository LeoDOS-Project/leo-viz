"""Ground-sized rectangles in a local spherical equal-area projection.

The compact rectangle specification is authoritative; GeoJSON approximates its
curved geographic edges with segments no longer than 5 km in the local plane.
These are vector AOIs, not falsely labelled memberships in the global tile grid.
"""
from dataclasses import dataclass
from math import ceil, cos, floor, pi, radians, sin, sqrt

import numpy as np
from shapely.affinity import translate
from shapely.geometry import Polygon, box, mapping
from shapely.ops import unary_union

from .geometry import EARTH_RADIUS_KM


def local_frame(lon, lat):
    lon, lat = radians(lon), radians(lat)
    return (np.array([cos(lat)*cos(lon), cos(lat)*sin(lon), sin(lat)]),
            np.array([-sin(lon), cos(lon), 0.]),
            np.array([-sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat)]))


def split_geographic_ring(points, north_pole):
    """Unwrap and split at ±180°, closing a pole-enclosing ring via its pole."""
    points = np.asarray(points)
    longitude = np.degrees(np.unwrap(np.radians(points[:, 0])))
    ring = list(zip(longitude.tolist(), points[:, 1].tolist()))
    if abs(longitude[-1] - longitude[0]) > 180:
        pole = 90. if north_pole else -90.
        ring.extend([(longitude[-1], pole), (longitude[0], pole), ring[0]])
    else:
        ring[-1] = ring[0]
    polygon = Polygon(ring)
    if not polygon.is_valid:
        raise ValueError("Invalid oriented rectangle; reduce its area or aspect ratio")
    lo, _, hi, _ = polygon.bounds
    parts = []
    for turn in range(floor((lo + 180) / 360), floor((hi + 180) / 360) + 1):
        clipped = polygon.intersection(box(-180 + turn*360, -90, 180 + turn*360, 90))
        pieces = [clipped] if clipped.geom_type == "Polygon" else getattr(clipped, "geoms", [])
        parts.extend(translate(p, xoff=-turn*360) for p in pieces if p.geom_type == "Polygon" and p.area > 0)
    if not parts:
        raise ValueError("Oriented rectangle has no geographic area")
    # A pole cap can reconnect along the original ring start after wrapping.
    # Dissolve those common edges rather than emitting an invalid multipolygon.
    result = unary_union(parts)
    if not result.is_valid:
        raise ValueError("Could not form a valid geographic rectangle")
    return mapping(result)


@dataclass(frozen=True)
class OrientedAoi:
    id: str
    anchor_lon: float
    anchor_lat: float
    requested_area_km2: float
    hotspot: int | None
    aspect: float
    rotation_deg: float
    geometry: dict

    # There is no global-grid tile assignment until observation units are chosen.
    tile_count = None
    rectangles = ()

    def area_km2(self, _grid):
        return self.requested_area_km2

    def as_dict(self, _grid):
        return {"id": self.id, "anchor_lon": self.anchor_lon, "anchor_lat": self.anchor_lat,
                "requested_area_km2": self.requested_area_km2, "area_km2": self.requested_area_km2,
                "hotspot": self.hotspot, "representation": "local_equal_area_rectangle",
                "width_km": sqrt(self.requested_area_km2 * self.aspect),
                "height_km": sqrt(self.requested_area_km2 / self.aspect),
                "rotation_deg": self.rotation_deg, "radius_km": EARTH_RADIUS_KM,
                "geometry": self.geometry, "tile_count": None}

    def geojson(self, grid):
        return {"type": "Feature", "id": self.id,
                "properties": {k: v for k, v in self.as_dict(grid).items() if k != "geometry"},
                "geometry": self.geometry}


def oriented_rectangle(identifier, lon, lat, area_km2, hotspot, aspect, rotation_deg):
    width, height = sqrt(area_km2 * aspect), sqrt(area_km2 / aspect)
    radius = EARTH_RADIUS_KM
    if width*width + height*height >= (4*radius)**2:
        raise ValueError("Rectangle exceeds the local equal-area projection; reduce area or aspect ratio")
    corners = [(-width/2, -height/2), (width/2, -height/2),
               (width/2, height/2), (-width/2, height/2)]
    points = []
    for a, b in zip(corners, corners[1:] + corners[:1]):
        steps = max(1, ceil(max(abs(a[0]-b[0]), abs(a[1]-b[1])) / 5.0))
        points.extend((a[0] + (b[0]-a[0])*t/steps, a[1] + (b[1]-a[1])*t/steps) for t in range(steps))
    points.append(points[0])
    uv = np.asarray(points)
    theta = radians(rotation_deg)
    center, east, north = local_frame(lon, lat)
    def geographic_points(uv):
        x = uv[:, 0]*cos(theta) + uv[:, 1]*sin(theta)
        y = -uv[:, 0]*sin(theta) + uv[:, 1]*cos(theta)
        squared = (x*x + y*y) / radius**2
        # Inverse Lambert azimuthal equal-area projection on the simulator's sphere.
        xyz = (1 - squared[:, None]/2)*center + np.sqrt(np.maximum(0, 1 - squared[:, None]/4)) * (
            x[:, None]*east + y[:, None]*north) / radius
        result = np.column_stack((np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])),
                                  np.degrees(np.arcsin(np.clip(xyz[:, 2], -1, 1)))))
        result[-1] = result[0]
        return result
    geographic = geographic_points(uv)
    # Near a pole, a short ground edge can span many degrees of longitude.
    # Refine in the local plane before cutting at the seam; interpolating a
    # coarse geographic edge there would distort both its area and shape.
    for _ in range(12):
        delta = (np.diff(geographic[:, 0]) + 180) % 360 - 180
        split = np.abs(delta) > 1.0
        if not split.any():
            break
        indices = np.arange(len(uv)-1) + np.r_[0, np.cumsum(split[:-1])]
        refined = np.empty((len(uv)+int(split.sum()), 2))
        refined[indices] = uv[:-1]
        refined[indices[split]+1] = (uv[:-1][split]+uv[1:][split])/2
        refined[-1] = uv[-1]
        uv = refined
        geographic = geographic_points(uv)
    # Test which pole is inside the local rectangle, for correct GeoJSON caps.
    denominator = 1 + center[2]
    north_inside = False
    if denominator > 1e-15:
        k = radius * sqrt(2 / denominator)
        px, py = k*east[2], k*north[2]
        u, v = px*cos(theta) - py*sin(theta), px*sin(theta) + py*cos(theta)
        north_inside = abs(u) <= width/2 and abs(v) <= height/2
    geometry = split_geographic_ring(geographic, north_inside)
    return OrientedAoi(identifier, lon, lat, area_km2, hotspot, aspect, rotation_deg, geometry)
