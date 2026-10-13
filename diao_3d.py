"""Ultimate DIAO Drawer (UDD) — 3D version.

Generates a parametric phallus mesh from a Gradio UI and exports it as
.glb for the Model3D viewer. Sliders cover shape, color, surface
wrinkles, optional hair, optional surface veins (raised ridges, not
floating tubes) and a corona ridge. A few one-click presets are wired
in, plus material packs (gold, chrome, marble, …), texture-based skin
packs, a Random button and shareable recipe codes.
"""

import base64
import functools
import io
import json
import os
import re
import struct
import tempfile
import zlib

import gradio as gr
import trimesh
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes


# ----------------------------- utilities -----------------------------


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def hex_to_rgba(hex_color, alpha=255):
    return np.array(hex_to_rgb(hex_color) + (alpha,), dtype=np.uint8)


def color_mesh_uniform(mesh, hex_color):
    """Uniform per-vertex color (we keep everything per-vertex so vein
    displacement, skin-tone noise, and gradients can all stack on top)."""
    rgba = hex_to_rgba(hex_color)
    mesh.visual.vertex_colors = np.tile(rgba, (len(mesh.vertices), 1))
    return mesh


def _existing_or_uniform_rgb(mesh, hex_color):
    """Read the current per-vertex RGB if set, else fall back to uniform
    `hex_color`. Returns float (n, 3)."""
    n = len(mesh.vertices)
    visual = mesh.visual
    if (
        isinstance(visual, trimesh.visual.ColorVisuals)
        and visual.vertex_colors is not None
        and len(visual.vertex_colors) == n
    ):
        return visual.vertex_colors[:, :3].astype(float)
    return np.tile(np.array(hex_to_rgb(hex_color), dtype=float), (n, 1))


def _set_rgb(mesh, rgb):
    rgb = np.clip(rgb, 0, 255)
    rgba = np.hstack([rgb, 255 * np.ones((len(rgb), 1))]).astype(np.uint8)
    mesh.visual.vertex_colors = rgba


def sanitize_name(name):
    safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", name or "").strip("_")
    return safe[:32] or "diao"


# --------------------------- surface effects --------------------------


def apply_wrinkles(mesh, intensity, smoothness, seed):
    if intensity <= 0:
        return mesh
    rng = np.random.default_rng(seed)
    perturbations = rng.uniform(-intensity, intensity, mesh.vertices.shape)
    for axis in range(3):
        perturbations[:, axis] = gaussian_filter(
            perturbations[:, axis], sigma=max(smoothness, 0.01)
        )
    mesh.vertices = mesh.vertices + perturbations
    return mesh


def _tone_intensity(wrinkle_intensity):
    """Map a wrinkle slider value (0..0.5) to a skin-tone variation
    intensity (0..0.8). Includes a small baseline so even Smooth-preset
    surfaces have a hint of color variation rather than looking like
    flat plastic."""
    return float(np.clip(0.15 + wrinkle_intensity * 2.5, 0.0, 0.8))


def apply_skin_tone_variation(mesh, base_color_hex, intensity, seed):
    """Add spatially-coherent per-vertex color jitter so the surface
    isn't a flat single color. Mimics real skin: warmer (more vascularised)
    blotches and slightly cooler/paler patches."""
    n = len(mesh.vertices)
    if n == 0 or intensity <= 0:
        return mesh
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, 1.0, n)
    # Smooth across the flat vertex array — neighbouring vertices in
    # procedurally-built shapes tend to be neighbours in space, so this
    # gives roughly spatial coherence (good enough for blotchy skin).
    smoothed = gaussian_filter(raw, sigma=10.0)
    s = smoothed / max(smoothed.std(), 1e-6)
    s = np.clip(s, -1.6, 1.6)

    warm = np.array([22.0, -8.0, -10.0])  # toward red/pink
    cool = np.array([-5.0, 6.0, 10.0])     # toward pale/cool

    delta = np.where(
        s[:, None] > 0,
        s[:, None] * warm,
        -s[:, None] * cool,
    ) * intensity

    rgb = _existing_or_uniform_rgb(mesh, base_color_hex)
    _set_rgb(mesh, rgb + delta)
    return mesh


def apply_shaft_gradient(mesh, base_color_hex, shaft_len, top_redden=14.0):
    """Subtle z-axis color gradient — tip a touch redder than the base
    (mimics blood pooling toward the glans)."""
    n = len(mesh.vertices)
    if n == 0 or shaft_len <= 0:
        return mesh
    z = mesh.vertices[:, 2]
    t = np.clip(z / shaft_len, 0.0, 1.0)
    direction = np.array([top_redden, -top_redden * 0.4, -top_redden * 0.4])
    rgb = _existing_or_uniform_rgb(mesh, base_color_hex)
    _set_rgb(mesh, rgb + t[:, None] * direction)
    return mesh


def apply_glans_tip_redden(mesh, head_radius, head_z_center,
                           redden_intensity=70.0):
    """Apply a localized red tint to vertices very near the central tip
    of the glans — where the urethral meatus would be. Real glans is
    mostly skin-tone with just this one spot vivid red, so we get the
    accent without painting the entire head."""
    n = len(mesh.vertices)
    if n == 0:
        return mesh
    top_z = head_z_center + head_radius
    z_dist = np.abs(mesh.vertices[:, 2] - top_z)
    radial = np.linalg.norm(mesh.vertices[:, :2], axis=1)
    proximity = np.exp(
        -((z_dist / (head_radius * 0.22)) ** 2 + (radial / (head_radius * 0.25)) ** 2)
    )
    # Push toward red: +R, -G, -B (saturates toward red without just
    # adding luminance).
    delta = np.outer(proximity, np.array([redden_intensity,
                                          -redden_intensity * 0.45,
                                          -redden_intensity * 0.45]))
    rgb = _existing_or_uniform_rgb(mesh, "#000000")
    _set_rgb(mesh, rgb + delta)
    return mesh


def apply_curvature(mesh, curvature, top_z):
    """Bend the mesh along +y as a function of z. Anything at z<=0 stays put,
    anything at z>=top_z gets shifted by curvature*top_z. Quadratic in between."""
    if abs(curvature) < 1e-6 or top_z <= 0:
        return mesh
    v = mesh.vertices.copy()
    t = np.clip(v[:, 2] / top_z, 0, None)
    v[:, 1] += curvature * top_z * (t ** 2)
    mesh.vertices = v
    return mesh


def apply_regional_wrinkles(mesh, shaft_intensity, head_intensity, scrotum_intensity,
                            shaft_len, head_z_center, smoothness, seed):
    """Apply wrinkles whose intensity blends smoothly between regions
    along z. The body is one mesh now (SDF + marching cubes), so we can't
    apply per-part wrinkle intensities independently — instead we mix
    them per-vertex using z-coordinate."""
    if max(shaft_intensity, head_intensity, scrotum_intensity) <= 0:
        return mesh
    rng = np.random.default_rng(seed)
    perturbations = rng.uniform(-1.0, 1.0, mesh.vertices.shape)
    for axis in range(3):
        perturbations[:, axis] = gaussian_filter(
            perturbations[:, axis], sigma=max(smoothness, 0.01)
        )

    z = mesh.vertices[:, 2]
    y = mesh.vertices[:, 1]
    head_low = head_z_center - 0.5
    head_high = head_z_center + 0.5
    # Scrotum wrinkles only kick in well below the neck along -Y.
    scrotum_high_y = -1.0
    scrotum_low_y = -2.5

    t_head = np.clip(
        (z - head_low) / max(head_high - head_low, 1e-6), 0, 1
    )
    t_scrotum = np.clip(
        (scrotum_high_y - y) / max(scrotum_high_y - scrotum_low_y, 1e-6), 0, 1
    )

    intensity = (1 - t_head) * shaft_intensity + t_head * head_intensity
    intensity = (1 - t_scrotum) * intensity + t_scrotum * scrotum_intensity

    mesh.vertices = mesh.vertices + perturbations * intensity[:, None]
    return mesh


def color_unified_by_region(mesh, shaft_color, head_color, scrotum_color,
                            shaft_len, head_z_center, head_radius,
                            scrotum_radius):
    """Blend shaft / head / scrotum colors smoothly across the body.
    head ↔ shaft transition runs along Z (shaft axis); shaft ↔ scrotum
    transition runs along Y (scrotum hangs perpendicular in -Y)."""
    n = len(mesh.vertices)
    if n == 0:
        return mesh
    z = mesh.vertices[:, 2]
    y = mesh.vertices[:, 1]

    rgb_shaft = np.array(hex_to_rgb(shaft_color), dtype=float)
    rgb_head = np.array(hex_to_rgb(head_color), dtype=float)
    rgb_scrotum = np.array(hex_to_rgb(scrotum_color), dtype=float)

    head_low = shaft_len - head_radius * 0.6
    head_high = shaft_len + head_radius * 0.5
    # Scrotum centered at y = -1.1 * scrotum_radius. Color band runs
    # from the upper edge of the sphere down through its middle.
    scrotum_high_y = -scrotum_radius * 0.3
    scrotum_low_y = -scrotum_radius * 1.0

    t_head = np.clip(
        (z - head_low) / max(head_high - head_low, 1e-6), 0, 1
    )
    t_scrotum = np.clip(
        (scrotum_high_y - y) / max(scrotum_high_y - scrotum_low_y, 1e-6), 0, 1
    )

    rgb = (1 - t_head[:, None]) * rgb_shaft + t_head[:, None] * rgb_head
    rgb = (1 - t_scrotum[:, None]) * rgb + t_scrotum[:, None] * rgb_scrotum

    rgba = np.hstack([rgb, 255 * np.ones((n, 1))]).astype(np.uint8)
    mesh.visual.vertex_colors = rgba
    return mesh


# --------------------------- material packs ---------------------------


def _smooth_noise(points, seed, frequency=1.0, octaves=3, size=24):
    """Smooth 3D noise in roughly [-1, 1] (std ≈ 0.4): a random lattice
    read back with cubic-spline interpolation, summed over a few octaves.
    Cheap enough to evaluate per vertex, which is all the packs need."""
    rng = np.random.default_rng(seed)
    points = np.asarray(points, dtype=float)
    total = np.zeros(len(points))
    amp, norm = 1.0, 0.0
    for _ in range(octaves):
        lattice = rng.uniform(-1.0, 1.0, (size, size, size))
        coords = (points * frequency).T + size / 2.0
        total += amp * map_coordinates(lattice, coords, order=3, mode="grid-wrap")
        norm += amp
        amp *= 0.5
        frequency *= 2.0
    return total / norm


def _hex(color):
    return np.array(hex_to_rgb(color), dtype=float)


def _mix(a, b, t):
    """Per-vertex lerp between colors a and b (each (3,) or (n, 3))."""
    t = np.clip(t, 0.0, 1.0)[:, None]
    return a * (1.0 - t) + b * t


def _hsv_to_rgb(h, s, v):
    h = (np.asarray(h, dtype=float) % 1.0) * 6.0
    i = np.floor(h).astype(int) % 6
    f = h - np.floor(h)
    s = np.broadcast_to(s, h.shape)
    v = np.broadcast_to(v, h.shape)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=1) * 255.0


def _solid(color, jitter=0.0):
    def pattern(p, seed):
        rgb = np.tile(_hex(color), (len(p), 1))
        if jitter:
            rgb *= 1.0 + jitter * _smooth_noise(p, seed, 0.8)[:, None]
        return rgb
    return pattern


def _marble(p, seed):
    base = _mix(_hex("#F4F1EC"), _hex("#DAD4CA"), 0.5 + _smooth_noise(p, seed + 1, 0.6))
    t = p @ np.array([0.5, 0.3, 0.9]) + 3.0 * _smooth_noise(p, seed, 0.35)
    veins = np.exp(-(np.abs(np.sin(2.0 * t)) / 0.12) ** 2)
    return _mix(base, _hex("#76737A"), 0.85 * veins)


def _wood(p, seed):
    # Growth rings around the model's long axis, wobbled by noise, plus
    # fine grain streaks stretched along that axis.
    r = np.hypot(p[:, 0], p[:, 1]) + 0.35 * _smooth_noise(p, seed, 0.5)
    rings = (0.5 + 0.5 * np.sin(12.0 * r)) ** 3
    grain = _smooth_noise(p * np.array([4.0, 4.0, 0.4]), seed + 1)
    return _mix(_hex("#DDA56B"), _hex("#8E5427"), 0.8 * rings + 0.2 * (grain + 0.5))


def _rainbow(p, seed):
    z = p[:, 2]
    hue = 0.85 * (z - z.min()) / max(np.ptp(z), 1e-6) + 0.08 * _smooth_noise(p, seed, 0.4)
    return _hsv_to_rgb(hue, 0.75, 1.0)


def _candy_cane(p, seed):
    angle = np.arctan2(p[:, 1], p[:, 0])
    stripe = 0.5 + 2.0 * np.sin(2.0 * angle + 2.4 * p[:, 2])  # soft-edged helix
    return _mix(_hex("#FFFFFF"), _hex("#E0242C"), stripe)


def _camo(p, seed):
    a = _smooth_noise(p, seed, 0.45)
    b = _smooth_noise(p, seed + 7, 0.45)
    rgb = np.tile(_hex("#B9A97A"), (len(p), 1))  # sand
    rgb[a > 0.05] = _hex("#6B7A3A")              # olive
    rgb[b > 0.12] = _hex("#5A4030")              # brown
    rgb[a > 0.30] = _hex("#36432A")              # dark green
    return rgb


def _galaxy(p, seed):
    rgb = _mix(_hex("#0A0F2E"), _hex("#4C1D72"), 0.4 + 1.8 * _smooth_noise(p, seed, 0.4))
    nebula = 3.0 * (_smooth_noise(p, seed + 3, 0.7) - 0.25)
    rgb = _mix(rgb, _hex("#D9488F"), 0.6 * np.clip(nebula, 0.0, 1.0))
    stars = np.random.default_rng(seed + 5).random(len(p)) < 0.015
    rgb[stars] = 255.0
    return rgb


# Each pack recolors every vertex (hair included) and carries its own
# PBR settings, which get injected into the GLB. "Original" keeps the
# color pickers and the untouched export path.
MATERIAL_PACKS = {
    "Original": None,
    "Gold": dict(pattern=_solid("#FFC75A", jitter=0.03), metallic=1.0, roughness=0.22),
    "Chrome": dict(pattern=_solid("#F2F3F5"), metallic=1.0, roughness=0.06),
    "Marble": dict(pattern=_marble, metallic=0.0, roughness=0.18),
    "Wood": dict(pattern=_wood, metallic=0.0, roughness=0.6),
    "Rainbow": dict(pattern=_rainbow, metallic=0.0, roughness=0.3),
    "Candy cane": dict(pattern=_candy_cane, metallic=0.0, roughness=0.12),
    "Camo": dict(pattern=_camo, metallic=0.0, roughness=0.85),
    "Galaxy": dict(pattern=_galaxy, metallic=0.0, roughness=0.35),
    "Neon": dict(pattern=_solid("#FF2BD6"), metallic=0.0, roughness=0.25,
                 emissive=(0.9, 0.05, 0.6)),
    "Chocolate": dict(pattern=_solid("#4B2716", jitter=0.06), metallic=0.0, roughness=0.3),
}


def apply_material_pack(mesh, pack_name, seed):
    """Overwrite vertex colors with the pack's pattern (a no-op for
    "Original"). Patterns are authored in sRGB and stored linear, since
    glTF vertex colors are linear."""
    pack = MATERIAL_PACKS.get(pack_name)
    if pack is None:
        return mesh
    rgb = pack["pattern"](np.asarray(mesh.vertices, dtype=float), seed)
    rgb = _srgb_to_linear_uint8(np.clip(rgb, 0, 255))
    mesh.visual.vertex_colors = np.hstack([rgb, np.full((len(rgb), 1), 255, np.uint8)])
    return mesh


# ------------------------------ SDF body ------------------------------


def _sdf_sphere(p, center, radius):
    return np.linalg.norm(p - np.asarray(center, dtype=float), axis=1) - radius


def _sdf_capsule(p, a, b, radius):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pa = p - a
    ba = b - a
    h = np.clip((pa @ ba) / max(np.dot(ba, ba), 1e-9), 0.0, 1.0)
    return np.linalg.norm(pa - h[:, None] * ba, axis=1) - radius


def _sdf_torus(p, center, major_r, minor_r):
    pl = p - np.asarray(center, dtype=float)
    q = np.column_stack([np.linalg.norm(pl[:, :2], axis=1) - major_r, pl[:, 2]])
    return np.linalg.norm(q, axis=1) - minor_r


def _smooth_min(a, b, k):
    """Cubic polynomial smooth-min — produces a continuous, blended
    minimum that gives natural-looking junctions when used to combine
    SDFs of overlapping primitives."""
    h = np.clip(0.5 + 0.5 * (b - a) / max(k, 1e-9), 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def _phallus_sdf(points, shaft_len, shaft_radius, head_radius, scrotum_radius,
                 corona_intensity):
    """SDF for the whole body in MODEL space.

    Shape orientation in this space:
      - shaft along +Z   (head at +Z end)
      - scrotum hangs in -Y (perpendicular to the shaft, below it)
    A final rotation in create_model maps model +Z → world +X so the
    shaft ends up horizontal in the viewer with the scrotum hanging
    straight down.
    """
    shaft_eff_r = shaft_radius * 1.5
    shaft = _sdf_capsule(points, [0, 0, 0], [0, 0, shaft_len], shaft_eff_r)

    head_z = shaft_len + head_radius * 0.4  # sit head fairly low so it caps the shaft cleanly
    head = _sdf_sphere(points, [0, 0, head_z], head_radius)

    # Scrotum hangs PERPENDICULAR to the shaft, in -Y, near the shaft
    # root. Two balls offset on +X / -X (this lateral axis becomes
    # the view's depth after the final rotation, so balls show as
    # front/back from a 3/4 view).
    scrotum_offset_x = scrotum_radius / 1.2
    scrotum_y = -scrotum_radius * 1.1
    # Slightly forward of the shaft's root along Z (i.e. at z just
    # below shaft start) so the neck droops naturally rather than
    # poking out of the side of the shaft.
    scrotum_z = -shaft_radius * 0.4
    scr1 = _sdf_sphere(points, [-scrotum_offset_x, scrotum_y, scrotum_z], scrotum_radius)
    scr2 = _sdf_sphere(points, [+scrotum_offset_x, scrotum_y, scrotum_z], scrotum_radius)

    # Neck capsule: short tube from shaft root → scrotum top, in -Y.
    # Sit it deeper into both shapes so the visible stem is short.
    neck_top_y = -shaft_radius * 0.2
    neck_bottom_y = scrotum_y + scrotum_radius * 0.7
    neck = _sdf_capsule(
        points,
        [0, neck_top_y, 0],
        [0, neck_bottom_y, scrotum_z],
        shaft_radius * 0.7,
    )

    sd = _smooth_min(shaft, head, 0.55)
    sd = _smooth_min(sd, neck, 0.45)
    sd = _smooth_min(sd, scr1, 0.55)
    sd = _smooth_min(sd, scr2, 0.55)

    if corona_intensity > 0:
        corona_minor = head_radius * 0.10 * corona_intensity
        corona_major = head_radius * 1.0
        corona_z = head_z - head_radius * 0.55
        corona = _sdf_torus(points, [0, 0, corona_z], corona_major, corona_minor)
        sd = _smooth_min(sd, corona, 0.18)

    return sd, head_z


def _build_unified_body(shaft_len, shaft_radius, head_radius, scrotum_radius,
                        corona_intensity, resolution=0.15):
    """Sample the phallus SDF on a grid and extract the level-0 isosurface
    via marching cubes. Returns (mesh, head_z_center).

    Grid bounds use separate extents per axis since the model is no
    longer roughly cube-shaped:
      - x: ± lateral spread (scrotum balls)
      - y: shaft eff_r (slight) up to scrotum bottom (deep negative)
      - z: shaft length (long, +Z direction)
    """
    shaft_eff_r = shaft_radius * 1.5
    scrotum_offset_x = scrotum_radius / 1.2

    x_max = max(shaft_eff_r, scrotum_offset_x + scrotum_radius) + 0.5
    y_max = shaft_eff_r + 0.5
    # Scrotum is centered at y = -1.1 * scrotum_radius and extends
    # an additional scrotum_radius below.
    y_min = -(scrotum_radius * 1.1 + scrotum_radius) - 0.5
    z_max = shaft_len + head_radius * 2.2 + 0.5
    # Scrotum spheres are also offset along z by -shaft_radius * 0.4
    # and have radius scrotum_radius, so the grid has to reach below
    # -scrotum_radius - 0.4*shaft_radius (otherwise the spheres get
    # clipped against z_min and you see hollow openings).
    z_min = min(-shaft_eff_r, -scrotum_radius - shaft_radius * 0.4) - 0.5

    nx = int(2 * x_max / resolution) + 1
    ny = int((y_max - y_min) / resolution) + 1
    nz = int((z_max - z_min) / resolution) + 1

    x = np.linspace(-x_max, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    z = np.linspace(z_min, z_max, nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    sd, head_z = _phallus_sdf(
        pts, shaft_len, shaft_radius, head_radius, scrotum_radius, corona_intensity
    )
    sd = sd.reshape(nx, ny, nz)

    verts, faces, _, _ = marching_cubes(
        sd, level=0.0, spacing=(resolution, resolution, resolution)
    )
    verts = verts + np.array([-x_max, y_min, z_min])

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    return mesh, head_z


# ----------------------------- primitives -----------------------------


def create_helix_path(radius, turns, length, center, direction, n=None):
    """Helical polyline of given total axial `length` and `turns` revolutions,
    rotated to point along `direction` and translated to `center`."""
    if turns <= 0 or length <= 0:
        return np.empty((0, 3))
    n = n or max(8, 20 * int(np.ceil(turns)) + 1)
    t = np.linspace(0, 2 * np.pi * turns, n)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = length * t / (2 * np.pi * turns)
    points = np.column_stack((x, y, z))

    direction = np.asarray(direction, dtype=float)
    direction = direction / max(np.linalg.norm(direction), 1e-9)
    if not np.allclose(direction, [0, 0, 1]):
        axis = np.cross([0, 0, 1], direction)
        if np.linalg.norm(axis) > 1e-9:
            angle = np.arccos(np.clip(direction[2], -1.0, 1.0))
            R = trimesh.transformations.rotation_matrix(angle, axis)[:3, :3]
            points = points @ R.T
    return points + np.asarray(center)


def create_tube(points, tube_radius, num_segments=6):
    """Build a triangulated tube mesh around a polyline."""
    if len(points) < 2 or tube_radius <= 0:
        return trimesh.Trimesh()
    angle = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
    circle = np.stack(
        [np.cos(angle), np.sin(angle), np.zeros_like(angle)], axis=1
    ) * tube_radius
    rings = [circle + p for p in points]
    faces = []
    for i in range(len(rings) - 1):
        for j in range(num_segments):
            jn = (j + 1) % num_segments
            a = i * num_segments + j
            b = i * num_segments + jn
            c = (i + 1) * num_segments + j
            d = (i + 1) * num_segments + jn
            faces.append([a, b, c])
            faces.append([c, b, d])
    vertices = np.vstack(rings)
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces))


# ---------------------------- feature parts ---------------------------


def _vein_paths_on_shaft(shaft_len, shaft_radius, vein_count, seed):
    """Generate vein polylines on the unified shaft surface.
    Anchored to a single capsule of effective radius `shaft_radius * 1.5`,
    matching the SDF that built the unified body."""
    if vein_count <= 0:
        return []
    rng = np.random.default_rng(seed + 5)
    eff_r = shaft_radius * 1.5
    paths = []
    for _ in range(int(vein_count)):
        base_angle = rng.uniform(0, 2 * np.pi)
        wind_amp = rng.uniform(0.4, 1.0)
        wind_freq = rng.uniform(1.5, 3.0) * np.pi / max(shaft_len, 1e-3)
        phase = rng.uniform(0, 2 * np.pi)
        z_lo = rng.uniform(0.10, 0.25) * shaft_len
        z_hi = rng.uniform(0.80, 0.92) * shaft_len
        z = np.linspace(z_lo, z_hi, 50)
        angle = base_angle + wind_amp * np.sin(wind_freq * z + phase)
        x = eff_r * np.cos(angle)
        y = eff_r * np.sin(angle)
        paths.append(np.column_stack((x, y, z)))
    return paths


def _displace_and_color_veins(shaft_mesh, vein_paths, vein_radius, base_color):
    """Push shaft vertices outward along the local normal where they sit
    near a vein path point, and tint them darker. Only the
    near-vein vertices are modified; everything else keeps whatever
    color was already on it (so head red and scrotum tan survive)."""
    if not vein_paths or vein_radius <= 0:
        return shaft_mesh

    all_pts = np.vstack(vein_paths)
    tree = cKDTree(all_pts)
    dists, _ = tree.query(shaft_mesh.vertices)

    sigma_push = max(vein_radius * 1.4, 1e-4)
    sigma_color = max(vein_radius * 2.2, 1e-4)
    push_amp = vein_radius * 1.6

    push = push_amp * np.exp(-(dists / sigma_push) ** 2)
    normals = shaft_mesh.vertex_normals  # cached; trimesh invalidates on vertex change
    shaft_mesh.vertices = shaft_mesh.vertices + normals * push[:, None]

    # Vein vertex color: read whatever color was set previously and blend
    # ONLY the near-vein vertices toward a cool dark tone. Don't reset
    # far vertices to base_color — they already have their region color
    # (head red, scrotum tan, etc.).
    existing_rgb = _existing_or_uniform_rgb(shaft_mesh, base_color)
    cool_dark = np.array([55.0, 40.0, 80.0])  # desaturated dark purple
    weight = np.exp(-(dists / sigma_color) ** 2)[:, None]
    rgb = existing_rgb * (1 - weight) + cool_dark * weight
    _set_rgb(shaft_mesh, rgb)
    return shaft_mesh


def _random_walk_hair_path(length, center, base_direction, seed,
                           n_segments=14):
    """Build a hair polyline as a small random walk biased toward
    `base_direction`. Each strand grows outward but bends randomly along
    the way — every strand has its own seed so no two are identical.

    Direction at each step blends the previous direction with a fresh
    random perturbation and a pull back toward base_direction (so the
    hair doesn't curl back on itself)."""
    rng = np.random.default_rng(seed)
    base_direction = np.asarray(base_direction, dtype=float)
    base_direction = base_direction / max(np.linalg.norm(base_direction), 1e-9)
    step = length / n_segments

    # Initial direction: base + a small per-strand jitter (so different
    # strands set off at different angles).
    initial_jitter = rng.uniform(-0.25, 0.25, size=3)
    current_dir = base_direction + initial_jitter
    current_dir = current_dir / max(np.linalg.norm(current_dir), 1e-9)

    points = [np.asarray(center, dtype=float)]
    for i in range(n_segments):
        # Random bend that grows toward the tip of the strand.
        bend_strength = 0.18 + 0.05 * (i / n_segments)
        delta = rng.normal(0.0, bend_strength, size=3)
        current_dir = 0.85 * current_dir + delta
        # Keep the strand pointed roughly outward — don't let it loop.
        current_dir = 0.78 * current_dir + 0.22 * base_direction
        current_dir = current_dir / max(np.linalg.norm(current_dir), 1e-9)
        points.append(points[-1] + step * current_dir)
    return np.asarray(points)


def make_hair(surface_mesh, num_hairs, hair_length, seed, max_y=None):
    """Random-bent strands rooted at sampled vertices and growing along
    the local normal. Each strand has its own seed so no two are alike.

    `max_y` restricts sampling to vertices below that y height — used to
    keep hair on the scrotum region (which hangs in -Y in model space)."""
    n = len(surface_mesh.vertices)
    if n == 0 or num_hairs <= 0:
        return None
    candidate_idx = np.arange(n)
    if max_y is not None:
        candidate_idx = candidate_idx[surface_mesh.vertices[candidate_idx, 1] < max_y]
    if len(candidate_idx) == 0:
        return None
    rng = np.random.default_rng(seed)
    indices = rng.choice(candidate_idx, size=num_hairs, replace=True)
    strand_seeds = rng.integers(0, 2**31 - 1, size=num_hairs)

    hairs = []
    for idx, strand_seed in zip(indices, strand_seeds):
        point = surface_mesh.vertices[idx]
        normal = surface_mesh.vertex_normals[idx]
        # Skip the top half of the scrotum (normal[1] > 0.5 means the
        # vertex is on the side of the sac that faces the shaft) so we
        # don't get hair tufts pointing up into the body.
        if normal[1] > 0.5:
            continue
        direction = normal + rng.uniform(-0.2, 0.2, size=3)
        direction = direction / max(np.linalg.norm(direction), 1e-9)
        strand_len = float(hair_length) * float(rng.uniform(0.75, 1.15))
        path = _random_walk_hair_path(
            strand_len, point, direction, seed=int(strand_seed),
        )
        thickness = 0.018 * float(rng.uniform(0.85, 1.10))
        hairs.append(create_tube(path, tube_radius=thickness, num_segments=4))
    if not hairs:
        return None
    hair_mesh = trimesh.util.concatenate(hairs)
    return color_mesh_uniform(hair_mesh, "#1a1a1a")


def _srgb_to_linear_uint8(rgb_uint8):
    """Convert an sRGB-encoded uint8 color array to linear-light uint8.
    glTF 2.0 PBR shaders expect vertex colors in linear space, but the
    color values we author (and that color pickers produce) are sRGB.
    Without this conversion the lit result looks washed out / overbright."""
    s = np.asarray(rgb_uint8, dtype=float) / 255.0
    linear = np.where(
        s <= 0.04045,
        s / 12.92,
        ((s + 0.055) / 1.055) ** 2.4,
    )
    return np.clip(linear * 255.0, 0, 255).astype(np.uint8)


def _inject_pbr_material(glb_bytes,
                         metallic=0.0,
                         roughness=0.78,
                         emissive=(0.0, 0.0, 0.0),
                         base_color_factor=1.0,
                         double_sided=True,
                         unlit=False,
                         name="material"):
    """Inject a glTF material and point every primitive at it.

    trimesh's vertex-color export carries no PBR parameters, so this is
    how a material pack gets its metalness, roughness and glow; the
    vertex colors still supply the base color.

    With `unlit=True` we use the standard `KHR_materials_unlit`
    extension. Why: model-viewer's default lit pipeline runs ACES
    tonemapping on top of a bright neutral IBL, which compresses the
    region-distinct hues we're authoring (shaft brown / head deep red /
    scrotum tan) into similar peach-orange. Unlit makes the viewer
    display vertex colors directly (modulo gamma), preserving the hue
    differences. We lose the soft specular highlight, but the alternative
    is colors that all read the same.
    """
    # parse glb chunks
    if glb_bytes[:4] != b"glTF":
        return glb_bytes
    json_len = struct.unpack("<I", glb_bytes[12:16])[0]
    if glb_bytes[16:20] != b"JSON":
        return glb_bytes
    json_str = glb_bytes[20:20 + json_len].decode("utf-8").rstrip("\x00 ")
    gltf = json.loads(json_str)

    # binary chunk follows the JSON chunk
    bin_offset = 20 + json_len
    bin_len = struct.unpack("<I", glb_bytes[bin_offset:bin_offset + 4])[0]
    bin_data = glb_bytes[bin_offset + 8:bin_offset + 8 + bin_len]

    bcf = float(base_color_factor)
    material = {
        "name": name,
        "pbrMetallicRoughness": {
            "baseColorFactor": [bcf, bcf, bcf, 1.0],
            "metallicFactor": float(metallic),
            "roughnessFactor": float(roughness),
        },
        "doubleSided": bool(double_sided),
    }
    if any(emissive):
        material["emissiveFactor"] = [float(c) for c in emissive]
    if unlit:
        material["extensions"] = {"KHR_materials_unlit": {}}
        ext_used = gltf.setdefault("extensionsUsed", [])
        if "KHR_materials_unlit" not in ext_used:
            ext_used.append("KHR_materials_unlit")

    materials = gltf.setdefault("materials", [])
    materials.append(material)
    mat_idx = len(materials) - 1
    for m in gltf.get("meshes", []):
        for p in m.get("primitives", []):
            p["material"] = mat_idx
    return _pack_glb(gltf, bin_data)


def _pack_glb(gltf, bin_data):
    """glTF JSON + binary buffer → .glb bytes. The JSON chunk is padded
    to a 4-byte boundary with spaces, the binary chunk with zeros."""
    json_chunk = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * (-len(json_chunk) % 4)
    bin_data = bytes(bin_data) + b"\0" * (-len(bin_data) % 4)
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_data)
    return (
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<I4s", len(json_chunk), b"JSON") + json_chunk
        + struct.pack("<I4s", len(bin_data), b"BIN\0") + bin_data
    )


# ------------------------------ skin packs ------------------------------
#
# A skin pack swaps the flat per-vertex color for baked textures: the body
# gets a UV atlas (see "texture baking" below) and every texel is rendered
# from the procedural layers in `_skin_layers` — mottling, veins under the
# skin, creases, follicles, freckles, fine grain — into albedo, normal and
# roughness maps. A pack is just a strength for each layer.


SKIN_PACKS = {
    "None": None,
    "Natural": dict(mottle=1.0, veins=0.6, creases=1.0, follicles=1.0,
                    freckles=0.0, goosebumps=0.0, grain=1.0, shine=0.0),
    "Silky": dict(mottle=0.4, veins=0.3, creases=0.25, follicles=0.3,
                  freckles=0.0, goosebumps=0.0, grain=0.3, shine=0.16),
    "Weathered": dict(mottle=2.0, veins=1.0, creases=2.5, follicles=1.4,
                      freckles=0.35, goosebumps=0.0, grain=2.2, shine=-0.12),
    "Freckled": dict(mottle=0.9, veins=0.4, creases=0.8, follicles=0.8,
                     freckles=1.0, goosebumps=0.0, grain=1.0, shine=0.0),
    "Goosebumps": dict(mottle=0.8, veins=0.5, creases=1.2, follicles=1.2,
                       freckles=0.0, goosebumps=1.0, grain=1.0, shine=-0.05),
}


def _smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@functools.lru_cache(maxsize=1)
def _noise_lattice(size=64):
    """The random lattice every `_detail_noise` call reads: gaussian-
    smoothed so trilinear reads look smooth, scaled to unit variance."""
    rng = np.random.default_rng(1)
    lattice = gaussian_filter(rng.normal(size=(size, size, size)), 1.0, mode="wrap")
    return (lattice / lattice.std()).astype(np.float32)


def _random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def _detail_noise(points, scale, seed):
    """Unit-variance 3D noise with blobs about `scale` model units across.
    `_smooth_noise` is too slow for ~1M texels, so this reads the shared
    lattice trilinearly — randomly rotated and offset per seed, which
    hides both the lattice axes and its repeats."""
    lattice = _noise_lattice()
    rng = np.random.default_rng(seed)
    coords = points @ _random_rotation(rng) * (2.5 / scale) + rng.uniform(0, len(lattice), 3)
    return map_coordinates(lattice, coords.T, order=1, mode="grid-wrap")


def _skin_regions(points, shaft_len, shaft_radius, head_radius, scrotum_radius):
    """Soft (shaft, head, scrotum) weights per point, by whichever part of
    the SDF (same primitives as `_phallus_sdf`) the point is closest to."""
    shaft = _sdf_capsule(points, [0, 0, 0], [0, 0, shaft_len], shaft_radius * 1.5)
    head = _sdf_sphere(points, [0, 0, shaft_len + head_radius * 0.4], head_radius)
    sx, sy, sz = scrotum_radius / 1.2, -scrotum_radius * 1.1, -shaft_radius * 0.4
    scrotum = np.minimum(
        _sdf_sphere(points, [-sx, sy, sz], scrotum_radius),
        _sdf_sphere(points, [sx, sy, sz], scrotum_radius),
    )
    w_scrotum = _smoothstep(-0.3, 0.3, shaft - scrotum)
    w_head = _smoothstep(-0.15, 0.15, shaft - head) * (1.0 - w_scrotum)
    return 1.0 - w_head - w_scrotum, w_head, w_scrotum


def _scatter_spots(points, covered, weight, count, radius, rng):
    """Drop `count` round spots on the surface, denser where `weight` is
    high. Returns, per point, the distance to the nearest spot center in
    units of that spot's radius (inf when out of reach) and the spot's
    random strength."""
    n = len(points)
    candidates = np.flatnonzero(covered & (weight > 0))
    if int(count) < 1 or len(candidates) == 0:
        return np.full(n, np.inf), np.zeros(n)
    p = weight[candidates] / weight[candidates].sum()
    centers = points[rng.choice(candidates, size=int(count), p=p)]
    radii = radius * rng.uniform(0.7, 1.3, len(centers))
    strength = rng.uniform(0.5, 1.0, len(centers))
    dist, idx = cKDTree(centers).query(points, distance_upper_bound=radius * 3.9, workers=-1)
    hit = np.isfinite(dist)
    d, s = np.full(n, np.inf), np.zeros(n)
    d[hit] = dist[hit] / radii[idx[hit]]
    s[hit] = strength[idx[hit]]
    return d, s


def _skin_layers(points, covered, shape, pack, seed, scale, area):
    """Procedural skin at rest-space texel `points` (`covered` marks the
    ones actually on the surface, not in a gutter). Feature sizes are
    multiplied by `scale`, so a big or a small model gets the same look.

    Returns (tint, height, roughness) per point: a multiplicative
    linear-RGB tint for the albedo, a bump height in model units and a
    roughness value."""
    shaft_len, shaft_radius, head_radius, scrotum_radius = shape[:4]
    w_shaft, w_head, w_scrotum = _skin_regions(
        points, shaft_len, shaft_radius, head_radius, scrotum_radius
    )
    x, y, z = points.T
    s = scale
    rng = np.random.default_rng(seed)
    tint = np.ones((len(points), 3))
    height = np.zeros(len(points))
    rough = 0.5 * w_shaft + 0.34 * w_head + 0.62 * w_scrotum - pack["shine"]

    # Mottling: broad warm (vascular) and cool (pale) blotches, plus a
    # finer patchiness in plain brightness.
    blotch = (0.7 * _detail_noise(points, 0.9 * s, seed)
              + 0.3 * _detail_noise(points, 0.3 * s, seed + 1))
    tint *= 1 + pack["mottle"] * (np.outer(np.maximum(blotch, 0), [0.08, -0.03, -0.04])
                                  + np.outer(np.maximum(-blotch, 0), [-0.03, 0.0, 0.025]))
    tint *= (1 + 0.05 * pack["mottle"] * _detail_noise(points, 0.5 * s, seed + 2))[:, None]

    # Veins under the skin: the zero crossings of a noise field form a
    # network of thin lines — stretched lengthwise off the scrotum.
    stretch = np.column_stack([np.ones_like(z), np.ones_like(z), 0.35 + 0.65 * w_scrotum])
    lines = np.exp(-(_detail_noise(points * stretch, 0.8 * s, seed + 3) / 0.14) ** 2)
    patches = _smoothstep(-0.3, 0.8, _detail_noise(points, 1.6 * s, seed + 4))
    vein = pack["veins"] * lines * patches * (1 - 0.85 * w_head)
    tint *= 1 - np.outer(vein, [0.10, 0.08, -0.02])

    # Creases: wavy rings around the shaft and meandering folds (rugae)
    # across the scrotum, each fading in and out in patches; plus the
    # raphe, a faint seam down the scrotum and along under the shaft.
    warp = _detail_noise(points, 0.9 * s, seed + 5)
    rings = np.exp(-(1 + np.cos(2 * np.pi * z / (0.42 * s) + 2.2 * warp)) / 0.22)
    folds = np.exp(-(1 + np.cos(2 * np.pi * y / (0.36 * s) + 1.6 * warp
                                + 0.6 * _detail_noise(points, 0.3 * s, seed + 6))) / 0.3)
    fade = _smoothstep(-0.4, 0.6, _detail_noise(points, 1.2 * s, seed + 7))
    crease = pack["creases"] * fade * (rings * w_shaft + 1.4 * folds * w_scrotum)
    underside = _smoothstep(0.0, -0.8, y / (1.5 * shaft_radius))
    raphe = pack["creases"] * np.exp(-(x / (0.06 * s)) ** 2) * (w_scrotum + w_shaft * underside)
    height += s * (0.006 * raphe - 0.01 * crease)
    tint *= (1 - 0.1 * crease - 0.06 * raphe)[:, None]
    rough += 0.04 * crease

    # Follicles: small raised bumps on the scrotum — and all over the
    # shaft with goosebumps.
    goose = pack["goosebumps"]
    where = w_scrotum + goose * w_shaft
    count = pack["follicles"] * (14 + 16 * goose) / s ** 2 * area * where[covered].mean()
    d, strength = _scatter_spots(points, covered, where, count, 0.032 * s * (1 + 0.9 * goose), rng)
    bump = strength * np.exp(-d ** 2)
    height += s * 0.01 * (1 + goose) * bump
    tint *= 1 - np.outer(0.1 * bump, [0.8, 1.0, 1.0])
    rough += 0.06 * bump

    # Freckles: flat brown spots, sharper-edged than follicles.
    if pack["freckles"] > 0:
        count = pack["freckles"] * 5 / s ** 2 * area
        d, strength = _scatter_spots(points, covered, 1 - 0.7 * w_head, count, 0.075 * s, rng)
        tint *= 1 - np.outer(strength * _smoothstep(1.0, 0.45, d), [0.30, 0.40, 0.48])

    # Grain: a fine network of furrows plus speckle — the micro-relief
    # that breaks up highlights. The glans is smoother.
    furrows = np.exp(-(_detail_noise(points, 0.07 * s, seed + 8) / 0.35) ** 2)
    speckle = _detail_noise(points, 0.03 * s, seed + 9)
    grain = pack["grain"] * (1 - 0.7 * w_head)
    height += s * 0.0025 * grain * (0.6 * speckle - furrows)
    tint *= (1 + grain * (0.025 * speckle - 0.05 * furrows))[:, None]
    rough += 0.05 * grain * furrows

    return tint, height, np.clip(rough, 0.12, 0.95)


# ---------------------------- texture baking ----------------------------


SKIN_TEXTURE_SIZE = 1024
_ATLAS_PAD = 4  # gutter texels around each chart, so mipmaps don't bleed


def _skyline_pack(widths, heights, size, pad):
    """Bottom-left skyline packing of boxes (in texels, each grown by
    `pad` on every side) into a size×size square, trying both
    orientations. Returns (x, y, rotated) per box, or None if they
    don't all fit."""
    n = len(widths)
    sky = np.zeros(size, dtype=int)
    x, y, rotated = np.zeros(n, int), np.zeros(n, int), np.zeros(n, bool)
    for i in np.argsort(-np.maximum(widths, heights)):
        best = None
        for rot in (False, True):
            w = int(np.ceil(heights[i] if rot else widths[i])) + 2 * pad
            h = int(np.ceil(widths[i] if rot else heights[i])) + 2 * pad
            if w > size:
                continue
            tops = sliding_window_view(sky, w).max(axis=1)
            at = int(np.argmin(tops))
            if tops[at] + h <= size and (best is None or tops[at] + h < best[0]):
                best = (tops[at] + h, at, w, h, rot)
        if best is None:
            return None
        top, at, w, h, rot = best
        x[i], y[i], rotated[i] = at, top - h, rot
        sky[at:at + w] = top
    return x, y, rotated


def _uv_atlas(mesh, size, pad):
    """Box-projection UV atlas. Each face goes to the axis (±X/±Y/±Z) its
    normal points along most, and every connected patch of same-axis
    faces becomes a chart, flattened by dropping that axis — cheap, and
    no face flips over. Long charts get cut into pieces, then all of
    them are skyline-packed as large as they fit.

    Returns (vmap, faces, xy): vertices are split along chart seams
    (`vmap` = original vertex of each new one) and `xy` is in texels,
    rows running down the image (glTF's UV convention)."""
    verts, faces = mesh.vertices, mesh.faces
    every = np.arange(len(faces))

    def axis_label(normals):  # 0..5 = +X, -X, +Y, -Y, +Z, -Z
        axis = np.abs(normals).argmax(axis=1)
        return axis * 2 + (normals[every, axis] < 0)

    def facing(f, label):  # faces f's own normals along the axes of `label`
        return mesh.face_normals[f, label // 2] * np.where(label % 2 == 1, -1.0, 1.0)

    # Labels come from the smoothed (vertex) normals, so chart borders
    # are clean curves rather than following every wobble.
    label = axis_label(mesh.vertex_normals[faces].sum(axis=1))

    # Charts = connected same-label patches. Tiny ones (where two axes
    # nearly tie) are folded into a neighboring chart, a ring of faces
    # per pass — but only into one whose axis the face itself looks along.
    adj = mesh.face_adjacency

    def patches(label):
        same = label[adj[:, 0]] == label[adj[:, 1]]
        graph = coo_matrix((np.ones(same.sum()), (adj[same, 0], adj[same, 1])),
                           shape=(len(faces), len(faces)))
        return connected_components(graph, directed=False)[1]

    chart = patches(label)
    for _ in range(3):
        small = np.bincount(chart)[chart] < 24
        if not small.any():
            break
        for a, b in ((adj[:, 0], adj[:, 1]), (adj[:, 1], adj[:, 0])):
            take = small[a] & ~small[b]
            take[take] = facing(a[take], label[b[take]]) > 0.3
            label[a[take]] = label[b[take]]
        chart = patches(label)
    # A face turned away from its chart's axis (in tight curls like the
    # corona) would flip over in the projection: give it its own axis.
    # The deep undercut under a huge corona can still overlap a few
    # texels within a chart — slightly misplaced grain, out of sight.
    away = facing(every, label) < 0.1
    if away.any():
        label[away] = axis_label(mesh.face_normals)[away]
        chart = patches(label)

    def project(points, label):
        # Drop the chart's axis; (u, v) stay right-handed around the
        # outward normal, so no chart comes out mirrored.
        ax, r = label // 2, np.arange(len(points))
        u = points[r, (ax + 1) % 3] * np.where(label % 2 == 1, -1.0, 1.0)
        return u, points[r, (ax + 2) % 3]

    # Cut charts longer than `longest` into equal pieces along their
    # long side (the shaft strips would otherwise waste most of the
    # square texture).
    cu, cv = project(verts[faces].mean(axis=1), label)
    longest = 0.55 * np.sqrt(mesh.area)
    piece = np.zeros(len(faces), dtype=np.int64)
    for c in range(chart.max() + 1):
        sel = chart == c
        t = cu[sel] if np.ptp(cu[sel]) >= np.ptp(cv[sel]) else cv[sel]
        pieces = int(np.ceil(np.ptp(t) / longest))
        if pieces > 1:
            piece[sel] = np.minimum((t - t.min()) / np.ptp(t) * pieces, pieces - 1).astype(int)
    chart = np.unique(chart * (piece.max() + 1) + piece, return_inverse=True)[1]
    n_charts = chart.max() + 1

    # One new vertex per (chart, vertex) pair.
    n = len(verts)
    keys, inverse = np.unique(np.repeat(chart, 3) * n + faces.ravel(), return_inverse=True)
    vmap, vchart = keys % n, keys // n
    chart_label = np.zeros(n_charts, dtype=int)
    chart_label[chart] = label
    u, v = project(verts[vmap], chart_label[vchart])

    lo_u, hi_u = np.full(n_charts, np.inf), np.full(n_charts, -np.inf)
    lo_v, hi_v = np.full(n_charts, np.inf), np.full(n_charts, -np.inf)
    np.minimum.at(lo_u, vchart, u)
    np.maximum.at(hi_u, vchart, u)
    np.minimum.at(lo_v, vchart, v)
    np.maximum.at(hi_v, vchart, v)
    w, h = hi_u - lo_u, hi_v - lo_v

    # Largest scale (texels per model unit) at which everything fits.
    lo, hi = 0.0, 2.0 * size / np.sqrt(mesh.area)
    for _ in range(24):
        k = 0.5 * (lo + hi)
        if _skyline_pack(w * k, h * k, size, pad) is None:
            hi = k
        else:
            lo = k
    k = lo
    ox, oy, rotated = _skyline_pack(w * k, h * k, size, pad)

    du, dv = u - lo_u[vchart], v - lo_v[vchart]
    rot = rotated[vchart]
    # A rotated chart turns 90° — a rotation, so still not mirrored.
    du, dv = np.where(rot, h[vchart] - dv, du), np.where(rot, du, dv)
    top = np.where(rotated, w, h)[vchart]
    xy = np.column_stack([
        ox[vchart] + pad + du * k,
        oy[vchart] + pad + (top - dv) * k,  # flip: image rows run down
    ])
    return vmap, inverse.reshape(-1, 3), xy


def _barycentric(tri, pts):
    """Barycentric weights of 2D points `pts` in triangles `tri` (n, 3, 2).
    Outside a triangle some weights go negative (that's how gutter texels
    extrapolate the surface); degenerate triangles give NaN."""
    a = tri[:, 0]
    e1, e2, q = tri[:, 1] - a, tri[:, 2] - a, pts - a
    det = e1[:, 0] * e2[:, 1] - e2[:, 0] * e1[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        l1 = (q[:, 0] * e2[:, 1] - e2[:, 0] * q[:, 1]) / det
        l2 = (e1[:, 0] * q[:, 1] - q[:, 0] * e1[:, 1]) / det
        return np.column_stack([1.0 - l1 - l2, l1, l2])


def _rasterize(xy, faces, size):
    """Index of the triangle covering each texel center (-1 where none
    does): every triangle tests the texels in its bounding box at once."""
    tri = xy[faces]
    lo = np.ceil(tri.min(axis=1) - 0.5).astype(int)
    span = np.maximum(np.floor(tri.max(axis=1) - 0.5).astype(int) - lo + 1, 0)
    count = span[:, 0] * span[:, 1]
    t = np.repeat(np.arange(len(faces)), count)
    local = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    cols = lo[t, 0] + local % span[t, 0]
    rows = lo[t, 1] + local // span[t, 0]
    inside = (_barycentric(tri[t], np.column_stack([cols, rows]) + 0.5) >= -1e-6).all(axis=1)
    image = np.full((size, size), -1)
    image[rows[inside], cols[inside]] = t[inside]
    return image


@functools.lru_cache(maxsize=2)
def _skin_atlas(shaft_len, shaft_radius, head_radius, scrotum_radius, corona_intensity):
    """UV atlas of one body shape, plus for every texel near a chart the
    rest-space point, normal and triangle it stands for. Only the Shape
    sliders change the body's topology, so this is cached on them."""
    body, _ = _build_unified_body(
        shaft_len, shaft_radius, head_radius, scrotum_radius, corona_intensity
    )
    size, pad = SKIN_TEXTURE_SIZE, _ATLAS_PAD
    vmap, faces, xy = _uv_atlas(body, size, pad)

    covered = _rasterize(xy, faces, size)
    # Gutter texels (within `pad` of a chart) take their nearest covered
    # texel's triangle, extrapolated — so filtering at a seam sees the
    # surface continue rather than a hard edge.
    dist, (near_r, near_c) = distance_transform_edt(covered < 0, return_indices=True)
    rows, cols = np.nonzero(dist <= pad)
    tri = covered[near_r[rows, cols], near_c[rows, cols]]
    bary = _barycentric(xy[faces[tri]], np.column_stack([cols, rows]) + 0.5)
    # Any texel → the active texel whose value it shows (the empty space
    # between charts copies the nearest chart edge).
    index = np.full((size, size), -1)
    index[rows, cols] = np.arange(len(rows))
    fill = np.where(index >= 0, index, index[near_r, near_c]).ravel()

    corner = vmap[faces[tri]].astype(np.int32)  # original vertices of each texel's triangle
    normals = np.einsum("nk,nkc->nc", bary, body.vertex_normals[corner])
    # World-space step per texel along +x (right) and +y (down) on each
    # texel's triangle — what turns height-map gradients into slopes.
    p = body.vertices[corner]
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    d = xy[faces[tri]]
    d1, d2 = d[:, 1] - d[:, 0], d[:, 2] - d[:, 0]
    det = (d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1])[:, None]
    return dict(
        size=size, vmap=vmap, faces=faces, uv=(xy / size).astype(np.float32),
        n_vertices=len(body.vertices), texels=(rows * size + cols).astype(np.int32),
        fill=fill.astype(np.int32),
        corner=corner, bary=bary.astype(np.float32),
        covered=dist[rows, cols] == 0,
        points=np.einsum("nk,nkc->nc", bary, p).astype(np.float32),
        normals=(normals / np.linalg.norm(normals, axis=1, keepdims=True)).astype(np.float32),
        step_x=((e1 * d2[:, 1:] - e2 * d1[:, 1:]) / det).astype(np.float32),
        step_y=((e2 * d1[:, :1] - e1 * d2[:, :1]) / det).astype(np.float32),
        area=body.area,
        # Skin features scale with the model (1 = Default preset's size).
        scale=float(np.clip(np.sqrt(body.area / 170.0), 0.4, 4.0)),
    )


def _encode_image(array, fmt, **options):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format=fmt, **options)
    return buffer.getvalue()


@functools.lru_cache(maxsize=4)
def _skin_maps(shape, skin_pack, seed):
    """Bake a skin pack onto a body shape: the per-texel albedo tint, and
    the normal and roughness maps as JPEG bytes ready to embed."""
    atlas = _skin_atlas(*shape)
    size, fill = atlas["size"], atlas["fill"]
    tint, height, rough = _skin_layers(
        atlas["points"], atlas["covered"], shape, SKIN_PACKS[skin_pack],
        seed, atlas["scale"], atlas["area"],
    )

    # Height → tangent-space normals. Gradients come from the height
    # image in texel steps; each texel's world-space steps turn them into
    # slopes along glTF's tangent frame (+X right, +Y up in the image).
    dy, dx = np.gradient(height[fill].reshape(size, size))
    gx, gy = dx.ravel()[atlas["texels"]], dy.ravel()[atlas["texels"]]
    n, sx, sy = atlas["normals"], atlas["step_x"], atlas["step_y"]
    t = sx - np.sum(sx * n, axis=1, keepdims=True) * n
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    b = np.cross(n, t)
    m00, m01 = np.sum(sx * t, axis=1), np.sum(sx * b, axis=1)
    m10, m11 = np.sum(sy * t, axis=1), np.sum(sy * b, axis=1)
    det = m00 * m11 - m01 * m10
    slope_t = (m11 * gx - m01 * gy) / det  # height gained per unit along t
    slope_b = (m00 * gy - m10 * gx) / det  # ... and along b
    normal = np.column_stack([-slope_t, -slope_b, np.ones(len(det))])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    normal_map = np.round(normal * 127.5 + 127.5).astype(np.uint8)
    rough_map = np.round(rough * 255).astype(np.uint8)
    return (
        tint.astype(np.float32),
        # JPEG without chroma subsampling: ~1/3 the size of PNG, ~1.5° off.
        _encode_image(normal_map[fill].reshape(size, size, 3), "JPEG", quality=94, subsampling=0),
        _encode_image(rough_map[fill].reshape(size, size), "JPEG", quality=90),
    )


def _bake_albedo(atlas, colors, tint=None, srgb=True):
    """Vertex colors interpolated onto the texels, times the (linear) skin
    tint, as an sRGB JPEG. `srgb` says how the vertex colors are encoded:
    the color pickers' sRGB, or linear (material packs convert theirs)."""
    rgb = np.einsum("nk,nkc->nc", atlas["bary"], colors[atlas["corner"]].astype(np.float32))
    rgb = np.clip(rgb / 255.0, 0.0, 1.0)
    if srgb:
        rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    rgb = np.clip(rgb * (1.0 if tint is None else tint), 0.0, 1.0)
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
    size = atlas["size"]
    image = np.round(rgb * 255).astype(np.uint8)[atlas["fill"]].reshape(size, size, 3)
    return _encode_image(image, "JPEG", quality=90)


def _textured_glb(parts, images):
    """Hand-rolled binary glTF writer — trimesh would store the textures
    as PNG, several times bigger than JPEG for noisy skin maps.

    `parts` become the primitives of one mesh: dicts of vertex
    `attributes` (glTF name → array), `indices` and a glTF `material`
    (None for glTF's default one) whose texture indices point into
    `images` ((bytes, mime) pairs)."""
    blob = bytearray()
    gltf = {
        "asset": {"version": "2.0", "generator": "Ultimate DIAO Drawer"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": []}], "accessors": [], "bufferViews": [],
    }
    materials = []

    def view(data, target=None):
        blob.extend(b"\0" * (-len(blob) % 4))
        gltf["bufferViews"].append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)})
        if target is not None:
            gltf["bufferViews"][-1]["target"] = target
        blob.extend(data)
        return len(gltf["bufferViews"]) - 1

    def accessor(array, target=34962):  # ARRAY_BUFFER
        array = np.ascontiguousarray(array)
        acc = {
            "bufferView": view(array.tobytes(), target),
            "componentType": {"float32": 5126, "uint8": 5121, "uint32": 5125}[array.dtype.name],
            "count": len(array),
            "type": "SCALAR" if array.ndim == 1 else f"VEC{array.shape[1]}",
        }
        if array.dtype == np.uint8:
            acc["normalized"] = True
        if acc["type"] == "VEC3":  # required for POSITION
            acc["min"], acc["max"] = array.min(axis=0).tolist(), array.max(axis=0).tolist()
        gltf["accessors"].append(acc)
        return len(gltf["accessors"]) - 1

    for part in parts:
        primitive = {
            "attributes": {k: accessor(a) for k, a in part["attributes"].items()},
            "indices": accessor(part["indices"].astype(np.uint32).ravel(), 34963),
        }
        if part["material"] is not None:
            materials.append(part["material"])
            primitive["material"] = len(materials) - 1
        gltf["meshes"][0]["primitives"].append(primitive)
    if materials:
        gltf["materials"] = materials
    if images:
        gltf["images"] = [{"bufferView": view(data), "mimeType": mime} for data, mime in images]
        gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987,  # trilinear,
                             "wrapS": 33071, "wrapT": 33071}]       # clamped
        gltf["textures"] = [{"source": i, "sampler": 0} for i in range(len(images))]
    gltf["buffers"] = [{"byteLength": len(blob)}]
    return _pack_glb(gltf, blob)


def _skin_glb(mesh, shape, skin_pack, seed, material_pack):
    """.glb of a finished model with a skin pack: the body (the first
    vertices — hair is concatenated after it) textured through its
    shape's atlas, the hair as a second, vertex-colored primitive.

    With "Original" the skin supplies color detail, relief and roughness.
    Any other material pack keeps its own colors and finish and only
    borrows the relief — painted skin, or a cast."""
    atlas = _skin_atlas(*shape)
    tint, normal_jpg, rough_jpg = _skin_maps(shape, skin_pack, seed)
    pack = MATERIAL_PACKS.get(material_pack)
    n_body, vmap = atlas["n_vertices"], atlas["vmap"]
    colors = mesh.visual.vertex_colors
    vertices = mesh.vertices.astype(np.float32)
    normals = mesh.vertex_normals.astype(np.float32)

    if pack is None:
        albedo = _bake_albedo(atlas, colors[:n_body, :3], tint)
    else:
        albedo = _bake_albedo(atlas, colors[:n_body, :3], srgb=False)
    images = [(albedo, "image/jpeg"), (normal_jpg, "image/jpeg")]
    if pack is None:
        images.append((rough_jpg, "image/jpeg"))
        body_finish = dict(metallicFactor=0.0, roughnessFactor=1.0,
                           metallicRoughnessTexture={"index": 2})
        hair_material = None  # glTF's default material, as without a skin pack
        extra = {}
    else:
        body_finish = dict(metallicFactor=float(pack["metallic"]),
                           roughnessFactor=float(pack["roughness"]))
        extra = {"emissiveFactor": [float(c) for c in pack["emissive"]]} if "emissive" in pack else {}
        hair_material = dict(name=material_pack, doubleSided=True,
                             pbrMetallicRoughness=body_finish, **extra)

    parts = [dict(
        attributes=dict(POSITION=vertices[:n_body][vmap], NORMAL=normals[:n_body][vmap],
                        TEXCOORD_0=atlas["uv"]),
        indices=atlas["faces"],
        material=dict(name=f"Skin ({skin_pack})", doubleSided=True, normalTexture={"index": 1},
                      pbrMetallicRoughness=dict(body_finish, baseColorTexture={"index": 0}),
                      **extra),
    )]
    if len(vertices) > n_body:
        parts.append(dict(
            attributes=dict(POSITION=vertices[n_body:], NORMAL=normals[n_body:],
                            COLOR_0=colors[n_body:]),
            indices=mesh.faces[len(atlas["faces"]):] - n_body,
            material=hair_material,
        ))
    return _textured_glb(parts, images)


# ------------------------------- assembly ------------------------------


_OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "diao_drawer_output")
os.makedirs(_OUTPUT_DIR, exist_ok=True)


def create_model(
    shaft_len, shaft_radius, head_radius, scrotum_radius,
    curvature, corona_intensity,
    shaft_color, head_color, scrotum_color,
    shaft_wrinkle_intensity, head_wrinkle_intensity, scrotum_wrinkle_intensity,
    wrinkle_smoothness, wrinkle_seed,
    hair_density, hair_length,
    vein_count, vein_radius,
    name,
    material_pack="Original",
    skin_pack="None",
):
    seed = int(wrinkle_seed) if wrinkle_seed is not None else 0

    # 1. Build a single watertight body from an SDF. Smooth-min unions
    # mean shaft↔head and shaft↔scrotum already blend continuously here.
    body, head_z_center = _build_unified_body(
        shaft_len, shaft_radius, head_radius, scrotum_radius, corona_intensity
    )

    # 2. Per-region wrinkles, with intensity blended smoothly across z so
    # the boundary between regions doesn't show as a wrinkle-density seam.
    body = apply_regional_wrinkles(
        body,
        shaft_wrinkle_intensity, head_wrinkle_intensity, scrotum_wrinkle_intensity,
        shaft_len, head_z_center, wrinkle_smoothness, seed,
    )

    # 3. Curvature (z-based bend; scrotum below z=0 stays put).
    top_z = shaft_len + 2 * head_radius
    body = apply_curvature(body, curvature, top_z)

    # 4. Smooth color blending across z (shaft <-> head <-> scrotum).
    body = color_unified_by_region(
        body, shaft_color, head_color, scrotum_color,
        shaft_len, head_z_center, head_radius, scrotum_radius,
    )

    # 5. Subtle z-gradient on the shaft (tip a touch redder).
    body = apply_shaft_gradient(body, shaft_color, shaft_len)

    # 6. Localized red tint at the urethral meatus (only the very tip).
    body = apply_glans_tip_redden(body, head_radius, head_z_center)

    # 7. Vein displacement — only the vertices physically near the vein
    # paths get pushed outward / tinted, so head and scrotum aren't
    # affected even though we operate on the unified mesh.
    paths = _vein_paths_on_shaft(shaft_len, shaft_radius, vein_count, seed)
    if paths and vein_radius > 0:
        body = _displace_and_color_veins(body, paths, vein_radius, shaft_color)

    # 8. Per-vertex skin-tone variation on top.
    avg_wrinkle = (
        shaft_wrinkle_intensity + head_wrinkle_intensity + scrotum_wrinkle_intensity
    ) / 3.0
    body = apply_skin_tone_variation(
        body, shaft_color, _tone_intensity(avg_wrinkle), seed + 11
    )

    # 9. Hair only on the scrotum sphere itself (below the neck — and
    # since the scrotum hangs in -Y in model space, we filter by y).
    if hair_density > 0 and hair_length > 0:
        hair = make_hair(
            body, int(hair_density), float(hair_length),
            seed=seed + 17, max_y=-scrotum_radius * 0.7,
        )
        if hair is not None:
            body = trimesh.util.concatenate([body, hair])

    # 9.2 Material pack: recolor everything (hair included). Done in
    # model space so stripes / rings follow the shaft axis.
    body = apply_material_pack(body, material_pack, seed)

    # 9.5 Reorient for the viewer.
    #   model:  shaft +Z (head up at +z), scrotum hanging -Y
    #   world:  shaft +X (horizontal), scrotum -Y (still down)
    # Rotate +π/2 about Y so model +Z → world +X (shaft becomes
    # horizontal). The scrotum's -Y direction is on the rotation axis,
    # so it stays as -Y (hanging straight down) in the viewer.
    rotate_shaft_horizontal = trimesh.transformations.rotation_matrix(
        np.pi / 2, [0, 1, 0]
    )
    body.apply_transform(rotate_shaft_horizontal)

    # 10. Skip sRGB→linear conversion. We tried it and ACES tonemap +
    # bright IBL still squished all hues to similar peach in
    # model-viewer. By leaving vertex_colors as sRGB hex, the viewer
    # treats them as already-linear, the relative R/G/B ratios are
    # preserved through tonemapping, and distinct hues come through.

    file_path = os.path.join(_OUTPUT_DIR, f"{sanitize_name(name)}_model.glb")
    if SKIN_PACKS.get(skin_pack) is not None:
        # 11. Skin pack: bake textures through the body's UV atlas.
        shape = (shaft_len, shaft_radius, head_radius, scrotum_radius, corona_intensity)
        with open(file_path, "wb") as f:
            f.write(_skin_glb(body, shape, skin_pack, seed, material_pack))
        return file_path

    glb_bytes = trimesh.exchange.gltf.export_glb(body)
    # "Original" keeps trimesh's default (no PBR material) — the baseline
    # the color pickers were tuned against. Packs inject their own.
    pack = MATERIAL_PACKS.get(material_pack)
    if pack is not None:
        glb_bytes = _inject_pbr_material(
            glb_bytes,
            metallic=pack["metallic"],
            roughness=pack["roughness"],
            emissive=pack.get("emissive", (0.0, 0.0, 0.0)),
            name=material_pack,
        )
    with open(file_path, "wb") as f:
        f.write(glb_bytes)
    return file_path


# ------------------------------- presets -------------------------------


# create_model's positional parameters, in order. The UI inputs,
# presets, the Random button and recipe codes all go through this list.
PARAM_NAMES = [
    "shaft_len", "shaft_radius", "head_radius", "scrotum_radius",
    "curvature", "corona_intensity",
    "shaft_color", "head_color", "scrotum_color",
    "shaft_wrinkle_intensity", "head_wrinkle_intensity", "scrotum_wrinkle_intensity",
    "wrinkle_smoothness", "wrinkle_seed",
    "hair_density", "hair_length",
    "vein_count", "vein_radius",
    "name", "material_pack", "skin_pack",
]

# (min, max, step) for every slider — shared by the UI and by recipe
# validation, so a pasted code can never push a slider out of range.
SLIDERS = {
    "shaft_len": (1, 20, 1),
    "shaft_radius": (0.5, 5, 0.1),
    "head_radius": (0.5, 5, 0.1),
    "scrotum_radius": (1, 5, 0.1),
    "curvature": (-0.4, 0.4, 0.01),
    "corona_intensity": (0, 2, 0.05),
    "shaft_wrinkle_intensity": (0, 0.5, 0.01),
    "head_wrinkle_intensity": (0, 0.5, 0.01),
    "scrotum_wrinkle_intensity": (0, 0.5, 0.01),
    "wrinkle_smoothness": (0.1, 5, 0.1),
    "wrinkle_seed": (0, 1000, 1),
    "hair_density": (0, 100, 1),
    "hair_length": (0.0, 3.0, 0.05),
    "vein_count": (0, 20, 1),
    "vein_radius": (0.01, 0.20, 0.005),
}
COLOR_PARAMS = ("shaft_color", "head_color", "scrotum_color")


def _preset_args(preset_name):
    p = PRESETS[preset_name]
    return [p[k] for k in PARAM_NAMES]


PRESETS = {
    "Default": dict(
        shaft_len=7, shaft_radius=1.0, head_radius=1.5, scrotum_radius=2.0,
        curvature=0.0, corona_intensity=0.6,
        # head/shaft/scrotum keep distinct LUMINANCE bands but only
        # subtle hue difference between head and shaft (head is just
        # slightly deeper/redder skin). The vivid red is reserved for
        # the urethral meatus tip via apply_glans_tip_redden.
        shaft_color="#7A4218", head_color="#682E10", scrotum_color="#C89B6A",
        shaft_wrinkle_intensity=0.06, head_wrinkle_intensity=0.0,
        scrotum_wrinkle_intensity=0.10, wrinkle_smoothness=2.5, wrinkle_seed=42,
        hair_density=15, hair_length=0.6,
        vein_count=4, vein_radius=0.06,
        name="diao", material_pack="Original", skin_pack="None",
    ),
    "Smooth": dict(
        shaft_len=6, shaft_radius=1.0, head_radius=1.4, scrotum_radius=1.8,
        curvature=0.0, corona_intensity=0.3,
        shaft_color="#A06A48", head_color="#8C5234", scrotum_color="#DEB48E",
        shaft_wrinkle_intensity=0.0, head_wrinkle_intensity=0.0,
        scrotum_wrinkle_intensity=0.0, wrinkle_smoothness=2.5, wrinkle_seed=0,
        hair_density=0, hair_length=0.5,
        vein_count=0, vein_radius=0.05,
        name="smooth", material_pack="Original", skin_pack="None",
    ),
    "Realistic": dict(
        shaft_len=8, shaft_radius=1.1, head_radius=1.6, scrotum_radius=2.0,
        curvature=0.06, corona_intensity=1.0,
        shaft_color="#6E3818", head_color="#5A2410", scrotum_color="#B89070",
        shaft_wrinkle_intensity=0.10, head_wrinkle_intensity=0.0,
        scrotum_wrinkle_intensity=0.16, wrinkle_smoothness=2.0, wrinkle_seed=7,
        hair_density=30, hair_length=0.7,
        vein_count=8, vein_radius=0.07,
        name="realistic", material_pack="Original", skin_pack="Natural",
    ),
    "Cartoon": dict(
        shaft_len=5, shaft_radius=1.4, head_radius=2.0, scrotum_radius=2.2,
        curvature=0.0, corona_intensity=0.4,
        shaft_color="#C87858", head_color="#A85838", scrotum_color="#FFC8A0",
        shaft_wrinkle_intensity=0.0, head_wrinkle_intensity=0.0,
        scrotum_wrinkle_intensity=0.0, wrinkle_smoothness=2.5, wrinkle_seed=0,
        hair_density=0, hair_length=0.5,
        vein_count=0, vein_radius=0.04,
        name="cartoon", material_pack="Original", skin_pack="None",
    ),
}


def _theme(base, **changes):
    # Themes are flat material looks, so they pin skin_pack to "None"
    # rather than inheriting whatever skin pack the base preset uses.
    return {**PRESETS[base], "skin_pack": "None", **changes}


# One-click themes: a material pack plus shape tweaks that suit it. They
# are ordinary presets; THEME_BUTTONS only adds the emoji for the UI.
PRESETS.update({
    "Trophy": _theme("Smooth", material_pack="Gold", corona_intensity=0.8, name="trophy"),
    "Statue": _theme("Smooth", material_pack="Marble", name="statue"),
    "Robot": _theme("Smooth", material_pack="Chrome", vein_count=10, vein_radius=0.08,
                    name="robot"),
    "Galaxy": _theme("Default", material_pack="Galaxy", hair_density=0, curvature=0.08,
                     name="galaxy"),
    "Candy": _theme("Cartoon", material_pack="Candy cane", curvature=0.4, name="candy"),
    "Rainbow": _theme("Cartoon", material_pack="Rainbow", name="rainbow"),
})
THEME_BUTTONS = {
    "Trophy": "🏆", "Statue": "🗿", "Robot": "🤖",
    "Galaxy": "🌌", "Candy": "🍭", "Rainbow": "🌈",
}


# ------------------------- random & recipe codes -------------------------


def _snap(key, x):
    lo, hi, step = SLIDERS[key]
    return round(float(np.clip(round(x / step) * step, lo, hi)), 4)


def random_args(name):
    """A random but sensible parameter set (narrower than the slider
    ranges), a random material and skin pack, and one of the presets'
    palettes. Keeps the current name."""
    rng = np.random.default_rng()
    palettes = [k for k, p in PRESETS.items() if p["material_pack"] == "Original"]
    palette = PRESETS[str(rng.choice(palettes))]
    ranges = dict(
        shaft_len=(4, 14), shaft_radius=(0.7, 1.6), head_radius=(1.0, 2.4),
        scrotum_radius=(1.3, 2.8), curvature=(-0.15, 0.15), corona_intensity=(0.0, 1.2),
        shaft_wrinkle_intensity=(0.0, 0.15), head_wrinkle_intensity=(0.0, 0.0),
        scrotum_wrinkle_intensity=(0.0, 0.2), wrinkle_smoothness=(1.5, 3.5),
        wrinkle_seed=(0, 1000), hair_density=(0, 40), hair_length=(0.3, 1.0),
        vein_count=(0, 8), vein_radius=(0.03, 0.09),
    )
    values = {k: _snap(k, rng.uniform(lo, hi)) for k, (lo, hi) in ranges.items()}
    values.update({k: palette[k] for k in COLOR_PARAMS})
    values["name"] = name
    values["material_pack"] = str(rng.choice(list(MATERIAL_PACKS)))
    values["skin_pack"] = str(rng.choice(list(SKIN_PACKS)))
    return [values[k] for k in PARAM_NAMES]


_RECIPE_PREFIX = "UDD1-"


def encode_recipe(*values):
    """All parameters → a copy-pasteable code (raw-deflated JSON in
    base64url, ~300 chars — half the size of plain base64 JSON)."""
    raw = json.dumps(dict(zip(PARAM_NAMES, values)), separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    deflate = zlib.compressobj(9, zlib.DEFLATED, -15)
    packed = deflate.compress(raw) + deflate.flush()
    return _RECIPE_PREFIX + base64.urlsafe_b64encode(packed).decode("ascii").rstrip("=")


def decode_recipe(code):
    """Recipe code → parameter list. Anything missing or invalid falls
    back to the Default preset; numbers are clamped to slider ranges."""
    code = (code or "").strip()
    if code.startswith(_RECIPE_PREFIX):
        code = code[len(_RECIPE_PREFIX):]
    try:
        # Size caps: real codes are ~300 chars / ~450 bytes inflated; the
        # caps stop a crafted "deflate bomb" from eating the server's RAM.
        if len(code) > 4096:
            raise ValueError("code too long")
        packed = base64.urlsafe_b64decode(code + "=" * (-len(code) % 4))
        inflate = zlib.decompressobj(-15)
        raw = inflate.decompress(packed, 65536)
        if inflate.unconsumed_tail:
            raise ValueError("inflates past the size cap")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
    except (ValueError, zlib.error):  # bad base64 / deflate / UTF-8 / JSON
        raise gr.Error("That doesn't look like a UDD recipe code.") from None

    default = PRESETS["Default"]
    values = {}
    for key in PARAM_NAMES:
        value = data.get(key, default[key])
        if key in SLIDERS:
            try:
                value = _snap(key, float(value))
            except (TypeError, ValueError):
                value = default[key]
        elif key in COLOR_PARAMS:
            if not re.fullmatch(r"#[0-9A-Fa-f]{6}", str(value)):
                value = default[key]
        elif key == "material_pack":
            if not isinstance(value, str) or value not in MATERIAL_PACKS:
                value = "Original"
        elif key == "skin_pack":
            if not isinstance(value, str) or value not in SKIN_PACKS:
                value = "None"
        elif key == "name":
            value = str(value)[:64]
        values[key] = value
    return [values[k] for k in PARAM_NAMES]


def generate(*values):
    """UI entry point: the model plus the recipe code that rebuilds it."""
    return create_model(*values), encode_recipe(*values)


# --------------------------------- UI ----------------------------------


def _slider(key, label):
    lo, hi, step = SLIDERS[key]
    return gr.Slider(lo, hi, value=PRESETS["Default"][key], step=step, label=label)


def build_ui():
    default = PRESETS["Default"]
    with gr.Blocks(title="Ultimate DIAO Drawer (UDD)") as demo:
        gr.Markdown(
            "# Ultimate DIAO Drawer (UDD)\n"
            "Tweak parameters → click **Generate** (or use a preset, a "
            "material or skin pack, or 🎲 Random).  "
            "Sliders auto-update on release; color pickers and the name "
            "field require an explicit Generate to keep the UI snappy."
        )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tab("Shape"):
                    shaft_len = _slider("shaft_len", "Shaft length")
                    shaft_radius = _slider("shaft_radius", "Shaft radius")
                    head_radius = _slider("head_radius", "Head radius")
                    scrotum_radius = _slider("scrotum_radius", "Scrotum radius")
                    curvature = _slider("curvature", "Curvature (bend along +y)")
                    corona_intensity = _slider("corona_intensity", "Corona ridge")

                with gr.Tab("Color"):
                    shaft_color = gr.ColorPicker(label="Shaft color", value=default["shaft_color"])
                    head_color = gr.ColorPicker(label="Head color", value=default["head_color"])
                    scrotum_color = gr.ColorPicker(label="Scrotum color", value=default["scrotum_color"])
                    gr.Markdown("_Click **Generate** below to apply colors._")

                with gr.Tab("Material"):
                    material_pack = gr.Dropdown(
                        list(MATERIAL_PACKS),
                        value=default["material_pack"],
                        label="Material pack",
                        info="Original uses the Color tab; every other pack replaces those colors.",
                    )

                with gr.Tab("Texture"):
                    skin_pack = gr.Dropdown(
                        list(SKIN_PACKS),
                        value=default["skin_pack"],
                        label="Skin pack",
                        info="Bakes mottling, veins, creases, follicles and fine grain into "
                             "textures. None keeps the flat vertex colors. With a material "
                             "pack other than Original, only the skin's relief carries over.",
                    )

                with gr.Tab("Surface"):
                    shaft_wrinkle_intensity = _slider("shaft_wrinkle_intensity", "Shaft wrinkles")
                    head_wrinkle_intensity = _slider("head_wrinkle_intensity", "Head wrinkles")
                    scrotum_wrinkle_intensity = _slider("scrotum_wrinkle_intensity", "Scrotum wrinkles")
                    wrinkle_smoothness = _slider("wrinkle_smoothness", "Wrinkle smoothness (gaussian σ)")
                    wrinkle_seed = _slider("wrinkle_seed", "Wrinkle seed")

                with gr.Tab("Hair"):
                    hair_density = _slider("hair_density", "Hair density")
                    hair_length = _slider("hair_length", "Hair length")

                with gr.Tab("Veins"):
                    vein_count = _slider("vein_count", "Vein count")
                    vein_radius = _slider("vein_radius", "Vein thickness")
                    gr.Markdown("_Veins are now raised ridges on the shaft surface, not floating tubes._")

                name = gr.Textbox(label="Name", value=default["name"])

                with gr.Row():
                    btn_generate = gr.Button("Generate", variant="primary")

                gr.Markdown("**Presets:**")
                with gr.Row():
                    btn_default = gr.Button("Default")
                    btn_smooth = gr.Button("Smooth")
                    btn_realistic = gr.Button("Realistic")
                    btn_cartoon = gr.Button("Cartoon")
                    btn_random = gr.Button("🎲 Random")

                gr.Markdown("**Themes:**")
                theme_buttons = {}
                theme_names = list(THEME_BUTTONS)
                for row_start in range(0, len(theme_names), 3):
                    with gr.Row():
                        for key in theme_names[row_start:row_start + 3]:
                            theme_buttons[key] = gr.Button(
                                f"{THEME_BUTTONS[key]} {key}", min_width=0
                            )

                with gr.Row():
                    recipe = gr.Textbox(
                        label="Recipe code",
                        value=encode_recipe(*_preset_args("Default")),
                        info="Send this to a friend, or paste one and click Load to rebuild that model.",
                        scale=4,
                    )
                    btn_load = gr.Button("Load", scale=1)

            with gr.Column(scale=2):
                # Pre-generate the Default mesh so the viewer has something
                # on initial page load WITHOUT a server-side `demo.load`
                # event firing (which would race with any early UI clicks).
                initial_glb = create_model(*_preset_args("Default"))
                output = gr.Model3D(
                    value=initial_glb,
                    label="Output (.glb)",
                    # Light-gray background so the dark hair strands are
                    # visible (against black they disappeared).
                    clear_color=(0.85, 0.85, 0.85, 1.0),
                )

        all_inputs = [
            shaft_len, shaft_radius, head_radius, scrotum_radius,
            curvature, corona_intensity,
            shaft_color, head_color, scrotum_color,
            shaft_wrinkle_intensity, head_wrinkle_intensity, scrotum_wrinkle_intensity,
            wrinkle_smoothness, wrinkle_seed,
            hair_density, hair_length,
            vein_count, vein_radius,
            name, material_pack, skin_pack,
        ]
        outputs = [output, recipe]

        # Slider auto-update on release (drag freely, only re-render when you let go).
        slider_inputs = [
            shaft_len, shaft_radius, head_radius, scrotum_radius,
            curvature, corona_intensity,
            shaft_wrinkle_intensity, head_wrinkle_intensity, scrotum_wrinkle_intensity,
            wrinkle_smoothness, wrinkle_seed,
            hair_density, hair_length,
            vein_count, vein_radius,
        ]
        for s in slider_inputs:
            s.release(generate, inputs=all_inputs, outputs=outputs)

        # Manual Generate button — also wired by preset buttons via .then().
        btn_generate.click(generate, inputs=all_inputs, outputs=outputs)

        # Picking a pack re-renders straight away. `.input` (not `.change`)
        # so presets / Random / Load setting the dropdown don't render twice.
        material_pack.input(generate, inputs=all_inputs, outputs=outputs)
        skin_pack.input(generate, inputs=all_inputs, outputs=outputs)

        for btn, key in [
            (btn_default, "Default"),
            (btn_smooth, "Smooth"),
            (btn_realistic, "Realistic"),
            (btn_cartoon, "Cartoon"),
            *[(b, k) for k, b in theme_buttons.items()],
        ]:
            btn.click(
                lambda k=key: _preset_args(k), outputs=all_inputs
            ).then(generate, inputs=all_inputs, outputs=outputs)

        btn_random.click(random_args, inputs=name, outputs=all_inputs).then(
            generate, inputs=all_inputs, outputs=outputs
        )
        # .success: a bad code shows an error and leaves the model alone.
        btn_load.click(decode_recipe, inputs=recipe, outputs=all_inputs).success(
            generate, inputs=all_inputs, outputs=outputs
        )

    return demo


if __name__ == "__main__":
    build_ui().launch()
