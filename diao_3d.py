"""Ultimate DIAO Drawer (UDD) — 3D version.

Generates a parametric phallus mesh from a Gradio UI and exports it as
.glb for the Model3D viewer. Sliders cover shape, color, surface
wrinkles, optional hair, optional surface veins (raised ridges, not
floating tubes) and a corona ridge. A few one-click presets are wired
in.
"""

import os
import re
import tempfile

import gradio as gr
import trimesh
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


# ----------------------------- utilities -----------------------------


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def hex_to_rgba(hex_color, alpha=255):
    return np.array(hex_to_rgb(hex_color) + (alpha,), dtype=np.uint8)


def color_mesh_uniform(mesh, hex_color):
    rgba = hex_to_rgba(hex_color)
    mesh.visual.face_colors = np.tile(rgba, (len(mesh.faces), 1))
    return mesh


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


# layout of the three cylinders that make up the shaft. Anchors used by
# vein placement so paths sit on a real cylinder surface, not an
# idealised circle.
def _shaft_cylinders(shaft_radius):
    return [
        # (cx, cy, radius_scale)
        (+shaft_radius / 2, 0.0, 1.0),
        (-shaft_radius / 2, 0.0, 1.0),
        (0.0, shaft_radius * 0.8, 0.8),
    ]


def _make_subdivided_cylinder(radius, height, sections=18, longitudinal=16):
    """Cylinder with vertices in BOTH radial and longitudinal directions.
    `trimesh.creation.cylinder` only puts vertices at the caps, which means
    surface displacement (e.g. vein ridges) has nothing to push in the
    middle band. This generator gives a dense quad-strip side wall plus
    fan-triangulated caps."""
    angle = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    z = np.linspace(0, height, longitudinal)
    AA, ZZ = np.meshgrid(angle, z)
    X = radius * np.cos(AA)
    Y = radius * np.sin(AA)
    side = np.column_stack([X.ravel(), Y.ravel(), ZZ.ravel()])

    faces = []
    for i in range(longitudinal - 1):
        for j in range(sections):
            jn = (j + 1) % sections
            a = i * sections + j
            b = i * sections + jn
            c = (i + 1) * sections + j
            d = (i + 1) * sections + jn
            faces.append([a, b, c])
            faces.append([c, b, d])

    bottom_idx = len(side)
    top_idx = bottom_idx + 1
    vertices = np.vstack([side, [0, 0, 0], [0, 0, height]])
    for j in range(sections):
        jn = (j + 1) % sections
        faces.append([bottom_idx, jn, j])  # outward normal -z
    base = (longitudinal - 1) * sections
    for j in range(sections):
        jn = (j + 1) % sections
        faces.append([top_idx, base + j, base + jn])  # outward normal +z

    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces))


def _build_shaft_geometry(shaft_len, shaft_radius):
    parts = []
    # Cap longitudinal at 20 — more than that just slows the browser
    # without visibly improving the vein-displacement smoothness.
    longitudinal = int(np.clip(shaft_len * 2.5, 8, 20))
    for cx, cy, r_scale in _shaft_cylinders(shaft_radius):
        c = _make_subdivided_cylinder(
            radius=shaft_radius * r_scale, height=shaft_len,
            sections=18, longitudinal=longitudinal,
        )
        c.apply_translation([cx, cy, 0])
        parts.append(c)
    return trimesh.util.concatenate(parts)


def _vein_paths_on_shaft(shaft_len, shaft_radius, vein_count, seed):
    """Generate vein polylines that genuinely lie on the shaft cylinder
    surface (each vein is anchored to one of the three cylinders)."""
    if vein_count <= 0:
        return []
    rng = np.random.default_rng(seed + 5)
    cylinders = _shaft_cylinders(shaft_radius)
    paths = []
    for _ in range(int(vein_count)):
        host = cylinders[rng.integers(0, len(cylinders))]
        cx, cy, r_scale = host
        host_r = shaft_radius * r_scale
        base_angle = rng.uniform(0, 2 * np.pi)
        wind_amp = rng.uniform(0.4, 1.0)
        wind_freq = rng.uniform(1.5, 3.0) * np.pi / max(shaft_len, 1e-3)
        phase = rng.uniform(0, 2 * np.pi)
        z_lo = rng.uniform(0.05, 0.2) * shaft_len
        z_hi = rng.uniform(0.85, 0.97) * shaft_len
        z = np.linspace(z_lo, z_hi, 50)
        angle = base_angle + wind_amp * np.sin(wind_freq * z + phase)
        x = cx + host_r * np.cos(angle)
        y = cy + host_r * np.sin(angle)
        paths.append(np.column_stack((x, y, z)))
    return paths


def _displace_and_color_veins(shaft_mesh, vein_paths, vein_radius, base_color):
    """Push shaft vertices outward along the local normal where they sit
    near a vein path point, and tint them darker. This embeds the vein
    as a raised ridge on the shaft surface (real veins, not floating
    tubes)."""
    if not vein_paths or vein_radius <= 0:
        return color_mesh_uniform(shaft_mesh, base_color)

    all_pts = np.vstack(vein_paths)
    tree = cKDTree(all_pts)
    dists, _ = tree.query(shaft_mesh.vertices)

    sigma_push = max(vein_radius * 1.4, 1e-4)
    sigma_color = max(vein_radius * 2.2, 1e-4)
    push_amp = vein_radius * 1.6

    push = push_amp * np.exp(-(dists / sigma_push) ** 2)
    normals = shaft_mesh.vertex_normals  # cached; trimesh invalidates on vertex change
    shaft_mesh.vertices = shaft_mesh.vertices + normals * push[:, None]

    # Vein vertex color: blend base toward a cool dark tone (real
    # subdermal veins look bluish/purple due to subsurface scattering),
    # not a pure darken of the base which reads as a scar.
    rgb_base = np.array(hex_to_rgb(base_color), dtype=float)
    cool_dark = np.array([55.0, 40.0, 80.0])  # desaturated dark purple
    rgb_vein = 0.55 * rgb_base * 0.45 + 0.45 * cool_dark  # half-mix toward cool
    weight = np.exp(-(dists / sigma_color) ** 2)[:, None]
    rgb = (1 - weight) * rgb_base + weight * rgb_vein
    rgba = np.hstack([rgb, 255 * np.ones((len(rgb), 1))]).astype(np.uint8)
    shaft_mesh.visual.vertex_colors = rgba
    return shaft_mesh


def make_shaft(shaft_len, shaft_radius, color, wrinkle_intensity,
               wrinkle_smoothness, vein_count, vein_radius, seed):
    shaft = _build_shaft_geometry(shaft_len, shaft_radius)
    shaft = apply_wrinkles(shaft, wrinkle_intensity, wrinkle_smoothness, seed=seed)
    paths = _vein_paths_on_shaft(shaft_len, shaft_radius, vein_count, seed)
    shaft = _displace_and_color_veins(shaft, paths, vein_radius, color)
    return shaft


def make_head(head_radius, shaft_len, color, wrinkle_intensity,
              wrinkle_smoothness, seed):
    head = trimesh.creation.icosphere(subdivisions=3, radius=head_radius)
    head.apply_translation([0, 0, shaft_len + head_radius / 1.4])
    head = apply_wrinkles(head, wrinkle_intensity, wrinkle_smoothness, seed=seed + 1)
    return color_mesh_uniform(head, color)


def make_corona(head_radius, shaft_len, color, ridge_intensity):
    """Torus around the base of the glans."""
    if ridge_intensity <= 0:
        return None
    minor = head_radius * 0.12 * ridge_intensity
    torus = trimesh.creation.torus(
        major_radius=head_radius * 1.02, minor_radius=minor
    )
    torus.apply_translation([0, 0, shaft_len + head_radius * 0.25])
    return color_mesh_uniform(torus, color)


def make_scrotum(scrotum_radius, color, wrinkle_intensity, wrinkle_smoothness,
                 hair_density, hair_length, seed):
    spheres = []
    for x in [-scrotum_radius / 1.2, scrotum_radius / 1.2]:
        s = trimesh.creation.icosphere(subdivisions=3, radius=scrotum_radius)
        s.apply_translation([x, 0.3, -scrotum_radius / 1.2])
        s = apply_wrinkles(s, wrinkle_intensity, wrinkle_smoothness, seed=seed + 2)
        spheres.append(s)
    scrotum = trimesh.util.concatenate(spheres)
    scrotum = color_mesh_uniform(scrotum, color)

    if hair_density > 0 and hair_length > 0:
        hair = make_hair(scrotum, int(hair_density), float(hair_length), seed=seed + 3)
        if hair is not None:
            scrotum = trimesh.util.concatenate([scrotum, hair])
    return scrotum


def make_hair(surface_mesh, num_hairs, hair_length, seed):
    """Curly helical strands rooted at sampled vertices, oriented along
    the local normal at that same vertex (the original code sampled
    `point` and `normal` from independent random indices)."""
    n = len(surface_mesh.vertices)
    if n == 0 or num_hairs <= 0:
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=num_hairs)
    hairs = []
    for idx in indices:
        point = surface_mesh.vertices[idx]
        normal = surface_mesh.vertex_normals[idx]
        # Skip the top half of the scrotum so we don't get bald-spot tufts on the upper surface.
        if normal[2] > 0.5:
            continue
        direction = normal + rng.uniform(-0.2, 0.2, size=3)
        direction = direction / max(np.linalg.norm(direction), 1e-9)
        path = create_helix_path(
            radius=0.06, turns=2.5, length=hair_length, center=point, direction=direction
        )
        hairs.append(create_tube(path, tube_radius=0.018, num_segments=4))
    if not hairs:
        return None
    hair_mesh = trimesh.util.concatenate(hairs)
    return color_mesh_uniform(hair_mesh, "#1a1a1a")


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
):
    seed = int(wrinkle_seed) if wrinkle_seed is not None else 0

    shaft = make_shaft(
        shaft_len, shaft_radius, shaft_color,
        shaft_wrinkle_intensity, wrinkle_smoothness,
        vein_count, vein_radius, seed,
    )
    head = make_head(
        head_radius, shaft_len, head_color,
        head_wrinkle_intensity, wrinkle_smoothness, seed,
    )
    scrotum = make_scrotum(
        scrotum_radius, scrotum_color,
        scrotum_wrinkle_intensity, wrinkle_smoothness,
        hair_density, hair_length, seed,
    )

    upper_parts = [shaft, head]
    corona = make_corona(head_radius, shaft_len, head_color, corona_intensity)
    if corona is not None:
        upper_parts.append(corona)

    upper = trimesh.util.concatenate(upper_parts)
    top_z = shaft_len + 2 * head_radius
    upper = apply_curvature(upper, curvature, top_z)

    final_model = trimesh.util.concatenate([upper, scrotum])
    file_path = os.path.join(_OUTPUT_DIR, f"{sanitize_name(name)}_model.glb")
    final_model.export(file_path)
    return file_path


# ------------------------------- presets -------------------------------


def _preset_args(preset_name):
    p = PRESETS[preset_name]
    return [
        p["shaft_len"], p["shaft_radius"], p["head_radius"], p["scrotum_radius"],
        p["curvature"], p["corona_intensity"],
        p["shaft_color"], p["head_color"], p["scrotum_color"],
        p["shaft_wrinkle_intensity"], p["head_wrinkle_intensity"], p["scrotum_wrinkle_intensity"],
        p["wrinkle_smoothness"], p["wrinkle_seed"],
        p["hair_density"], p["hair_length"],
        p["vein_count"], p["vein_radius"],
        p["name"],
    ]


PRESETS = {
    "Default": dict(
        shaft_len=7, shaft_radius=1.0, head_radius=1.5, scrotum_radius=2.0,
        curvature=0.0, corona_intensity=0.6,
        shaft_color="#8B4513", head_color="#B44444", scrotum_color="#FFDAB9",
        shaft_wrinkle_intensity=0.06, head_wrinkle_intensity=0.04,
        scrotum_wrinkle_intensity=0.10, wrinkle_smoothness=2.5, wrinkle_seed=42,
        hair_density=15, hair_length=0.6,
        vein_count=4, vein_radius=0.06,
        name="diao",
    ),
    "Smooth": dict(
        shaft_len=6, shaft_radius=1.0, head_radius=1.4, scrotum_radius=1.8,
        curvature=0.0, corona_intensity=0.3,
        shaft_color="#D6A07A", head_color="#C97070", scrotum_color="#E8C4A0",
        shaft_wrinkle_intensity=0.0, head_wrinkle_intensity=0.0,
        scrotum_wrinkle_intensity=0.0, wrinkle_smoothness=2.5, wrinkle_seed=0,
        hair_density=0, hair_length=0.5,
        vein_count=0, vein_radius=0.05,
        name="smooth",
    ),
    "Realistic": dict(
        shaft_len=8, shaft_radius=1.1, head_radius=1.6, scrotum_radius=2.0,
        curvature=0.06, corona_intensity=1.0,
        shaft_color="#A06A4D", head_color="#9C4A4A", scrotum_color="#C9A185",
        shaft_wrinkle_intensity=0.10, head_wrinkle_intensity=0.05,
        scrotum_wrinkle_intensity=0.16, wrinkle_smoothness=2.0, wrinkle_seed=7,
        hair_density=30, hair_length=0.7,
        vein_count=8, vein_radius=0.07,
        name="realistic",
    ),
    "Cartoon": dict(
        shaft_len=5, shaft_radius=1.4, head_radius=2.0, scrotum_radius=2.2,
        curvature=0.0, corona_intensity=0.4,
        shaft_color="#FFB07A", head_color="#FF6F61", scrotum_color="#FFD8B5",
        shaft_wrinkle_intensity=0.0, head_wrinkle_intensity=0.0,
        scrotum_wrinkle_intensity=0.0, wrinkle_smoothness=2.5, wrinkle_seed=0,
        hair_density=0, hair_length=0.5,
        vein_count=0, vein_radius=0.04,
        name="cartoon",
    ),
}


# --------------------------------- UI ----------------------------------


def build_ui():
    with gr.Blocks(title="Ultimate DIAO Drawer (UDD)") as demo:
        gr.Markdown(
            "# Ultimate DIAO Drawer (UDD)\n"
            "Tweak parameters → click **Generate** (or use a preset).  "
            "Sliders auto-update on release; color pickers and the name "
            "field require an explicit Generate to keep the UI snappy."
        )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tab("Shape"):
                    shaft_len = gr.Slider(1, 20, value=7, step=1, label="Shaft length")
                    shaft_radius = gr.Slider(0.5, 5, value=1.0, step=0.1, label="Shaft radius")
                    head_radius = gr.Slider(0.5, 5, value=1.5, step=0.1, label="Head radius")
                    scrotum_radius = gr.Slider(1, 5, value=2.0, step=0.1, label="Scrotum radius")
                    curvature = gr.Slider(-0.4, 0.4, value=0.0, step=0.01, label="Curvature (bend along +y)")
                    corona_intensity = gr.Slider(0, 2, value=0.6, step=0.05, label="Corona ridge")

                with gr.Tab("Color"):
                    shaft_color = gr.ColorPicker(label="Shaft color", value="#8B4513")
                    head_color = gr.ColorPicker(label="Head color", value="#B44444")
                    scrotum_color = gr.ColorPicker(label="Scrotum color", value="#FFDAB9")
                    gr.Markdown("_Click **Generate** below to apply colors._")

                with gr.Tab("Surface"):
                    shaft_wrinkle_intensity = gr.Slider(0, 0.5, value=0.06, step=0.01, label="Shaft wrinkles")
                    head_wrinkle_intensity = gr.Slider(0, 0.5, value=0.04, step=0.01, label="Head wrinkles")
                    scrotum_wrinkle_intensity = gr.Slider(0, 0.5, value=0.10, step=0.01, label="Scrotum wrinkles")
                    wrinkle_smoothness = gr.Slider(0.1, 5, value=2.5, step=0.1, label="Wrinkle smoothness (gaussian σ)")
                    wrinkle_seed = gr.Slider(0, 1000, value=42, step=1, label="Wrinkle seed")

                with gr.Tab("Hair"):
                    hair_density = gr.Slider(0, 100, value=15, step=1, label="Hair density")
                    hair_length = gr.Slider(0.0, 3.0, value=0.6, step=0.05, label="Hair length")

                with gr.Tab("Veins"):
                    vein_count = gr.Slider(0, 20, value=4, step=1, label="Vein count")
                    vein_radius = gr.Slider(0.01, 0.20, value=0.06, step=0.005, label="Vein thickness")
                    gr.Markdown("_Veins are now raised ridges on the shaft surface, not floating tubes._")

                name = gr.Textbox(label="Name", value="diao")

                with gr.Row():
                    btn_generate = gr.Button("Generate", variant="primary")

                gr.Markdown("**Presets:**")
                with gr.Row():
                    btn_default = gr.Button("Default")
                    btn_smooth = gr.Button("Smooth")
                    btn_realistic = gr.Button("Realistic")
                    btn_cartoon = gr.Button("Cartoon")

            with gr.Column(scale=2):
                # Pre-generate the Default mesh so the viewer has something
                # on initial page load WITHOUT a server-side `demo.load`
                # event firing (which would race with any early UI clicks).
                initial_glb = create_model(*_preset_args("Default"))
                output = gr.Model3D(value=initial_glb, label="Output (.glb)")

        all_inputs = [
            shaft_len, shaft_radius, head_radius, scrotum_radius,
            curvature, corona_intensity,
            shaft_color, head_color, scrotum_color,
            shaft_wrinkle_intensity, head_wrinkle_intensity, scrotum_wrinkle_intensity,
            wrinkle_smoothness, wrinkle_seed,
            hair_density, hair_length,
            vein_count, vein_radius,
            name,
        ]

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
            s.release(create_model, inputs=all_inputs, outputs=output)

        # Manual Generate button — also wired by preset buttons via .then().
        btn_generate.click(create_model, inputs=all_inputs, outputs=output)

        for btn, key in [
            (btn_default, "Default"),
            (btn_smooth, "Smooth"),
            (btn_realistic, "Realistic"),
            (btn_cartoon, "Cartoon"),
        ]:
            btn.click(
                lambda k=key: _preset_args(k), outputs=all_inputs
            ).then(create_model, inputs=all_inputs, outputs=output)

    return demo


if __name__ == "__main__":
    build_ui().launch()
