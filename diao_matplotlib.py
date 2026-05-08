"""DIAO Drawer — 2D version (legacy).

Renders a static PNG using matplotlib's 3D toolkit. Kept around as a
lightweight fallback to the trimesh + Gradio Model3D version in
`diao_3d.py`.
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams


sns.set(style="whitegrid")

# Try a few CJK fonts that exist on common systems. matplotlib will use the
# first one it actually finds; if none of these are installed the title
# falls back to the default font and CJK glyphs render as boxes.
rcParams["font.sans-serif"] = [
    "PingFang SC",        # macOS
    "Heiti SC", "STHeiti", # macOS
    "Microsoft YaHei",     # Windows
    "SimHei",              # Windows (older)
    "WenQuanYi Zen Hei",   # Linux
    "Noto Sans CJK SC",    # Linux (modern)
    "Arial Unicode MS",    # cross-platform fallback
]
rcParams["axes.unicode_minus"] = False

darker_lightcoral = (180 / 255, 68 / 255, 68 / 255)


def create_model(shaft_len, shaft_radius, head_radius, scrotum_radius, name):
    def create_cylinder(ax, radius, height, x_offset=0, y_offset=0, z_offset=0,
                        color="tan", alpha=1.0):
        z = np.linspace(z_offset, z_offset + height, 100)
        theta = np.linspace(0, 2 * np.pi, 100)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = radius * np.cos(theta_grid) + x_offset
        y_grid = radius * np.sin(theta_grid) + y_offset
        ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha, edgecolor="none")

    def create_sphere(ax, radius, center, color="pink", alpha=1.0):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
        ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor="none")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(name + "的屌", fontsize=20, pad=20)
    ax.view_init(elev=30, azim=30)

    create_cylinder(ax, radius=shaft_radius, height=shaft_len, x_offset=shaft_radius / 2, color="saddlebrown")
    create_cylinder(ax, radius=shaft_radius, height=shaft_len, x_offset=-shaft_radius / 2, color="saddlebrown")
    create_cylinder(ax, radius=shaft_radius * 0.8, height=shaft_len, y_offset=shaft_radius * 0.8, color="rosybrown")

    create_sphere(ax, radius=head_radius / 1.2, center=(0, 0, shaft_len + head_radius / 1.4), color="lightcoral")
    create_sphere(ax, radius=head_radius, center=(0, 0, shaft_len + head_radius / 3),
                  color=darker_lightcoral, alpha=0.9)
    create_sphere(ax, radius=0.4, center=(0, 0, shaft_len + head_radius / 1.4 + 1.1), color="black")

    create_sphere(ax, radius=scrotum_radius, center=(-scrotum_radius / 1.2, 0.3, -scrotum_radius / 1.2), color="navajowhite")
    create_sphere(ax, radius=scrotum_radius, center=(scrotum_radius / 1.2, 0.3, -scrotum_radius / 1.2), color="navajowhite")

    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([-5, 10])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 2])
    ax.set_axis_off()
    ax.grid(False)

    file_path = "output.png"
    plt.savefig(file_path)
    plt.close()
    return file_path


def build_ui():
    return gr.Interface(
        fn=create_model,
        inputs=[
            gr.Slider(minimum=1, maximum=20, step=1, label="Shaft Length", value=7),
            gr.Slider(minimum=0.5, maximum=5, step=0.1, label="Shaft Radius", value=1),
            gr.Slider(minimum=0.5, maximum=5, step=0.1, label="Head Radius", value=1.5),
            gr.Slider(minimum=1, maximum=5, step=0.5, label="Scrotum Radius", value=2),
            gr.Textbox(label="Name", value="xyf"),
        ],
        outputs=gr.Image(type="filepath"),
        title="DIAO Drawer (2D)",
        live=True,
    )


if __name__ == "__main__":
    build_ui().launch()
