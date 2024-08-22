import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns
from matplotlib import rcParams

# Apply seaborn style for the plot
sns.set(style="whitegrid")

# Set font to SimHei (or another Chinese font) to support Chinese characters
rcParams['font.sans-serif'] = ['SimHei']  # For Chinese characters
rcParams['axes.unicode_minus'] = False  # To prevent the minus sign from being displayed as a square

darker_lightcoral = (180/255, 68/255, 68/255)

def create_model(shaft_len, shaft_radius, head_radius, scrotum_radius, name):
    def create_cylinder(ax, radius, height, x_offset=0, y_offset=0, z_offset=0, color='tan', alpha=1.0):
        z = np.linspace(z_offset, z_offset + height, 100)
        theta = np.linspace(0, 2 * np.pi, 100)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = radius * np.cos(theta_grid) + x_offset
        y_grid = radius * np.sin(theta_grid) + y_offset
        ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha, edgecolor='none')

    def create_sphere(ax, radius, center, color='pink', alpha=1.0):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
        ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor='none')

    # Create the figure and axis
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Add a title with Chinese characters
    ax.set_title(name + '的屌', fontsize=20, pad=20)

    # Adjust the viewing angle to zoom in and position above xy-plane
    ax.view_init(elev=30, azim=30)

    # Draw the three cylinders for the shaft
    create_cylinder(ax, radius=shaft_radius, height=shaft_len, x_offset=shaft_radius/2, y_offset=0, z_offset=0, color='saddlebrown')
    create_cylinder(ax, radius=shaft_radius, height=shaft_len, x_offset=-shaft_radius/2, y_offset=0, z_offset=0, color='saddlebrown')
    create_cylinder(ax, radius=shaft_radius*0.8, height=shaft_len, x_offset=0, y_offset=shaft_radius*0.8, z_offset=0, color='rosybrown')

    # Draw the glans as a sphere with corona
    create_sphere(ax, radius=head_radius/1.2, center=(0, 0, shaft_len+head_radius/1.4), color='lightcoral')
    create_sphere(ax, radius=head_radius, center=(0, 0, shaft_len+head_radius/3), color=darker_lightcoral, alpha=0.9)

    create_sphere(ax, radius=0.4, center=(0, 0, shaft_len+head_radius/1.4+1.1), color='black')

    # Draw the testicles (scrotum) as two spheres
    create_sphere(ax, radius=scrotum_radius, center=(-scrotum_radius/1.2, 0.3, -scrotum_radius/1.2), color='navajowhite')
    create_sphere(ax, radius=scrotum_radius, center=(scrotum_radius/1.2, 0.3, -scrotum_radius/1.2), color='navajowhite')

    # Set the limits and labels
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([-5, 10])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Make sure the entire model is above the xy-plane
    ax.set_box_aspect([1, 1, 2])  # Aspect ratio for better visualization

    ax.set_axis_off()

    # Remove the grid
    ax.grid(False)

    # Save the plot to a file and return the file path
    file_path = "output.png"
    plt.savefig(file_path)
    plt.close()
    
    return file_path

# Gradio interface
iface = gr.Interface(
    fn=create_model,
    inputs=[
        gr.Slider(minimum=1, maximum=20, step=1, label="Shaft Length", value=7),
        gr.Slider(minimum=0.5, maximum=5, step=0.1, label="Shaft Radius", value=1),
        gr.Slider(minimum=0.5, maximum=5, step=0.1, label="Head Radius", value=1.5),
        gr.Slider(minimum=1, maximum=5, step=0.5, label="Scrotum Radius", value=2),
        gr.Textbox(label="Name", value="xyf")
    ],
    outputs=gr.Image(type="filepath"),
    title="Dick Creator"
)

# Launch the Gradio interface
iface.launch()