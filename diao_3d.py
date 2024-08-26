import gradio as gr
import trimesh
import numpy as np

def hex_to_rgb(hex_color):
    """Convert hex color string to an RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def create_colored_cylinder(radius, height, x_offset=0, y_offset=0, z_offset=0, color="#FFFFFF"):
    # Convert hex color to RGB
    rgb_color = hex_to_rgb(color)
    rgba_color = np.array(rgb_color + (255,))  # Add alpha channel
    
    # Create a cylinder mesh
    cylinder = trimesh.creation.cylinder(radius=radius, height=height)
    # Translate the cylinder to the correct offset
    cylinder.apply_translation([x_offset, y_offset, z_offset])
    # Apply the color to all faces of the cylinder
    cylinder.visual.face_colors = np.tile(rgba_color, (len(cylinder.faces), 1))
    return cylinder

def create_colored_sphere(radius, center, color="#FFFFFF"):
    # Convert hex color to RGB
    rgb_color = hex_to_rgb(color)
    rgba_color = np.array(rgb_color + (255,))  # Add alpha channel
    
    # Create a sphere mesh
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=radius)
    # Translate the sphere to the center
    sphere.apply_translation(center)
    # Apply the color to all faces of the sphere
    sphere.visual.face_colors = np.tile(rgba_color, (len(sphere.faces), 1))
    return sphere

def create_model(shaft_len, shaft_radius, head_radius, scrotum_radius, shaft_color, head_color, scrotum_color, name):
    # Create the shafts as cylinders with color
    shaft1 = create_colored_cylinder(shaft_radius, shaft_len, x_offset=shaft_radius/2, y_offset=0, z_offset=shaft_len/2, color=shaft_color)
    shaft2 = create_colored_cylinder(shaft_radius, shaft_len, x_offset=-shaft_radius/2, y_offset=0, z_offset=shaft_len/2, color=shaft_color)
    shaft3 = create_colored_cylinder(shaft_radius * 0.8, shaft_len, x_offset=0, y_offset=shaft_radius * 0.8, z_offset=shaft_len/2, color=shaft_color)
    
    # Combine the shafts
    shaft = trimesh.util.concatenate([shaft1, shaft2, shaft3])
    
    # Create the head as a sphere with color
    head = create_colored_sphere(head_radius, center=[0, 0, shaft_len + head_radius/1.4], color=head_color)
    
    # Create the scrotum as two spheres with color
    scrotum1 = create_colored_sphere(scrotum_radius, center=[-scrotum_radius/1.2, 0.3, -scrotum_radius/1.2], color=scrotum_color)
    scrotum2 = create_colored_sphere(scrotum_radius, center=[scrotum_radius/1.2, 0.3, -scrotum_radius/1.2], color=scrotum_color)
    
    # Combine all parts into one mesh
    final_model = trimesh.util.concatenate([shaft, head, scrotum1, scrotum2])
    
    # Save the mesh as a GLB file (supports colors)
    file_path = f"{name}_model.glb"
    final_model.export(file_path)
    
    return file_path

# Gradio interface with live updates
iface = gr.Interface(
    fn=create_model,
    inputs=[
        gr.Slider(minimum=1, maximum=20, step=1, label="Shaft Length", value=7),
        gr.Slider(minimum=0.5, maximum=5, step=0.1, label="Shaft Radius", value=1),
        gr.Slider(minimum=0.5, maximum=5, step=0.1, label="Head Radius", value=1.5),
        gr.Slider(minimum=1, maximum=5, step=0.5, label="Scrotum Radius", value=2),
        gr.ColorPicker(label="Shaft Color", value="#8B4513"),  # Default to saddlebrown
        gr.ColorPicker(label="Head Color", value="#B44444"),   # Default to darker_lightcoral
        gr.ColorPicker(label="Scrotum Color", value="#FFDAB9"), # Default to navajowhite
        gr.Textbox(label="Name", value="xyf")
    ],
    outputs=gr.Model3D(),  # Use Model3D to display the 3D model
    title="3D Model Creator",
    live=True  # Enable live updates
)

# Launch the Gradio interface
iface.launch()
