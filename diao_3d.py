import gradio as gr
import trimesh
import numpy as np

def create_colored_cylinder(radius, height, x_offset=0, y_offset=0, z_offset=0, color=(1, 1, 1)):
    cylinder = trimesh.creation.cylinder(radius, height)
    # Translate the cylinder to the correct offset
    cylinder.apply_translation([x_offset, y_offset, z_offset])
    # Apply color to each face
    cylinder.visual.face_colors = np.array([color] * len(cylinder.faces)) * 255
    return cylinder

def create_colored_sphere(radius, center, color=(1, 1, 1)):
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=radius)
    # Translate the sphere to the center
    sphere.apply_translation(center)
    # Apply color to each face
    sphere.visual.face_colors = np.array([color] * len(sphere.faces)) * 255
    return sphere

def create_model(shaft_len, shaft_radius, head_radius, scrotum_radius, name):
    # Create the shafts as cylinders with color
    shaft1 = create_colored_cylinder(shaft_radius, shaft_len, x_offset=shaft_radius/2, y_offset=0, z_offset=shaft_len/2, color=(139/255, 69/255, 19/255))  # saddlebrown
    shaft2 = create_colored_cylinder(shaft_radius, shaft_len, x_offset=-shaft_radius/2, y_offset=0, z_offset=shaft_len/2, color=(139/255, 69/255, 19/255))  # saddlebrown
    shaft3 = create_colored_cylinder(shaft_radius * 0.8, shaft_len, x_offset=0, y_offset=shaft_radius * 0.8, z_offset=shaft_len/2, color=(188/255, 143/255, 143/255))  # rosybrown
    
    # Combine the shafts
    shaft = trimesh.util.concatenate([shaft1, shaft2, shaft3])
    
    # Create the head as a sphere with color
    head = create_colored_sphere(head_radius, center=[0, 0, shaft_len + head_radius/1.4], color=(180/255, 68/255, 68/255))  # darker_lightcoral
    
    # Create the scrotum as two spheres with color
    scrotum1 = create_colored_sphere(scrotum_radius, center=[-scrotum_radius/1.2, 0.3, -scrotum_radius/1.2], color=(255/255, 222/255, 173/255))  # navajowhite
    scrotum2 = create_colored_sphere(scrotum_radius, center=[scrotum_radius/1.2, 0.3, -scrotum_radius/1.2], color=(255/255, 222/255, 173/255))  # navajowhite
    
    # Combine all parts into one mesh
    final_model = trimesh.util.concatenate([shaft, head, scrotum1, scrotum2])
    
    # Save the mesh as a GLB file (supports colors)
    file_path = f"{name}_model.glb"
    final_model.export(file_path)
    
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
    outputs=gr.Model3D(),
    title="3D Model Creator"
)

# Launch the Gradio interface
iface.launch()
