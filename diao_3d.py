import gradio as gr
import trimesh
import numpy as np
from scipy.ndimage import gaussian_filter

def hex_to_rgb(hex_color):
    """Convert hex color string to an RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def apply_wrinkles(mesh, intensity, smoothness):
    vertices = mesh.vertices.copy()
    perturbations = np.random.uniform(-intensity, intensity, vertices.shape)
    for i in range(3):
        perturbations[:, i] = gaussian_filter(perturbations[:, i], sigma=smoothness)
    mesh.vertices = vertices + perturbations
    return mesh

def create_helix(radius, pitch, height, turns, center=(0,0,0), direction=(0,0,1)):
    t = np.linspace(0, 2*np.pi*turns, 100*turns)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = (pitch/(2*np.pi)) * t
    points = np.column_stack((x, y, z))

    # Normalize the direction vector
    direction = direction / np.linalg.norm(direction)
    
    # Apply the direction vector to the helix
    points = points @ trimesh.transformations.rotation_matrix(
        np.arccos(direction[2]), np.cross([0, 0, 1], direction))[:3, :3]
    
    # Translate to the center point
    points += center

    return points

def create_tube(points, tube_radius):
    segments = []
    num_segments = 12
    
    for point in points:
        angle = np.linspace(0, 2 * np.pi, num_segments)
        circle = np.array([np.cos(angle), np.sin(angle)]).T * tube_radius
        circle_3d = np.column_stack((circle, np.zeros(circle.shape[0])))
        circle_3d += point
        segments.append(circle_3d)
    
    faces = []
    for i in range(len(segments) - 1):
        for j in range(num_segments):
            next_j = (j + 1) % num_segments
            faces.append([i * num_segments + j, i * num_segments + next_j, (i + 1) * num_segments + j])
            faces.append([(i + 1) * num_segments + j, i * num_segments + next_j, (i + 1) * num_segments + next_j])
    
    vertices = np.vstack(segments)
    faces = np.array(faces)
    
    return trimesh.Trimesh(vertices=vertices, faces=faces)

def create_hair_distribution(surface_mesh, num_hairs, radius, pitch, turns, hair_length):
    hair_meshes = []
    
    # Sample points on the surface and use the mesh's vertex normals for direction
    for _ in range(num_hairs):
        point = surface_mesh.vertices[np.random.choice(len(surface_mesh.vertices))]
        normal = surface_mesh.vertex_normals[np.random.choice(len(surface_mesh.vertices))]

        # Generate a random direction by perturbing the normal vector slightly
        random_direction = normal + np.random.uniform(-0.2, 0.2, size=3)
        random_direction = random_direction / np.linalg.norm(random_direction)  # Normalize the direction
        
        helix_points = create_helix(radius=radius, pitch=pitch, height=hair_length, turns=turns, center=point, direction=random_direction)
        hair_mesh = create_tube(helix_points, tube_radius=radius*0.1)
        hair_meshes.append(hair_mesh)
    
    return trimesh.util.concatenate(hair_meshes)

def create_colored_cylinder(radius, height, x_offset=0, y_offset=0, z_offset=0, color="#FFFFFF", wrinkle_intensity=0, wrinkle_smoothness=1.5):
    rgb_color = hex_to_rgb(color)
    rgba_color = np.array(rgb_color + (255,))
    cylinder = trimesh.creation.cylinder(radius=radius, height=height)
    cylinder.apply_translation([x_offset, y_offset, z_offset])
    cylinder.visual.face_colors = np.tile(rgba_color, (len(cylinder.faces), 1))
    cylinder = apply_wrinkles(cylinder, wrinkle_intensity, wrinkle_smoothness)
    
    return cylinder

def create_colored_sphere(radius, center, color="#FFFFFF", wrinkle_intensity=0, wrinkle_smoothness=1.5, hair_density=0, hair_length=0):
    rgb_color = hex_to_rgb(color)
    rgba_color = np.array(rgb_color + (255,))
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=radius)
    sphere.apply_translation(center)
    sphere.visual.face_colors = np.tile(rgba_color, (len(sphere.faces), 1))
    sphere = apply_wrinkles(sphere, wrinkle_intensity, wrinkle_smoothness)
    
    # Add curly hair strands only on the scrotum
    if hair_density > 0 and hair_length > 0:
        hair_mesh = create_hair_distribution(sphere, num_hairs=hair_density, radius=0.1, pitch=0.2, turns=3, hair_length=hair_length)
        sphere = trimesh.util.concatenate([sphere, hair_mesh])
    
    return sphere

def create_model(shaft_len, shaft_radius, head_radius, scrotum_radius, shaft_color, head_color, scrotum_color, shaft_wrinkle_intensity, head_wrinkle_intensity, scrotum_wrinkle_intensity, wrinkle_smoothness, hair_density, hair_length, name):
    shaft1 = create_colored_cylinder(shaft_radius, shaft_len, x_offset=shaft_radius/2, y_offset=0, z_offset=shaft_len/2, color=shaft_color, wrinkle_intensity=shaft_wrinkle_intensity, wrinkle_smoothness=wrinkle_smoothness)
    shaft2 = create_colored_cylinder(shaft_radius, shaft_len, x_offset=-shaft_radius/2, y_offset=0, z_offset=shaft_len/2, color=shaft_color, wrinkle_intensity=shaft_wrinkle_intensity, wrinkle_smoothness=wrinkle_smoothness)
    shaft3 = create_colored_cylinder(shaft_radius * 0.8, shaft_len, x_offset=0, y_offset=shaft_radius * 0.8, z_offset=shaft_len/2, color=shaft_color, wrinkle_intensity=shaft_wrinkle_intensity, wrinkle_smoothness=wrinkle_smoothness)
    shaft = trimesh.util.concatenate([shaft1, shaft2, shaft3])
    
    head = create_colored_sphere(head_radius, center=[0, 0, shaft_len + head_radius/1.4], color=head_color, wrinkle_intensity=head_wrinkle_intensity, wrinkle_smoothness=wrinkle_smoothness)
    
    scrotum1 = create_colored_sphere(scrotum_radius, center=[-scrotum_radius/1.2, 0.3, -scrotum_radius/1.2], color=scrotum_color, wrinkle_intensity=scrotum_wrinkle_intensity, wrinkle_smoothness=wrinkle_smoothness, hair_density=hair_density, hair_length=hair_length)
    scrotum2 = create_colored_sphere(scrotum_radius, center=[scrotum_radius/1.2, 0.3, -scrotum_radius/1.2], color=scrotum_color, wrinkle_intensity=scrotum_wrinkle_intensity, wrinkle_smoothness=wrinkle_smoothness, hair_density=hair_density, hair_length=hair_length)
    
    final_model = trimesh.util.concatenate([shaft, head, scrotum1, scrotum2])
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
        gr.ColorPicker(label="Shaft Color", value="#8B4513"),
        gr.ColorPicker(label="Head Color", value="#B44444"),
        gr.ColorPicker(label="Scrotum Color", value="#FFDAB9"),
        gr.Slider(minimum=0, maximum=0.5, step=0.01, label="Shaft Wrinkle Intensity", value=0.1),
        gr.Slider(minimum=0, maximum=0.5, step=0.01, label="Head Wrinkle Intensity", value=0.1),
        gr.Slider(minimum=0, maximum=0.5, step=0.01, label="Scrotum Wrinkle Intensity", value=0.1),
        gr.Slider(minimum=0.1, maximum=5, step=0.1, label="Wrinkle Smoothness", value=2.5),
        gr.Slider(minimum=0, maximum=100, step=1, label="Hair Density", value=20),
        gr.Slider(minimum=0.1, maximum=20, step=0.1, label="Hair Length", value=1.0),  # Hair length can now be as long as the shaft
        gr.Textbox(label="Name", value="xyf")
    ],
    outputs=gr.Model3D(),
    title="3D Model Creator with Hair on Scrotum",
    live=True
)

iface.launch()
