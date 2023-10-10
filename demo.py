import bpy

# Clear existing scene
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

# Create a sphere
bpy.ops.mesh.primitive_uv_sphere_add(radius=5, location=(0, 0, 0))

# Apply a material to the sphere
bpy.context.object.data.materials.append(bpy.data.materials['Planet Material'])

# Set up the rendering settings
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.samples = 100

# Render the 3D model
bpy.ops.render.render(write_still=True)

# Save the rendered image
bpy.data.images['Render Result'].save_render(filepath='planet.png')
