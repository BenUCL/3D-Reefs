# See docs here: https://github.com/wildflowai/splat
import pycolmap
from wildflow import splat

# Paths
colmap_path = "/home/bwilliams/encode/code/lichtfeld-studio/data/reef_test/sparse/0"
colmap_points_bin = "/home/bwilliams/encode/code/lichtfeld-studio/data/reef_test/sparse/0/points3D.bin"  # COLMAP tie points
input_splat = "/home/bwilliams/encode/code/lichtfeld-studio/output/reef_test1/splat_100000.ply"
output_splat = "/home/bwilliams/encode/code/lichtfeld-studio/output/reef_test1/splat_100000_cleaned.ply"
disposed_splat = "/home/bwilliams/encode/code/lichtfeld-studio/output/reef_test1/splat_100000_disposed.ply"

# Load the colmap reconstruction model to get camera bounds
print("Loading COLMAP reconstruction...")
model = pycolmap.Reconstruction(colmap_path)

# Get 3d camera positions to determine reasonable bounds
camera_poses = [img.projection_center() for img in model.images.values()]
print(f"Found {len(camera_poses)} camera positions")

# Get min and max z bounds (similar to your dev.py)
cameras_z_values = [pos[2] for pos in camera_poses]
min_z = min(cameras_z_values) - 2.0  # add 2m buffer
max_z = max(cameras_z_values) + 0.5  # add 0.5m buffer

# Get x, y bounds from camera positions
cameras_x_values = [pos[0] for pos in camera_poses]
cameras_y_values = [pos[1] for pos in camera_poses]

min_x = min(cameras_x_values) - 2.0  # add buffer
max_x = max(cameras_x_values) + 2.0
min_y = min(cameras_y_values) - 2.0
max_y = max(cameras_y_values) + 2.0

print(f"\nCamera bounds:")
print(f"  X: {min_x:.2f} to {max_x:.2f}")
print(f"  Y: {min_y:.2f} to {max_y:.2f}")
print(f"  Z: {min_z:.2f} to {max_z:.2f}")

# Run cleanup on the splat
print(f"\nCleaning up splat: {input_splat}")
print(f"Output will be saved to: {output_splat}")
print("\nCleanup settings (optimized - no core boundaries needed for single splat):")
print("  - max_area: 0.003 (more aggressive on oversized splats)")
print("  - min_neighbors: 15 (moderate balance)")
print("  - radius: 0.15m (tighter clustering)")
print("\nNote: COLMAP points reference disabled due to binary read error")

splat.cleanup_splats({
    "input_file": input_splat,
    "output_file": output_splat,
    "disposed_file": disposed_splat,  # Save disposed splats for examination
    # Spatial bounds
    "min_x": min_x,
    "max_x": max_x,
    "min_y": min_y,
    "max_y": max_y,
    "min_z": min_z,
    "max_z": max_z,
    # Cleanup settings - balanced approach
    "max_area": 0.003,       # More aggressive (was 0.004)
    "min_neighbors": 15,     # Moderate (between 10 default and 20)
    "radius": 0.15           # Tighter (was 0.2)
})

print("\n✅ Cleanup complete!")
print(f"Clean splat saved to: {output_splat}")
print(f"Disposed splats saved to: {disposed_splat}")
