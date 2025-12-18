import pycolmap
from wildflow import splat

colmap_path = "/home/bwilliams/encode/code/lichtfeld-studio/data/reef_test/sparse/0"
# save path for patches
patches_save_path = "/home/bwilliams/encode/code/lichtfeld-studio/data/reef_test/patches"
pc_path = "/home/bwilliams/encode/code/lichtfeld-studio/data/reef_test/" #add dense point cloud later

# output path for merged splat
output_file = "/home/bwilliams/encode/code/lichtfeld-studio/LichtFeld-Studio/output/merged_splat.ply"

# Load the colmap reconstruction model (contains cameras, images, points3D)
model = pycolmap.Reconstruction(colmap_path)

# Get 3d camera positions
camera_poses = [img.projection_center() for img in model.images.values()]

# prints array with 3d positions of all cameras
print(camera_poses[:10])

# get min and max z, this was to remove splats outside of sensible range
cameras_z_values = [pos[2] for pos in camera_poses]
min_z = min(cameras_z_values) - 2.0 # add 2m
max_z = max(cameras_z_values) + 0.5 # add 0.5m

# split into 2d patches
cameras_2d = [(pos[0], pos[1]) for pos in camera_poses]
patches_list = splat.patches(cameras_2d, 
                             # get these values from the matplot visualiser
                             max_cameras=250, # use this many images max per patch
                             buffer_meters=0.5, # add this much buffer to each patch
)
print(patches_list[0])

# # split cameras. Makes the images, cameras and points3D. so splits the original bigger version.
# result = splat.split_cameras({
#     "input_path": colmap_path,
#     "save_points3d": True, #optional
#     "min_z": min_z,
#     "max_z": max_z,
#     "patches": [
#         {**patch, "output_path": f"{patches_save_path}/p{i}/sparse/0"}
#         for i, patch in enumerate(patches_list)
#     ]
# })
# print(result)



# # Sub sample points in big point cloud made in metashape, then splits and places these for each patch
# # into the patch folders.
# coords = lambda p: {k: p[k] for k in ('min_x', 'max_x', 'min_y', 'max_y')}
# result = splat.split_point_cloud({
#     "input_file": pc_path,
#     "min_z": min_z,
#     "max_z": max_z,
#     "sample_percentage": 5.0 , # only use 10% of points
#     "patches": [
#         {**coords(patch), "output_file": f"{save_path}/p{i}/sparse/0/points3D.bin"}
#         for i, patch in enumerate(patches_list)
#     ]
# })

def get_core_boundaries(patch):
    buffer = 0.3 # in metres
    return {
        "min_x": patch["min_x"] + buffer,
        "max_x": patch["max_x"] - buffer,
        "min_y": patch["min_y"] + buffer,
        "max_y": patch["max_y"] - buffer 
    }


# #TODO: dont hard code the patch paths here
# uncomment o clean up individual splats
# patch_id = 1
# splat.cleanup_splats({
#     "input_file": f"/home/bwilliams/encode/code/lichtfeld-studio/LichtFeld-Studio/output/p{patch_id}/splat_10000.ply",
#     "output_file": f"/home/bwilliams/encode/code/lichtfeld-studio/LichtFeld-Studio/output/p{patch_id}/splat_clean.ply",
#     "disposed_file": f"/home/bwilliams/encode/code/lichtfeld-studio/LichtFeld-Studio/output/p{patch_id}/splat_disposed.ply",  # Save disposed splats for examination
#     **get_core_boundaries(patches_list[patch_id]),
#     "min_z": min_z,
#     "max_z": max_z,
#     # Cleanup settings - moved from nested dict to top level
#     "max_area": 0.004,
#     "min_neighbors": 20,
#     "radius": 0.2
# })


# Run merge
config = {
    "input_files": [f"/home/bwilliams/encode/code/lichtfeld-studio/LichtFeld-Studio/output/p{patch_id}/splat_clean.ply" for patch_id in [0,1]],
    "output_file": str(output_file)
}

# Call Rust merge function
splat.merge_ply_files(config)