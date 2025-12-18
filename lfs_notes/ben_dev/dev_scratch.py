import numpy as np
from plyfile import PlyData

# Load and examine the PLY file header
ply_path = "/home/bwilliams/encode/code/lichtfeld-studio/LichtFeld-Studio/output/truck_full/splat_10000.ply"

try:
    plydata = PlyData.read(ply_path)
    
    print("PLY Header Information:")
    print(f"Number of elements: {len(plydata.elements)}")
    
    for element in plydata.elements:
        print(f"\nElement: {element.name}")
        print(f"Count: {element.count}")
        print("Properties:")
        for prop in element.properties:
            print(f"  - {prop.name}: {prop.val_dtype}")
            
except Exception as e:
    print(f"Error reading PLY file: {e}")