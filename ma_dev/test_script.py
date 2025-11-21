# NOTE: Run in the map-anything conda env AND in the map-anything directory

import numpy as np
import open3d
import os
import torch

#For getting mapanything
from mapanything.models import MapAnything
from mapanything.utils.image import load_images

# TODO: Optional config for better memory efficiency (sacrifice speed to prevent OOM)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # Set True to enable

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Load model
model = MapAnything.from_pretrained("facebook/map-anything").to(device)
# NOTE: For Apache 2.0 license model, use "facebook/map-anything-apache"

# path to images
images = "/home/ben/encode/data/troll"  

# resize, normalize, convert numpy array to pytorch tensors, batch
# TODO: blog post says consider using pytorch Dataset and DataLoader for large datasets
views = load_images(images)

# Inference parameters
predictions = model.infer(
    views,                            # Input views
    memory_efficient_inference=False, # TODO use for larger datasets? # Trades off speed for more views (up to 2000 views on 140 GB) 
    use_amp=True,                     # Use mixed precision inference (recommended)
    amp_dtype="bf16",                 # bf16 inference (recommended; falls back to fp16 if bf16 not supported)
    apply_mask=True,                  # Apply masking to dense geometry outputs
    mask_edges=True,                  # Remove edge artifacts by using normals and depth
    apply_confidence_mask=False,      # Filter low-confidence regions
    confidence_percentile=10,         # Remove bottom 10 percentile confidence pixels
)
