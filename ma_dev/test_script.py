import numpy as np
import open3d
import os
import torch

#For getting mapanything
from mapanything.models import MapAnything
from mapanything.utils.image import load_images

# Optional config for better memory efficiency (sacrifice speed to prevent OOM)
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # Set True to enable

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Load model
model = MapAnything.from_pretrained("facebook/map-anything").to(device)
# NOTE: For Apache 2.0 license model, use "facebook/map-anything-apache"