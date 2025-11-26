#!/bin/bash
# 1. Create the output folder
mkdir -p /home/ben/encode/data/KIOST_vids/images_DJI_20250924_Camera01_D

# 2. Extract frames (2 per second) using ffmpeg
ffmpeg -i /home/ben/encode/data/KIOST_vids/DJI_20250924_Camera01_D.mp4 -vf fps=2 /home/ben/encode/data/KIOST_vids/images_DJI_20250924_Camera01_D/%05d.png