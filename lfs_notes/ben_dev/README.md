# First try at splatting

### Attempt 1 (fail), what we did:
- Downloaded some reef data from gc. I think this was raw images and then also the other colmap outputs required, like `cameras.bin`, `images.bin` and `points3D.bin`, that were output by agisoft (or created from its outputs). The gc download command:
```
gsutil -o GSUtil:sliced_object_download_threshold=1G \
       -o GSUtil:sliced_object_download_max_components=12 \
       -m cp gs://wildflow/konstantz/pc.ply ./
```

- We then downsampled the resolution of the raw images as this was too high for my GPU. I think it was still the raw gopro images. We used: `code/lichtfeld-studio/LichtFeld-Studio/dev/downsample_img.sh`
- Ran `code/lichtfeld-studio/LichtFeld-Studio/dev/dev.py`, this takes the colmap outputs and splits it into patches, so it makes a smaller area that we can work with by taking a single patch.
- We then tried to run the lichfeld studio code on a single patch but hit some issues. A first issue was my GPU had disapeared. For a sanity check, tryo to run it with the truck.
- To run the demo with the truck do:
```
# Nav to the right directory
cd /home/bwilliams/encode/code/lichtfeld-studio/LichtFeld-Studio
# Set library path so LichtFeld-Studio can find PyTorch C++ libraries at runtime
export LD_LIBRARY_PATH="$PWD/external/libtorch/lib:${LD_LIBRARY_PATH}"

# Run the splat!
./build/LichtFeld-Studio -d ../data/tandt/truck -o output/truck_full --eval --headless -i 10000 # might want to increase iters to 30k or more
```

### Attempt 2 (pass), what we did:
- I downloaded the data from the wilfdlow cloud bucket here: https://console.cloud.google.com/storage/browser/wildflow/workshop-tabuhan-patch, with:
```
gsutil -m cp -r \
  "gs://wildflow/workshop-tabuhan-patch/colmap" \
  "gs://wildflow/workshop-tabuhan-patch/images-1600" \
  "gs://wildflow/workshop-tabuhan-patch/images-800" \
  "gs://wildflow/workshop-tabuhan-patch/images" \
  "gs://wildflow/workshop-tabuhan-patch/points3D_30percent.bin" \
  "gs://wildflow/workshop-tabuhan-patch/points3D_5percent.bin" \
  .
  ```
- I then organised it to match the truck data exactly. I took the low res images-800 folder and made this the images folder. Only thing missing is the project.ini file which the truck data has but the reef data does not, this didn't seem to matter.
- Now try to splat:
```
./build/LichtFeld-Studio -d ../data/reef_test -o output/reef_test --eval --headless -i 50000 # might want to increase iters 
```
- This worked! 
- Open supersplat in chrome and drag and drop the .ply file in. I can see a clear diff between the quality with 10k vs 50k iters. 10k struggled much more on things like table coral.
- Note, Sergei was able to edit it in supesplat to do things like remove the largest splats (removing artifacts), crop around and delete anything outside (as it can produce artifacts really far away), and then export that as a new .ply which is better for serving.
- Note, I did not have to run the downsample used in Attempt 1 above, or the `dev.py` which is used for splitting up the colmap ouputs of larger areas into patches. This is because this new data was relatively small (a 4x4m area perhaps). However, this patching did seem to work well before so keep it in mind for the future.
- I just tried with the higher res images-1600 and got an error.

### Extras like patching, cleaning merging
See some notes in my step by step research doc. We can use the patch visualiser script to figure out what params to use for dev.py to then set patch images count etc. There is a bunch of code commented out in dev.py as we were running each step (like patch, clean, merge) one by one. Really we need to batch the stuff once we've run the visualiser.

```
# command to run splat on second patch after the dev.py which did patching
/home/bwilliams/encode/code/lichtfeld-studio/LichtFeld-Studio/build/LichtFeld-Studio \
-d /home/bwilliams/encode/code/lichtfeld-studio/data/reef_test/patches/p1/sparse/0 \
--images /home/bwilliams/encode/code/lichtfeld-studio/data/reef_test/images \
-o /home/bwilliams/encode/code/lichtfeld-studio/LichtFeld-Studio/output/p1 \
--eval \
--headless \
-i 10000
```

