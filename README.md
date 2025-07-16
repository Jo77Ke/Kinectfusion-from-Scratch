# KinectFusion

## Installation

```bash
    sudo apt update -qq -y && sudo apt upgrade -qq -y && sudo apt install -qq -y build-essentials libeigen3-dev libopencv-dev
```

## Setting environment variables to configure OpenMP parallelization

```bash
    export OMP_NUM_THREADS=<N_THREADS>;OMP_PROC_BIND=true;OMP_PLACES=cores
```
Replace `<N_THREADS>` with the number of threads you want to use.

## Checklist

### Input
- [x] Read depth and rgb images from sensor into depth and color maps
- [x] Read camera intrinsics and extrinsics

### Output
- [x] Project depth map into 3D vertex map using camera intrinsics
- [x] Triangulate vertices into a mesh and write it to a file

### Pre-processing (per frame)
- [x] Apply a bilateral filter on the depth map to reduce noise
- [x] Compute a normal map from the depth map
- [ ] Implement sub-sampling (block averaging with depth values within $3\sigma_r$ of the central pixel)

### Volumetric Fusion
- [x] Compute projective TSDF
- [ ] Implement running average for incrementally adding a new frame

### Surface Prediction
- [x] Implement ray-casting

### Pose Estimation
- [ ] Implement ICP