# PY_SPRT_RANSAC:R-RANSAC with Sequential Probability Ratio Test

This project implements several variants of the SPRT (Sequential Probability Ratio Test) RANSAC algorithm for robust plane fitting in 3D point clouds. The algorithms are optimized for speed(using [numba](https://numba.readthedocs.io/en/stable/index.html)) and accuracy, and support both standard and normal-assisted sampling strategies.

Based on Matas and Chum's 2008 paper [link](https://cmp.felk.cvut.cz/~matas/papers/chum-waldsac-iccv05.pdf) and fiskrt's Matlab implementation [link](https://github.com/fiskrt/RRANSAC). (But have some differents and new features)


## 1.Features:

four variants of the SPRT-RANSAC algorithm:

- SPRT-RANSAC : Standard SPRT-RANSAC for robust plane fitting 
- SPRT-RANSAC-with-Normal: Normal-assisted SPRT-RANSAC for scenarios with known point normals
- Fast SPRT-RANSAC&SPRT-RANSAC-with-Normal: Fast approximate versions of both SPRT-RANSAC and SPRT-RANSAC-with-Normal, but accuracy may be lower.

Speed up by Numba.

## 2.Main Functions

### Funtion: SPRT_RANSAC_RAW

Fits a plane to 3D points using standard SPRT-RANSAC with three-point random sampling.

#### Parameters

- **points**: (N, 3) array of 3D points
- **threshold**: Distance threshold for inliers
- **eta0, epsilon, delta**: SPRT parameters
- **max_iterations, m, tm, ms**: Algorithm parameters

#### Returns

- **best_model**: Plane coefficients [A, B, C, D] (Ax + By + Cz + D = 0)
- **best_support**: Number of inliers
- **mean_distance_error**: Mean absolute distance to the plane
  
### SPRT_RANSAC_RAW_NOR

SPRT-RANSAC using point normals for model estimation (no random sampling).

#### Parameters

- **points**: (N, 3) array of 3D points
- **point_normals**: (N, 3) array of normals
- **eta0, epsilon, delta**: SPRT parameters
- **max_iterations, m, tm, ms**: Algorithm parameters

#### Returns

- **best_model**: Plane coefficients [A, B, C, D] (Ax + By + Cz + D = 0)
- **best_support**: Number of inliers
- **mean_distance_error**: Mean absolute distance to the plane

### SPRT_RANSAC_FAST

A faster, approximate SPRT_RANSAC_RAW version.
Parameters/Returns: Same as SPRT_RANSAC_RAW

### SPRT_RANSAC_FAST_NOR

A faster, approximate SPRT_RANSAC_RAW_NOR version.

Parameters/Returns: Same as SPRT_RANSAC_RAW_NOR

## Dependencies

numpy 
numba


## Example Usage

~~~ python 
import numpy as np
from RRANSAC import SPRT_RANSAC_RAW, SPRT_RANSAC_RAW_NOR, SPRT_RANSAC_FAST, SPRT_RANSAC_FAST_NOR

points = np.random.rand(1000, 3).astype(np.float32)
normals = np.random.rand(1000, 3).astype(np.float32)

model, support, mean_err = SPRT_RANSAC_RAW(points)
print("Best plane:", model)
print("Inliers:", support)
print("Mean error:", mean_err)

model_nor, support_nor, mean_err_nor = SPRT_RANSAC_RAW_NOR(points, normals)
print("Best plane with normals:", model_nor)
print("Inliers with normals:", support_nor)
print("Mean error with normals:", mean_err_nor)

model_fast, support_fast, mean_err_fast = SPRT_RANSAC_FAST(points)
print("Best plane (fast):", model_fast)
print("Inliers (fast):", support_fast)
print("Mean error (fast):", mean_err_fast)

model_fast_nor, support_fast_nor, mean_err_fast_nor = SPRT_RANSAC_FAST_NOR(points, normals)
print("Best plane with normals (fast):", model_fast_nor)
print("Inliers with normals (fast):", support_fast_nor)
print("Mean error with normals (fast):", mean_err_fast_nor)

~~~