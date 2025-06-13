# Overview
Reimplementation of scikit-image's (https://scikit-image.org/docs/0.24.x/api/skimage.restoration.html#skimage.restoration.richardson_lucy)  Lucy-Richardson deconvolution using AMD Hip for GPUs.

This GPU implementation provides a 20 to 30 times speedup over the conventional version for large images [1000 x 400 x 400]. For smaller images, there will be less (if any) resulting speedup due to the speed transfer limits between main memory and the GPU.

In theory this should be compatible for both NVIDIA and AMD Gpus, but has only been tested with a AMD Radeon 6700XT using rocm 5.3.0.


# Requirements:
- AMD Hip (https://rocm.docs.amd.com/projects/HIP/en/docs-develop/install/install.html)
- pybind11 (https://github.com/pybind/pybind11)
- cmake >= 3.2.1

# Building
```bash
git clone https://github.com/PeterQLee/HIP_lucy_richardson
cd HIP_lucy_richardson
mkdir build
cd build
cmake ../
make
```

# Usage:

## C++ interface
Link to lucy_richardson_deconv. Example usage is shown in tests/test.cc

```c++
typedef struct {
  int dims[3]; // Channel, height, width
  f32 *imageData;

  int psfdims[2]; //height, width
  f32 *psfData;
  f32 *flipPsfData;
  
} ImageData;

void lucy_richardson(ImageData *imData, int num_iter, f32 epsilon , bool clip = true, bool flag_denom_filter = false, f32 denom_filter = 0.0, uint channelbatch = 1);
```

## Python interface
```
LR_GPU_wrapper.run_LR_deconv(image, psf, num_iter, clip=True, filter_epsilon=None, channelbatch=1)

   image: shape [channel, height, width] if three dimensional, or [height, width] if two dimensional. image must be a float32 array that is C contiguous.

  psf: a two dimensional point spread function. If image has three dimensions, the deconvolution is applied onto each channel independently.

  num_iter: the number of iterations the deconvolution is ran for.

  clip : Controls if values are bounded between -1 and 1

  filter_epsilon: Value below which intermediate results become 0 to avoid division by small numbers.

  channelbatch: number of channels to concurrently process.

---------------
  
  The result of the deconvolution is returned by overwriting the data in image.

```


Example:
```python
import LR_GPU_wrapper
import numpy as np
channel, height, width = (3,50,50)
image = np.ones((channel,height,width), dtype=np.float32)
psf = np.ones((1,3), dtype=np.float32)/3
LR_GPU_wrapper.run_LR_deconv(image, psf, 10)
```
