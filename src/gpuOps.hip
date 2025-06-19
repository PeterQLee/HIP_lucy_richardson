#include "gpuOps.hpp"
#include <hip/hip_runtime.h>
#include <iostream>
/* Kernels */

#define HIP_CHECK(condition)         \
{                                    \
    hipError_t error = condition;    \
   if(error != hipSuccess){         \
        std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
}

// Adapted from https://github.com/ROCm/HIP-Examples/blob/master/HIP-Examples-Applications/SimpleConvolution/SimpleConvolution.cpp
//extern __shared__ f32 buf[];
__global__ void Convolution(f32 *input,
			    f32 *psf, f32 *output,
			    uint3 inputDimensions, uint2 maskDimensions,
			    uint imWidth, f32 epsilon) {
  HIP_DYNAMIC_SHARED(f32, buf);
   uint tid = hipBlockDim_x*hipBlockIdx_x + hipThreadIdx_x;
   uint width = inputDimensions.x;
   uint height = inputDimensions.y;
   uint channel_width = inputDimensions.z;

   uint chan = tid / (width*height);
   uint x = tid%width;
   uint y = (tid % (width*height))/width;

   uint maskWidth = maskDimensions.x;
   uint maskHeight = maskDimensions.y;

   uint rad_x = maskWidth/2;
   uint rad_y = maskHeight/2;

   // Transfer to shared memory
   for (int m=0; m*hipBlockDim_x < maskWidth*maskHeight; m++) {
     if ( m*hipBlockDim_x + hipThreadIdx_x < maskWidth*maskHeight) {
       buf[m*hipBlockDim_x+hipThreadIdx_x] = psf[m*hipBlockDim_x+hipThreadIdx_x];
     }
   }
   __syncthreads();
   
   if(x >= width || y >= height || chan >= channel_width)
     return;

   f32 sum = 0.0f;
   uint m = 0 , n = 0, start_n = 0;


   // TODO: further optimize by utilizing shared memory.

   
   // Ensure bounds are correct.
   if (x < rad_x) {
       start_n = rad_x - x;
   }
   if (y < rad_y) {
     m = rad_y - y;
   }
   uint maxY = maskHeight, maxX = maskWidth;
   if ( x + rad_x >= width) {
     maxX = rad_x + (width - x);
   }
   if ( y + rad_y >= height) {
     maxY = rad_y + (height - y);
   }
   
   for(; m < maxY; m++) {
       n = start_n;
       for(; n < maxX; n++)
	 { 
	  uint maskIndex = m * maskWidth + n;
	  uint index = chan*height*width + (y + m - rad_y) * imWidth + (x + n - rad_x);
            
	  sum += input[index] * buf[maskIndex];
	  //sum += input[index] * psf[maskIndex];
	 }
   }
   output[tid] = sum + epsilon;
}
 
/*
  Element-wise division, overwriting the denominator
 */
__global__ void ElemDivision(f32 *num, f32 *den,
			     uint size
			     ) {
  
  uint tid = hipBlockDim_x*hipBlockIdx_x + hipThreadIdx_x;

  // bounds check
  if (tid >= size) return;
  den[tid] = num[tid] / den[tid];

}


__global__ void FiltElemDivision(f32 *num, f32 *den, f32 thresh,
			     uint size
			     ) {
  
  uint tid = hipBlockDim_x*hipBlockIdx_x + hipThreadIdx_x;

  // bounds check
  if (tid >= size) return;
  if (den[tid] < thresh) {den[tid] = 0.0f;}
  else {den[tid] = num[tid] / den[tid];}

}
/* Element-wise multiplication, overwriting b*/
__global__ void ElemMult(f32 *a, f32 *b, uint size) {
  uint tid = hipBlockDim_x*hipBlockIdx_x + hipThreadIdx_x;

  // bounds check.
  if (tid >= size) return;
  b[tid] = b[tid] * a[tid];
}

__global__ void Clip(f32 *a, f32 lower, f32 upper,  uint size) {
  uint tid = hipBlockDim_x*hipBlockIdx_x + hipThreadIdx_x;

  if (tid >= size) return;
  if (a[tid] < lower) a[tid] = lower;
  if (a[tid] > upper) a[tid] = upper;
}

__global__ void SetVal(f32 *a, f32 val, uint size) {
  uint tid = hipBlockDim_x*hipBlockIdx_x + hipThreadIdx_x;
  if (tid >= size) return;
  a[tid] = val;
}

uint localSize = 256;
int callConvolution(f32 *gIn, f32 *gMask, f32 *gOut, uint channelbatch, uint height, uint width,  uint filt_height, uint filt_width, f32 epsilon){

  
  uint localThreads = localSize;
  uint globalThreads = (channelbatch*width*height + localThreads - 1) / localThreads;
  globalThreads *= localThreads;

  uint3 input_dim = make_uint3(width, height, channelbatch);
  uint2 mask_dim  = make_uint2(filt_width, filt_height);

  hipLaunchKernelGGL(Convolution,
		     dim3(globalThreads/localThreads),
                    dim3(localThreads),
		     filt_width*filt_height*sizeof(f32), 0,
		     gIn, gMask, gOut, input_dim, mask_dim, width, epsilon);
  hipDeviceSynchronize();

  return 1;
}

int callDivision(f32 *num, f32 *den, uint size) {
   uint localThreads = localSize;
   uint globalThreads = (size + localThreads - 1) / localThreads;
   globalThreads *= localThreads;

   hipLaunchKernelGGL(ElemDivision,
		      dim3(globalThreads/localThreads),
                    dim3(localThreads),
                    0, 0,
		      num, den, size);
   hipDeviceSynchronize();
   return 1;
}


int callFiltDivision(f32 *num, f32 *den, f32 thresh, uint size) {
   uint localThreads = localSize;
   uint globalThreads = (size + localThreads - 1) / localThreads;
   globalThreads *= localThreads;

   hipLaunchKernelGGL(FiltElemDivision,
		      dim3(globalThreads/localThreads),
		      dim3(localThreads),
		      0, 0,
		      num, den, thresh, size);
   hipDeviceSynchronize();
   return 1;
}

int callMult(f32 *a, f32 *b, uint size) {
   uint localThreads = localSize;
   uint globalThreads = (size + localThreads - 1) / localThreads;
   globalThreads *= localThreads;

   hipLaunchKernelGGL(ElemMult,
		      dim3(globalThreads/localThreads),
		      dim3(localThreads),
		      0, 0,
		      a, b, size);
   hipDeviceSynchronize();
   return 1;
}

int callClip(f32 *a, f32 lower, f32 upper, uint size) {
   uint localThreads = localSize;
   uint globalThreads = (size + localThreads - 1) / localThreads;
   globalThreads *= localThreads;

   hipLaunchKernelGGL(Clip,
		      dim3(globalThreads/localThreads),
		      dim3(localThreads),
		      0, 0,
		      a, lower, upper, size);
   hipDeviceSynchronize();
   return 1;
}


int setupHip(ImageData *imData, GpuData *gData, uint channelbatch) {
   // Allocate buffers based on sizes from the input struct
  
  int filt_height = imData->psfdims[0];
  int filt_width = imData->psfdims[1];
  int channel = imData->dims[0];
  int height = imData->dims[1];
  int width = imData->dims[2];
  f32 *buffer;
  hipMalloc( (void**) &(buffer) , sizeof(float)*((channel+3*channelbatch)*height*width + filt_height*filt_width*2));

  gData->g_image = buffer;
  hipMemcpy(gData->g_image, imData->imageData, channel*height*width*sizeof(float), hipMemcpyHostToDevice);

  gData->g_im_deconv = &buffer[channel*height*width];

  gData->g_relBlur = &buffer[(channel+1*channelbatch)*height*width];

  gData->g_tmp = &buffer[(channel+2*channelbatch)*height*width];

  gData->g_psf = &buffer[(channel+3*channelbatch)*height*width];

  gData->g_flip_psf = &buffer[(channel+3*channelbatch)*height*width + filt_height*filt_width];


  hipMemcpy(gData->g_psf, imData->psfData, filt_height*filt_width*sizeof(float), hipMemcpyHostToDevice );
  hipMemcpy(gData->g_flip_psf, imData->flipPsfData, filt_height*filt_width*sizeof(float), hipMemcpyHostToDevice);
  hipDeviceSynchronize();
  return 1;
}

int cleanupHip(GpuData *gData) {
  hipFree(gData->g_image);
  return 1;
}

int resetDeconv(f32 *a, f32 val, uint size) {
   uint localThreads = localSize;
   uint globalThreads = (size + localThreads - 1) / localThreads;
   globalThreads *= localThreads;

   hipLaunchKernelGGL(SetVal,
		      dim3(globalThreads/localThreads),
		      dim3(localThreads),
		      0, 0,
		      a, val, size);

  
   return 1;
}

int transferResult(ImageData *imData, GpuData *gData) {
  hipMemcpy(imData->imageData, gData->g_image, imData->dims[0]*imData->dims[1]*imData->dims[2]*sizeof(float), hipMemcpyDeviceToHost);
  return 1;
}

int copyBuffers(f32 *dest, f32 *src, size_t n) {
  hipMemcpy(dest, src, n, hipMemcpyDeviceToDevice);
  return 1;
}

