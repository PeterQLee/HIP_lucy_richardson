#ifndef GPUOPS_H
#define GPUOPS_H
#include <stdlib.h>
typedef float f32;
typedef unsigned int uint;

typedef struct {
  int dims[3]; // Channel, height, width
  f32 *imageData;

  int psfdims[2];
  f32 *psfData;
  f32 *flipPsfData;
  
} ImageData;


typedef struct {
  f32 *g_image; //Image data an eventual output for output
  f32 *g_psf, *g_flip_psf; 
  f32 *g_im_deconv, *g_relBlur, *g_tmp;
} GpuData;


int callConvolution(f32 *gIn, f32 *mask, f32 *gOut, uint channelbatch, uint height, uint width,  uint filt_height, uint filt_width, f32 epsilon);
int callDivision(f32 *num, f32 *den, uint size);
int callFiltDivision(f32 *num, f32 *den, f32 thresh, uint size);
int callMult(f32 *a, f32 *b, uint size);
int callClip(f32 *a, f32 lower, f32 upper, uint size);
int setupHip(ImageData *im_data, GpuData *gData, uint channelbatch);
int cleanupHip(GpuData *gData);
int transferResult(ImageData *imData, GpuData *gData);
int transferResult(f32 *output, GpuData *gData, size_t n);
int copyBuffers(f32 *a, f32 *b, size_t n);
int resetDeconv(f32 *a, f32 val, uint size);
#endif
