#include "deconv.hpp"
#include <hip/hip_runtime.h>
#include <time.h>
#define DEBUG_RUNTIME 0
void lucy_richardson(ImageData *imData, int num_iter, f32 epsilon , bool clip, bool flag_denom_filter, f32 denom_filter, uint channelbatch, f32 *output) {

  GpuData gdata;
  #if DEBUG_RUNTIME
  clock_t before = clock();
#endif
  
  setupHip(imData, &gdata, channelbatch);
  
#if DEBUG_RUNTIME
  clock_t difference = clock() - before;
  int msec = difference * 1000000 / CLOCKS_PER_SEC;
  printf("Total setup time %d\n", msec);
  #endif

  int filt_height = imData->psfdims[0];
  int filt_width = imData->psfdims[1];
  int channels = imData->dims[0];
  int height = imData->dims[1];
  int width = imData->dims[2];

  
  #if DEBUG_RUNTIME
  before = clock();
  #endif


  f32 *slice;

  for (int c=0;c<channels;c+= channelbatch){

    uint cb = channelbatch;
    if (cb + c > channels ) {
      cb = channels - c;
    }
    slice = &gdata.g_image[c*height*width]; // pointer to slice of the image buffer to output result at.
    resetDeconv(gdata.g_im_deconv, 0.5, cb*height*width);
    
    for (int i=0;i<num_iter; i++) {

      callConvolution(gdata.g_im_deconv, gdata.g_flip_psf, gdata.g_relBlur, cb, height, width, filt_height, filt_width, epsilon); // Forgot to flip psf in convolution, technically callConvolution does correlation.

     
      if (flag_denom_filter) {
	callFiltDivision(slice, gdata.g_relBlur, denom_filter, cb*width*height);
      }
      else{
	callDivision(slice, gdata.g_relBlur, cb*width*height);
      }

      if (i < num_iter - 1) {

	callConvolution(gdata.g_relBlur, gdata.g_psf, gdata.g_tmp, cb, height, width, filt_height, filt_width, 0.0f);
      
	callMult(gdata.g_tmp, gdata.g_im_deconv, cb*width*height);
      }
      else {
	// Last  iteration, use image as final buffer

	callConvolution(gdata.g_relBlur, gdata.g_psf, slice, cb, height, width, filt_height, filt_width, 0.0f);
	callMult(gdata.g_im_deconv, slice, cb*width*height);
      }
   
    }

    if (clip) {
      callClip(slice, -1.0, 1.0, cb*width*height);
    }
  }
  
  #if DEBUG_RUNTIME
  difference = clock() - before;
  msec = difference * 1000000 / CLOCKS_PER_SEC;
  printf("Total restore time %d\n", msec);
    before = clock();
  #endif

  // output assumed as the input buffer.
  if (output == NULL) {
    transferResult(imData, &gdata);
  }
  // Transfer data to specific output buffer
  else {
    transferResult(output, &gdata, cb*width*height);
  }
  
  #if DEBUG_RUNTIME
  difference = clock() - before;
  msec = difference * 1000000 / CLOCKS_PER_SEC;
  printf("Total transfer time %d\n", msec);
  before = clock();
  #endif
  
  cleanupHip(&gdata);
  
  #if DEBUG_RUNTIME
  difference = clock() - before;
  msec = difference * 1000000 / CLOCKS_PER_SEC;
  printf("Total cleanup time %d\n", msec);
  #endif
}
