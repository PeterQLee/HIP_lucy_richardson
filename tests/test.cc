#include "deconv.hpp"
#include "gpuOps.hpp"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
int main() {
  // Load data
  ImageData imData;
  int num_iter = 10;
  float epsilon = 1e-12;

  int height = 401;
  int width = 421;
  int filt_height = 29;
  int filt_width = 1;
  int channel = 3;
  int dims[3] = {channel,height, width};

  int filt_dims[2] = {filt_height, filt_width};

  for (int i=0;i<3;i++) {imData.dims[i] = dims[i];}
  imData.imageData = (f32*) malloc(sizeof(f32)*dims[0]*dims[1]*dims[2]);
  // Read in file.
  FILE *fp = fopen("rgbimg.bin","rb");
  size_t ret = fread( (void*) imData.imageData, sizeof(f32), dims[0]*dims[1]*dims[2], fp);
  printf("ret=%zu\n", ret);
  fclose(fp);


  f32 *restored = (f32*) malloc(sizeof (f32)*dims[0]*dims[1]*dims[2]);
  fp = fopen("rgbrestored.bin","rb");
  ret = fread( (void*) restored, sizeof(f32), dims[0]*dims[1]*dims[2], fp);
  fclose(fp);
  printf("ret=%zu\n", ret);


  for (int i=0;i<2;i++) {imData.psfdims[i] = filt_dims[i];}
  imData.psfData = (f32*) malloc(sizeof(f32)*filt_dims[0]*filt_dims[1]);
  imData.flipPsfData = (f32*) malloc(sizeof(f32)*filt_dims[0]*filt_dims[1]);

  // Read in file.
  fp = fopen("impulse.bin","rb");
  ret = fread( (void*) imData.psfData, sizeof(f32), filt_dims[0]*filt_dims[1], fp);
  fclose(fp);
  printf("ret=%zu\n", ret);

  for (int i=0;i<filt_dims[0];i++) {
    imData.flipPsfData[i] = imData.psfData[filt_dims[0]  - 1 - i];
    printf("%f ", imData.psfData[i]);
  }
  printf("\n");

  clock_t before = clock();
  lucy_richardson(&imData, num_iter, epsilon, true, false, 0.0, 1);
  clock_t difference = clock() - before;
  int msec = difference * 1000 / CLOCKS_PER_SEC;

  printf("Total restore time %d", msec);

  bool same = true;
  f32 maxr = 0.0f;
  f32 maxi = 0.0f;
  
  for (int i=0;i<channel*height*width;i++) {
    if (fabs(restored[i] - imData.imageData[i]) >1e-5 ){
      same = false;

    }
    if (maxr < restored[i]) maxr = restored[i];
    if (maxi < imData.imageData[i]) maxi = imData.imageData[i];
  }
  printf("MAx %f %f\n", maxr, maxi);
  if (!same) {
    printf("Not the same\n");
  }
}



