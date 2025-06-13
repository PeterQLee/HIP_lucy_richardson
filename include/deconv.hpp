#include "gpuOps.hpp"


void lucy_richardson(ImageData *imData, int num_iter, f32 epsilon , bool clip = true, bool flag_denom_filter = false, f32 denom_filter = 0.0, uint channelbatch = 1);
