#include "gpuOps.hpp"

/// imData: input image data structure.
/// num_iter: number of iterations
/// epsilon: value added to denominator to prevent divide by 0
/// clip: clips range from -1 to 1
/// flag_denom_filter: indicates if "filter_epsilon" strategy is active
/// denom_filter: equivalent to "filter_epsilon" in python implementation
/// channelbatch: grouping of channels during gpu processing for saturating bandwidth of the device
/// output: if null, result is written to the input buffer. otherwise copied to the output pointer
void lucy_richardson(ImageData *imData, int num_iter, f32 epsilon , bool clip = true, bool flag_denom_filter = false, f32 denom_filter = 0.0, uint channelbatch = 1, f32 *output = NULL);

