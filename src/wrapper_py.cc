#include "deconv.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
 
py::array_t<float, py::array::c_style> run_LR_deconv(py::array_t<float, py::array::c_style | py::array::forcecast> image,
		   py::array_t<float, py::array::c_style | py::array::forcecast> psf,
		   int num_iter,
		   bool clip = true,
		   py::object filter_epsilon = py::none(),
		   uint channelbatch = 1
) {

  // Check dimensions
  ssize_t imdim = image.ndim();
  size_t channels, height, width;
  if (imdim == 2) {
    channels = 1;
    height =  image.shape(0);
    width =  image.shape(1);
  }
  
  else if (imdim == 3){
    channels = image.shape(0);
    height =  image.shape(1);
    width =  image.shape(2);
  }

  else {
    throw std::invalid_argument("Error: Image required to have 2 or 3 dimensions\n");
  }
  

  ssize_t psfdim = psf.ndim();
  size_t psfheight, psfwidth;
  if (psfdim == 2) {
    psfheight = psf.shape(0);
    psfwidth = psf.shape(1);
  }
  else {
    throw std::invalid_argument("Error: PSF required to have 2 dimensions\n");
  }

  if (psfheight %2 ==0 || psfwidth%2==0) {
    throw std::invalid_argument("Error: PSF dimensions must be odd\n");
  }

  if (psfheight > height || psfwidth > width) {
    throw std::invalid_argument("Error: PSF dimension is larger than the input image\n");
  }
  
  // Allocate buffers
  ImageData imdata;
  imdata.dims[0] = channels; imdata.dims[1] = height; imdata.dims[2] = width;
  imdata.imageData = (f32*) image.data(0);


  imdata.psfData = (f32*) psf.data(0);
  imdata.psfdims[0] = psfheight; imdata.psfdims[1] = psfwidth;
  // Make flipped psf
  imdata.flipPsfData = (f32*) malloc(sizeof(float)*psfheight*psfwidth);

  for (int i=0;i<psfheight;i++) {
    for (int j=0;j<psfwidth;j++) {
      imdata.flipPsfData[ i*psfwidth + j] = imdata.psfData[ (psfheight - 1 - i)*psfwidth + (psfwidth - 1 - j)];
    }
  }

  f32 eps = 1e-12;
  f32 denom_filter = 0;
  bool flag_denom_filter = false;
  if (!filter_epsilon.is(py::none())) {
    flag_denom_filter = true;
    denom_filter = py::cast<float>(filter_epsilon);
    
  }

  //Allocate output buffer.
  py::array_t<float, py::array::c_style> output = py::array_t<float>({channels, height, width});
  
  lucy_richardson(&imdata, num_iter, eps, clip, flag_denom_filter, denom_filter, channelbatch, (f32*) output.data(0));

  free(imdata.flipPsfData);
  
  // TODO: check ownership rules.
  return output;
}



PYBIND11_MODULE(LR_GPU_wrapper, m) {
    m.doc() = R"pbdoc(
        GPU Lucy richardson through HIP
        -----------------------
           
    )pbdoc";
    
    m.def("run_LR_deconv", &run_LR_deconv, R"pbdoc(
        -LucyRichardson deconvolution-

image: shape [channel, height, width] if three dimensional, or [height, width] if two dimensional. image must be a float32 array that is C contiguous.

psf: a two dimensional point spread function. If image has three dimensions, the deconvolution is applied onto each channel independently.

num_iter: the number of iterations the deconvolution is ran for.

clip : Controls if values are bounded between -1 and 1

filter_epsilon: Value below which intermediate results become 0 to avoid division by small numbers.

channelbatch: number of channels to concurrently process.

---------------

Returns: result ndarray with shape [channel, height, width]
)pbdoc", py::arg("image"), py::arg("psf"), py::arg("num_iter"), py::arg("clip") = true, py::arg("filter_epsilon") = py::none(), py::arg("channelbatch")=1,
	   py::return_value_policy::take_ownership);

    m.attr("__version__") = "0.1";

}
