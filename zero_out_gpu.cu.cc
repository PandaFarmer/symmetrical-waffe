#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "zero_out.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"


namespace tensorflow {

namespace functor {

using GPUDevice = Eigen::GpuDevice;

__global__ void ZeroOutKernel(const float* in, float* out, const int N) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    if (i == 0) {
      out[i] = in[i];
    } else {
      out[i] = 0;
    }
  }
}

template <>
struct ZeroOutFunctor<GPUDevice> {
  void operator()(const GPUDevice& d,
          typename TTypes<float>::ConstFlat input,
          typename TTypes<float>::Flat output,
          const int N) {
    // How to compute the optimal block count and threads per block?
    // tensorflow/core/util/cuda_kernel_helper.h isn;t included in the binary
    // distribution
    ZeroOutKernel<<<32, 256, 0, d.stream()>>>(input.data(), output.data(), N);
  }
};

template struct ZeroOutFunctor<GPUDevice>;  
} // namespace functor 
} // namespace tensorflow
#endif // GOOGLE_CUDA