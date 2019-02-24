#define EIGEN_USE_THREADS
#include "zero_out.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

REGISTER_OP("ZeroOut")
.Input("to_zero: float")
.Output("zeroed: float")
.Doc(R"doc(
Zeros all elements of the tensor except the first.
zeroed: A Tensor.
  output[0] = input[0]
  output[1:N] = 0
)doc");;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename Device>
struct ZeroOutFunctor {
  void operator()(const Device& d,
          typename TTypes<float>::ConstFlat input,
          typename TTypes<float>::Flat output,
          const int N);
};

template <>
struct ZeroOutFunctor<CPUDevice> {
  void operator()(const CPUDevice& d,
          typename TTypes<float>::ConstFlat input,
          typename TTypes<float>::Flat output,
          const int N) {
    for (int i = 1; i < N; i++) {
      output(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output(0) = input(0);
  }
};
} // namespace functor    

template <typename Device>
class ZeroOutOp : public OpKernel {
public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                             &output_tensor));

    auto output = output_tensor->template flat<float>();
    const int N = input.size();
    functor::ZeroOutFunctor<Device>()(context->eigen_device<Device>(),
                      input, output, N);
  }
};

REGISTER_KERNEL_BUILDER(Name("ZeroOut")                 \
            .Device(DEVICE_CPU),        \
            ZeroOutOp<CPUDevice>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("ZeroOut")                 \
            .Device(DEVICE_GPU),        \
            ZeroOutOp<GPUDevice>);
#endif // GOOGLE_CUDA
} // namespace tensoroflow