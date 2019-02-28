#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output_flat(0) = input(0);
  }
};

class SomeDimOp : public OpKernel {
 public:
  explicit SomeDimOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output_flat(0) = input(0);
  }
};

/*
InvalidArgumentError (see above for traceback): 
No OpKernel was registered to support Op 'CudnnRNN' with these attrs.  
Registered devices: [CPU,XLA_CPU], Registered kernels:
  <no registered kernels>

         [[node cu_dnngru_1/CudnnRNN 
         (defined at /home/iqiao/.local/lib/python3.6/site-packages/
         tensorflow/contrib/cudnn_rnn/python/ops/cudnn_rnn_ops.py:922)  
         = CudnnRNN[T=DT_FLOAT, 
         direction="unidirectional", 
         dropout=0, input_mode="linear_input", 
         is_training=true, rnn_mode="gru", seed=87654321, seed2=0]
         (cu_dnngru_1/transpose, cu_dnngru_1/ExpandDims_1, 
         cu_dnngru_1/Const_1, cu_dnngru_1/concat)]]
         */