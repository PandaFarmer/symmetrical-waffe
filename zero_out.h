#ifndef TENSORFLOW_KERNELS_ZERO_OUT_OP_H_
#define TENSORFLOW_KERNELS_ZERO_OUT_OP_H_

namespace tensorflow {

namespace functor {

// Generic helper functor for the ZeroOut Op.
template <typename Device>
struct ZeroOutFunctor;

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_ZERO_OUT_OP_H_