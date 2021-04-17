// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "snpe.h"

namespace onnxruntime {
namespace contrib {
namespace snpe {

template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Snpe,                                                       \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kSnpeExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Snpe<T>);

REGISTER_KERNEL_TYPED(uint8_t)
REGISTER_KERNEL_TYPED(float)
}  // namespace snpe
}  // namespace contrib
}  // namespace onnxruntime
