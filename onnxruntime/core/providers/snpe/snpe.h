// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "SnpeLib.h"
#include "snpe_execution_provider.h"

namespace onnxruntime {
namespace contrib {
namespace snpe {
template <typename T>
class Snpe : public OpKernel {
 public:
  explicit Snpe(const OpKernelInfo& info) : OpKernel(info) {
    auto snpeEp = static_cast<const SNPEExecutionProvider*>(info.GetExecutionProvider());
    auto payload = info.GetAttrOrDefault<std::string>("payload", "");
    output_dims_ = info.GetAttrsOrDefault<int64_t>("output1_shape");
    bool enforeDsp = snpeEp->GetEnforceDsp();
    snpe_rt_ = SnpeLib::SnpeLibFactory(reinterpret_cast<const unsigned char*>(payload.c_str()), payload.length(), nullptr, enforeDsp);
  }

  Status Compute(OpKernelContext* context) const override {
    auto input_tensor = context->Input<Tensor>(0);
    auto input_data = input_tensor->DataRaw();
    size_t input_size = input_tensor->Shape().Size();

    TensorShape output_shape = TensorShape(output_dims_);
    auto output_tensor = context->Output(0, output_shape);
    auto output_data = output_tensor->MutableDataRaw();

    snpe_rt_->SnpeProcess(static_cast<const unsigned char*>(input_data), input_size * sizeof(T), static_cast<unsigned char*>(output_data), output_shape.Size() * sizeof(T));

    return Status::OK();
  }

 private:
  std::vector<int64_t> output_dims_;
  std::unique_ptr<SnpeLib> snpe_rt_;
  //std::vector<std::string> input_names_;
  //std::vector<std::string> output_names_;
};
}  // namespace snpe
}  // namespace contrib
}  // namespace onnxruntime
