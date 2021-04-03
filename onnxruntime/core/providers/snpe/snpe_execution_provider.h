// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/execution_provider.h"
#include "core/providers/snpe/snpe_provider_factory.h"
#include "SnpeLib.h"

namespace onnxruntime {

struct SnpeFuncState {
  size_t output_size = 0;
  std::unique_ptr<SnpeLib> snpe_rt = nullptr;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
};

class SNPEExecutionProvider : public IExecutionProvider {
 public:
  SNPEExecutionProvider(bool enforce_dsp);
  virtual ~SNPEExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const onnxruntime::GraphViewer& graph,
      const std::vector<const KernelRegistry*>& ) const override;

  // we implement the Compile that takes FusedNodeAndGraph instances
  FusionStyle GetFusionStyle() const override { return FusionStyle::FilteredGraphViewer; }

  common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;


 private:
  bool enforce_dsp_;
  size_t output_size_;
  std::unique_ptr<SnpeLib> snpe_rt_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
};
}  // namespace onnxruntime
