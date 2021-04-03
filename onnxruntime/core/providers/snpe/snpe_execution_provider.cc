// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "snpe_execution_provider.h"

#include "core/framework/allocatormgr.h"
#include "core/framework/compute_capability.h"
#include "core/graph/graph_viewer.h"
#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {

constexpr const char* SNPE = "SNPE";

SNPEExecutionProvider::SNPEExecutionProvider(bool enforce_dsp)
    :IExecutionProvider{onnxruntime::kSnpeExecutionProvider}, enforce_dsp_(enforce_dsp) {
  AllocatorCreationInfo device_info(
      [](int) {
        return onnxruntime::make_unique<CPUAllocator>(OrtMemoryInfo(SNPE, OrtAllocatorType::OrtDeviceAllocator));
      });

  InsertAllocator(CreateAllocator(device_info));
}

SNPEExecutionProvider::~SNPEExecutionProvider() {}

std::vector<std::unique_ptr<ComputeCapability>>
SNPEExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                       const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;

  //const auto& logger = *GetLogger();
  // the model should has only 1 node with type PerceptionCoreNode, otherwise, report error
  std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::make_unique<IndexedSubGraph>();
  for (auto& node_index : graph.GetNodesInTopologicalOrder()) {
    sub_graph->nodes.push_back(graph.GetNode(node_index)->Index());
  }

  auto meta_def = onnxruntime::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
  meta_def->name = "snpe_container";
  meta_def->domain = kMSDomain;
  meta_def->since_version = 1;
  meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;

  const auto& graph_input_list = graph.GetInputs();
  for (const auto& input : graph_input_list) {
    meta_def->inputs.push_back(input->Name());
  }

  const auto& graph_output_list = graph.GetOutputs();
  for (const auto& output : graph_output_list) {
    meta_def->outputs.push_back(output->Name());
  }

  sub_graph->SetMetaDef(std::move(meta_def));

  result.push_back(onnxruntime::make_unique<ComputeCapability>(std::move(sub_graph)));

  return result;
}

// Another option: implement it in Contrib Op, but need to get the EP option like enforce_dsp from EP
// can get SNPE EP from OpKernelInfo
common::Status SNPEExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                              std::vector<NodeComputeInfo>& node_compute_funcs) {
  // There should be only one node
  const onnxruntime::GraphViewer& graph_viewer(fused_nodes_and_graphs.at(0).filtered_graph);
  auto perceptioncore_node = graph_viewer.GetNode(graph_viewer.GetNodesInTopologicalOrder().at(0));

  // Not sure whether the input_names & output_names are required or not
  const auto& input_defs = perceptioncore_node->InputDefs();
  for (size_t i = 0, end = input_defs.size(); i < end; ++i) {
    input_names_.push_back(input_defs[i]->Name());
  }

  const auto& output_defs = perceptioncore_node->OutputDefs();
  for (size_t i = 0, end = output_defs.size(); i < end; ++i) {
    output_names_.push_back(output_defs[i]->Name());
  }

  // if implement the snpe runtime code in Contrib OP, we can get the attributes in Op Construct from OpKernelInfo
  const auto& attributes = perceptioncore_node->GetAttributes();
  auto output_shape_attr = attributes.find("output_shape");
  if (output_shape_attr == attributes.end()) {
      //report error
    return Status::OK();
  }
  auto output_shpe = output_shape_attr->second.ints();
  output_size_ = static_cast<size_t>(std::accumulate(output_shpe.cbegin(), output_shpe.cend(), 1LL, std::multiplies<int64_t>()));

  auto payload_attr = attributes.find("payload");
  auto payload = payload_attr->second.s();
  snpe_rt_ = SnpeLib::SnpeLibFactory(reinterpret_cast<const unsigned char*>(payload.c_str()), payload.length(), nullptr, enforce_dsp_);

  NodeComputeInfo compute_info;
  compute_info.create_state_func = [&](ComputeContext* context, FunctionState* state) {
  std::unique_ptr<SnpeFuncState> snpe_func_state = onnxruntime::make_unique<SnpeFuncState>();
    snpe_func_state->output_size = output_size_;
    snpe_func_state->snpe_rt = std::move(snpe_rt_);
    copy(input_names_.begin(), input_names_.end(), back_inserter(snpe_func_state->input_names));
    copy(output_names_.begin(), output_names_.end(), back_inserter(snpe_func_state->output_names));
    *state = snpe_func_state.release();
    return 0;
  };

  compute_info.release_state_func = [](FunctionState state) {
    if (state)
      delete static_cast<SnpeFuncState*>(state);
  };

  // Create compute function
  compute_info.compute_func = [this](FunctionState state, const OrtCustomOpApi* api, OrtKernelContext* context) {
    Ort::CustomOpApi ort{*api};
    SnpeFuncState* snpe_func_sate = reinterpret_cast<SnpeFuncState*>(state);

    // Get SNPE model buffer from the 1st input
    const OrtValue* input_tensor = ort.KernelContext_GetInput(context, 0);
    auto* tensor_info = ort.GetTensorTypeAndShape(input_tensor);

    auto intput_shape = ort.GetTensorShape(tensor_info);
    //if (shape.empty()) error
    size_t inputSize = static_cast<size_t>(std::accumulate(intput_shape.cbegin(), intput_shape.cend(), 1LL, std::multiplies<int64_t>()));
    const void* inputBuffer = ort.GetTensorData<void>(input_tensor);

    std::vector<int64_t> output_shapes;
    output_shapes.push_back(snpe_func_sate->output_size);
    OrtValue* output_tensor = ort.KernelContext_GetOutput(context, 0, output_shapes.data(), output_shapes.size());
    auto output_tensor_ptr = ort.GetTensorMutableData<uint8_t>(output_tensor);

    snpe_func_sate->snpe_rt->SnpeProcess(static_cast<const unsigned char*>(inputBuffer), inputSize, static_cast<unsigned char*>(output_tensor_ptr), snpe_func_sate->output_size);
    return Status::OK();
  };

  node_compute_funcs.push_back(compute_info);

  return Status::OK();
}

}  // namespace onnxruntime
