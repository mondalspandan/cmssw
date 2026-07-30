#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/UnifiedParticleTransformerAK4Features.h"
#include "DataFormats/BTauReco/interface/UnifiedParticleTransformerAK4TagInfo.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "RecoBTag/ONNXRuntime/interface/tensor_configs.h"
#include "RecoBTag/ONNXRuntime/interface/tensor_fillers.h"
#include "onnx_model_editor.pb.h"

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {
using TagInfoCollection = std::vector<reco::UnifiedParticleTransformerAK4TagInfo>;
using JetTagCollection = reco::JetTagCollection;
using FloatArrays = std::vector<std::vector<float>>;

constexpr char kEncoderOutput[] = "/Encoder/layers.5/Add_1_output_0";
constexpr char kCLSOutput[] = "/CLS_EncoderLayer2/Add_1_output_0";
constexpr char kLinearOutput[] = "/Linear/Gemm_output_0";
constexpr char kIndexProduct[] = "UParTLatentTable";
constexpr char kCLSProduct[] = "UParTCLSValuesTable";
constexpr char kMLPProduct[] = "UParTMLPValuesTable";
constexpr char kEncoderProduct[] = "UParTEncoderValuesTable";
struct TensorResult {
  std::vector<float> values;
  std::vector<int64_t> shape;
};

struct LatentTables {
  std::vector<int> jetIdx;
  std::vector<uint32_t> clsOffset, clsLength;
  std::vector<uint32_t> mlpOffset, mlpLength;
  std::vector<uint32_t> encoderOffset, encoderLength;
  std::vector<float> clsValues, mlpValues, encoderValues;
};

std::vector<char> readBinary(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw cms::Exception("LatentFeaturesModel") << "Cannot open ONNX model " << path;
  return std::vector<char>((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
}

std::vector<char> addGraphOutputs(const std::string& path, const std::vector<std::string>& outputNames) {
  auto modelBytes = readBinary(path);
  physics_tools_nanoaod::latentfeatures::ModelProto model;
  if (!model.ParseFromArray(modelBytes.data(), static_cast<int>(modelBytes.size())))
    throw cms::Exception("LatentFeaturesModel") << "Cannot parse ONNX model " << path;

  auto* graph = model.mutable_graph();
  std::set<std::string> existing;
  for (const auto& output : graph->output())
    existing.insert(output.name());
  for (const auto& name : outputNames) {
    if (existing.count(name))
      continue;
    const auto info = std::find_if(graph->value_info().begin(), graph->value_info().end(), [&](const auto& valueInfo) {
      return valueInfo.name() == name;
    });
    if (info != graph->value_info().end()) {
      *graph->add_output() = *info;
    } else {
      // UParTAK4 V01 has no intermediate value_info entries. ONNX Runtime
      // resolves the shape after the in-memory FLOAT output is added.
      auto* output = graph->add_output();
      output->set_name(name);
      output->mutable_type()->mutable_tensor_type()->set_elem_type(1);  // ONNX FLOAT
    }
    existing.insert(name);
  }

  std::string serialized;
  if (!model.SerializeToString(&serialized))
    throw cms::Exception("LatentFeaturesModel") << "Cannot serialize modified ONNX model " << path;
  return std::vector<char>(serialized.begin(), serialized.end());
}

Ort::SessionOptions makeSessionOptions() {
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  return options;
}

TensorResult copyTensor(Ort::Value& output) {
  auto info = output.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();
  const auto size = info.GetElementCount();
  std::size_t shapeSize = 1;
  for (const auto dimension : shape) {
    if (dimension < 0)
      throw cms::Exception("LatentFeaturesOutput") << "ONNX output has an unresolved dimension";
    if (static_cast<std::size_t>(dimension) > std::numeric_limits<std::size_t>::max() / shapeSize)
      throw cms::Exception("LatentFeaturesOutput") << "ONNX output shape is too large";
    shapeSize *= static_cast<std::size_t>(dimension);
  }
  if (shapeSize != size)
    throw cms::Exception("LatentFeaturesOutput") << "ONNX output shape does not match its element count";
  const auto* data = output.GetTensorData<float>();
  return {std::vector<float>(data, data + size), std::move(shape)};
}

uint32_t checkedSize(std::size_t size, const char* name) {
  if (size > std::numeric_limits<uint32_t>::max())
    throw cms::Exception("LatentFeaturesOutput") << name << " exceeds the uint32_t FlatTable index range";
  return static_cast<uint32_t>(size);
}

void appendValues(const TensorResult& tensor,
                  std::vector<float>& values,
                  uint32_t& offset,
                  uint32_t& length,
                  const char* name) {
  offset = checkedSize(values.size(), name);
  if (tensor.values.empty()) {
    length = 0;
    return;
  }
  values.insert(values.end(), tensor.values.begin(), tensor.values.end());
  length = checkedSize(tensor.values.size(), name);
  checkedSize(values.size(), name);
}

std::size_t validateFixedVector(const TensorResult& tensor, const char* name) {
  if (tensor.shape.empty() || tensor.shape.back() <= 0)
    throw cms::Exception("LatentFeaturesOutput") << name << " output has no resolved final dimension";
  std::size_t leading = 1;
  for (std::size_t i = 0; i + 1 < tensor.shape.size(); ++i) {
    if (tensor.shape[i] <= 0 || static_cast<std::size_t>(tensor.shape[i]) > std::numeric_limits<std::size_t>::max() / leading)
      throw cms::Exception("LatentFeaturesOutput") << name << " output has an invalid shape";
    leading *= static_cast<std::size_t>(tensor.shape[i]);
  }
  const auto width = static_cast<std::size_t>(tensor.shape.back());
  if (leading != 1 || tensor.values.size() != width)
    throw cms::Exception("LatentFeaturesOutput") << name << " output is not one fixed-length vector: shape "
                                                   << tensor.shape.size() << "D with " << tensor.values.size()
                                                   << " values";
  return width;
}

void validateEncoder(const TensorResult& tensor) {
  if (tensor.shape.size() < 2 || tensor.shape.front() != 1 || tensor.shape.back() <= 0)
    throw cms::Exception("LatentFeaturesOutput") << "Expected encoder shape [1, ..., width]";
  const auto width = static_cast<std::size_t>(tensor.shape.back());
  if (tensor.values.size() % width != 0)
    throw cms::Exception("LatentFeaturesOutput") << "Final particle encoder output is not divisible by its final dimension";
}

class LatentFeaturesSession {
public:
  LatentFeaturesSession(const std::string& modelPath, const std::vector<std::string>& outputNames)
      : env_(ORT_LOGGING_LEVEL_WARNING, "LatentFeatures"),
        model_(addGraphOutputs(modelPath, outputNames)),
        session_(env_, model_.data(), model_.size(), makeSessionOptions()) {}

  std::vector<TensorResult> run(const std::vector<std::string>& inputNames,
                                const FloatArrays& data,
                                const std::vector<std::vector<int64_t>>& shapes,
                                const std::vector<std::string>& outputNames) const {
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> inputs;
    inputs.reserve(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
      inputs.emplace_back(Ort::Value::CreateTensor<float>(
          memoryInfo, const_cast<float*>(data[i].data()), data[i].size(), shapes[i].data(), shapes[i].size()));
    }

    std::vector<const char*> inputNamesC, outputNamesC;
    for (const auto& name : inputNames)
      inputNamesC.push_back(name.c_str());
    for (const auto& name : outputNames)
      outputNamesC.push_back(name.c_str());

    auto outputs = session_.Run(Ort::RunOptions{nullptr},
                                inputNamesC.data(),
                                inputs.data(),
                                inputs.size(),
                                outputNamesC.data(),
                                outputNamesC.size());
    std::vector<TensorResult> result;
    result.reserve(outputs.size());
    for (auto& output : outputs)
      result.emplace_back(copyTensor(output));
    return result;
  }

private:
  Ort::Env env_;
  std::vector<char> model_;
  mutable Ort::Session session_;
};
}  // namespace

class LatentFeaturesJetTagsProducer : public edm::stream::EDProducer<edm::GlobalCache<LatentFeaturesSession>> {
public:
  explicit LatentFeaturesJetTagsProducer(const edm::ParameterSet&, const LatentFeaturesSession*);
  ~LatentFeaturesJetTagsProducer() override = default;

  static std::unique_ptr<LatentFeaturesSession> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(const LatentFeaturesSession*) {}
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  void prepareInputs(const btagbtvdeep::UnifiedParticleTransformerAK4Features&);
  void putLatentTables(edm::Event&, LatentTables&) const;

  const edm::EDGetTokenT<TagInfoCollection> src_;
  const std::vector<std::string> flavNames_;
  const std::vector<std::string> inputNames_;
  const bool useDynamicAxes_;
  const bool saveCLS_;
  const bool saveMLP_;
  const bool saveInputEncoder_;
  const StringCutObjectSelector<reco::Jet> jetCut_;
  unsigned nCpf_ = 1, nLt_ = 1, nNpf_ = 1, nSv_ = 1;
  FloatArrays data_;
  std::vector<std::vector<int64_t>> inputShapes_;
  std::vector<std::string> outputNames_;
};

LatentFeaturesJetTagsProducer::LatentFeaturesJetTagsProducer(const edm::ParameterSet& config,
                                                             const LatentFeaturesSession*)
    : src_(consumes<TagInfoCollection>(config.getParameter<edm::InputTag>("src"))),
      flavNames_(config.getParameter<std::vector<std::string>>("flav_names")),
      inputNames_(config.getParameter<std::vector<std::string>>("input_names")),
      useDynamicAxes_(config.getParameter<edm::FileInPath>("model_path").fullPath().find("V01") != std::string::npos),
      saveCLS_(config.getParameter<bool>("CLS")),
      saveMLP_(config.getParameter<bool>("MLP")),
      saveInputEncoder_(config.getParameter<bool>("InputEncoder")),
      jetCut_(config.getParameter<std::string>("jet_cut")) {
  outputNames_.emplace_back("softmax");
  if (saveInputEncoder_)
    outputNames_.emplace_back(kEncoderOutput);
  if (saveCLS_)
    outputNames_.emplace_back(kCLSOutput);
  if (saveMLP_)
    outputNames_.emplace_back(kLinearOutput);

  for (const auto& name : flavNames_)
    produces<JetTagCollection>(name);
  produces<nanoaod::FlatTable>(kIndexProduct);
  if (saveCLS_)
    produces<nanoaod::FlatTable>(kCLSProduct);
  if (saveMLP_)
    produces<nanoaod::FlatTable>(kMLPProduct);
  if (saveInputEncoder_)
    produces<nanoaod::FlatTable>(kEncoderProduct);
}

std::unique_ptr<LatentFeaturesSession> LatentFeaturesJetTagsProducer::initializeGlobalCache(
    const edm::ParameterSet& config) {
  std::vector<std::string> outputs{"softmax"};
  if (config.getParameter<bool>("InputEncoder"))
    outputs.emplace_back(kEncoderOutput);
  if (config.getParameter<bool>("CLS"))
    outputs.emplace_back(kCLSOutput);
  if (config.getParameter<bool>("MLP"))
    outputs.emplace_back(kLinearOutput);
  return std::make_unique<LatentFeaturesSession>(config.getParameter<edm::FileInPath>("model_path").fullPath(), outputs);
}

void LatentFeaturesJetTagsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src");
  desc.add<std::vector<std::string>>("flav_names");
  desc.add<std::vector<std::string>>("input_names");
  desc.add<edm::FileInPath>("model_path");
  desc.add<bool>("CLS", false);
  desc.add<bool>("MLP", false);
  desc.add<bool>("InputEncoder", false);
  desc.add<std::string>("jet_cut", "pt > 15 && abs(eta) < 2.5");
  descriptions.add("LatentFeaturesJetTagsProducer", desc);
}

void LatentFeaturesJetTagsProducer::produce(edm::Event& event, const edm::EventSetup&) {
  edm::Handle<TagInfoCollection> tagInfos;
  event.getByToken(src_, tagInfos);

  std::vector<std::unique_ptr<JetTagCollection>> outputTags;
  if (!tagInfos->empty()) {
    auto refToProd = edm::makeRefToBaseProdFrom(tagInfos->front().jet(), event);
    for (std::size_t i = 0; i < flavNames_.size(); ++i)
      outputTags.emplace_back(std::make_unique<JetTagCollection>(refToProd));
  } else {
    for (std::size_t i = 0; i < flavNames_.size(); ++i)
      outputTags.emplace_back(std::make_unique<JetTagCollection>());
  }

  LatentTables tables;
  int selectedJetIndex = 0;
  for (const auto& tagInfo : *tagInfos) {
    std::vector<float> standardOutput(flavNames_.size(), -1.f);
    TensorResult encoder, cls, mlp;
    if (tagInfo.features().is_filled) {
      prepareInputs(tagInfo.features());
      auto tensors = globalCache()->run(inputNames_, data_, inputShapes_, outputNames_);
      std::size_t outputIndex = 0;
      standardOutput = std::move(tensors[outputIndex++].values);
      if (saveInputEncoder_)
        encoder = std::move(tensors[outputIndex++]);
      if (saveCLS_)
        cls = std::move(tensors[outputIndex++]);
      if (saveMLP_)
        mlp = std::move(tensors[outputIndex++]);
      if (standardOutput.size() != flavNames_.size())
        throw cms::Exception("LatentFeaturesOutput") << "Expected " << flavNames_.size()
                                                       << " standard UParT outputs, got " << standardOutput.size();
      if (saveCLS_)
        validateFixedVector(cls, "CLS");
      if (saveMLP_)
        validateFixedVector(mlp, "MLP");
      if (saveInputEncoder_)
        validateEncoder(encoder);
    }

    const auto& jetRef = tagInfo.jet();
    for (std::size_t i = 0; i < flavNames_.size(); ++i)
      (*outputTags[i])[jetRef] = standardOutput[i];

    const bool selected = jetRef.isNonnull() && jetCut_(*jetRef);
    if (selected) {
      tables.jetIdx.push_back(selectedJetIndex);
      if (saveCLS_)
        appendValues(cls, tables.clsValues, tables.clsOffset.emplace_back(), tables.clsLength.emplace_back(), "CLS");
      if (saveMLP_)
        appendValues(mlp, tables.mlpValues, tables.mlpOffset.emplace_back(), tables.mlpLength.emplace_back(), "MLP");
      if (saveInputEncoder_)
        appendValues(encoder,
                     tables.encoderValues,
                     tables.encoderOffset.emplace_back(),
                     tables.encoderLength.emplace_back(),
                     "encoder");
      ++selectedJetIndex;
    }
  }

  for (std::size_t i = 0; i < flavNames_.size(); ++i)
    event.put(std::move(outputTags[i]), flavNames_[i]);
  putLatentTables(event, tables);
}

void LatentFeaturesJetTagsProducer::putLatentTables(edm::Event& event, LatentTables& tables) const {
  auto index = std::make_unique<nanoaod::FlatTable>(tables.jetIdx.size(), "JetUParTLatent", false);
  index->addColumn<int>("jetIdx", tables.jetIdx, "Index of the corresponding Jet row");
  if (saveCLS_) {
    index->addColumn<uint32_t>("clsOffset", tables.clsOffset, "Offset into JetUParTCLSValues");
    index->addColumn<uint32_t>("clsLength", tables.clsLength, "Length in JetUParTCLSValues");
  }
  if (saveMLP_) {
    index->addColumn<uint32_t>("mlpOffset", tables.mlpOffset, "Offset into JetUParTMLPValues");
    index->addColumn<uint32_t>("mlpLength", tables.mlpLength, "Length in JetUParTMLPValues");
  }
  if (saveInputEncoder_) {
    index->addColumn<uint32_t>("encoderOffset", tables.encoderOffset, "Offset into JetUParTEncoderValues");
    index->addColumn<uint32_t>("encoderLength", tables.encoderLength, "Length in JetUParTEncoderValues");
  }
  event.put(std::move(index), kIndexProduct);

  if (saveCLS_) {
    auto values = std::make_unique<nanoaod::FlatTable>(tables.clsValues.size(), "JetUParTCLSValues", false);
    values->addColumn<float>("value", tables.clsValues, "Selected CLS encoder representation; width is read from ONNX Runtime shape metadata.", 10);
    event.put(std::move(values), kCLSProduct);
  }
  if (saveMLP_) {
    auto values = std::make_unique<nanoaod::FlatTable>(tables.mlpValues.size(), "JetUParTMLPValues", false);
    values->addColumn<float>("value", tables.mlpValues, "Selected pre-softmax Linear/Gemm representation; width is read from ONNX Runtime shape metadata.", 10);
    event.put(std::move(values), kMLPProduct);
  }
  if (saveInputEncoder_) {
    auto values = std::make_unique<nanoaod::FlatTable>(tables.encoderValues.size(), "JetUParTEncoderValues", false);
    values->addColumn<float>(
        "value", tables.encoderValues, "Selected per-token encoder representation, flattened in ONNX row-major order.", 10);
    event.put(std::move(values), kEncoderProduct);
  }
}

void LatentFeaturesJetTagsProducer::prepareInputs(
    const btagbtvdeep::UnifiedParticleTransformerAK4Features& features) {
  if (useDynamicAxes_) {
    nCpf_ = std::clamp<unsigned>(features.c_pf_features.size(), 1, UparT::n_cpf_accept);
    nLt_ = std::clamp<unsigned>(features.lt_features.size(), 1, UparT::n_lt_accept);
    nNpf_ = std::clamp<unsigned>(features.n_pf_features.size(), 1, UparT::n_npf_accept);
    nSv_ = std::clamp<unsigned>(features.sv_features.size(), 1, UparT::n_sv_accept);
  } else {
    nCpf_ = UparT::n_cpf_accept;
    nLt_ = UparT::n_lt_accept;
    nNpf_ = UparT::n_npf_accept;
    nSv_ = UparT::n_sv_accept;
  }

  data_.clear();
  inputShapes_.clear();
  const std::vector<unsigned> counts{nCpf_, nLt_, nNpf_, nSv_, nCpf_, nLt_, nNpf_, nSv_};
  for (unsigned i = 0; i < UparT::kEnd; ++i) {
    data_.emplace_back(counts[i] * UparT::N_InputFeatures.at(i), 0.f);
    inputShapes_.push_back({1, static_cast<int64_t>(counts[i]), static_cast<int64_t>(UparT::N_InputFeatures.at(i))});
  }

  const float* start = nullptr;
  const auto nCpf = std::min(features.c_pf_features.size(), static_cast<std::size_t>(nCpf_));
  const auto nLt = std::min(features.lt_features.size(), static_cast<std::size_t>(nLt_));
  const auto nNpf = std::min(features.n_pf_features.size(), static_cast<std::size_t>(nNpf_));
  const auto nSv = std::min(features.sv_features.size(), static_cast<std::size_t>(nSv_));
  btagbtvdeep::UParT_tensor_filler(data_, UparT::kChargedCandidates, features.c_pf_features, nCpf, start, 0);
  btagbtvdeep::UParT_tensor_filler(data_, UparT::kLostTracks, features.lt_features, nLt, start, 0);
  btagbtvdeep::UParT_tensor_filler(data_, UparT::kNeutralCandidates, features.n_pf_features, nNpf, start, 0);
  btagbtvdeep::UParT_tensor_filler(data_, UparT::kVertices, features.sv_features, nSv, start, 0);
  btagbtvdeep::UParT_tensor_filler(data_, UparT::kChargedCandidates4Vec, features.c_pf_features, nCpf, start, 0);
  btagbtvdeep::UParT_tensor_filler(data_, UparT::kLostTracks4Vec, features.lt_features, nLt, start, 0);
  btagbtvdeep::UParT_tensor_filler(data_, UparT::kNeutralCandidates4Vec, features.n_pf_features, nNpf, start, 0);
  btagbtvdeep::UParT_tensor_filler(data_, UparT::kVertices4Vec, features.sv_features, nSv, start, 0);
}

DEFINE_FWK_MODULE(LatentFeaturesJetTagsProducer);
