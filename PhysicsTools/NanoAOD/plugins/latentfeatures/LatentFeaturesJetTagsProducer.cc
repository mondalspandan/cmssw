#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/BTauReco/interface/UnifiedParticleTransformerAK4Features.h"
#include "DataFormats/BTauReco/interface/UnifiedParticleTransformerAK4TagInfo.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "onnx_model_editor.pb.h"

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iterator>
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
constexpr char kCLSProduct[] = "UParTCLSTable";
constexpr char kMLPProduct[] = "UParTMLPTable";
constexpr char kEncoderProduct[] = "UParTEncodedInputsTable";
constexpr unsigned kCLSWidth = 192;
constexpr unsigned kLinearWidth = 24;

struct TensorResult {
  std::vector<float> values;
  std::vector<int64_t> shape;
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
    if (!existing.count(name))
      graph->add_output()->set_name(name);
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
  const auto* data = output.GetTensorData<float>();
  return {std::vector<float>(data, data + size), std::move(shape)};
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

    std::vector<const char*> inputNamesC;
    std::vector<const char*> outputNamesC;
    inputNamesC.reserve(inputNames.size());
    outputNamesC.reserve(outputNames.size());
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
  void makeInputs(const btagbtvdeep::UnifiedParticleTransformerAK4Features&);
  void putTable(edm::Event&, const std::string&, const std::string&, std::vector<int>&, std::vector<float>&) const;
  void putEncodedTable(edm::Event&, std::vector<int>&, std::vector<int>&, std::vector<float>&) const;

  const edm::EDGetTokenT<TagInfoCollection> src_;
  const std::vector<std::string> flavNames_;
  const std::vector<std::string> inputNames_;
  const bool useDynamicAxes_;
  const bool saveCLS_;
  const bool saveMLP_;
  const bool saveInputEncoder_;
  const StringCutObjectSelector<reco::Jet> jetCut_;

  enum InputIndexes {
    kChargedCandidates = 0,
    kLostTracks = 1,
    kNeutralCandidates = 2,
    kVertices = 3,
    kChargedCandidates4Vec = 4,
    kLostTracks4Vec = 5,
    kNeutralCandidates4Vec = 6,
    kVertices4Vec = 7
  };
  unsigned nCpf_ = 1;
  unsigned nLt_ = 1;
  unsigned nNpf_ = 1;
  unsigned nSv_ = 1;
  constexpr static unsigned nFeaturesCpf_ = 25;
  constexpr static unsigned nFeaturesLt_ = 18;
  constexpr static unsigned nFeaturesNpf_ = 8;
  constexpr static unsigned nFeaturesSv_ = 14;
  constexpr static unsigned nPairwiseFeatures_ = 4;
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
  outputNames_.push_back("softmax");
  if (saveInputEncoder_)
    outputNames_.push_back(kEncoderOutput);
  if (saveCLS_)
    outputNames_.push_back(kCLSOutput);
  if (saveMLP_)
    outputNames_.push_back(kLinearOutput);

  for (const auto& name : flavNames_)
    produces<JetTagCollection>(name);
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

  std::vector<int> clsJetIdx, mlpJetIdx, encodedJetIdx, encodedParticleIdx;
  std::vector<float> clsValues, mlpValues, encodedValues;
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
        throw cms::Exception("LatentFeaturesOutput") << "Expected " << flavNames_.size() << " standard UParT outputs, got "
                                             << standardOutput.size();
    }

    const auto& jetRef = tagInfo.jet();
    for (std::size_t i = 0; i < flavNames_.size(); ++i)
      (*outputTags[i])[jetRef] = standardOutput[i];

    const bool selected = jetRef.isNonnull() && jetCut_(*jetRef);
    if (selected && tagInfo.features().is_filled) {
      if (saveCLS_) {
        if (cls.values.size() != kCLSWidth)
          throw cms::Exception("LatentFeaturesOutput") << "Expected a 192-element CLS vector, got " << cls.values.size();
        for (float value : cls.values) {
          clsJetIdx.push_back(selectedJetIndex);
          clsValues.push_back(value);
        }
      }
      if (saveMLP_) {
        if (mlp.values.size() != kLinearWidth)
          throw cms::Exception("LatentFeaturesOutput") << "Expected a 24-element Linear vector, got " << mlp.values.size();
        for (float value : mlp.values) {
          mlpJetIdx.push_back(selectedJetIndex);
          mlpValues.push_back(value);
        }
      }
      if (saveInputEncoder_) {
        if (encoder.values.size() % kCLSWidth != 0)
          throw cms::Exception("LatentFeaturesOutput") << "Final particle encoder output is not divisible by 192";
        const auto nParticles = encoder.values.size() / kCLSWidth;
        for (std::size_t particle = 0; particle < nParticles; ++particle) {
          for (unsigned feature = 0; feature < kCLSWidth; ++feature) {
            encodedJetIdx.push_back(selectedJetIndex);
            encodedParticleIdx.push_back(static_cast<int>(particle));
            encodedValues.push_back(encoder.values[particle * kCLSWidth + feature]);
          }
        }
      }
    }
    if (selected)
      ++selectedJetIndex;
  }

  for (std::size_t i = 0; i < flavNames_.size(); ++i)
    event.put(std::move(outputTags[i]), flavNames_[i]);
  if (saveCLS_)
    putTable(event, kCLSProduct, "JetUParTCLS", clsJetIdx, clsValues);
  if (saveMLP_)
    putTable(event, kMLPProduct, "JetUParTMLP", mlpJetIdx, mlpValues);
  if (saveInputEncoder_)
    putEncodedTable(event, encodedJetIdx, encodedParticleIdx, encodedValues);
}

void LatentFeaturesJetTagsProducer::putTable(edm::Event& event,
                                                const std::string& product,
                                                const std::string& tableName,
                                                std::vector<int>& jetIndices,
                                                std::vector<float>& values) const {
  auto table = std::make_unique<nanoaod::FlatTable>(values.size(), tableName, false);
  table->addColumn<int>("jetIdx", jetIndices, "Index of the corresponding Jet row");
  table->addColumn<float>("value", values, "UParT intermediate value", 10);
  event.put(std::move(table), product);
}

void LatentFeaturesJetTagsProducer::putEncodedTable(edm::Event& event,
                                                        std::vector<int>& jetIndices,
                                                        std::vector<int>& particleIndices,
                                                        std::vector<float>& values) const {
  auto table = std::make_unique<nanoaod::FlatTable>(values.size(), "JetUParTEncodedInputs", false);
  table->addColumn<int>("jetIdx", jetIndices, "Index of the corresponding Jet row");
  table->addColumn<int>("particleIdx", particleIndices, "Particle index within the jet");
  table->addColumn<float>("value", values, "Final UParT particle encoder value", 10);
  event.put(std::move(table), kEncoderProduct);
}

void LatentFeaturesJetTagsProducer::prepareInputs(
    const btagbtvdeep::UnifiedParticleTransformerAK4Features& features) {
  if (useDynamicAxes_) {
    nCpf_ = std::clamp<unsigned>(features.c_pf_features.size(), 1, 29);
    nLt_ = std::clamp<unsigned>(features.lt_features.size(), 1, 5);
    nNpf_ = std::clamp<unsigned>(features.n_pf_features.size(), 1, 25);
    nSv_ = std::clamp<unsigned>(features.sv_features.size(), 1, 5);
  } else {
    nCpf_ = 29;
    nLt_ = 5;
    nNpf_ = 25;
    nSv_ = 5;
  }

  const std::vector<unsigned> sizes = {nCpf_ * nFeaturesCpf_, nLt_ * nFeaturesLt_, nNpf_ * nFeaturesNpf_,
                                       nSv_ * nFeaturesSv_, nCpf_ * nPairwiseFeatures_, nLt_ * nPairwiseFeatures_,
                                       nNpf_ * nPairwiseFeatures_, nSv_ * nPairwiseFeatures_};
  data_.clear();
  for (const auto size : sizes)
    data_.emplace_back(size, 0.f);
  inputShapes_ = {{1, nCpf_, nFeaturesCpf_},
                  {1, nLt_, nFeaturesLt_},
                  {1, nNpf_, nFeaturesNpf_},
                  {1, nSv_, nFeaturesSv_},
                  {1, nCpf_, nPairwiseFeatures_},
                  {1, nLt_, nPairwiseFeatures_},
                  {1, nNpf_, nPairwiseFeatures_},
                  {1, nSv_, nPairwiseFeatures_}};
  makeInputs(features);
}

void LatentFeaturesJetTagsProducer::makeInputs(
    const btagbtvdeep::UnifiedParticleTransformerAK4Features& features) {
  auto fill = [&](unsigned group, std::size_t index, const auto& values) {
    std::copy(values.begin(), values.end(), data_[group].begin() + index);
  };

  for (std::size_t i = 0; i < std::min(features.c_pf_features.size(), static_cast<std::size_t>(nCpf_)); ++i) {
    const auto& f = features.c_pf_features[i];
    fill(kChargedCandidates, i * nFeaturesCpf_, std::vector<float>{
                                                   f.btagPf_trackEtaRel, f.btagPf_trackPtRel, f.btagPf_trackPPar,
                                                   f.btagPf_trackDeltaR, f.btagPf_trackPParRatio, f.btagPf_trackSip2dVal,
                                                   f.btagPf_trackSip2dSig, f.btagPf_trackSip3dVal, f.btagPf_trackSip3dSig,
                                                   f.btagPf_trackJetDistVal, f.ptrel, f.drminsv, f.vtx_ass, f.puppiw,
                                                   f.chi2, f.quality, f.charge, f.dz, f.btagPf_trackDecayLen,
                                                   f.HadFrac, f.CaloFrac, f.pdgID, f.lostInnerHits,
                                                   f.numberOfPixelHits, f.numberOfStripHits});
    fill(kChargedCandidates4Vec, i * nPairwiseFeatures_,
         std::vector<float>{f.px, f.py, f.pz, f.e});
  }
  for (std::size_t i = 0; i < std::min(features.lt_features.size(), static_cast<std::size_t>(nLt_)); ++i) {
    const auto& f = features.lt_features[i];
    fill(kLostTracks, i * nFeaturesLt_, std::vector<float>{
                                              f.btagPf_trackEtaRel, f.btagPf_trackPtRel, f.btagPf_trackPPar,
                                              f.btagPf_trackDeltaR, f.btagPf_trackPParRatio, f.btagPf_trackSip2dVal,
                                              f.btagPf_trackSip2dSig, f.btagPf_trackSip3dVal, f.btagPf_trackSip3dSig,
                                              f.btagPf_trackJetDistVal, f.drminsv, f.charge, f.puppiw, f.chi2,
                                              f.quality, f.lostInnerHits, f.numberOfPixelHits, f.numberOfStripHits});
    fill(kLostTracks4Vec, i * nPairwiseFeatures_, std::vector<float>{f.pt, f.eta, f.phi, f.e});
  }
  for (std::size_t i = 0; i < std::min(features.n_pf_features.size(), static_cast<std::size_t>(nNpf_)); ++i) {
    const auto& f = features.n_pf_features[i];
    fill(kNeutralCandidates, i * nFeaturesNpf_,
         std::vector<float>{f.ptrel, f.etarel, f.phirel, f.deltaR, f.isGamma, f.hadFrac, f.drminsv, f.puppiw});
    fill(kNeutralCandidates4Vec, i * nPairwiseFeatures_,
         std::vector<float>{f.px, f.py, f.pz, f.e});
  }
  for (std::size_t i = 0; i < std::min(features.sv_features.size(), static_cast<std::size_t>(nSv_)); ++i) {
    const auto& f = features.sv_features[i];
    fill(kVertices, i * nFeaturesSv_, std::vector<float>{
                                           f.pt, f.deltaR, f.mass, f.etarel, f.phirel, f.ntracks, f.chi2, f.normchi2,
                                           f.dxy, f.dxysig, f.d3d, f.d3dsig, f.costhetasvpv, f.enratio});
    fill(kVertices4Vec, i * nPairwiseFeatures_, std::vector<float>{f.px, f.py, f.pz, f.e});
  }
}

DEFINE_FWK_MODULE(LatentFeaturesJetTagsProducer);
