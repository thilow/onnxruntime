#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4244)
#pragma warning(disable : 4541)
#endif

#ifndef _WIN32
#define dynamic_cast static_cast
#endif
#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "DlSystem/ITensorFactory.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "DlSystem/DlError.hpp"
#ifdef _WIN32
#pragma warning(pop)
#endif

//#include "Toolbox/TelemetryLib/AsgTelemetry.h"
//#include "Toolbox/UtilLib/AsgLogging.h"
#include "SnpeLib.h"

#include <iostream>
#include <unordered_map>
#include <memory>

bool SnpeLib::IsSnpeAvailable()
{
    // fallback cpu should always be available:
    zdl::DlSystem::Runtime_t runtime = { zdl::DlSystem::Runtime_t::CPU_FLOAT32 };
    return zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime);
}

static std::string s_getRuntimeString(const zdl::DlSystem::Runtime_t& t) {
    std::unordered_map<zdl::DlSystem::Runtime_t, std::string> s_names;
    s_names[zdl::DlSystem::Runtime_t::AIP_FIXED8_TF] = "AIP_FIXED8_TF";
    s_names[zdl::DlSystem::Runtime_t::DSP_FIXED8_TF] = "DSP_FIXED8_TF";
    s_names[zdl::DlSystem::Runtime_t::GPU_FLOAT16] = "GPU_FLOAT16";
    s_names[zdl::DlSystem::Runtime_t::GPU_FLOAT32_16_HYBRID] = "GPU_FLOAT32_16_HYBRID";
    s_names[zdl::DlSystem::Runtime_t::CPU_FLOAT32] = "CPU_FLOAT32";
    if (s_names.find(t) != s_names.end()) {
        return s_names[t];
    }
    return "RUNTIME_UNKNOWN";
}

#ifndef _WIN32
#include <sys/system_properties.h>
/* Get device name
    NOTE : these properties can be queried via adb :
adb shell getprop ro.product.manufacturer
adb shell getprop ro.product.model
*/
static void s_device_get_make_and_model(std::string& make, std::string& model) {
    std::vector<char> man(PROP_VALUE_MAX + 1, 0);
    std::vector<char> mod(PROP_VALUE_MAX + 1, 0);
    /* A length 0 value indicates that the property is not defined */
    int man_len = __system_property_get("ro.product.manufacturer", man.data());
    int mod_len = __system_property_get("ro.product.model", mod.data());
    std::string manufacturer(man.data(), man.data() + man_len);
    std::string mmodel(mod.data(), mod.data() + mod_len);
    make = manufacturer;
    model = mmodel;
}

static bool s_device_uses_dsp_only() {
    std::string make, model;
    s_device_get_make_and_model(make, model);
    // enforce DSP only
    if (make == "Microsoft") {
        return true;
    }
    // Epsilon Selfhost LKG
    if (make == "oema0") {
        return true;
    }
    // Zeta EV2
    if (make == "oemc1" && model == "sf c1") {
        return true;
    }
    // Zeta EV1.2
    if (make == "QUALCOMM" && model == "oemc1") {
        return true;
    }
    // OnePlus 7
    if (make == "OnePlus" && model == "GM1903") {
        return true;
    }
    // OnePlus 7T
    if (make == "OnePlus" && model == "HD1903") {
        return true;
    }
    return false;
}

static bool s_device_must_not_use_dsp()
{
    std::string make, model;
    s_device_get_make_and_model(make, model);
    // do not use DSP
    // OnePlus 7Pro
    if (make == "OnePlus" && model == "GM1925") {
        return true;
    }
    return false;
}

#else
static bool s_device_uses_dsp_only() {
    return true;
}
static bool s_device_must_not_use_dsp() {
    return false;
}
#endif

static zdl::DlSystem::Runtime_t s_getPreferredRuntime(bool enforce_dsp)
{
    zdl::DlSystem::Runtime_t runtimes[] = { zdl::DlSystem::Runtime_t::DSP_FIXED8_TF,
                                            zdl::DlSystem::Runtime_t::AIP_FIXED8_TF,
                                            zdl::DlSystem::Runtime_t::GPU_FLOAT16,
                                            zdl::DlSystem::Runtime_t::GPU_FLOAT32_16_HYBRID,
                                            zdl::DlSystem::Runtime_t::CPU_FLOAT32 };
    static zdl::DlSystem::Version_t version = zdl::SNPE::SNPEFactory::getLibraryVersion();
    zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::CPU;
    //AsgTraceLogNote("SNPE Version %s", version.asString().c_str()); 

    bool ignore_dsp = s_device_must_not_use_dsp() | !enforce_dsp;
    bool ignore_others = s_device_uses_dsp_only() & enforce_dsp;
    int start = ignore_dsp * 2;
    int end = ignore_others ? 2 : sizeof(runtimes) / sizeof(*runtimes);

    if (ignore_others) {
        runtime = zdl::DlSystem::Runtime_t::DSP;
    }
    // start with skipping aip and dsp if specified. 
    for ( int i=start; i<end; ++i ) {
        //AsgTraceLogNote("testing runtime %d", (int)runtimes[i]);
        if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtimes[i])) {
            runtime = runtimes[i];
            break;
        }
    }
    //AsgTraceLogNote("using runtime %d", (int)runtime);
    return runtime;
}

std::string getSnpePreferredRuntimeString(bool enforce_dsp)
{
    return s_getRuntimeString(s_getPreferredRuntime(enforce_dsp));
}


class SnpeLibImpl : public SnpeLib
{
    zdl::DlSystem::Runtime_t _runtime;
public:
    /*! if false, dsp use is not necessary even if requested by given platform. Not used on Windows. */
    SnpeLibImpl(bool enforce_dsp)
        : _runtime(zdl::DlSystem::Runtime_t::CPU)
    {
#if defined(_WIN32)
        (void)enforce_dsp; // get rid of unused variable warning
#if !defined(_M_ARM64)
        _runtime = zdl::DlSystem::Runtime_t::CPU;
#else
    if (enforce_dsp) {
        // force DSP on ARM64 WIN32
        _runtime = zdl::DlSystem::Runtime_t::DSP;
    }
#endif
#else
        // ANDROID
        _runtime = s_getPreferredRuntime(enforce_dsp);
#endif
        //AsgTraceLogNote("PerceptionCore using runtime %s", s_getRuntimeString(_runtime).c_str());
    }
    ~SnpeLibImpl() override {}

    std::unique_ptr<zdl::SNPE::SNPE> InitializeSnpe(zdl::DlContainer::IDlContainer* container, const std::vector<std::string>* pOutputTensorNames = nullptr, const std::vector<std::string>* pInputTensorNames = nullptr)
    {
        zdl::SNPE::SNPEBuilder snpeBuilder(container);

        // use setOutputTensors instead, also try zdl::DlSystem::Runtime_t::AIP_FIXED8_TF
        //return snpeBuilder.setOutputLayers({}).setRuntimeProcessor(zdl::DlSystem::Runtime_t::DSP_FIXED8_TF).build();

        zdl::DlSystem::StringList outputTensorNames = {};
        if ((nullptr != pOutputTensorNames) && (pOutputTensorNames->size() != 0))
        {
            for (auto layerName : *pOutputTensorNames)
            {
                outputTensorNames.append(layerName.c_str());
            }
        }

        std::unique_ptr<zdl::SNPE::SNPE> snpe = snpeBuilder.setOutputTensors(outputTensorNames).setRuntimeProcessor(_runtime).build();

        _inputTensorMap.clear();
        _inputTensors.clear();
        if ((snpe != nullptr) && (pInputTensorNames != nullptr) && (pInputTensorNames->size() != 0))
        {
            _inputTensors.resize(pInputTensorNames->size());
            for (size_t i=0; i < pInputTensorNames->size(); ++i)
            {
                zdl::DlSystem::Optional<zdl::DlSystem::TensorShape> inputShape = snpe->getInputDimensions(pInputTensorNames->at(i).c_str());
                if (!inputShape)
                {
                    //::AsgTelemetryError("Snpe cannot get input shape for input name %s", pInputTensorNames->at(i).c_str());
                    _inputTensorMap.clear();
                    _inputTensors.clear();
                    return nullptr;
                }
                _inputTensors[i] = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(*inputShape);
                zdl::DlSystem::ITensor* inputTensor = _inputTensors[i].get();
                if (!inputTensor)
                {
                    //::AsgTelemetryError("Snpe cannot create ITensor");
                    _inputTensorMap.clear();
                    _inputTensors.clear();
                    return nullptr;
                }
                _inputTensorMap.add(pInputTensorNames->at(i).c_str(), inputTensor);
            }
        }

        return snpe;
    }

    bool Initialize(const char* dlcPath, const std::vector<std::string>* pOutputLayerNames = nullptr, const std::vector<std::string>* pInputLayerNames = nullptr)
    {
        std::unique_ptr<zdl::DlContainer::IDlContainer> container = zdl::DlContainer::IDlContainer::open(zdl::DlSystem::String(dlcPath));
        if (!container)
        {
            //::AsgTelemetryError("failed open %s container file", dlcPath);
            return false;
        }

        _snpe = InitializeSnpe(container.get(), pOutputLayerNames, pInputLayerNames);
        if (!_snpe)
        {
            //::AsgTelemetryError("failed to build snpe");
            return false;
        }

        return true;
    }

    bool Initialize(const unsigned char* dlcData, size_t size, const std::vector<std::string>* pOutputLayerNames = nullptr, const std::vector<std::string>* pInputLayerNames = nullptr)
    {
        std::unique_ptr<zdl::DlContainer::IDlContainer> container = zdl::DlContainer::IDlContainer::open(dlcData, size);
        if (container == nullptr)
        {
            //::AsgTelemetryError("failed open container buffer");
            return false;
        }

        _snpe = InitializeSnpe(container.get(), pOutputLayerNames, pInputLayerNames);
        if (!_snpe)
        {
            //::AsgTelemetryError("failed to build snpe %s", zdl::DlSystem::getLastErrorString());
            return false;
        }

        return true;
    }

    bool GetInputDimensions(int which, std::vector<int>& sizes) override
    {
        try {
            zdl::DlSystem::Optional<zdl::DlSystem::TensorShape> inputShape;
            if (which != 0) {
                zdl::DlSystem::Optional<zdl::DlSystem::StringList> pnames = _snpe->getInputTensorNames();
                if (!pnames) {
                    //::AsgTelemetryError("Snpe cannot get input names");
                    return false;
                }
                const zdl::DlSystem::StringList& names(*pnames);
                if (names.size() <= which) {
                    //::AsgTelemetryError("Snpe cannot find input %ul", which);
                    return false;
                }
                inputShape = _snpe->getInputDimensions(names.at(which));
            }
            else {
                inputShape = _snpe->getInputDimensions();
            }
            if (!inputShape) {
                //::AsgTelemetryError("Snpe cannot get input shape for input %ul", which);
                return false;
            }
            zdl::DlSystem::TensorShape shape(*inputShape);
            sizes.resize(shape.rank());
            for (size_t i = 0; i < shape.rank(); ++i ) {
                sizes[i] = (int) shape[i]; // todo: sizes should be of type size_t
            }
        }
        catch (...) {
            //::AsgTelemetryError("Snpe threw exception");
            return false;
        }
        return true;
    }

    bool GetOutputDimensions(int which, std::vector<int>& sizes) override
    {
        (void)which; (void)sizes;

        /*
        zdl::DlSystem::Optional<zdl::DlSystem::TensorShape> inputShape = _snpe->getOutputDimensions();
        zdl::DlSystem::StringList tensorNames = outputTensorMap.getTensorNames();
        zdl::DlSystem::ITensor* tensor = outputTensorMap.getTensor(tensorNames.at(tensorNames.size() - 1));
        */

        return true;
    }

    bool SnpeProcessMultipleOutput(const unsigned char* input, size_t inputSize, size_t output_number, unsigned char* outputs[], size_t outputSizes[]) override
    {
        try {
            zdl::DlSystem::Optional<zdl::DlSystem::TensorShape> inputShape = _snpe->getInputDimensions();
            if (!inputShape)
            {
                //::AsgTelemetryError("Snpe cannot get input shape");
                return false;
            }
            std::unique_ptr<zdl::DlSystem::ITensor> inputTensor = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(*inputShape);
            //std::unique_ptr<zdl::DlSystem::ITensor> inputTensor = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(*inputShape, input, inputSize);
            if (!inputTensor)
            {
                //::AsgTelemetryError("Snpe cannot create ITensor");
                return false;
            }
            // ensure size of the input buffer matches input shape buffer size
            if (inputTensor->getSize()*4 != inputSize) {
                //::AsgTelemetryError("Snpe input size incorrect: expected %d, given %d bytes", inputTensor->getSize()*4, inputSize);
                return false;
            }
            memcpy(inputTensor->begin().dataPointer(), input, inputSize);

            zdl::DlSystem::TensorMap outputTensorMap;
            bool result = _snpe->execute(inputTensor.get(), outputTensorMap);
            if (!result)
            {
                //::AsgTelemetryError("Snpe Error while executing the network.");
                return false;
            }
            if ( outputTensorMap.size() == 0 ) {
                return false;
            }

            zdl::DlSystem::StringList tensorNames = outputTensorMap.getTensorNames();

            for (size_t i=0; i < output_number; i++)
            {
                zdl::DlSystem::ITensor* tensor = outputTensorMap.getTensor(tensorNames.at(i));
                // ensure size of the output buffer matches output shape buffer size
                if ( tensor->getSize()*sizeof(float) > outputSizes[i] ) {
                    //::AsgTelemetryError("Snpe output size incorrect: output_layer: %s, expected %d, given %d bytes", tensorNames.at(i), tensor->getSize()*4, outputSizes[i]);
                    return false;
                }
                memcpy(outputs[i], tensor->cbegin().dataPointer(), tensor->getSize() * sizeof(float));
            }

            return true;
        }
        catch (...){
            //::AsgTelemetryError("Snpe threw exception");
            return false;
        }
    }


    bool SnpeProcess(const unsigned char* input, size_t inputSize, unsigned char* output, size_t outputSize) override
    {
        // Use SnpeProcessMultipleOutput with 1 output layer
        unsigned char* outputsArray[1];
        size_t outputSizesArray[1];
        outputsArray[0] = output;
        outputSizesArray[0] = outputSize;
        return SnpeProcessMultipleOutput(input, inputSize, 1, outputsArray, outputSizesArray);
    }


    bool SnpeProcessMultipleInputsMultipleOutputs(const unsigned char** inputs, const size_t* inputSizes, size_t input_number,
                                                  unsigned char** outputs, const size_t* outputSizes, size_t output_number) override
    {
        try {
            if (input_number != _inputTensors.size())
            {
                //::AsgTelemetryError("Snpe number of inputs doesn't match");
                return false;
            }
            for (size_t i=0; i < input_number; ++i)
            {
                zdl::DlSystem::ITensor* inputTensor = _inputTensors[i].get();
                // ensure size of the input buffer matches input shape buffer size
                if (inputTensor->getSize()*4 != inputSizes[i]) {
                    //::AsgTelemetryError("Snpe input size incorrect: expected %d, given %d bytes", inputTensor->getSize()*4, inputSizes[i]);
                    return false;
                }
                memcpy(inputTensor->begin().dataPointer(), inputs[i], inputSizes[i]);
            }
            zdl::DlSystem::TensorMap outputTensorMap;
            bool result = _snpe->execute(_inputTensorMap, outputTensorMap);
            if (!result)
            {
                //::AsgTelemetryError("Snpe Error while executing the network.");
                return false;
            }
            if ( outputTensorMap.size() == 0 ) {
                return false;
            }

            zdl::DlSystem::StringList tensorNames = outputTensorMap.getTensorNames();

            for (size_t i=0; i < output_number; i++)
            {
                zdl::DlSystem::ITensor* tensor = outputTensorMap.getTensor(tensorNames.at(i));
                // ensure size of the output buffer matches output shape buffer size
                if ( tensor->getSize()*sizeof(float) > outputSizes[i] ) {
                    //::AsgTelemetryError("Snpe output size incorrect: output_layer: %s, expected %d, given %d bytes", tensorNames.at(i), tensor->getSize()*4, outputSizes[i]);
                    return false;
                }
                memcpy(outputs[i], tensor->cbegin().dataPointer(), tensor->getSize() * sizeof(float));
            }

            return true;
        }
        catch (...){
            //::AsgTelemetryError("Snpe threw exception");
            return false;
        }
    }

private:
    std::unique_ptr<zdl::SNPE::SNPE> _snpe;
    std::vector<std::unique_ptr<zdl::DlSystem::ITensor>> _inputTensors;
    zdl::DlSystem::TensorMap _inputTensorMap;
};

std::unique_ptr<SnpeLib> SnpeLib::SnpeLibFactory(const char* dlcPath, const std::vector<std::string>* pOutputLayerNames, bool enforce_dsp, const std::vector<std::string>* pInputLayerNames)
{
    std::unique_ptr<SnpeLibImpl> object(new SnpeLibImpl(enforce_dsp));

    if (!object)
    {
        //::AsgTelemetryError("failed to make snpe library");
        return nullptr;
    }

    if (!object->Initialize(dlcPath, pOutputLayerNames, pInputLayerNames))
    {
        //::AsgTelemetryError("failed to initialize dlc from path");
        return nullptr;
    }

    return object;
}

std::unique_ptr<SnpeLib> SnpeLib::SnpeLibFactory(const unsigned char* dlcData, size_t size, const std::vector<std::string>* pOutputLayerNames, bool enforce_dsp, const std::vector<std::string>* pInputLayerNames)
{
    std::unique_ptr<SnpeLibImpl> object(new SnpeLibImpl(enforce_dsp));

    if (!object)
    {
        //::AsgTelemetryError("failed to make snpe library");
        return nullptr;
    }

    if (!object->Initialize(dlcData, size, pOutputLayerNames, pInputLayerNames))
    {
        //::AsgTelemetryError("failed to initialize dlc from buffer");
        return nullptr;
    }

    return object;
}
