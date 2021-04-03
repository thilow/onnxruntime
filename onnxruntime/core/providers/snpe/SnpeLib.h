#pragma once

#include <memory>
#include <string>
#include <vector>

class SnpeLib
{
public:
    virtual ~SnpeLib() {}

    /*!
     * function to see if SNPE is available without need for any network data
     * @return true if SNPE is supported on at least CPU
     */
    static bool IsSnpeAvailable();

    /*!
     * function to see if get the preferred runtime on the given platform
     * @return "RUNTIME_UNKNOWN" if unsupported or not handled by linked SNPE API. Other strings
     *   according to snpe docs
     */
    static std::string GetSnpePreferredRuntimeString(bool enforce_dsp = true);

    static std::unique_ptr<SnpeLib> SnpeLibFactory(const char* dlcPath, const std::vector<std::string>* pOutputLayerNames = nullptr, bool enforce_dsp = true, const std::vector<std::string>* pInputLayerNames = nullptr);
    static std::unique_ptr<SnpeLib> SnpeLibFactory(const unsigned char* dlcData, size_t size, const std::vector<std::string>* pOutputLayerNames = nullptr, bool enforce_dsp = true, const std::vector<std::string>* pInputLayerNames = nullptr);

    virtual bool SnpeProcess(const unsigned char* input, size_t inputSize, unsigned char* output, size_t outputSize) = 0;
    virtual bool SnpeProcessMultipleOutput(const unsigned char* input, size_t inputSize, size_t output_number, unsigned char* outputs[], size_t outputSizes[]) = 0;
    virtual bool SnpeProcessMultipleInputsMultipleOutputs(const unsigned char** inputs, const size_t* inputSizes, size_t input_number,
                                                          unsigned char** outputs, const size_t* outputSizes, size_t output_number) = 0;

    virtual bool GetInputDimensions(int which, std::vector<int>& shape) = 0;
    virtual bool GetOutputDimensions(int which, std::vector<int>& shape) = 0;
};
