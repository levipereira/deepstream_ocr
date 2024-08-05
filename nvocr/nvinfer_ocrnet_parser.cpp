#include <string>
#include <vector>
#include <iostream>
#include <locale>
#include <cstring>
#include "nvdsinfer_custom_impl.h"

using namespace std;
using std::string;
using std::vector;

static bool ocr_dict_ready = false;
std::vector<string> ocr_dict_table;

/* C-linkage to prevent name-mangling */
extern "C"
bool NvDsInferParseOCRNetCTC(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                NvDsInferNetworkInfo const &networkInfo, float classifierThreshold,
                                std::vector<NvDsInferAttribute> &attrList, std::string &attrString);

extern "C" 
bool NvDsInferParseOCRNetCTC(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                NvDsInferNetworkInfo const &networkInfo, float classifierThreshold,
                                std::vector<NvDsInferAttribute> &attrList, std::string &attrString)
{
    NvDsInferAttribute OCR_attr;

    if (!ocr_dict_ready) {
        static const char* hardcodedOCRDict[] = {
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
            "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
            "u", "v", "w", "x", "y", "z"
        };
        ocr_dict_table.emplace_back("CTCBlank");
        for (size_t i = 0; i < std::extent<decltype(hardcodedOCRDict)>::value; ++i) {
            ocr_dict_table.emplace_back(hardcodedOCRDict[i]);
        } 
        ocr_dict_ready = true;
    }
    

    if (outputLayersInfo.size() != 3)
    {
        std::cerr << "Mismatch in the number of output buffers."
                  << "Expected 3 output buffers, detected in the network: "
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    auto layerFinder = [&outputLayersInfo](const std::string &name)
        -> const NvDsInferLayerInfo *{
        for (auto &layer : outputLayersInfo) {
            if (layer.layerName && name == layer.layerName) {
                return &layer;
            }
        }
        return nullptr;
    };

    const NvDsInferLayerInfo *output_id = layerFinder("output_id");
    const NvDsInferLayerInfo *output_prob = layerFinder("output_prob");
    const NvDsInferLayerInfo *_798 = layerFinder("798");


    if (!output_id || !output_prob || !_798 ) {
        if (!output_id) {
            std::cerr << "  - output_id: Missing or unsupported data type." << std::endl;
        }

        if (!output_prob) {
            std::cerr << "  - output_prob: Missing or unsupported data type." << std::endl;
        }

        if (!_798) {
            std::cerr << "  - 798: Missing or unsupported data type." << std::endl;
        }
        return false;
    }

    if(output_id->inferDims.numDims != 1U) {
        std::cerr << "Network output_id dims is : " <<
            output_id->inferDims.numDims << " expect is 1"<< std::endl;
        return false;
    }
    if(output_prob->inferDims.numDims != 1U) {
        std::cerr << "Network output_prob dims is : " <<
            output_prob->inferDims.numDims << " expect is 1"<< std::endl;
        return false;
    }
    if(_798->inferDims.numDims != 1U) {
        std::cerr << "Network 798 dims is : " <<
            _798->inferDims.numDims << " expect is 1"<< std::endl;
        return false;
    }

    int batch_size = 1;
    int output_len = output_prob->inferDims.d[0];

    //std::cout << "Batch size: " << batch_size << std::endl;
    //std::cout << "Output length: " << output_len << std::endl;
    //std::cout << "networkInfo.width: " << networkInfo.width << std::endl;
    
    std::vector<std::pair<std::string, float>> temp_de_texts;

    int *output_id_data = reinterpret_cast<int*>(output_id->buffer);
    float *output_prob_data = reinterpret_cast<float*>(output_prob->buffer);

    for(int batch_idx = 0; batch_idx < batch_size; ++batch_idx)
        {
            int b_offset = batch_idx * output_len; 
            int prev = output_id_data[b_offset];
            std::vector<int> temp_seq_id = {prev};
            std::vector<float> temp_seq_prob = {output_prob_data[b_offset]};
            for(int i = 1 ; i < output_len; ++i)
            {
                if (output_id_data[b_offset + i] != prev)
                {
                    temp_seq_id.push_back(output_id_data[b_offset + i]);
                    temp_seq_prob.push_back(output_prob_data[b_offset + i]);
                    prev = output_id_data[b_offset + i];
                }
            }
            std::string de_text = "";
            float prob = 1.0;
            for(size_t i = 0; i < temp_seq_id.size(); ++i)
            {
                if (temp_seq_id[i] != 0)
                {
                    if (temp_seq_id[i] <= static_cast<int>(ocr_dict_table.size()) - 1)
                    {
                        de_text += ocr_dict_table[temp_seq_id[i]];
                        prob *= temp_seq_prob[i];
                    }
                    else
                    {
                        std::cerr << "[ERROR] Character dict is not compatible with OCRNet TRT engine." << std::endl;
                    }
                }
            }
            temp_de_texts.emplace_back(std::make_pair(de_text, prob));
        }

    attrString = "";
    for (const auto& temp_text : temp_de_texts) {
        if (temp_text.second >= classifierThreshold) {
            attrString += temp_text.first;
        }
        //std::cout << "Decoded text: " << temp_text.first << ", Probability: " << temp_text.second <<  ", Threshold: " << classifierThreshold << std::endl;
    }

    OCR_attr.attributeIndex = 0;
    OCR_attr.attributeValue = 1;
    OCR_attr.attributeLabel = strdup(attrString.c_str()); 
    OCR_attr.attributeConfidence = 1.0;
    
    for (const auto& temp_text : temp_de_texts) {
        OCR_attr.attributeConfidence *= temp_text.second;
    }

    std::cout << "attributeIndex: " << OCR_attr.attributeIndex << std::endl;
    std::cout << "attributeValue: " << OCR_attr.attributeValue << std::endl;
    std::cout << "attributeLabel: " << OCR_attr.attributeLabel << std::endl;
    std::cout << "attributeConfidence: " << OCR_attr.attributeConfidence << std::endl;

    attrList.push_back(OCR_attr);

    return true;
}

CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferParseOCRNetCTC);