#include <vector>
#include <string>
#include "Network.hpp"

Network* createNN(const std::vector<int> &layers);
Network* clone(Network* net);
std::vector<float> getWeights(Network* net, int stepIdx);
void setWeights(Network* net, int stepId, const std::vector<float> &weights);
void setActivation(Network* net, int layerId, const std::string &name);
void mutate(Network* net, float mutationRate);
std::vector<float> forward(Network* net, const std::vector<float> &input);
int numLayers(Network* net);
Network* load(const std::string &path);
void save(Network* net, const std::string &path);
void destroy(Network* net);
int weightCount(Network* net);
void copyWeights(Network* dst, const Network* src);
std::vector<std::vector<float>> forwardBatch(
    const std::vector<Network*>& nets,
    const std::vector<std::vector<float>>& inputs);