#include "nn/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <random>

// Returns the index in weights[] where layer i's data starts
static int layerOffset(const Network* net, int layerIdx) {
    int offset = 0;

    for (int i = 0; i < layerIdx; ++i) {
        offset += net->topology[i] * net->topology[i + 1];
    }

    return offset;
}

/**
 * @param layers neuron count per layer, e.g. {4, 5, 1} = 4 inputs, hidden 5, 1 output
 */
Network* createNN(const std::vector<int>& layers) {
    auto* net = new Network();
    net->topology = layers;

    int total = 0;
    for (int i = 0; i + 1 < (int)layers.size(); ++i) {
        total += layers[i] * layers[i + 1];
    }
    net->weights.resize(total);

    std::mt19937 rng(std::random_device{}());
    for (int i = 0; i + 1 < (int)layers.size(); ++i) {
        std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / layers[i])); // He init
        int offset = layerOffset(net, i);
        int count = layers[i] * layers[i + 1];
        for (int j = 0; j < count; ++j) {
            net->weights[offset + j] = dist(rng);
        }
    }

    int maxLayer = *std::max_element(layers.begin(), layers.end());
    net->neurons.resize(2 * maxLayer); // ping-pong scratchpad for forward pass
    net->activations.assign(layers.size() - 1, "relu");

    return net;
}

Network* clone(Network* net) {
    return new Network(*net);
}

/**
 * @param layerIdx transition index (0 = input→first hidden, 1 = first→second hidden, ...)
 */
std::vector<float> getWeights(Network* net, int layerIdx) {
    int offset = layerOffset(net, layerIdx);
    int count = net->topology[layerIdx] * net->topology[layerIdx + 1];

    return {
        net->weights.begin() + offset, net->weights.begin() + offset + count
    };
}

/**
 * @param layerIdx transition index (0 = input→first hidden, 1 = first→second hidden, ...)
 */
void setWeights(Network* net, int layerIdx, const std::vector<float>& weights) {
    int offset = layerOffset(net, layerIdx);

    std::copy(weights.begin(), weights.end(), net->weights.begin() + offset);
}

void setActivation(Network* net, int layerId, const std::string& name) {
    net->activations[layerId] = name;
}
