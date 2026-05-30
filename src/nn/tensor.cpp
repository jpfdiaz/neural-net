#include "nn/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>

// Layout per layer i:
//   i == 0: topology[0] bias placeholders (stored, unused in forward)
//   i  > 0: topology[i-1]*topology[i] connection weights, then topology[i] biases
static int layerOffset(const Network* net, int layerIdx) {
    if (layerIdx == 0) return 0;

    int offset = net->topology[0];
    for (int i = 1; i < layerIdx; ++i) {
        offset += net->topology[i-1] * net->topology[i];
    }

    return offset;
}

static int layerParamCount(const Network* net, int layerIdx) {
    if (layerIdx == 0) return net->topology[0];

    return net->topology[layerIdx-1] * net->topology[layerIdx];
}

static float applyActivation(float x, const std::string& act) {
    if (act == "sigmoid")  return 1.0f / (1.0f + std::exp(-x));
    if (act == "tanh")     return std::tanh(x);
    if (act == "hardtanh") return std::max(-1.0f, std::min(1.0f, x));

    return std::max(0.0f, x); // relu default
}

Network* createNN(const std::vector<int>& layers) {
    auto* net = new Network();
    net->topology = layers;

    int total = layers[0];
    for (int i = 1; i < (int)layers.size(); ++i) {
        total += layers[i-1] * layers[i];
    }
    net->weights.resize(total, 0.0f);

    std::mt19937 rng(std::random_device{}());
    for (int i = 1; i < (int)layers.size(); ++i) {
        std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / layers[i-1]));
        int offset = layerOffset(net, i);
        int connCount = layers[i-1] * layers[i];
        for (int j = 0; j < connCount; ++j) {
            net->weights[offset + j] = dist(rng);
        }
    }

    int maxLayer = *std::max_element(layers.begin(), layers.end());
    net->neurons.resize(2 * maxLayer);
    net->activations.assign(layers.size(), "relu");

    return net;
}

Network* clone(Network* net) {
    return new Network(*net);
}

std::vector<float> getWeights(Network* net, int layerIdx) {
    int offset = layerOffset(net, layerIdx);
    int count = layerParamCount(net, layerIdx);

    return {net->weights.begin() + offset, net->weights.begin() + offset + count};
}

void setWeights(Network* net, int layerIdx, const std::vector<float>& weights) {
    int offset = layerOffset(net, layerIdx);

    std::copy(weights.begin(), weights.end(), net->weights.begin() + offset);
}

void setActivation(Network* net, int layerId, const std::string& name) {
    net->activations[layerId] = name;
}

int numLayers(Network* net) {
    return (int)net->topology.size();
}

int weightCount(Network* net) {
    return (int)net->weights.size();
}

std::vector<float> forward(Network* net, const std::vector<float>& input) {
    int size = (int)net->topology.size();
    std::vector<float> current = input;

    for (int i = 1; i < size; ++i) {
        int inSize = net->topology[i-1];
        int outSize = net->topology[i];
        int offset = layerOffset(net, i);
        std::vector<float> next(outSize);

        for (int j = 0; j < outSize; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < inSize; ++k) {
                sum += net->weights[offset + j * inSize + k] * current[k];
            }
            next[j] = applyActivation(sum, net->activations[i]);
        }

        current = std::move(next);
    }

    return current;
}

void mutate(Network* net, float mutationRate) {
    if (mutationRate <= 0.0f) return;

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);
    std::normal_distribution<float> delta(0.0f, 0.1f);

    for (auto& w : net->weights) {
        if (prob(rng) < mutationRate) {
            w += delta(rng);
        }
    }
}

void copyWeights(Network* dst, const Network* src) {
    dst->weights = src->weights;
}

static constexpr char MAGIC[4] = {'N', 'N', 'F', 'T'};

void save(Network* net, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    f.write(MAGIC, sizeof(MAGIC));
    int n = (int)net->topology.size();
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));
    f.write(reinterpret_cast<const char*>(net->topology.data()), n * sizeof(int));

    int wcount = (int)net->weights.size();
    f.write(reinterpret_cast<const char*>(&wcount), sizeof(wcount));
    f.write(reinterpret_cast<const char*>(net->weights.data()), wcount * sizeof(float));
    
    for (const auto& act : net->activations) {
        int len = (int)act.size();
        f.write(reinterpret_cast<const char*>(&len), sizeof(len));
        f.write(act.data(), len);
    }
}

Network* load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return nullptr;

    char magic[4];
    f.read(magic, sizeof(magic));
    if (!std::equal(magic, magic + 4, MAGIC)) return nullptr;

    int n;
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    std::vector<int> topology(n);
    f.read(reinterpret_cast<char*>(topology.data()), n * sizeof(int));

    Network* net = createNN(topology);
    int wcount;
    f.read(reinterpret_cast<char*>(&wcount), sizeof(wcount));
    net->weights.resize(wcount);
    f.read(reinterpret_cast<char*>(net->weights.data()), wcount * sizeof(float));

    for (auto& act : net->activations) {
        int len;
        f.read(reinterpret_cast<char*>(&len), sizeof(len));
        act.resize(len);
        f.read(&act[0], len);
    }

    return net;
}

void destroy(Network* net) {
    delete net;
}

std::vector<std::vector<float>> forwardBatch(
    const std::vector<Network*>& nets,
    const std::vector<std::vector<float>>& inputs) {
    std::vector<std::vector<float>> results(nets.size());

    for (size_t i = 0; i < nets.size(); ++i) {
        results[i] = forward(nets[i], inputs[i]);
    }

    return results;
}