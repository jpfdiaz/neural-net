#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "nn/tensor.hpp"

// ── createNN / destroy ────────────────────────────────────────────────────────

TEST_CASE("createNN returns a valid pointer") {
    Network* net = createNN({2, 3, 1});
    CHECK(net != nullptr);
    destroy(net);
}

TEST_CASE("createNN: numLayers matches topology") {
    Network* net = createNN({4, 5, 3, 1});
    CHECK(numLayers(net) == 4);
    destroy(net);
}

// ── weightCount ───────────────────────────────────────────────────────────────

TEST_CASE("weightCount is positive") {
    Network* net = createNN({2, 3, 1});
    CHECK(weightCount(net) > 0);
    destroy(net);
}

TEST_CASE("weightCount grows with network size") {
    Network* small = createNN({2, 2});
    Network* large = createNN({2, 8});
    CHECK(weightCount(large) > weightCount(small));
    destroy(small);
    destroy(large);
}

// ── getWeights / setWeights ───────────────────────────────────────────────────

TEST_CASE("setWeights / getWeights round-trip") {
    Network* net = createNN({2, 3, 1});
    std::vector<float> w = getWeights(net, 1);
    for (auto& v : w) v = 0.42f;
    setWeights(net, 1, w);
    CHECK(getWeights(net, 1) == w);
    destroy(net);
}

TEST_CASE("getWeights size is non-zero for every layer") {
    Network* net = createNN({3, 4, 2});
    for (int i = 0; i < numLayers(net); ++i)
        CHECK(getWeights(net, i).size() > 0);
    destroy(net);
}

// ── setActivation ─────────────────────────────────────────────────────────────

TEST_CASE("setActivation does not crash on known activations") {
    Network* net = createNN({2, 4, 1});
    CHECK_NOTHROW(setActivation(net, 1, "relu"));
    CHECK_NOTHROW(setActivation(net, 1, "sigmoid"));
    CHECK_NOTHROW(setActivation(net, 1, "tanh"));
    destroy(net);
}

// ── forward ───────────────────────────────────────────────────────────────────

TEST_CASE("forward output size matches last layer") {
    Network* net = createNN({4, 5, 3});
    auto out = forward(net, {0.1f, 0.2f, 0.3f, 0.4f});
    CHECK(out.size() == 3);
    destroy(net);
}

TEST_CASE("forward produces known values with manual weights") {
    // topology {2,2,1}, relu, no biases
    // layer 1: [[0.5, 0.5], [0.5, 0.5]]  → hidden = {1.0, 1.0}
    // layer 2: [[1.0, 1.0]]              → output = {2.0}
    Network* net = createNN({2, 2, 1});
    setActivation(net, 1, "relu");
    setActivation(net, 2, "relu");
    setWeights(net, 1, {0.5f, 0.5f, 0.5f, 0.5f});
    setWeights(net, 2, {1.0f, 1.0f});
    auto out = forward(net, {1.0f, 1.0f});
    CHECK(out.size() == 1);
    CHECK(out[0] == doctest::Approx(2.0f));
    destroy(net);
}

TEST_CASE("forward applies bias correctly") {
    // topology {1,1}: single weight w=0, bias b=3.0 → relu(0*input + 3) = 3
    Network* net = createNN({1, 1});
    setActivation(net, 1, "relu");
    // layer 1 params: [weight, bias]
    setWeights(net, 1, {0.0f, 3.0f});
    auto out = forward(net, {99.0f});
    CHECK(out.size() == 1);
    CHECK(out[0] == doctest::Approx(3.0f));
    destroy(net);
}

TEST_CASE("forward is deterministic") {
    Network* net = createNN({3, 4, 1});
    std::vector<float> input = {0.1f, 0.5f, 0.9f};
    CHECK(forward(net, input) == forward(net, input));
    destroy(net);
}

// ── clone ─────────────────────────────────────────────────────────────────────

TEST_CASE("clone produces same output as original") {
    Network* net  = createNN({2, 3, 1});
    Network* copy = clone(net);
    std::vector<float> input = {0.3f, 0.7f};
    CHECK(forward(net, input) == forward(copy, input));
    destroy(net);
    destroy(copy);
}

TEST_CASE("clone is independent: mutating clone does not affect original") {
    Network* net  = createNN({2, 3, 1});
    Network* copy = clone(net);
    std::vector<float> input = {0.3f, 0.7f};
    auto before = forward(net, input);
    mutate(copy, 1.0f);
    CHECK(forward(net, input) == before);
    destroy(net);
    destroy(copy);
}

// ── copyWeights ───────────────────────────────────────────────────────────────

TEST_CASE("copyWeights: dst produces same output as src") {
    Network* src = createNN({2, 3, 1});
    Network* dst = createNN({2, 3, 1});
    mutate(dst, 1.0f);  // ensure dst starts different
    copyWeights(dst, src);
    std::vector<float> input = {0.5f, 0.5f};
    CHECK(forward(dst, input) == forward(src, input));
    destroy(src);
    destroy(dst);
}

// ── mutate ────────────────────────────────────────────────────────────────────

TEST_CASE("mutate with rate=0 leaves weights unchanged") {
    Network* net = createNN({2, 3, 1});
    auto w = getWeights(net, 1);
    mutate(net, 0.0f);
    CHECK(getWeights(net, 1) == w);
    destroy(net);
}

TEST_CASE("mutate with rate=1 changes weights") {
    Network* net = createNN({2, 3, 1});
    auto before = getWeights(net, 1);
    mutate(net, 1.0f);
    CHECK(getWeights(net, 1) != before);
    destroy(net);
}

// ── save / load ───────────────────────────────────────────────────────────────

TEST_CASE("save and load round-trip preserves output") {
    const std::string path = "/tmp/test_net.bin";
    Network* net = createNN({3, 4, 1});
    std::vector<float> input = {0.1f, 0.2f, 0.3f};
    auto before = forward(net, input);
    save(net, path);
    destroy(net);

    Network* loaded = load(path);
    CHECK(loaded != nullptr);
    CHECK(forward(loaded, input) == before);
    destroy(loaded);
}

// ── forwardBatch ──────────────────────────────────────────────────────────────

TEST_CASE("forwardBatch result count matches network count") {
    Network* a = createNN({2, 3, 1});
    Network* b = createNN({2, 3, 1});
    auto results = forwardBatch({a, b}, {{0.1f, 0.2f}, {0.3f, 0.4f}});
    CHECK(results.size() == 2);
    destroy(a);
    destroy(b);
}

TEST_CASE("forwardBatch matches individual forward calls") {
    Network* a = createNN({2, 3, 1});
    Network* b = createNN({2, 3, 1});
    std::vector<float> inputA = {0.1f, 0.2f};
    std::vector<float> inputB = {0.3f, 0.4f};
    auto results = forwardBatch({a, b}, {inputA, inputB});
    CHECK(results[0] == forward(a, inputA));
    CHECK(results[1] == forward(b, inputB));
    destroy(a);
    destroy(b);
}