#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"
#include "activation.h"

#include <vector>
#include <atomic>

class Network
{
public:
    Network(unsigned int input_size);
    ~Network();

    // Add a layer to the network.
    void addLayer(int num_neurons, Activation activation);

    // Initialize the network and its layers.
    void initialize();

    // Initialize the network and its layers with custom weights and biases.
    void initialize(const std::vector<std::vector<double>> &bias, const std::vector<std::vector<std::vector<double>>> &weights);

    // Get layers in the network.
    std::vector<Layer> getLayers() const;

    // Export the network's weights and biases.
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>> exportWeightsBiases() const;

    // Import the network's weights and biases.
    void importWeightsBiases(const std::vector<std::vector<double>> &bias, const std::vector<std::vector<std::vector<double>>> &weights);

    // Get the number of layers in the network.
    unsigned int size() const;

    // Forward pass through the network.
    std::vector<double> forward(const std::vector<double> &inputs);

    // Backpropagate and update weights and biases using gradient descent.
    void train(const std::vector<std::vector<double>> &input_data, const std::vector<std::vector<double>> &target_data, double learning_rate, int epochs, int num_threads = 1);

    // Make predictions using the trained network.
    std::vector<double> predict(const std::vector<double> &input);

private:
    unsigned int input_size;   // Number of inputs to the network.
    std::vector<Layer> layers; // Layers in the network.
};

#endif // NETWORK_H