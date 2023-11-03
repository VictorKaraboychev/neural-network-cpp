#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"
#include "activation.h"

#include <vector>

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

    // Export the network's weights and biases.
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>> exportWeightsBiases() const;

    // Import the network's weights and biases.
    void importWeightsBiases(const std::vector<std::vector<double>> &bias, const std::vector<std::vector<std::vector<double>>> &weights);

    // Get the number of layers in the network.
    unsigned int size() const;

    // Backpropagate and update weights and biases using gradient descent.
    void train(const std::vector<std::vector<double>> &input_data, const std::vector<std::vector<double>> &target_data, double learning_rate, int epochs);

    // Make predictions using the trained network.
    std::vector<double> predict(const std::vector<double> &input);

private:
    unsigned int input_size;    // Number of inputs to the network.
    std::vector<Layer> layers;  // Layers in the network.

    // Compute deltas for the network.
    void computeDeltas(const std::vector<double> &target);

    // Forward pass through the network.
    std::vector<double> forward(const std::vector<double> &inputs);

    // Update weights and biases using gradient descent.
    void backward(const std::vector<double> &inputs, double learning_rate);
};

#endif // NETWORK_H