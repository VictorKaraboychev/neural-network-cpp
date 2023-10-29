#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"
#include <vector>

class Network {
public:
    Network();
    ~Network();

    // Add a layer to the network.
    void addLayer(int num_neurons, Activation activation);

    // Initialize the network and its layers.
    void initialize();

    // Forward pass through the network.
    std::vector<double> forward(const std::vector<double>& inputs);

    // Backpropagate and update weights and biases using gradient descent.
    void train(const std::vector<std::vector<double>>& input_data, const std::vector<std::vector<double>>& target_data, double learning_rate, int epochs);

    // Make predictions using the trained network.
    std::vector<double> predict(const std::vector<double>& input);

private:
    std::vector<Layer> layers;
};

#endif // NETWORK_H