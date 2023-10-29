#include "network.h"

Network::Network()
{
    // Constructor, if necessary
}

Network::~Network()
{
    // Destructor, if necessary
}

void Network::addLayer(int num_neurons, Activation activation)
{
    layers.emplace_back(num_neurons, layers.empty() ? 0 : layers.back().size(), activation);
}

void Network::initialize()
{
    for (Layer &layer : layers)
    {
        layer.initialize();
    }
}

std::vector<double> Network::forward(const std::vector<double> &inputs)
{
    std::vector<double> current_inputs = inputs;
    for (Layer &layer : layers)
    {
        current_inputs = layer.forward(current_inputs);
    }
    return current_inputs;
}

void Network::train(const std::vector<std::vector<double>> &input_data, const std::vector<std::vector<double>> &target_data, double learning_rate, int epochs)
{
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        for (size_t i = 0; i < input_data.size(); ++i)
        {
            const std::vector<double> &input = input_data[i];
            const std::vector<double> &target = target_data[i];

            // Forward pass
            std::vector<double> output = forward(input);

            // Compute loss (e.g., mean squared error) and deltas
            std::vector<double> deltas(output.size());
            for (size_t j = 0; j < output.size(); ++j)
            {
                deltas[j] = output[j] - target[j];
            }

            // Backpropagate and update weights and biases
            for (int l = layers.size() - 1; l >= 0; --l)
            {
                const std::vector<double> &prev_outputs = (l == 0) ? input : layers[l - 1].forward();
                layers[l].backward(prev_outputs, deltas, learning_rate);
                // Calculate new deltas for the previous layer
                deltas = layers[l].computeDeltas(deltas);
            }
        }
    }
}

std::vector<double> Network::predict(const std::vector<double> &input)
{
    return forward(input);
}