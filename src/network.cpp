#include "network.h"

#include <iostream>
#include <stdexcept> // For runtime_error

Network::Network(unsigned int input_size) : input_size(input_size)
{
    // Constructor, if necessary
}

Network::~Network()
{
    // Destructor, if necessary
}

void Network::addLayer(int num_neurons, Activation activation)
{
    unsigned int num_inputs = layers.size() == 0 ? this->input_size : layers.back().size();
    Layer layer(num_neurons, num_inputs, activation);
    layers.push_back(layer);
}

void Network::initialize()
{
    for (Layer &layer : layers)
    {
        layer.initialize();
    }
}

void Network::initialize(const std::vector<std::vector<double>> &bias, const std::vector<std::vector<std::vector<double>>> &weights)
{
    if (bias.size() != layers.size() || weights.size() != layers.size())
    {
        throw std::runtime_error("Input size does not match layer size.");
    }

    for (size_t i = 0; i < layers.size(); ++i)
    {
        layers[i].initialize(bias[i], weights[i]);
    }
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>> Network::exportWeightsBiases() const
{
    std::vector<std::vector<double>> bias;
    std::vector<std::vector<std::vector<double>>> weights;

    for (const Layer &layer : layers)
    {
        std::pair<std::vector<double>, std::vector<std::vector<double>>> layer_weights_biases = layer.getWeightsBiases();
        bias.push_back(layer_weights_biases.first);
        weights.push_back(layer_weights_biases.second);
    }

    return std::make_pair(bias, weights);
}

void Network::importWeightsBiases(const std::vector<std::vector<double>> &bias, const std::vector<std::vector<std::vector<double>>> &weights)
{
    if (bias.size() != this->layers.size() || weights.size() != this->layers.size())
    {
        throw std::runtime_error("Input size does not match layer size.");
    }

    for (size_t i = 0; i < this->layers.size(); ++i)
    {
        this->layers[i].setWeightsBiases(bias[i], weights[i]);
    }
}

unsigned int Network::size() const
{
    return layers.size();
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
        double epoch_loss = 0.0; // Initialize epoch loss to 0

        for (size_t i = 0; i < input_data.size(); ++i)
        {
            const std::vector<double> &input = input_data[i];
            const std::vector<double> &target = target_data[i];

            // Forward pass
            std::vector<double> output = forward(input);

            // Compute loss (e.g., mean squared error) and deltas
            std::vector<double> deltas(output.size());
            double instance_loss = 0.0; // Initialize instance loss to 0
            for (size_t j = 0; j < output.size(); ++j)
            {
                deltas[j] = output[j] - target[j];
                instance_loss += deltas[j] * deltas[j]; // Add squared error to instance loss
            }
            instance_loss /= output.size(); // Divide by number of outputs to get mean squared error
            epoch_loss += instance_loss;    // Add instance loss to epoch loss

            // Backpropagate and update weights and biases
            for (int l = layers.size() - 1; l >= 0; --l)
            {
                // Get previous outputs and deltas
                const std::vector<double> &prev_outputs = (l == 0) ? input : layers[l - 1].getValues();
                const std::vector<double> &prev_deltas = layers[l].computeDeltas(deltas);

                // Update weights and biases
                layers[l].backward(prev_outputs, deltas, learning_rate);

                // Calculate new deltas for the previous layer
                deltas = prev_deltas; //layers[l].computeDeltas(deltas);
            }
        }

        epoch_loss /= input_data.size();                                            // Divide by number of instances to get mean epoch loss
        std::cout << "Epoch " << epoch + 1 << " loss: " << epoch_loss << std::endl; // Print epoch loss
    }
}

std::vector<double> Network::predict(const std::vector<double> &input)
{
    return forward(input);
}