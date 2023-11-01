#ifndef NEURON_H
#define NEURON_H

#include "activation.h"

#include <vector>

class Neuron
{
public:
    Neuron(unsigned int num_inputs);
    ~Neuron();

    // Initialize neuron weights and bias with random values or custom initialization.
    void initialize();

    // Initialize neuron weights and bias with custom values.
    void initialize(double bias, const std::vector<double> &weights);

    // Get the neuron's output value.
    double getValue() const;

    // Set the neuron's output value.
    void setValue(double value);

    // Get the neuron's bias.
    double getBias() const;

    // Set the neuron's bias.
    void setBias(double bias);

    // Gets the neurons weight for the index.
    std::vector<double> getWeights() const;

    // Set the neuron's weights.
    void setWeights(const std::vector<double> &weights);

    // Compute the delta for the neuron.
    double computeDelta(double target, Activation activation);

    // Compute the weighted sum of inputs and apply the activation function.
    double activate(const std::vector<double> &inputs, Activation activation);

    // Update the neuron's weights and bias during backpropagation.
    void updateWeightsBias(double learning_rate, double delta, const std::vector<double> &inputs);

private:
    double value; // Output of the activation function.

    unsigned int num_inputs;     // Number of inputs to the neuron.
    std::vector<double> weights; // Weights for each input.
    double bias;                 // Bias for the neuron.

    // Initialize neuron weights and bias with random values.
    void randomInitialization();
};

#endif // NEURON_H
