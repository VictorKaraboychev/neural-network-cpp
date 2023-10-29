#ifndef NEURON_H
#define NEURON_H

#include "activation.h"
#include <vector>

class Neuron {
public:
    Neuron(unsigned int num_inputs);
    ~Neuron();

    // Initialize neuron weights and bias with random values or custom initialization.
    void initialize();

    // Compute the weighted sum of inputs and apply the activation function.
    double activate(const std::vector<double>& inputs, Activation activation);

    // Update the neuron's weights and bias during backpropagation.
    void updateWeightsBias(double learning_rate, double delta, const std::vector<double>& inputs, Activation activation);

private:
    unsigned int num_inputs;		// Number of inputs to the neuron.
    std::vector<double> weights; 	// Weights for each input.
    double bias;				 	// Bias for the neuron.

	double activation_output;		// Output of the activation function.

	// Initialize neuron weights and bias with random values.
    void randomInitialization();
};

#endif // NEURON_H
