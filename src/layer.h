#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include "neuron.h"

class Layer {
public:
    Layer(unsigned int num_neurons, unsigned int num_inputs, Activation activation);
    ~Layer();

    // Initialize the neurons in the layer.
    void initialize();

	unsigned int getNumNeurons() const;

    // Forward pass through the layer.
    std::vector<double> forward(const std::vector<double>& inputs);

    // Backward pass through the layer to update weights and biases.
    void backward(const std::vector<double>& inputs, const std::vector<double>& deltas, double learning_rate);

private:
    unsigned int num_neurons;		// Number of neurons in the layer.
	unsigned int num_inputs;		// Number of inputs to each neuron.
    std::vector<Neuron> neurons;	// Neurons in the layer.
	Activation activation;			// Activation function for the layer.
};

#endif // LAYER_H
