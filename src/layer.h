#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"
#include "activation.h"

#include <vector>

class Layer
{
public:
	Layer(unsigned int num_neurons, unsigned int num_inputs, Activation activation);
	~Layer();

	// Initialize the neurons in the layer.
	void initialize();

	// Initialize the neurons in the layer with custom weights and biases.
	void initialize(const std::vector<double> bias, const std::vector<std::vector<double>> &weights);

	// Export the weights and biases of the layer.
	std::pair<std::vector<double>, std::vector<std::vector<double>>> getWeightsBiases() const;

	// Import the weights and biases of the layer.
	void setWeightsBiases(const std::vector<double> bias, const std::vector<std::vector<double>> &weights);

	// Get the number of neurons in the layer.
	unsigned int size() const;

	// Get the values of the neurons in the layer.
	std::vector<double> getValues() const;

	// Set the values of the neurons in the layer.
	void setValues(const std::vector<double> &values);

	// Compute loss for the layer.
	double computeLoss(const std::vector<double> &targets) const;

	// Compute the deltas for the current layer (l) based on the next layer (l + 1).
	void computeDeltas(Layer &next_layer);

	// Set deltas for the last layer using the target values.
	void computeDeltas(const std::vector<double> &targets);

	// Forward pass through the layer.
	std::vector<double> forward(const std::vector<double> &inputs);

	// Backward pass through the layer to update weights and biases.
	void backward(const std::vector<double> &inputs, double learning_rate);

private:
	unsigned int num_neurons;       // Number of neurons in the layer.
	unsigned int num_inputs;        // Number of inputs to each neuron.
	std::vector<Neuron> neurons;    // Neurons in the layer.
	Activation activation;          // Activation function for the layer.

	std::vector<double> deltas;     // Deltas for the layer.
};

#endif // LAYER_H
