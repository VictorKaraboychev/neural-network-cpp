#include "layer.h"

#include <stdexcept> // For runtime_error

Layer::Layer(unsigned int num_neurons, unsigned int num_inputs, Activation activation): 
	num_neurons(num_neurons), 
	activation(activation) 
{
    this->initialize();
}

Layer::~Layer() {
    // Destructor, if necessary
}

void Layer::initialize() {
    this->neurons.clear();

    for (int i = 0; i < this->num_neurons; i++) {
        Neuron neuron(num_inputs);
        this->neurons.push_back(neuron);
    }
}

unsigned int Layer::getNumNeurons() const {
	return this->num_neurons;
}

std::vector<double> Layer::forward(const std::vector<double>& inputs) {
	std::vector<double> outputs;
	for (Neuron& neuron : neurons) {
		outputs.push_back(neuron.activate(inputs, activation));
	}
	return outputs;
}

void Layer::backward(const std::vector<double>& inputs, const std::vector<double>& deltas, double learning_rate) {
    if (inputs.size() != this->num_neurons || deltas.size() != this->num_neurons) {
        throw std::runtime_error("Input size does not match layer size.");
    }

    for (int i = 0; i < num_neurons; i++) {
        neurons[i].updateWeightsBias(learning_rate, deltas[i], inputs, activation);
    }
}