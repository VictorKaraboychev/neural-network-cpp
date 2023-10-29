#include "neuron.h"

#include <ctime> // For random weight initialization
#include <stdexcept> // For runtime_error

Neuron::Neuron(unsigned int num_inputs): 
	num_inputs(num_inputs)
{
    this->initialize();
}

Neuron::~Neuron() {
    // Destructor, if necessary
}

void Neuron::initialize() {
    this->randomInitialization();
}

double Neuron::activate(const std::vector<double>& inputs, Activation activation) {
    if (inputs.size() != this->num_inputs) {
        throw std::runtime_error("Input size does not match weight size.");
    }

    // Calculate the weighted sum
    double weighted_sum = this->bias;
    for (int i = 0; i < num_inputs; i++) {
        weighted_sum += inputs[i] * weights[i];
    }

    // Apply the activation function
    this->activation_output = activation.function(weighted_sum);
	return this->activation_output;
}

void Neuron::updateWeightsBias(double learning_rate, double delta, const std::vector<double>& inputs, Activation activation) {
    if (inputs.size() != this->num_inputs) {
        throw std::runtime_error("Input size does not match weight size.");
    }

	double gradient = delta * activation.derivative(this->activation_output);

    // Update the bias
    this->bias += learning_rate * gradient;

    // Update weights
    for (int i = 0; i < num_inputs; i++) {
        this->weights[i] += learning_rate * gradient * inputs[i];
    }
}

void Neuron::randomInitialization() {
    // Initialize weights and bias with random values
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    for (int i = 0; i < num_inputs; i++) {
        this->weights.push_back(static_cast<double>(rand()) / RAND_MAX);
    }
    this->bias = static_cast<double>(rand()) / RAND_MAX;
}