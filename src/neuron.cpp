#include "neuron.h"

#include <stdexcept> // For runtime_error

Neuron::Neuron(unsigned int num_inputs) : num_inputs(num_inputs)
{
	// Constructor, if necessary
}

Neuron::~Neuron()
{
	// Destructor, if necessary
}

void Neuron::initialize()
{
	this->value = 0.0;
	this->randomInitialization();
}

void Neuron::initialize(double bias, const std::vector<double> &weights)
{
	if (weights.size() != this->num_inputs)
	{
		throw std::runtime_error("Input size does not match weight size.");
	}

	this->value = 0.0;
	this->bias = bias;
	this->weights = weights;
}

double Neuron::getValue() const
{
	return this->value;
}

void Neuron::setValue(double value)
{
	this->value = value;
}

double Neuron::getBias() const
{
	return this->bias;
}

void Neuron::setBias(double bias)
{
	this->bias = bias;
}

std::vector<double> Neuron::getWeights() const
{
	return this->weights;
}

void Neuron::setWeights(const std::vector<double> &weights)
{
	if (weights.size() != this->num_inputs)
	{
		throw std::runtime_error("Input size does not match weight size.");
	}

	this->weights = weights;
}

double Neuron::activate(const std::vector<double> &inputs, Activation activation)
{
	if (inputs.size() != this->num_inputs)
	{
		throw std::runtime_error("Input size does not match weight size.");
	}

	// Calculate the weighted sum
	double weighted_sum = this->bias;
	for (int i = 0; i < num_inputs; i++)
	{
		weighted_sum += inputs[i] * weights[i];
	}

	// Apply the activation function
	this->value = activation.function(weighted_sum);
	return this->value;
}

void Neuron::updateWeightsBias(double learning_rate, double delta, const std::vector<double> &inputs)
{
	if (inputs.size() != this->num_inputs)
	{
		throw std::runtime_error("Input size does not match weight size.");
	}

	// Calculate the gradient
	double gradient = learning_rate * delta;

	// Update the bias
	this->bias -= gradient;

	// Update weights
	for (int i = 0; i < num_inputs; i++)
	{
		this->weights[i] -= gradient * inputs[i];
	}
}

void Neuron::randomInitialization()
{
	// Initialize weights and bias with random values
	this->bias = 2 * ((double)rand() / RAND_MAX) - 1;

	for (int i = 0; i < num_inputs; i++)
	{
		this->weights.push_back(2 * ((double)rand() / RAND_MAX) - 1);
	}
}