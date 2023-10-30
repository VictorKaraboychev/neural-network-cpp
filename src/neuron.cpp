#include "neuron.h"

#include <ctime>     // For random weight initialization
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

std::vector<double> Neuron::getWeights() const
{
    return this->weights;
}

double Neuron::getBias() const
{
    return this->bias;
}

double Neuron::computeDelta(double target, Activation activation)
{
    return (target - this->value) * activation.derivative(this->value);
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
    this->bias += gradient;

    // Update weights
    for (int i = 0; i < num_inputs; i++)
    {
        this->weights[i] += gradient * inputs[i];
    }
}

void Neuron::randomInitialization()
{
    // Initialize weights and bias with random values
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    this->bias = static_cast<double>(rand()) / RAND_MAX;
    for (int i = 0; i < num_inputs; i++)
    {
        this->weights.push_back(static_cast<double>(rand()) / RAND_MAX);
    }
}