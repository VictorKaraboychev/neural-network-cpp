#include "layer.h"

#include <stdexcept> // For runtime_error

Layer::Layer(unsigned int num_neurons, unsigned int num_inputs, Activation activation) : num_neurons(num_neurons), activation(activation)
{
    this->initialize();
}

Layer::~Layer()
{
    // Destructor, if necessary
}

void Layer::initialize()
{
    this->neurons.clear();

    for (int i = 0; i < this->num_neurons; i++)
    {
        Neuron neuron(num_inputs);
        this->neurons.push_back(neuron);
    }
}

unsigned int Layer::size() const
{
    return this->num_neurons;
}

std::vector<double> Layer::computeDeltas(const std::vector<double> &nextLayerDeltas)
{
    std::vector<double> deltas(num_neurons, 0.0);

    for (Neuron &neuron : neurons)
    {
        double weighted_sum = 0.0;
        for (int j = 0; j < nextLayerDeltas.size(); j++)
        {
            weighted_sum += neuron.getWeight(j) * nextLayerDeltas[j];
        }
        deltas.push_back(weighted_sum * this->activation.derivative(neuron.getValue()));
    }

    return deltas;
}

std::vector<double> Layer::forward(const std::vector<double> &inputs)
{
    std::vector<double> outputs;
    for (Neuron &neuron : neurons)
    {
        outputs.push_back(neuron.activate(inputs, activation));
    }
    return outputs;
}

void Layer::backward(const std::vector<double> &inputs, const std::vector<double> &deltas, double learning_rate)
{
    if (inputs.size() != this->num_neurons || deltas.size() != this->num_neurons)
    {
        throw std::runtime_error("Input size does not match layer size.");
    }

    for (Neuron &neuron : neurons)
    {
        double weighted_sum = 0.0;
        for (int j = 0; j < nextLayerDeltas.size(); j++)
        {
            weighted_sum += neuron.getWeight(j) * nextLayerDeltas[j];
        }
        neuron.updateWeightsBias(learning_rate, deltas[i], inputs, activation);
    }
}