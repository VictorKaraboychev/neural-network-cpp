#include "layer.h"

#include <stdexcept> // For runtime_error

Layer::Layer(unsigned int num_neurons, unsigned int num_inputs, Activation activation) : num_neurons(num_neurons), activation(activation)
{
    for (int i = 0; i < this->num_neurons; i++)
    {
        Neuron neuron(this->num_inputs);
        this->neurons.push_back(neuron);
    }
}

Layer::~Layer()
{
    // Destructor, if necessary
}

void Layer::initialize()
{
    for (Neuron &neuron : this->neurons)
    {
        neuron.initialize();
    }
}

void Layer::initialize(const std::vector<double> bias, const std::vector<std::vector<double>> &weights)
{
    if (bias.size() != this->num_neurons || weights.size() != this->num_neurons)
    {
        throw std::runtime_error("Input size does not match layer size.");
    }

    for (int i = 0; i < this->num_neurons; i++)
    {
        this->neurons[i].initialize(bias[i], weights[i]);
    }
}

std::pair<std::vector<double>, std::vector<std::vector<double>>> Layer::exportWeightsBiases() const
{
    std::vector<double> bias;
    std::vector<std::vector<double>> weights;

    for (const Neuron &neuron : this->neurons)
    {
        bias.push_back(neuron.getBias());
        weights.push_back(neuron.getWeights());
    }

    return std::make_pair(bias, weights);
}

unsigned int Layer::size() const
{
    return this->num_neurons;
}

std::vector<double> Layer::getValues() const
{
    std::vector<double> values;
    for (Neuron neuron : this->neurons)
    {
        values.push_back(neuron.getValue());
    }
    return values;
}

std::vector<double> Layer::computeDeltas(const std::vector<double> &nextLayerDeltas)
{
    std::vector<double> deltas(num_neurons, 0.0);

    for (Neuron &neuron : this->neurons)
    {
        double weighted_sum = 0.0;
        for (int j = 0; j < nextLayerDeltas.size(); j++)
        {
            weighted_sum += neuron.getWeights()[j] * nextLayerDeltas[j];
        }
        deltas.push_back(weighted_sum * this->activation.derivative(neuron.getValue()));
    }

    return deltas;
}

std::vector<double> Layer::forward(const std::vector<double> &inputs)
{
    std::vector<double> outputs;
    for (Neuron &neuron : this->neurons)
    {
        outputs.push_back(neuron.activate(inputs, this->activation));
    }
    return outputs;
}

void Layer::backward(const std::vector<double> &inputs, const std::vector<double> &deltas, double learning_rate)
{
    if (inputs.size() != this->num_neurons || deltas.size() != this->num_neurons)
    {
        throw std::runtime_error("Input size does not match layer size.");
    }

    for (int i = 0; i < this->num_neurons; i++)
    {
        this->neurons[i].updateWeightsBias(learning_rate, deltas[i], inputs);
    }
}