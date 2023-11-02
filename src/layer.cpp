#include "layer.h"

#include <stdexcept> // For runtime_error

Layer::Layer(unsigned int num_neurons, unsigned int num_inputs, Activation activation) : num_inputs(num_inputs), num_neurons(num_neurons), activation(activation)
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

std::pair<std::vector<double>, std::vector<std::vector<double>>> Layer::getWeightsBiases() const
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

void Layer::setWeightsBiases(const std::vector<double> bias, const std::vector<std::vector<double>> &weights)
{
    if (bias.size() != this->num_neurons || weights.size() != this->num_neurons)
    {
        throw std::runtime_error("Input size does not match layer size.");
    }

    for (int i = 0; i < this->num_neurons; i++)
    {
        this->neurons[i].setBias(bias[i]);
        this->neurons[i].setWeights(weights[i]);
    }
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

void Layer::setValues(const std::vector<double> &values)
{
    if (values.size() != this->num_neurons)
    {
        throw std::runtime_error("Input size does not match layer size.");
    }

    for (int i = 0; i < this->num_neurons; i++)
    {
        this->neurons[i].setValue(values[i]);
    }
}

double Layer::computeLoss(const std::vector<double> &targets) const
{
    if (targets.size() != this->num_neurons)
    {
        throw std::runtime_error("Input size does not match layer size.");
    }

    double loss = 0.0;
    for (int i = 0; i < this->num_neurons; i++)
    {
        loss += std::pow(this->neurons[i].getValue() - targets[i], 2);
    }
    return loss / this->num_neurons;
}

void Layer::computeDeltas(Layer &next_layer) // deltas for layer l + 1
{
    std::vector<double> deltas(this->num_neurons, 0.0); // deltas for layer l

    for (int i = 0; i < this->num_neurons; i++)
    {
        double delta = 0.0;
        for (int j = 0; j < next_layer.size(); j++)
        {
            delta += next_layer.deltas[j] * next_layer.neurons[j].getWeights()[i];
        }
        deltas[i] = delta * this->activation.derivative(this->neurons[i].getValue());
    }

    this->deltas = deltas;
}

void Layer::computeDeltas(const std::vector<double> &targets)
{
    if (targets.size() != this->num_neurons)
    {
        throw std::runtime_error("Input size does not match layer size.");
    }

    std::vector<double> deltas(this->num_neurons, 0.0); // deltas for layer l

     for (size_t i = 0; i < this->num_neurons; ++i)
    {
        deltas[i] = (this->neurons[i].getValue() - targets[i]) * this->activation.derivative(this->neurons[i].getValue());
    }

    this->deltas = deltas;
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

void Layer::backward(const std::vector<double> &inputs, double learning_rate)
{
    if (inputs.size() != this->num_inputs)
    {
        throw std::runtime_error("Input size does not match layer size.");
    }

    for (int i = 0; i < this->num_neurons; i++)
    {
        this->neurons[i].updateWeightsBias(learning_rate, this->deltas[i], inputs);
    }
}