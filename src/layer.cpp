#include "layer.h"
#include <stdexcept> // For runtime_error

Layer::Layer(unsigned int num_neurons, unsigned int num_inputs, Activation activation): activation(activation)
{
    this->weights.resize(num_neurons, num_inputs);
    this->bias.resize(num_neurons);
    this->values.resize(num_neurons);
    this->deltas.resize(num_neurons);
}

Layer::~Layer()
{
    // Destructor, if necessary
}

Layer* Layer::initialize()
{
    // Initialize weights and biases randomly or with a specific strategy
    this->weights = Eigen::MatrixXd::Random(weights.rows(), weights.cols());
    this->bias = Eigen::VectorXd::Random(bias.size());

    return this;
}

Layer* Layer::initialize(const Eigen::VectorXd &init_bias, const Eigen::MatrixXd &init_weights)
{
    if (init_bias.size() != this->bias.size() || init_weights.rows() != this->weights.rows() || init_weights.cols() != this->weights.cols())
    {
        throw std::runtime_error("Input size does not match layer size.");
    }

    bias = init_bias;
    weights = init_weights;

    return this;
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> Layer::getWeightsBiases() const
{
    return std::make_pair(bias, weights);
}

void Layer::setWeightsBiases(const Eigen::VectorXd &new_bias, const Eigen::MatrixXd &new_weights)
{
    if (new_bias.size() != this->bias.size() || new_weights.rows() != this->weights.rows() || new_weights.cols() != this->weights.cols())
    {
        throw std::runtime_error("Input size does not match layer size.");
    }

    bias = new_bias;
    weights = new_weights;
}

unsigned int Layer::size() const
{
    return this->values.size();
}

Eigen::VectorXd Layer::getValues() const
{
    return values;
}

void Layer::setValues(const Eigen::VectorXd &new_values)
{
    if (new_values.size() != this->values.size())
    {
        throw std::runtime_error("Input size does not match layer size.");
    }

    this->values = new_values;
}

double Layer::computeLoss(const Eigen::VectorXd &targets) const
{
    if (targets.size() != this->size())
    {
        throw std::runtime_error("Input size does not match layer size.");
    }

    double loss = (values - targets).array().pow(2).sum();
    return loss / this->size();
}

void Layer::computeDeltas(const Eigen::VectorXd &targets)
{
    if (targets.size() != this->size())
    {
        throw std::runtime_error("Input size does not match layer size.");
    }

    deltas = (values - targets).array() * activation.derivative(values).array();
}

void Layer::computeDeltas(Layer &next_layer)
{
    Eigen::VectorXd next_layer_weights = next_layer.weights.transpose() * next_layer.deltas;
    this->deltas = next_layer_weights.array() * activation.derivative(values).array();
}

Eigen::VectorXd Layer::forward(const Eigen::VectorXd &inputs)
{
    this->values = activation.function(this->weights * inputs + this->bias);
    return this->values;
}

Eigen::VectorXd Layer::backward(const Eigen::VectorXd &inputs, double learning_rate)
{
    if (inputs.size() != this->weights.cols())
    {
        throw std::runtime_error("Input size does not match layer size.");
    }

    Eigen::MatrixXd weight_updates = this->deltas * inputs.transpose();
    this->weights -= learning_rate * weight_updates;
    this->bias -= learning_rate * this->deltas;

    return this->values;
}
