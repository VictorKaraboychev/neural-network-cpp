#ifndef LAYER_H
#define LAYER_H

#include "activation.h"
#include <Eigen/Dense>

class Layer
{
public:
    Layer(unsigned int num_neurons, unsigned int num_inputs, Activation activation);
    ~Layer();

    // Initialize the neurons in the layer.
    Layer *initialize();

    // Initialize the neurons in the layer with custom weights and biases.
    Layer *initialize(const Eigen::VectorXd &bias, const Eigen::MatrixXd &weights);

    // Export the weights and biases of the layer.
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> getWeightsBiases() const;

    // Import the weights and biases of the layer.
    void setWeightsBiases(const Eigen::VectorXd &bias, const Eigen::MatrixXd &weights);

    // Get the number of neurons in the layer.
    unsigned int size() const;

    // Get the values of the neurons in the layer.
    Eigen::VectorXd getValues() const;

    // Set the values of the neurons in the layer.
    void setValues(const Eigen::VectorXd &values);

    // Compute loss for the layer.
    double computeLoss(const Eigen::VectorXd &targets) const;

    // Compute the deltas for the current layer (l) based on the next layer (l + 1).
    void computeDeltas(Layer &next_layer);

    // Set deltas for the last layer using the target values.
    void computeDeltas(const Eigen::VectorXd &targets);

    // Forward pass through the layer.
    Eigen::VectorXd forward(const Eigen::VectorXd &inputs);

    // Backward pass through the layer to update weights and biases.
    Eigen::VectorXd backward(const Eigen::VectorXd &inputs, double learning_rate);

private:
	Activation activation;   // Activation function for the layer.

    Eigen::MatrixXd weights; // Weights for the layer.
    Eigen::VectorXd bias;    // Biases for the layer.

    Eigen::VectorXd values;  // Values of the neurons in the layer.

    Eigen::VectorXd deltas;  // Deltas for the layer.
};

#endif // LAYER_H
