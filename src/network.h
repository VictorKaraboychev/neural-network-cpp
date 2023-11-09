#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"
#include "activation.h"

#include <vector>
#include <Eigen/Dense>

class Network
{
public:
    Network(unsigned int input_size);
    ~Network();

    // Add a layer to the network.
    Network* addLayer(int num_neurons, Activation activation);

    // Initialize the network and its layers.
    Network* initialize();

    // Initialize the network and its layers with custom weights and biases.
    Network* initialize(const std::vector<std::vector<double>>& bias, const std::vector<std::vector<std::vector<double>>>& weights);

    // Export the network's weights and biases.
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>> exportWeightsBiases() const;

    // Import the network's weights and biases.
    void importWeightsBiases(const std::vector<std::vector<double>>& bias, const std::vector<std::vector<std::vector<double>>>& weights);

    // Get the number of layers in the network.
    unsigned int size() const;

    // Backpropagate and update weights and biases using gradient descent.
    void train(const std::vector<std::vector<double>>& input_data, const std::vector<std::vector<double>>& target_data, double learning_rate, int epochs);

    // Make predictions using the trained network.
    std::vector<double> predict(const std::vector<double>& input);

private:
    unsigned int input_size;    // Number of inputs to the network.
    std::vector<Layer> layers;  // Layers in the network.

    // Helper function to convert Eigen to std::vector
    static std::vector<double> eigenToVector(const Eigen::VectorXd& eigenVector);
	static std::vector<std::vector<double>> eigenToVector(const Eigen::MatrixXd& eigenVector);

    // Helper function to convert std::vector to Eigen
    static Eigen::VectorXd vectorToEigen(const std::vector<double>& stdVector);
	static Eigen::MatrixXd vectorToEigen(const std::vector<std::vector<double>>& stdVector);

    // Compute deltas for the network.
    void computeDeltas(const Eigen::VectorXd& target);

    // Forward pass through the network.
    Eigen::VectorXd forward(const Eigen::VectorXd& inputs);

    // Update weights and biases using gradient descent.
    void backward(const Eigen::VectorXd& inputs, double learning_rate);
};

#endif // NETWORK_H