#include "network.h"

#include <iostream>
#include <stdexcept> // For runtime_error
#include <chrono>	 // For steady_clock

Network::Network(unsigned int input_size) : input_size(input_size)
{
	// Constructor, if necessary
}

Network::~Network()
{
	// Destructor, if necessary
}

Network *Network::addLayer(int num_neurons, Activation activation)
{
	unsigned int num_inputs = layers.size() == 0 ? this->input_size : layers.back().size();
	Layer layer(num_neurons, num_inputs, activation);
	this->layers.push_back(layer);

	return this;
}

Network *Network::initialize()
{
	for (Layer &layer : this->layers)
	{
		layer.initialize();
	}

	return this;
}

Network *Network::initialize(const std::vector<std::vector<double>> &bias, const std::vector<std::vector<std::vector<double>>> &weights)
{
	if (bias.size() != this->layers.size() || weights.size() != this->layers.size())
	{
		throw std::runtime_error("Input size does not match layer size.");
	}

	for (size_t i = 0; i < layers.size(); ++i)
	{
		this->layers[i].initialize(vectorToEigen(bias[i]), vectorToEigen(weights[i]));
	}

	return this;
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>> Network::exportWeightsBiases() const
{
	std::vector<std::vector<double>> bias;
	std::vector<std::vector<std::vector<double>>> weights;

	bias.reserve(layers.size());
	weights.reserve(layers.size());

	for (const Layer &layer : this->layers)
	{
		std::pair<Eigen::VectorXd, Eigen::MatrixXd> layer_weights_biases = layer.getWeightsBiases();
		bias.emplace_back(eigenToVector(layer_weights_biases.first));
		weights.emplace_back(eigenToVector(layer_weights_biases.second));
	}

	return std::make_pair(bias, weights);
}

void Network::importWeightsBiases(const std::vector<std::vector<double>> &bias, const std::vector<std::vector<std::vector<double>>> &weights)
{
	if (bias.size() != this->layers.size() || weights.size() != this->layers.size())
	{
		throw std::runtime_error("Input size does not match layer size.");
	}

	for (size_t i = 0; i < this->layers.size(); ++i)
	{
		this->layers[i].setWeightsBiases(vectorToEigen(bias[i]), vectorToEigen(weights[i]));
	}
}

unsigned int Network::size() const
{
	return this->layers.size();
}

void Network::computeDeltas(const Eigen::VectorXd &target)
{
	// Compute deltas for the output layer
	this->layers.back().computeDeltas(target);

	// Compute deltas for the hidden layers
	for (int l = this->layers.size() - 2; l >= 0; --l)
	{
		this->layers[l].computeDeltas(this->layers[l + 1]);
	}
}

Eigen::VectorXd Network::forward(const Eigen::VectorXd &inputs)
{
	Eigen::VectorXd current_inputs = inputs;

	for (Layer &layer : layers)
	{
		current_inputs = layer.forward(current_inputs);
	}

	return current_inputs;
}

void Network::backward(const Eigen::VectorXd &input, double learning_rate)
{
	Eigen::VectorXd current_inputs = input;

	for (Layer &layer : layers)
	{
		current_inputs = layer.backward(current_inputs, learning_rate);
	}
}

void Network::train(const std::vector<std::vector<double>> &input_data, const std::vector<std::vector<double>> &target_data, double learning_rate, int epochs)
{
	// Check if input and target data have the same size
	if (input_data.size() != target_data.size())
	{
		throw std::runtime_error("Input and target data have different sizes.");
	}

	// Check if input data has the same size as the input layer
	if (input_data[0].size() != this->input_size)
	{
		throw std::runtime_error("Input data size does not match input layer size.");
	}

	// Check if target data has the same size as the output layer
	if (target_data[0].size() != this->layers.back().size())
	{
		throw std::runtime_error("Target data size does not match output layer size.");
	}

	// Start timer
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	// Train the network for the specified number of epochs
	printf("\nTraining network...\n\n");

	for (int epoch = 0; epoch < epochs; ++epoch)
	{
		double epoch_loss = 0.0; // Initialize epoch loss to 0

		// Train the network on each instance
		for (size_t i = 0; i < input_data.size(); ++i)
		{
			const Eigen::VectorXd &input = vectorToEigen(input_data[i]);
			const Eigen::VectorXd &target = vectorToEigen(target_data[i]);

			// Forward pass
			Eigen::VectorXd output = this->forward(input);

			// Compute deltas
			this->computeDeltas(target);

			// Backpropagate and update weights and biases
			this->backward(input, learning_rate);

			// Compute loss
			epoch_loss += this->layers.back().computeLoss(target);
		}

		// Divide by number of instances to get mean epoch loss
		epoch_loss /= input_data.size();

		// Print epoch loss every 1% of epochs
		if (epoch % (std::max(epochs, 100) / 100) == 0 || epoch == epochs - 1)
		{
			// Calculate time elapsed and predicted time to completion
			std::chrono::steady_clock::time_point current_time = std::chrono::steady_clock::now();
			double time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - begin).count() / 1000.0;
			double time_per_epoch = time_elapsed / (epoch + 1);
			double time_remaining = time_per_epoch * (epochs - epoch - 1);

			// Make a progress bar and display current epoch loss and predicted time to completion
			printf("\r[");
			int pos = 50 * epoch / epochs;
			for (int i = 0; i <= 50; ++i)
			{
				if (i < pos)
				{
					printf("=");
				}
				else if (i == pos)
				{
					printf(">");
				}
				else
				{
					printf(" ");
				}
			}
			printf("] %d%% - Loss: %.2e - Elapsed: %.2fs - Remaining: %.2fs", epoch * 100 / epochs, epoch_loss, time_elapsed, time_remaining);

			// Flush stdout
			fflush(stdout);

			// Print 100% and a full bar on the last epoch
			if (epoch == epochs - 1)
			{
				printf("\r[");
				for (int i = 0; i <= 50; ++i)
				{
					printf("=");
				}
				printf("] 100%% - Loss: %.4e - Elapsed: %.2fs - Total: %.2fs\n", epoch_loss, time_elapsed, time_elapsed + time_remaining);
			}
		}
	}

	printf("\nTraining complete for %d epochs with a learning rate of %.2f.\n\n", epochs, learning_rate);

	// Stop timer
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	// Print training time
	printf("Training time: %.3fs\n\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0);
}

std::vector<double> Network::predict(const std::vector<double> &input)
{
	return eigenToVector(this->forward(vectorToEigen(input)));
}

// Helper function to convert Eigen to std::vector
std::vector<double> Network::eigenToVector(const Eigen::VectorXd &eigenVector)
{
	std::vector<double> stdVector(eigenVector.data(), eigenVector.data() + eigenVector.size());
	return stdVector;
}

std::vector<std::vector<double>> Network::eigenToVector(const Eigen::MatrixXd &eigenVector)
{
	std::vector<std::vector<double>> stdVector(eigenVector.rows(), std::vector<double>(eigenVector.cols()));

	for (int i = 0; i < eigenVector.rows(); ++i)
	{
		for (int j = 0; j < eigenVector.cols(); ++j)
		{
			stdVector[i][j] = eigenVector(i, j);
		}
	}

	return stdVector;
}

// Helper function to convert std::vector to Eigen
Eigen::VectorXd Network::vectorToEigen(const std::vector<double> &stdVector)
{
	Eigen::VectorXd eigenVector = Eigen::Map<const Eigen::VectorXd>(stdVector.data(), stdVector.size());
	return eigenVector;
}

Eigen::MatrixXd Network::vectorToEigen(const std::vector<std::vector<double>> &stdVector)
{
	Eigen::MatrixXd eigenVector(stdVector.size(), stdVector[0].size());

	for (int i = 0; i < stdVector.size(); ++i)
	{
		for (int j = 0; j < stdVector[0].size(); ++j)
		{
			eigenVector(i, j) = stdVector[i][j];
		}
	}

	return eigenVector;
}