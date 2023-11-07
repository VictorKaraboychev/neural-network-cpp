#include "network.h"

#include <iostream>
#include <stdexcept> // For runtime_error
#include <chrono>	 // For steady_clock
#include <atomic>

Network::Network(unsigned int input_size) : input_size(input_size)
{
	// Constructor, if necessary
}

Network::~Network()
{
	// Destructor, if necessary
}

void Network::addLayer(int num_neurons, Activation activation)
{
	unsigned int num_inputs = layers.size() == 0 ? this->input_size : layers.back().size();
	Layer layer(num_neurons, num_inputs, activation);
	this->layers.push_back(layer);
}

void Network::initialize()
{
	for (Layer &layer : this->layers)
	{
		layer.initialize();
	}
}

void Network::initialize(const std::vector<std::vector<double>> &bias, const std::vector<std::vector<std::vector<double>>> &weights)
{
	if (bias.size() != this->layers.size() || weights.size() != this->layers.size())
	{
		throw std::runtime_error("Input size does not match layer size.");
	}

	for (size_t i = 0; i < layers.size(); ++i)
	{
		this->layers[i].initialize(bias[i], weights[i]);
	}
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>> Network::exportWeightsBiases() const
{
	std::vector<std::vector<double>> bias;
	std::vector<std::vector<std::vector<double>>> weights;

	for (const Layer &layer : this->layers)
	{
		std::pair<std::vector<double>, std::vector<std::vector<double>>> layer_weights_biases = layer.getWeightsBiases();
		bias.push_back(layer_weights_biases.first);
		weights.push_back(layer_weights_biases.second);
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
		this->layers[i].setWeightsBiases(bias[i], weights[i]);
	}
}

unsigned int Network::size() const
{
	return this->layers.size();
}

void Network::computeDeltas(const std::vector<double> &target)
{
	// Compute deltas for the output layer
	this->layers.back().computeDeltas(target);

	// Compute deltas for the hidden layers
	for (int l = this->layers.size() - 2; l >= 0; --l)
	{
		this->layers[l].computeDeltas(this->layers[l + 1]);
	}
}

std::vector<double> Network::forward(const std::vector<double> &inputs)
{
	std::vector<double> current_inputs = inputs;
	for (Layer &layer : this->layers)
	{
		current_inputs = layer.forward(current_inputs);
	}
	return current_inputs;
}

void Network::backward(const std::vector<double> &input, double learning_rate)
{
	// Update weights and biases for hidden layers
	for (int l = this->layers.size() - 1; l > 0; --l)
	{
		this->layers[l].backward(this->layers[l - 1].getValues(), learning_rate);
	}

	// Update weights and biases for input layer
	this->layers.front().backward(input, learning_rate);
}

void Network::train(const std::vector<std::vector<double>> &input_train_data, const std::vector<std::vector<double>> &target_train_data, const std::vector<std::vector<double>> &input_test_data, const std::vector<std::vector<double>> &target_test_data, double learning_rate, int epochs, std::string filename)
{
	// Check if input and target data have the same size
	if (input_train_data.size() != target_train_data.size())
	{
		throw std::runtime_error("Input and target data have different sizes.");
	}

	// Check if input data has the same size as the input layer
	if (input_train_data[0].size() != this->input_size)
	{
		throw std::runtime_error("Input data size does not match input layer size.");
	}

	// Check if target data has the same size as the output layer
	if (target_train_data[0].size() != this->layers.back().size())
	{
		throw std::runtime_error("Target data size does not match output layer size.");
	}

	// Start timer
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	// Create CSV string
	std::string csv = "epoch,loss,accuracy\n";

	// Train the network for the specified number of epochs
	printf("\nTraining network...\n\n");

	for (int epoch = 0; epoch < epochs; ++epoch)
	{
		double epoch_loss = 0.0; // Initialize epoch loss to 0

		for (size_t i = 0; i < input_train_data.size(); ++i)
		{
			const std::vector<double> &input = input_train_data[i];
			const std::vector<double> &target = target_train_data[i];

			// Forward pass
			std::vector<double> output = this->forward(input);

			// Compute deltas
			this->computeDeltas(target);

			// Backpropagate and update weights and biases
			this->backward(input, learning_rate);

			// Compute loss
			epoch_loss += this->layers.back().computeLoss(target);
		}

		// Divide by number of instances to get mean epoch loss
		epoch_loss /= input_train_data.size();

		double epoch_accuracy = this->test(input_test_data, target_test_data);

		// Add epoch loss to CSV string
		csv += std::to_string(epoch) + "," + std::to_string(epoch_loss) + "," + std::to_string(epoch_accuracy) + "\n";

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
			printf("] %d%% - Accuracy: %.2f%% - Elapsed: %.2fs - Remaining: %.2fs", epoch * 100 / epochs, epoch_accuracy * 100, time_elapsed, time_remaining);

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
				printf("] 100%% - Accuracy: %.2f%% - Elapsed: %.2fs - Total: %.2fs\n", epoch_accuracy * 100, time_elapsed, time_elapsed + time_remaining);
			}
		}
	}

	std::ofstream file(filename + "_lr_" + std::to_string(learning_rate) + "_time_" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(begin.time_since_epoch()).count()) + ".csv");
	if (file.is_open())
	{
		file << csv;
		file.close();
	}
	else
	{
		std::cout << "Unable to open file `" << filename << "`!" << std::endl;
	}

	printf("\nTraining complete for %d epochs with a learning rate of %.2f.\n\n", epochs, learning_rate);

	// Stop timer
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	// Print training time
	printf("Training time: %ld ms\n\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
}

double Network::test(const std::vector<std::vector<double>> &input_data, const std::vector<std::vector<double>> &target_data)
{
	// Check if input and target data have the same size
	if (input_data.size() != target_data.size())
	{
		throw std::runtime_error("Input and target data have different sizes.");
	}

	int correct = 0;

	for (size_t i = 0; i < input_data.size(); i++)
	{
		const std::vector<double> &input = input_data[i];
		const std::vector<double> &target = target_data[i];

		std::vector<double> prediction = this->forward(input);

		// Find max index
		int max_index = 0;
		for (size_t j = 0; j < 10; j++)
		{
			if (prediction[j] > prediction[max_index])
			{
				max_index = j;
			}
		}

		// Check if prediction is correct
		int max_target_index = 0;
		for (size_t j = 0; j < 10; j++)
		{
			if (target[j] > target[max_target_index])
			{
				max_target_index = j;
			}
		}

		if (max_index == max_target_index)
		{
			correct++;
		}
	}

	return (double)correct / input_data.size();
}

std::vector<double> Network::predict(const std::vector<double> &input)
{
	return this->forward(input);
}