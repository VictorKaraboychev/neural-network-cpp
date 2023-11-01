#include "network.h"
#include "activation.h"

#include "data.h"

#include <iostream>
#include <fstream>
#include <stdint.h> // For uint8_t

#define TRAINING_SIZE 60
#define TESTING_SIZE 10

int main()
{
	printf("Start\n");

	// Create network
	Network network(784);

	network.addLayer(16, ActivationFunctions::sigmoid);
	network.addLayer(16, ActivationFunctions::sigmoid);
	network.addLayer(10, ActivationFunctions::sigmoid);

	printf("Network size: %d\n", network.size());

	network.initialize();

	// Import training data
	uint8_t **train_images = read_mnist_images("../data/train/train-images.idx3-ubyte", TRAINING_SIZE, 784);
	uint8_t *train_labels = read_mnist_labels("../data/train/train-labels.idx1-ubyte", TRAINING_SIZE);

	std::vector<std::vector<double>> input_data;
	std::vector<std::vector<double>> target_data;

	// Convert data to vectors
	for (size_t i = 0; i < TRAINING_SIZE; i++)
	{
		std::vector<double> input;
		for (size_t j = 0; j < 784; j++)
		{
			input.push_back(train_images[i][j] / 255.0);
		}
		input_data.push_back(input);

		std::vector<double> target(10, 0.0);
		target[train_labels[i]] = 1.0;
		target_data.push_back(target);
	}

	// size of data
	printf("Input data size: %d\n\n", input_data.size());

	printf("Training...\n");

	// Train network
	network.train(input_data, target_data, 0.01, 10);

	printf("Training complete\n\n");

	// Import testing data
	uint8_t **test_images = read_mnist_images("../data/test/test-images.idx3-ubyte", TESTING_SIZE, 784);
	uint8_t *test_labels = read_mnist_labels("../data/test/test-labels.idx1-ubyte", TESTING_SIZE);

	printf("Testing...\n");

	// Print predictions
	for (size_t i = 0; i < TESTING_SIZE; i++)
	{
		std::vector<double> input;
		for (size_t j = 0; j < 784; j++)
		{
			input.push_back(test_images[i][j] / 255.0);
		}

		std::vector<double> prediction = network.predict(input);

		// Find max index
		int max_index = 0;
		for (size_t j = 0; j < 10; j++)
		{
			if (prediction[j] > prediction[max_index])
			{
				max_index = j;
			}
		}

		printf("Prediction: %d, Actual: %d\n", max_index, test_labels[i]);
	}

	printf("Testing complete\n\n");

	// Export weights and biases to file
	std::pair<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>> weights_biases = network.exportWeightsBiases();

	std::vector<std::vector<double>> bias = weights_biases.first;
	std::vector<std::vector<std::vector<double>>> weights = weights_biases.second;

	std::ofstream file("weights_biases.json");

	file << "{\n";

	file << "\t\"bias\": [\n";

	for (int i = 0; i < bias.size(); i++)
	{
		file << "\t\t[";
		for (int j = 0; j < bias[i].size(); j++)
		{
			file << bias[i][j];
			if (j != bias[i].size() - 1)
			{
				file << ", ";
			}
		}
		file << "]";
		if (i != bias.size() - 1)
		{
			file << ",";
		}
		file << "\n";
	}

	file << "\t],\n";

	file << "\t\"weights\": [\n";

	for (int i = 0; i < weights.size(); i++)
	{
		file << "\t\t[\n";
		for (int j = 0; j < weights[i].size(); j++)
		{
			file << "\t\t\t[";
			for (int k = 0; k < weights[i][j].size(); k++)
			{
				file << weights[i][j][k];
				if (k != weights[i][j].size() - 1)
				{
					file << ", ";
				}
			}
			file << "]";
			if (j != weights[i].size() - 1)
			{
				file << ",";
			}
			file << "\n";
		}
		file << "\t\t]";
		if (i != weights.size() - 1)
		{
			file << ",";
		}
		file << "\n";
	}

	file << "\t]\n";

	file << "}";

	file.close();

	return 0;
}