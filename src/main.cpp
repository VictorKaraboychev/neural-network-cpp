#include "network.h"
#include "activation.h"

#include "data.h"

#include <iostream>
#include <fstream>
#include <stdint.h> // For uint8_t

#define TRAINING_SIZE 10000
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

	// printf("Training...\n");

	// Train network
	network.train(input_data, target_data, 1, 100, 48);

	// printf("Training complete\n\n");

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
	// export_network(network, "network.json");

	return 0;
}

// int main()
// {
// 	Network network(3);
// 	network.addLayer(10, ActivationFunctions::sigmoid);
// 	network.addLayer(1, ActivationFunctions::sigmoid);

// 	network.initialize();

// 	// std::vector<std::vector<std::vector<double>>> weights = {{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}}, {{0.7, 0.8}}};
// 	// std::vector<std::vector<double>> bias = {{0.1, 0.2}, {0.3}};

// 	// network.importWeightsBiases(bias, weights);

// 	std::vector<std::vector<double>> input_data = {{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9}};
// 	std::vector<std::vector<double>> target_data = {{1}, {0.2}, {0.3}};

// 	network.train(input_data, target_data, 1, 100000, 10);

// 	// test prediction

// 	for (size_t i = 0; i < input_data.size(); i++)
// 	{
// 		std::vector<double> prediction = network.predict(input_data[i]);

// 		printf("Prediction: %f, Actual: %f\n", prediction[0], target_data[i][0]);
// 	}

// 	// export weights and biases
// 	// export_network(network, "test.json");

// 	return 0;
// }