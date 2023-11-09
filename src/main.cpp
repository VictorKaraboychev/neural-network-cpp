#include "network.h"
#include "activation.h"

#include "data.h"

#include <iostream>
#include <fstream>
#include <stdint.h> // For uint8_t

#define TRAINING_SIZE 60000
#define TESTING_SIZE 10000

#define LEARNING_RATE 1
#define EPOCHS 50

uint32_t shape[] = {784, 16, 10};

int main()
{
	// Create network
	Network network(shape[0]);

	// Add layers
	for (size_t i = 1; i < sizeof(shape) / sizeof(shape[0]); i++)
	{
		network.addLayer(shape[i], ActivationFunctions::sigmoid);
	}

	// Initialize network
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

	// Train network
	network.train(input_data, target_data, LEARNING_RATE, EPOCHS);

	// Import testing data
	uint8_t **test_images = read_mnist_images("../data/test/test-images.idx3-ubyte", TESTING_SIZE, 784);
	uint8_t *test_labels = read_mnist_labels("../data/test/test-labels.idx1-ubyte", TESTING_SIZE);

	printf("Testing...\n\n");

	int incorrect = 0;

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

		if (max_index != test_labels[i])
		{
			incorrect++;
		}
	}

	printf("Testing complete with %d incorrect predictions out of %d instances (%.2f%%)\n", incorrect, TESTING_SIZE, (double)incorrect / TESTING_SIZE * 100);

	// Export weights and biases to file in ../models/network-X-X-X-X.json
	std::string filename = "network-" + std::to_string(shape[0]);
	for (size_t i = 1; i < sizeof(shape) / sizeof(shape[0]); i++)
	{
		filename += "-" + std::to_string(shape[i]);
	}
	filename += ".json";
	export_network(network, filename);

	// Free memory
	for (size_t i = 0; i < TRAINING_SIZE; i++)
	{
		delete[] train_images[i];
	}
	delete[] train_images;

	for (size_t i = 0; i < TESTING_SIZE; i++)
	{
		delete[] test_images[i];
	}
	delete[] test_images;

	delete[] train_labels;
	delete[] test_labels;

	return 0;
}
