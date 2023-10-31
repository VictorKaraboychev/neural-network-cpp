#include "network.h"
#include "activation.h"

#include "data.h"

#include <iostream>
#include <stdint.h> // For uint8_t

#define TRAINING_SIZE 6000

int main()
{
	printf("Start\n");

	uint8_t** images = read_mnist_images("../data/train/train-images.idx3-ubyte", TRAINING_SIZE, 784);
	uint8_t*  labels = read_mnist_labels("../data/train/train-labels.idx1-ubyte", TRAINING_SIZE);

	Network network;

	network.addLayer(784, ActivationFunctions::sigmoid);
	network.addLayer(16, ActivationFunctions::sigmoid);
	network.addLayer(16, ActivationFunctions::sigmoid);
	network.addLayer(10, ActivationFunctions::sigmoid);

	printf("Network size: %d\n", network.size());

	network.initialize();

	std::vector<std::vector<double>> input_data;
	std::vector<std::vector<double>> target_data;

	for (int i = 0; i < TRAINING_SIZE; i++)
	{
		std::vector<double> input;
		for (int j = 0; j < 784; j++)
		{
			input.push_back(images[i][j] / 255.0);
		}
		input_data.push_back(input);

		std::vector<double> target(10, 0.0);
		target[labels[i]] = 1.0;
		target_data.push_back(target);
	}

	// size of data
	printf("Input data size: %d\n", input_data.size());

	network.train(input_data, target_data, 0.01, 10);

	// std::pair<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>> weights_biases = network.exportWeightsBiases();

	return 0;
}