#include "network.h"
#include "activation.h"

#include "data.h"

int main(int argc, char const *argv[])
{
	unsigned int** images = read_mnist_images("data/train-images-idx3-ubyte", 60000, 784);
	unsigned int* labels = read_mnist_labels("data/train-labels-idx1-ubyte", 60000);

	Network network;
	network.addLayer(784, ActivationFunctions::relu);
	network.addLayer(16, ActivationFunctions::relu);
	network.addLayer(16, ActivationFunctions::relu);
	network.addLayer(10, ActivationFunctions::softmax);

	network.initialize();

	std::vector<std::vector<double>> input_data;
	std::vector<std::vector<double>> target_data;

	for(int i = 0; i < 60000; i++) {
		std::vector<double> input;
		for(int j = 0; j < 784; j++) {
			input.push_back(images[i][j] / 255.0);
		}
		input_data.push_back(input);

		std::vector<double> target(10, 0.0);
		target[labels[i]] = 1.0;
		target_data.push_back(target);
	}

	network.train(input_data, target_data, 0.01, 10);

	std::pair<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>> weights_biases = network.exportWeightsBiases();

	return 0;
}