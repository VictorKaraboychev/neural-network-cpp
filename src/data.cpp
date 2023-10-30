#include "data.h"

#include <iostream>
#include <fstream>
#include <stdexcept> // For runtime_error

unsigned int** read_mnist_images(std::string full_path, int number_of_images, int image_size) {
	auto reverseInt = [](int i) {
		unsigned int c1, c2, c3, c4;
		c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
		return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
	};

	std::ifstream file(full_path, std::ios::binary);

	if(file.is_open()) {
		int magic_number = 0, n_rows = 0, n_cols = 0;

		file.read((char *)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		if(magic_number != 2051) throw std::runtime_error("Invalid MNIST image file!");

		file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
		file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
		file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

		image_size = n_rows * n_cols;

		unsigned int** _dataset = new unsigned int*[number_of_images];
		for(int i = 0; i < number_of_images; i++) {
			_dataset[i] = new unsigned int[image_size];
			file.read((char *)_dataset[i], image_size);
		}
		return _dataset;
	} else {
		throw std::runtime_error("Cannot open file `" + full_path + "`!");
	}
}

unsigned int* read_mnist_labels(std::string full_path, int number_of_labels) {
	auto reverseInt = [](int i) {
		unsigned int c1, c2, c3, c4;
		c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
		return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
	};

	std::ifstream file(full_path, std::ios::binary);

	if(file.is_open()) {
		int magic_number = 0;
		file.read((char *)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		if(magic_number != 2049) throw std::runtime_error("Invalid MNIST label file!");

		file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

		unsigned int* _dataset = new unsigned int[number_of_labels];
		for(int i = 0; i < number_of_labels; i++) {
			file.read((char*)&_dataset[i], 1);
		}
		return _dataset;
	} else {
		throw std::runtime_error("Unable to open file `" + full_path + "`!");
	}
}