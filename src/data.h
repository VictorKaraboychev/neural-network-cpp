#ifndef DATA_H
#define DATA_H

#include <string>

// Function to read MNIST images.
unsigned int** read_mnist_images(std::string full_path, int number_of_images, int image_size);

// Function to read MNIST image labels.
unsigned int* read_mnist_labels(std::string full_path, int number_of_labels);

#endif // DATA_H