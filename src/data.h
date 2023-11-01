#ifndef DATA_H
#define DATA_H

#include "network.h"

#include <string>
#include <stdint.h> // For uint8_t

// Function to read MNIST images.
uint8_t **read_mnist_images(std::string full_path, int number_of_images, int image_size);

// Function to read MNIST image labels.
uint8_t *read_mnist_labels(std::string full_path, int number_of_labels);

// Function to export a network to a json file.
void export_network(Network network, std::string filename);

// Function to import a network from a json file.
void import_network(Network &network, std::string filename);

#endif // DATA_H