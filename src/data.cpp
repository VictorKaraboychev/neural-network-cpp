#include "data.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>   // For stringstream
#include <stdexcept> // For runtime_error

uint8_t **read_mnist_images(std::string full_path, int number_of_images, int image_size)
{
    auto reverseInt = [](int i)
    {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    std::ifstream file(full_path, std::ios::binary);

    if (file.is_open())
    {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2051)
            throw std::runtime_error("Invalid MNIST image file!");

        file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;

        uint8_t **_dataset = new uint8_t *[number_of_images];
        for (int i = 0; i < number_of_images; i++)
        {
            _dataset[i] = new uint8_t[image_size];
            file.read((char *)_dataset[i], image_size);
        }
        return _dataset;
    }
    else
    {
        throw std::runtime_error("Cannot open file `" + full_path + "`!");
    }
}

uint8_t *read_mnist_labels(std::string full_path, int number_of_labels)
{
    auto reverseInt = [](int i)
    {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    std::ifstream file(full_path, std::ios::binary);

    if (file.is_open())
    {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2049)
            throw std::runtime_error("Invalid MNIST label file!");

        file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        uint8_t *_dataset = new uint8_t[number_of_labels];
        for (int i = 0; i < number_of_labels; i++)
        {
            file.read((char *)&_dataset[i], 1);
        }
        return _dataset;
    }
    else
    {
        throw std::runtime_error("Unable to open file `" + full_path + "`!");
    }
}

void print_image(std::vector<double> image, int width, int height)
{
    std::string edges[8] = {"┌", "┐", "└", "┘", "─", "─", "│", "│"};
    std::string shades[5] = {"  ", "░░", "▒▒", "▓▓", "██"};

    std::cout << edges[0];
    for (int j = 0; j < width; j++)
    {
        std::cout << edges[4] << edges[4];
    }
    std::cout << edges[1] << std::endl;

    for (int i = 0; i < height; i++)
    {
        std::cout << edges[6];
        for (int j = 0; j < width; j++)
        {
            double pixel = image[i * width + j];
            std::cout << shades[(int)(pixel * 4)];
        }
        std::cout << edges[7] << std::endl;
    }

    std::cout << edges[2];
    for (int j = 0; j < width; j++)
    {
        std::cout << edges[5] << edges[5];
    }
    std::cout << edges[3] << std::endl;
}

void export_network(Network network, std::string filename)
{
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>> weights_biases = network.exportWeightsBiases();

    std::vector<std::vector<double>> bias = weights_biases.first;
    std::vector<std::vector<std::vector<double>>> weights = weights_biases.second;

    // Create a JSON object to store the weights and biases
    std::string json = "{\n";
    json += "\"weights\": [\n";

    // Add the weights to the JSON object
    for (int i = 0; i < weights.size(); i++)
    {
        json += "[\n";
        for (int j = 0; j < weights[i].size(); j++)
        {
            json += "[";
            for (int k = 0; k < weights[i][j].size(); k++)
            {
                json += std::to_string(weights[i][j][k]);
                if (k != weights[i][j].size() - 1)
                {
                    json += ",";
                }
            }
            json += "]";
            if (j != weights[i].size() - 1)
            {
                json += ",";
            }
            json += "\n";
        }
        json += "]";
        if (i != weights.size() - 1)
        {
            json += ",";
        }
        json += "\n";
    }

    json += "],\n";
    json += "\"biases\": [\n";

    // Add the biases to the JSON object
    for (int i = 0; i < bias.size(); i++)
    {
        json += "[";
        for (int j = 0; j < bias[i].size(); j++)
        {
            json += std::to_string(bias[i][j]);
            if (j != bias[i].size() - 1)
            {
                json += ",";
            }
        }
        json += "]";
        if (i != bias.size() - 1)
        {
            json += ",";
        }
        json += "\n";
    }

    json += "]\n";
    json += "}\n";

    // Write the JSON object to a file
    std::ofstream file(filename);
    if (file.is_open())
    {
        file << json;
        file.close();
    }
    else
    {
        std::cout << "Unable to open file `" << filename << "`!" << std::endl;
    }
}

// Helper function to split a string by a delimiter
std::vector<std::string> split(const std::string &str, char delimiter)
{
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}

void import_network(Network &network, std::string filename)
{
    // Read the JSON file
    std::ifstream file(filename);
    std::string json_string;
    file >> json_string;
    file.close();

    // Parse the JSON string
    std::vector<std::vector<double>> biases;
    std::vector<std::vector<std::vector<double>>> weights;

    // Extract biases
    size_t pos = json_string.find("\"biases\": [");
    if (pos != std::string::npos)
    {
        pos += 11;
        size_t end_pos = json_string.find("]", pos);
        if (end_pos != std::string::npos)
        {
            std::string biases_string = json_string.substr(pos, end_pos - pos);
            std::vector<std::string> biases_strings = split(biases_string, ';');
            for (const std::string &bias_string : biases_strings)
            {
                std::vector<double> bias;
                std::vector<std::string> bias_strings = split(bias_string, ',');
                for (const std::string &bias_value_string : bias_strings)
                {
                    bias.push_back(std::stod(bias_value_string));
                }
                biases.push_back(bias);
            }
        }
    }

    // Extract weights
    pos = json_string.find("\"weights\": [");
    if (pos != std::string::npos)
    {
        pos += 12;
        size_t end_pos = json_string.find("]", pos);
        if (end_pos != std::string::npos)
        {
            std::string weights_string = json_string.substr(pos, end_pos - pos);
            std::vector<std::string> weights_strings = split(weights_string, ';');
            for (const std::string &layer_weights_string : weights_strings)
            {
                std::vector<std::vector<double>> layer_weights;
                std::vector<std::string> neuron_weights_strings = split(layer_weights_string, '|');
                for (const std::string &neuron_weights_string : neuron_weights_strings)
                {
                    std::vector<double> neuron_weights;
                    std::vector<std::string> weight_strings = split(neuron_weights_string, ',');
                    for (const std::string &weight_string : weight_strings)
                    {
                        neuron_weights.push_back(std::stod(weight_string));
                    }
                    layer_weights.push_back(neuron_weights);
                }
                weights.push_back(layer_weights);
            }
        }
    }

    network.importWeightsBiases(biases, weights);
}