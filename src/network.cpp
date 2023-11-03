#include "network.h"

#include <iostream>
#include <stdexcept> // For runtime_error
#include <chrono>    // For steady_clock
#include <thread>

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

std::vector<Layer> Network::getLayers() const
{
    return this->layers;
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

std::vector<double> Network::forward(const std::vector<double> &inputs)
{
    std::vector<double> current_inputs = inputs;
    for (Layer &layer : this->layers)
    {
        current_inputs = layer.forward(current_inputs);
    }
    return current_inputs;
}

void trainBatch(Network &net, const std::vector<std::vector<double>>& input_data, const std::vector<std::vector<double>>& target_data, double learning_rate, size_t start_idx, size_t end_idx, std::atomic<double>& epoch_loss)
{
    double local_loss = 0.0;

    for (size_t i = start_idx; i < end_idx; ++i)
    {
        const std::vector<double>& input = input_data[i];
        const std::vector<double>& target = target_data[i];

        // Forward pass
        std::vector<double> output = net.forward(input);

        std::vector<Layer> layers = net.getLayers();

        // Compute deltas
        layers.back().computeDeltas(target);

        // Compute loss
        local_loss += layers.back().computeLoss(target);

        // Backpropagate and update weights and biases
        for (int l = layers.size() - 1; l >= 0; --l)
        {
            // Get previous outputs and deltas
            const std::vector<double>& prev_outputs = (l == 0) ? input : layers[l - 1].getValues();

            // Compute deltas for the current layer (l) based on the next layer (l + 1)
            if (l != layers.size() - 1)
            {
                layers[l].computeDeltas(layers[l + 1]);
            }

            // Update weights and biases
            layers[l].backward(prev_outputs, learning_rate);
        }
    }

    epoch_loss = epoch_loss + local_loss;
}

void Network::train(const std::vector<std::vector<double>>& input_data, const std::vector<std::vector<double>>& target_data, double learning_rate, int epochs, int num_threads)
{
    // Check if input and target data have the same size
    if (input_data.size() != target_data.size())
    {
        throw std::runtime_error("Input and target data have different sizes.");
    }

    // Check if input data has the same size as the input layer
    if (input_data[0].size() != this->input_size)
    {
        throw std::runtime_error("Input data size does not match input layer size.");
    }

    // Check if target data has the same size as the output layer
    if (target_data[0].size() != this->layers.back().size())
    {
        throw std::runtime_error("Target data size does not match output layer size.");
    }

    // Start timer
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // Train the network for the specified number of epochs
    printf("\nTraining network...\n\n");

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        std::atomic<double> epoch_loss(0.0); // Initialize epoch loss to 0 using atomic to avoid data race

        // Calculate batch size
        size_t batch_size = input_data.size() / num_threads;

        // Create and start threads
        std::vector<std::thread> threads;
        for (int i = 0; i < num_threads; ++i)
        {
            size_t start_idx = i * batch_size;
            size_t end_idx = (i == num_threads - 1) ? input_data.size() : start_idx + batch_size;
            threads.emplace_back(trainBatch, std::ref(*this), std::ref(input_data), std::ref(target_data), learning_rate, start_idx, end_idx, std::ref(epoch_loss));
        }

        // Join threads
        for (std::thread& t : threads)
        {
            t.join();
        }

        // Divide by number of instances to get mean epoch loss
        epoch_loss = epoch_loss / input_data.size();

        // Print epoch loss every 1% of epochs
        if (epoch % (epochs / 100) == 0 || epoch == epochs - 1)
        {
            // Make a progress bar and display current epoch loss
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
            printf("] %d%% - Loss: %.2e", epoch * 100 / epochs, epoch_loss.load());

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
                printf("] 100%% - Loss: %.4e\n", epoch_loss.load());
            }
        }
    }

    printf("\nTraining complete for %d epochs with a learning rate of %.2f.\n\n", epochs, learning_rate);

    // Stop timer
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    // Print training time
    printf("Training time: %ld ms\n\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
}

// void Network::train(const std::vector<std::vector<double>> &input_data, const std::vector<std::vector<double>> &target_data, double learning_rate, int epochs)
// {
//     // Check if input and target data have the same size
//     if (input_data.size() != target_data.size())
//     {
//         throw std::runtime_error("Input and target data have different sizes.");
//     }

//     // Check if input data has the same size as the input layer
//     if (input_data[0].size() != this->input_size)
//     {
//         throw std::runtime_error("Input data size does not match input layer size.");
//     }

//     // Check if target data has the same size as the output layer
//     if (target_data[0].size() != this->layers.back().size())
//     {
//         throw std::runtime_error("Target data size does not match output layer size.");
//     }

//     // Start timer
//     std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

//     // Train the network for the specified number of epochs
//     printf("\nTraining network...\n\n");

//     for (int epoch = 0; epoch < epochs; ++epoch)
//     {
//         double epoch_loss = 0.0; // Initialize epoch loss to 0

//         for (size_t i = 0; i < input_data.size(); ++i)
//         {
//             const std::vector<double> &input = input_data[i];
//             const std::vector<double> &target = target_data[i];

//             // Forward pass
//             std::vector<double> output = this->forward(input);

//             // Compute deltas
//             this->layers.back().computeDeltas(target);

//             // Compute loss
//             epoch_loss += this->layers.back().computeLoss(target);

//             // Backpropagate and update weights and biases
//             for (int l = this->layers.size() - 1; l >= 0; --l)
//             {
//                 // Get previous outputs and deltas
//                 const std::vector<double> &prev_outputs = (l == 0) ? input : this->layers[l - 1].getValues();

//                 // Compute deltas for the current layer (l) based on the next layer (l + 1)
//                 if (l != this->layers.size() - 1)
//                 {
//                     this->layers[l].computeDeltas(this->layers[l + 1]);
//                 }

//                 // Update weights and biases
//                 this->layers[l].backward(prev_outputs, learning_rate);
//             }
//         }

//         // Divide by number of instances to get mean epoch loss
//         epoch_loss /= input_data.size();

//         // Print epoch loss every 1% of epochs
//         if (epoch % (epochs / 100) == 0 || epoch == epochs - 1)
//         {
//             // Make a progress bar and display current epoch loss
//             printf("\r[");
//             int pos = 50 * epoch / epochs;
//             for (int i = 0; i <= 50; ++i)
//             {
//                 if (i < pos)
//                 {
//                     printf("=");
//                 }
//                 else if (i == pos)
//                 {
//                     printf(">");
//                 }
//                 else
//                 {
//                     printf(" ");
//                 }
//             }
//             printf("] %d%% - Loss: %.4e", epoch * 100 / epochs, epoch_loss);

//             // Flush stdout
//             fflush(stdout);

//             // Print 100% and a full bar on the last epoch
//             if (epoch == epochs - 1)
//             {
//                 printf("\r[");
//                 for (int i = 0; i <= 50; ++i)
//                 {
//                     printf("=");
//                 }
//                 printf("] 100%% - Loss: %.4e\n", epoch_loss);
//             }
//         }
//     }

//     printf("\nTraining complete for %d epochs with a learning rate of %.2f.\n\n", epochs, learning_rate);

//     // Stop timer
//     std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

//     // Print training time
//     printf("Training time: %ld ms\n\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
// }

std::vector<double> Network::predict(const std::vector<double> &input)
{
    return this->forward(input);
}