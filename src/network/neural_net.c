#include "core/matrix.h"
#include "math/matrix_operations.h"
#include "core/helpers.h"
#include "network/neural_net.h"
#include <stdio.h>
#include <stdlib.h>

NetworkLayer newNetworkLayer(int n_neurons, int n_neuronsLast)
{
    Matrix *biases = createMatrix(1, n_neurons);
    Matrix *weights = createMatrix(n_neurons, n_neuronsLast);
    Matrix *output = createMatrix(1, n_neurons);

    if (biases == NULL || biases->data == NULL ||
        weights == NULL || weights->data == NULL ||
        output == NULL || output->data == NULL)
    {
        if (biases != NULL)
        {
            freeMatrixData(biases);
            free(biases);
            biases = NULL;
        }
        if (weights != NULL)
        {
            freeMatrixData(weights);
            free(weights);
            weights = NULL;
        }
        if (output != NULL)
        {
            freeMatrixData(output);
            free(output);
            output = NULL;
        }

        fatalError("Erro: Allocation fail. Don't possible to create new NetworkLayer.");
    }

    fillMatrixRandom(biases);
    fillMatrixRandom(weights);

    NetworkLayer layer = {
        .n_neurons = n_neurons,
        .bias = cloneMatrix(biases),
        .weights = cloneMatrix(weights),
        .output = cloneMatrix(output)};

    return layer;
}

void appendLayerToNeuralNet(NeuralNetModel *net, NetworkLayer layer)
{
    NetworkLayer *temp = realloc(net->layers, (net->layerCount + 1) * sizeof(NetworkLayer));

    if (!temp)
        fatalError("Error: Allocation fail. It was not possible to append an layer to the net.");

    net->layers = temp;

    net->layers[net->layerCount] = layer;

    net->layerCount++;
}

NeuralNetModel newNeuralNet(int n_Layers, int n_neurons, int inputNeurons, int outputNeurons)
{
    NeuralNetModel neuralNet = {.layerCount = 0, .layers = NULL};

    int n_lastLayer = inputNeurons;

    for (int i = 0; i < n_Layers; i++)
    {
        NetworkLayer currentLayer;

        if (i == n_Layers - 1)
        {
            currentLayer = newNetworkLayer(outputNeurons, n_lastLayer);
        }
        else
        {
            currentLayer = newNetworkLayer(n_neurons, n_lastLayer);
        }

        appendLayerToNeuralNet(&neuralNet, currentLayer);

        n_lastLayer = neuralNet.layers[i].n_neurons;
    }

    return neuralNet;
}

void freeNeuralNet(NeuralNetModel *neuralNet)
{
    if (neuralNet == NULL)
        return;

    for (int i = 0; i < neuralNet->layerCount; i++)
    {
        freeMatrixData(neuralNet->layers[i].weights);
        freeMatrixData(neuralNet->layers[i].bias);
        freeMatrixData(neuralNet->layers[i].output);
    }

    free(neuralNet->layers);
    neuralNet->layers = NULL;
    neuralNet->layerCount = 0;
}

void printNeuralNet(NeuralNetModel *net)
{
    if (net == NULL)
    {
        printf("NeuralNet: NULL\n");
        return;
    }

    printf("NeuralNet: %d layers\n", net->layerCount);

    for (int i = 0; i < net->layerCount; i++)
    {
        NetworkLayer *layer = &net->layers[i];
        printf("\nLayer %d:\n", i);
        printf("  neurons: %d\n", layer->n_neurons);

        printf("  weights:\n");
        printMatrixFormatted(layer->weights);

        printf("  bias:\n");
        printMatrixFormatted(layer->bias);

        printf("  output:\n");
        printMatrixFormatted(layer->output);
    }
}
