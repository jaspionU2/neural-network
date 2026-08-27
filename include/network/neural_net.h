#pragma once

#include "core/matrix.h"
#include "data/data_loader.h"

typedef struct
{
    int n_neurons;
    Matrix *weights;
    Matrix *bias;
    Matrix *output;
} NetworkLayer;

typedef struct
{
    NetworkLayer *layers;
    int layerCount;
} NeuralNetModel;

NetworkLayer newNetworkLayer(int n_neurons, int n_neuronsLast);
NeuralNetModel newNeuralNet(int n_Layers, int n_neurons, int inputNeurons, int outuputNeurons);
void appendLayerToNeuralNet(NeuralNetModel *net, NetworkLayer layer);
void propagateForward(Matrix inputs, Matrix weights, Matrix bias, int layerCount, Matrix outputs);
void propagateBackward();
NeuralNetModel trainNeuralNetOnImages(NeuralNetModel *net, Image **dataset, int datasetSize, int epochs, float learningRate);
void printNeuralNet(NeuralNetModel *net);
void freeNeuralNet(NeuralNetModel *neuralNet);