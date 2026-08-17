#pragma once

#include "matrix.h"

typedef struct
{
    int neurons;
    Matrix2D weights;
    float* bias;
    float* output;
} Layer;

typedef struct
{
    Layer *layers;
    int layerCount;
} NeuralNetwork;

void forwardPropagation(Matrix2D inputs, Matrix2D weights, Matrix2D bias, int layerCount, Matrix2D outputs);
void backwardPropagation();
