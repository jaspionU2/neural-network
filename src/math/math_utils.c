#include "math/math_utils.h"
#include "core/matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

/*
 * sigmoid:
 *  - input: valor bruto da entrada.
 * Retorna: resultado da ativação sigmoide, que limita o valor ao intervalo (0, 1).
 * Essa função é usada para transformar sinais lineares em valores de ativação não lineares.
 */
void sigmoid(Matrix *matrix)
{
    int totalElements = matrix->rows * matrix->cols;
    for (int k = 0; k < totalElements; k++)
    {
        float z;

        if (matrix->data[k] < -88.0f)
            z = 0.0f;
        else if (matrix->data[k] > 88.0f)
            z = 1.0f;
        else
            z = 1.0f / (1 + expf(-(matrix->data[k])));

        matrix->data[k] = z;
    }
}

void softmax(Matrix *output)
{
    if (!output || !output->data)
        fatalError("Error: Invalid matrix passed to softmax.");

    for (int i = 0; i < output->rows; i++)
    {
        float maxVal = output->data[idxMatrix(i, 0, output->cols)];
        for (int j = 1; j < output->cols; j++)
        {
            float val = output->data[idxMatrix(i, j, output->cols)];
            if (val > maxVal)
                maxVal = val;
        }

        float sumExp = 0.0f;
        for (int j = 0; j < output->cols; j++)
        {
            float expVal = expf(output->data[idxMatrix(i, j, output->cols)] - maxVal);
            output->data[idxMatrix(i, j, output->cols)] = expVal;
            sumExp += expVal;
        }

        for (int j = 0; j < output->cols; j++)
        {
            output->data[idxMatrix(i, j, output->cols)] /= sumExp;
        }
    }
}

/*
 * crossEntropy:
 *  - idxExpectedClass: indíce da classe correta para a entrada processada.
 *  - output: matriz com a saída calculada pela rede.
 * Retorna: valor da perda de entropia cruzada entre a saída esperada e a saída prevista.
 * A função calcula a métrica de erro usada para avaliar o desempenho da rede neural.
 */
double crossEntropy(Matrix *output, int idxExpectedClass)
{
    return -log(getMatrixValue(output, 0, idxExpectedClass));
}

Matrix *sigmoidDerivative(Matrix *preActivation)
{
    if (!preActivation || !preActivation->data)
        fatalError("Error: Invalid matrix passed to sigmoidDerivative.");

    Matrix *matrix = createMatrix(preActivation->rows, preActivation->cols);

    int totalElements = preActivation->rows * preActivation->cols;
    float *data = preActivation->data;

    for (int i = 0; i < totalElements; i++)
    {
        matrix->data[i] = data[i] * (1 - data[i]);
    }

    return matrix;
}

Matrix *outputDelta(Matrix *output, Matrix *target)
{
    if (!output || !output->data)
        fatalError("Error: Invalid output passed to OutputDelta.");

    if (!target || !target->data)
        fatalError("Error: Invalid target passed to OutputDelta.");

    return subtractMatrices(output, target);
}

Matrix *hiddenDelta(Matrix *output, Matrix *nextWeights, Matrix *nextDelta)
{
    if (!output || !output->data)
        fatalError("Error: Invalid output passed to HiddenDelta.");

    if (!nextWeights || !nextWeights->data)
        fatalError("Error: Invalid nextWeights passed to HiddenDelta.");

    if (!nextDelta || !nextDelta->data)
        fatalError("Error: Invalid nextDelta passed to HiddenDelta.");

    Matrix *tWeights = transposeMatrix(nextWeights);
    Matrix *prodMatrix = multiplyMatrices(nextDelta, tWeights);

    Matrix *sigDerivate = sigmoidDerivative(output);

    Matrix *delta = hadamardProduct(prodMatrix, sigDerivate);

    freeMatrixData(tWeights);
    free(tWeights);

    freeMatrixData(prodMatrix);
    free(prodMatrix);

    freeMatrixData(sigDerivate);
    free(sigDerivate);

    return delta;
}

/*
 * weightedSum:
 *  - weights: vetor de pesos sinápticos.
 *  - wSize: tamanho do vetor de pesos.
 *  - inputs: vetor de entradas da camada.
 *  - iSize: tamanho do vetor de entradas.
 *  - bias: valor de bias somado ao resultado.
 * Retorna: soma ponderada das entradas mais o bias.
 * Essa função representa a parte linear da ativação de um neurônio.
 */
Matrix *weightedSum(Matrix *inputs, Matrix *weights, Matrix *bias)
{
    Matrix *dot = multiplyMatrices(inputs, weights);
    Matrix *z = addMatrices(dot, bias);

    freeMatrixData(dot);
    free(dot);

    return z;
}

void updateLayerParameters(NetworkLayer *layer, Matrix *prevInput, Matrix *delta, int batchSize, float learningRate)
{
    Matrix *tInput = transposeMatrix(prevInput);
    Matrix *gradW = multiplyMatrices(tInput, delta);

    float scale = learningRate / (float)batchSize;

    for (int k = 0; k < layer->weights->rows * layer->weights->cols; k++)
    {
        layer->weights->data[k] -= scale * gradW->data[k];
    }

    for (int j = 0; j < layer->bias->cols; j++)
    {
        float sumDelta = 0.0f;
        for (int b = 0; b < batchSize; b++)
        {
            sumDelta += getMatrixValue(delta, b, j);
        }
        layer->bias->data[j] -= scale * sumDelta;
    }

    freeMatrixData(tInput);
    free(tInput);
    
    freeMatrixData(gradW);
    free(gradW);
}