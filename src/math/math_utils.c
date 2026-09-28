#include "math/math_utils.h"
#include "core/matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

void sigmoid(Matrix *matrix)
{
    int total_elements = matrix->rows * matrix->cols;
    for (int element_index = 0; element_index < total_elements; element_index++)
    {
        float activated_value;

        if (matrix->data[element_index] < -88.0f)
            activated_value = 0.0f;
        else if (matrix->data[element_index] > 88.0f)
            activated_value = 1.0f;
        else
            activated_value = 1.0f / (1 + expf(-(matrix->data[element_index])));

        matrix->data[element_index] = activated_value;
    }
}

void softmax(Matrix *output)
{
    if (!output || !output->data)
        fatalError("Error: Invalid matrix passed to softmax.");

    for (int row_index = 0; row_index < output->rows; row_index++)
    {
        float max_value = output->data[idxMatrix(row_index, 0, output->cols)];
        for (int col_index = 1; col_index < output->cols; col_index++)
        {
            float current_value = output->data[idxMatrix(row_index, col_index, output->cols)];
            if (current_value > max_value)
                max_value = current_value;
        }

        float sum_exp = 0.0f;
        for (int col_index = 0; col_index < output->cols; col_index++)
        {
            float exp_value = expf(output->data[idxMatrix(row_index, col_index, output->cols)] - max_value);
            output->data[idxMatrix(row_index, col_index, output->cols)] = exp_value;
            sum_exp += exp_value;
        }

        for (int col_index = 0; col_index < output->cols; col_index++)
        {
            output->data[idxMatrix(row_index, col_index, output->cols)] /= sum_exp;
        }
    }
}

Matrix *sigmoidDerivative(Matrix *pre_activation)
{
    if (!pre_activation || !pre_activation->data)
        fatalError("Error: Invalid matrix passed to sigmoidDerivative.");

    Matrix *derivative_matrix = createMatrix(pre_activation->rows, pre_activation->cols);
    int total_elements = pre_activation->rows * pre_activation->cols;
    float *source_data = pre_activation->data;

    for (int element_index = 0; element_index < total_elements; element_index++)
    {
        derivative_matrix->data[element_index] = source_data[element_index] * (1 - source_data[element_index]);
    }

    return derivative_matrix;
}

Matrix *outputDelta(Matrix *output, Matrix *target)
{
    if (!output || !output->data)
        fatalError("Error: Invalid output passed to OutputDelta.");

    if (!target || !target->data)
        fatalError("Error: Invalid target passed to OutputDelta.");

    return subtractMatrices(output, target);
}

Matrix *hiddenDelta(Matrix *output, Matrix *next_weights, Matrix *next_delta)
{
    if (!output || !output->data)
        fatalError("Error: Invalid output passed to HiddenDelta.");

    if (!next_weights || !next_weights->data)
        fatalError("Error: Invalid nextWeights passed to HiddenDelta.");

    if (!next_delta || !next_delta->data)
        fatalError("Error: Invalid nextDelta passed to HiddenDelta.");

    Matrix *transposed_weights = transposeMatrix(next_weights);
    Matrix *product_matrix = multiplyMatrices(next_delta, transposed_weights);
    Matrix *sigmoid_derivative = sigmoidDerivative(output);
    Matrix *delta = hadamardProduct(product_matrix, sigmoid_derivative);

    freeMatrixData(transposed_weights);
    free(transposed_weights);

    freeMatrixData(product_matrix);
    free(product_matrix);

    freeMatrixData(sigmoid_derivative);
    free(sigmoid_derivative);

    return delta;
}

Matrix *weightedSum(Matrix *inputs, Matrix *weights, Matrix *bias)
{
    Matrix *dot_product = multiplyMatrices(inputs, weights);
    Matrix *activation = addMatrices(dot_product, bias);

    freeMatrixData(dot_product);
    free(dot_product);

    return activation;
}

void updateLayerParameters(NetworkLayer *layer, Matrix *prev_input, Matrix *delta, int batch_size, float learning_rate)
{
    Matrix *transposed_input = transposeMatrix(prev_input);
    Matrix *gradient_weights = multiplyMatrices(transposed_input, delta);
    float scale = learning_rate / (float)batch_size;

    int total_elements = layer->weights->rows * layer->weights->cols;
    for (int element_index = 0; element_index < total_elements; element_index++)
    {
        layer->weights->data[element_index] -= scale * gradient_weights->data[element_index];
    }

    for (int col_index = 0; col_index < layer->bias->cols; col_index++)
    {
        float sum_delta = 0.0f;
        for (int batch_index = 0; batch_index < batch_size; batch_index++)
        {
            sum_delta += delta->data[idxMatrix(batch_index, col_index, delta->cols)];
        }
        layer->bias->data[col_index] -= scale * sum_delta;
    }

    freeMatrixData(transposed_input);
    free(transposed_input);

    freeMatrixData(gradient_weights);
    free(gradient_weights);
}

float CategoricalCrossEntropy(Matrix *output, Matrix *target)
{
    if (!output || !output->data)
        fatalError("Error: Invalid output passed to computeMeanLoss.");

    if (!target || !target->data)
        fatalError("Error: Invalid target passed to computeMeanLoss.");

    if (output->rows != target->rows || output->cols != target->cols)
        fatalError("Error: Incompatible dimensions for computeMeanLoss.");

    int total_elements = output->rows * output->cols;
    float total_loss = 0.0f;

    for (int element_index = 0; element_index < total_elements; element_index++)
    {
        float prediction = output->data[element_index];
        float target_value = target->data[element_index];

        prediction = fmaxf(1e-7f, fminf(prediction, 1.0f - 1e-7f));
        total_loss += -target_value * logf(prediction);
    }

    return total_loss / (float)output->rows;
}