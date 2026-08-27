#include "core/matrix.h"
#include "math/matrix_operations.h"
#include "core/helpers.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int matricesHaveSameDimensions(Matrix *matrixA, Matrix *matrixB)
{
    return matrixA->rows == matrixB->rows && matrixA->cols == matrixB->cols;
}

Matrix *transposeMatrix(Matrix *matrix)
{
    Matrix *tMatrix = createMatrix(matrix->cols, matrix->rows);

    for (int i = 0; i < matrix->rows; i++)
    {
        for (int j = 0; j < matrix->cols; j++)
        {
            setMatrixValue(getMatrixValue(matrix, i, j), tMatrix, j, i);
        }
    }

    return tMatrix;
}

Matrix *multiplyMatrices(Matrix *matrixA, Matrix *matrixB)
{
    Matrix *dotMatrix = createMatrix(matrixA->rows, matrixB->cols);

    for (int i = 0; i < matrixA->rows; i++)
    {
        for (int j = 0; j < matrixB->cols; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < matrixA->cols; k++)
            {
                sum +=
                    getMatrixValue(matrixA, i, k) *
                    getMatrixValue(matrixB, k, j);
            }
            setMatrixValue(sum, dotMatrix, i, j);
        }
    }

    return dotMatrix;
}

Matrix *multiplyMatrixByScalar(Matrix *matrix, float scalar)
{
    Matrix *scalarMatrix = createMatrix(matrix->rows, matrix->cols);

    for (int i = 0; i < matrix->rows; i++)
    {
        for (int j = 0; j < matrix->cols; j++)
        {
            scalarMatrix->data[i * matrix->cols + j] = getMatrixValue(matrix, i, j) * scalar;
        }
    }

    return scalarMatrix;
}

Matrix *addMatrices(Matrix *matrixA, Matrix *matrixB)
{
    Matrix *result = createMatrix(matrixA->rows, matrixA->cols);

    for (int i = 0; i < matrixA->rows; i++)
    {
        for (int j = 0; j < matrixA->cols; j++)
        {
            result->data[i * matrixA->cols + j] =
                getMatrixValue(matrixA, i, j) +
                getMatrixValue(matrixB, i, j);
        }
    }

    return result;
}

Matrix *subtractMatrices(Matrix *matrixA, Matrix *matrixB)
{
    Matrix *result = createMatrix(matrixA->rows, matrixA->cols);

    for (int i = 0; i < matrixA->rows; i++)
    {
        for (int j = 0; j < matrixA->cols; j++)
        {
            result->data[i * matrixA->cols + j] =
                getMatrixValue(matrixA, i, j) -
                getMatrixValue(matrixB, i, j);
        }
    }

    return result;
}

Matrix *applyFunctionToMatrix(float (*func)(float), Matrix *matrix)
{
    Matrix *appliedMatrix = createMatrix(matrix->rows, matrix->cols);

    for (int i = 0; i < matrix->rows; i++)
    {
        for (int j = 0; j < matrix->cols; j++)
        {
            float value = getMatrixValue(matrix, i, j);
            setMatrixValue((*func)(value), appliedMatrix, i, j);
        }
    }

    return appliedMatrix;
}