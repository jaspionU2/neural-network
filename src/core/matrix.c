#include "core/matrix.h"
#include "core/helpers.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

Matrix *createMatrix(int rows, int cols)
{
    Matrix *matrix = (Matrix *)malloc(sizeof(Matrix));

    matrix->rows = rows <= 0 ? 1 : rows;
    matrix->cols = cols <= 0 ? 1 : cols;
    matrix->data = (float *)malloc(rows * cols * sizeof(float));

    if (!matrix->data)
        fatalError("Error: allocation from malloc get wrong.");

    return matrix;
}

void fillMatrixWithValue(Matrix *matrix, float value)
{
    for (int i = 0; i < matrix->rows; i++)
    {
        for (int j = 0; j < matrix->cols; j++)
        {
            setMatrixValue(value, matrix, i, j);
        }
    }
}

void setMatrixValue(float value, Matrix *matrix, int row, int col)
{
    matrix->data[row * matrix->cols + col] = value;
}

float getMatrixValue(Matrix *matrix, int row, int col)
{
    return matrix->data[row * matrix->cols + col];
}

float randomUniformFloat(float min, float max)
{
    double scale = (double)rand() / RAND_MAX;
    return min + scale * (max - min);
}

void fillMatrixRandom(Matrix *matrix)
{
    for (int i = 0; i < matrix->rows; i++)
    {
        for (int j = 0; j < matrix->cols; j++)
        {
            setMatrixValue(randomUniformFloat(0.0f, 1.0f), matrix, i, j);
        }
    }
}

Matrix *cloneMatrix(Matrix *matrix)
{
    Matrix *dupMatrix = createMatrix(matrix->rows, matrix->cols);

    memcpy(dupMatrix->data, matrix->data, matrix->rows * matrix->cols * sizeof(float));

    return dupMatrix;
}

void printMatrixFormatted(Matrix *matrix)
{
    printf("rows: %d cols: %d\n", matrix->rows, matrix->cols);
    for (int i = 0; i < matrix->rows; i++)
    {
        printf("[");
        for (int j = 0; j < matrix->cols; j++)
        {
            printf(" %.1f", getMatrixValue(matrix, i, j));
        }
        printf(" ]\n");
    }
}

void freeMatrixData(Matrix *matrix)
{
    if (matrix == NULL)
        return;

    free(matrix->data);
    matrix->data = NULL;

    matrix->rows = 0;
    matrix->cols = 0;
}