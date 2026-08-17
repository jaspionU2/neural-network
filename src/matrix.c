#include "matrix.h"
#include "helpers.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

Matrix2D newMatrix(int rows, int cols)
{
    Matrix2D matrix;

    matrix.rows = rows <= 0 ? 1 : rows;
    matrix.cols = cols <= 0 ? 1 : cols;
    matrix.data = (float *)malloc(rows * cols * sizeof(float));

    if (!matrix.data)
        die("Error: allocation from malloc get wrong.");
    
    return matrix;
}

void fillMatrix(Matrix2D matrix, float value)
{
    for (int i = 0; i < matrix.rows; i++)
    {
        for (int j = 0; j < matrix.cols; j++)
        {
            addToMatrix(value, matrix, i, j);
        }
    }
}

void addToMatrix(float value, Matrix2D matrix, int row, int col)
{
    matrix.data[row * matrix.cols + col] = value;
}

float getFromMatrix(Matrix2D matrix, int row, int col)
{
    return matrix.data[row * matrix.cols + col];
}

float randomUniform(float min, float max)
{
    double scale = (double)rand() / RAND_MAX; 
    return min + scale * (max - min);
}

void randMatrix(Matrix2D matrix)
{
    for (int i = 0; i < matrix.rows; i++)
    {
       for (int j = 0; j < matrix.cols; j++)
       {
           addToMatrix(randomUniform(0.0f, 5.0f), matrix, i, j);
       }
    }
}

void printMatrix(Matrix2D matrix)
{
    printf("rows: %d cols: %d\n", matrix.rows, matrix.cols);
    for (int i = 0; i < matrix.rows; i++)
    {
        printf("[");
        for (int j = 0; j < matrix.cols; j++)
        {
            printf(" %.1f", getFromMatrix(matrix, i, j));
        }
        printf(" ]\n");
    }
}

void freeMatrix(Matrix2D matrix)
{
    if (matrix.data != NULL) {
        free(matrix.data);
        matrix.data = NULL;
    }
    matrix.rows = 0;
    matrix.cols = 0;
}