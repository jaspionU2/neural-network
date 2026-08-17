#include "matrix.h"
#include "matrix_operations.h"
#include "helpers.h"
#include "stdio.h"
#include "stdlib.h"

int checkDimensions(Matrix2D matrixA, Matrix2D matrixB)
{
    if (matrixA.rows == matrixB.rows && matrixA.cols == matrixB.cols)
        return 1;
    return 0;
}

Matrix2D transposeMatrix(Matrix2D matrix)
{
    if (!matrix.data)
        die("Isn't possible to transpose a empty matrix.");

    Matrix2D tMatrix = newMatrix(matrix.cols, matrix.rows);

    for (int i = 0; i < matrix.rows; i++)
    {
        for (int j = 0; j < matrix.cols; j++)
        {
            tMatrix.data[j * matrix.cols + i] = matrix.data[i * matrix.cols + j];
        }
    }

    return tMatrix;
}

Matrix2D multiplyMatrix(Matrix2D matrixA, Matrix2D matrixB)
{
    if (matrixA.cols != matrixB.rows)
        die("The cols in matrix A, don't equal of rows in matrix B");

    Matrix2D dotMatrix = newMatrix(matrixA.rows, matrixB.cols);

    for (int i = 0; i < matrixA.rows; i++)
    {
        for (int j = 0; j < matrixB.cols; j++)
        {
            dotMatrix.data[i * matrixB.cols + j] = 0.0f;
            for (int k = 0; k < matrixA.cols; k++)
            {
                dotMatrix.data[i * matrixB.cols + j] += 
                getFromMatrix(matrixA, i, k) * 
                getFromMatrix(matrixB, k, j);
            }
        }
    }

    return dotMatrix;
}

Matrix2D multiplyMatrixScalar(Matrix2D matrix, float scalar)
{
    Matrix2D scalarMatrix = newMatrix(matrix.rows, matrix.cols);

    for (int i = 0; i < matrix.rows; i++)
    {
        for (int j = 0; j < matrix.cols; j++)
        {
            scalarMatrix.data[i * matrix.cols + j] = getFromMatrix(matrix, i, j) * scalar;
        }
    }

    return scalarMatrix;
}

Matrix2D sumMatrix(Matrix2D matrixA, Matrix2D matrixB)
{
    if (!checkDimensions(matrixA, matrixB))
        die("sumMatrix: The matrices do not have equivalent dimensions.");

    Matrix2D sumMatrix = newMatrix(matrixA.rows, matrixA.cols);

    for (int i = 0; i < matrixA.rows; i++)
    {
        for (int j = 0; j < matrixA.cols; j++)
        {
            sumMatrix.data[i * matrixB.cols + j] =
            getFromMatrix(matrixA, i, j) +
            getFromMatrix(matrixB, i, j);
        }
    }

    return sumMatrix;
}

Matrix2D subMatrix(Matrix2D matrixA, Matrix2D matrixB)
{
    if (!checkDimensions(matrixA, matrixB))
        die("The matrices do not have equivalent dimensions.");

    Matrix2D subMatrix = newMatrix(matrixA.rows, matrixA.cols);

    for (int i = 0; i < matrixA.rows; i++)
    {
        for (int j = 0; j < matrixA.cols; j++)
        {
            subMatrix.data[i * matrixB.cols + j] =
                getFromMatrix(matrixA, i, j) -
                getFromMatrix(matrixB, i, j);
        }
    }

    return subMatrix;
}

Matrix2D applyFunc(float (*func)(float), Matrix2D matrix)
{
    Matrix2D appliedMatrix = newMatrix(matrix.rows, matrix.cols);

    for (int i = 0; i < matrix.rows; i++)
    {
        for (int j = 0; j < matrix.cols; j++)
        {
            float value = getFromMatrix(matrix, i, j);
            addToMatrix((*func)(value), appliedMatrix, i, j);
        }
    }

    return appliedMatrix;
}