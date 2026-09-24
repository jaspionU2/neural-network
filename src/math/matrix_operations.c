#include "core/matrix.h"
#include "math/matrix_operations.h"
#include "core/helpers.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

/*
 * matricesHaveSameDimensions:
 *  - matrixA: primeira matriz a ser comparada.
 *  - matrixB: segunda matriz a ser comparada.
 * Retorna: 1 quando as dimensões são iguais e 0 em caso contrário.
 * Essa checagem é útil antes de operações entre matrizes para evitar erros de indexação.
 */
int matricesHaveSameDimensions(Matrix *matrixA, Matrix *matrixB)
{
    return matrixA->rows == matrixB->rows && matrixA->cols == matrixB->cols;
}

/*
 * transposeMatrix:
 *  - matrix: matriz original a ser transposta.
 * Retorna: nova matriz com linhas e colunas invertidas.
 * A transposição é usada em operações de álgebra linear e cálculo de gradientes.
 */
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

/*
 * multiplyMatrices:
 *  - matrixA: matriz à esquerda do produto.
 *  - matrixB: matriz à direita do produto.
 * Retorna: matriz resultante do produto matricial matrixA * matrixB.
 * Essa operação é fundamental para propagar sinais entre camadas da rede neural.
 */
Matrix *multiplyMatrices(Matrix *matrixA, Matrix *matrixB)
{
    if (!matrixA || !matrixB || !matrixA->data || !matrixB->data)
        fatalError("Error: Invalid matrix passed to multiplyMatrices.");

    if (matrixA->cols != matrixB->rows)
        fatalError("Error: Incompatible dimensions for matrix multiplication.");

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

/*
 * multiplyMatrixByScalar:
 *  - matrix: matriz de entrada.
 *  - scalar: valor escalar multiplicador.
 * Retorna: nova matriz com cada elemento multiplicado pelo escalar.
 * Essa operação é útil para ajustar magnitudes de valores em cálculos de aprendizado.
 */
Matrix *multiplyMatrixByScalar(Matrix *matrix, float scalar)
{
    Matrix *scalarMatrix = createMatrix(matrix->rows, matrix->cols);

    for (int i = 0; i < matrix->rows; i++)
    {
        for (int j = 0; j < matrix->cols; j++)
        {
            scalarMatrix->data[idxMatrix(i, j, matrix->cols)] = getMatrixValue(matrix, i, j) * scalar;
        }
    }

    return scalarMatrix;
}

Matrix *hadamardProduct(Matrix *matrixA, Matrix *matrixB)
{
    if (!matrixA || !matrixB || !matrixA->data || !matrixB->data)
        fatalError("Error: Invalid matrix passed to multiplyMatrices.");

    if (!matricesHaveSameDimensions(matrixA, matrixB))
        fatalError("Error: Incompatible dimensions for hadamard matrix multiplication.");

    Matrix *result = createMatrix(matrixA->rows, matrixA->cols);

    int totalElements = matrixA->rows * matrixA->cols;

    for (int k = 0; k < totalElements; k++)
    {
        result->data[k] = matrixA->data[k] * matrixB->data[k];
    }

    return result;
}

/*
 * addMatrices:
 *  - matrixA: primeira matriz.
 *  - matrixB: segunda matriz.
 * Retorna: matriz resultante da soma elemento a elemento.
 * Essa função combina os valores das matrizes posição por posição.
 */
Matrix *addMatrices(Matrix *matrixA, Matrix *matrixB)
{
    if (!matrixA || !matrixB || !matrixA->data || !matrixB->data)
        fatalError("Error: Invalid matrix passed to addMatrices.");

    bool rowsCompatible = (matrixA->rows == matrixB->rows) || (matrixA->rows == 1) || (matrixB->rows == 1);
    bool colsCompatible = (matrixA->cols == matrixB->cols) || (matrixA->cols == 1) || (matrixB->cols == 1);

    if (!rowsCompatible || !colsCompatible)
        fatalError("Error: Incompatible dimensions for matrix addition with broadcasting.");

    int outRows = (matrixA->rows > matrixB->rows) ? matrixA->rows : matrixB->rows;
    int outCols = (matrixA->cols > matrixB->cols) ? matrixA->cols : matrixB->cols;

    Matrix *result = createMatrix(outRows, outCols);

    for (int i = 0; i < outRows; i++)
    {
        for (int j = 0; j < outCols; j++)
        {
            int rowA = (matrixA->rows == 1) ? 0 : i;
            int colA = (matrixA->cols == 1) ? 0 : j;

            int rowB = (matrixB->rows == 1) ? 0 : i;
            int colB = (matrixB->cols == 1) ? 0 : j;

            float valA = getMatrixValue(matrixA, rowA, colA);
            float valB = getMatrixValue(matrixB, rowB, colB);

            setMatrixValue(valA + valB, result, i, j);
        }
    }

    return result;
}

/*
 * subtractMatrices:
 *  - matrixA: matriz minuendo.
 *  - matrixB: matriz subtraendo.
 * Retorna: matriz com a diferença elemento a elemento.
 * O resultado representa matrixA - matrixB em cada posição.
 */
Matrix *subtractMatrices(Matrix *matrixA, Matrix *matrixB)
{
    Matrix *result = createMatrix(matrixA->rows, matrixA->cols);

    int totalElements = matrixA->rows * matrixA->cols;

    for (int k = 0; k < totalElements; k++)
    {
        result->data[k] = matrixA->data[k] - matrixB->data[k];
    }

    return result;
}

/*
 * applyFunctionToMatrix:
 *  - func: ponteiro para a função que será aplicada a cada elemento.
 *  - matrix: matriz de entrada.
 * Retorna: nova matriz com a função aplicada em todos os elementos.
 * Isso facilita a aplicação de ativação ou transformações em toda a estrutura da matriz.
 */
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