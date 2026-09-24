#pragma once

#include "core/matrix.h"

/*
 * matricesHaveSameDimensions:
 *  - matrixA: primeira matriz.
 *  - matrixB: segunda matriz.
 * Retorna: 1 quando as matrizes têm dimensões iguais e 0 em caso contrário.
 */
int matricesHaveSameDimensions(Matrix *matrixA, Matrix *matrixB);

/*
 * transposeMatrix:
 *  - matrix: matriz a ser transposta.
 * Retorna: nova matriz com linhas e colunas invertidas.
 */
Matrix *transposeMatrix(Matrix *matrix);

/*
 * multiplyMatrices:
 *  - matrixA: matriz à esquerda no produto.
 *  - matrixB: matriz à direita no produto.
 * Retorna: produto matricial entre matrixA e matrixB.
 */
Matrix *multiplyMatrices(Matrix *matrixA, Matrix *matrixB);

/*
 * multiplyMatrixByScalar:
 *  - matrix: matriz que será escalada.
 *  - scalar: valor escalar multiplicador.
 * Retorna: nova matriz com cada elemento multiplicado pelo escalar.
 */
Matrix *multiplyMatrixByScalar(Matrix *matrix, float scalar);

/*
 * addMatrices:
 *  - matrixA: primeira matriz.
 *  - matrixB: segunda matriz.
 * Retorna: matriz resultante da soma elemento a elemento.
 */
Matrix *addMatrices(Matrix *matrixA, Matrix *matrixB);

/*
 * subtractMatrices:
 *  - matrixA: matriz minuendo.
 *  - matrixB: matriz subtraendo.
 * Retorna: matriz com a diferença elemento a elemento.
 */
Matrix *subtractMatrices(Matrix *matrixA, Matrix *matrixB);

/*
 * hadamardProduct:
 *  - matrixA: primeira matriz.
 *  - matrixB: segunda matriz.
 * Retorna: produto de Hadamard (multiplicação elemento a elemento).
 * As matrizes devem ter as mesmas dimensões.
 */
Matrix *hadamardProduct(Matrix *matrixA, Matrix *matrixB);

/*
 * applyFunctionToMatrix:
 *  - func: função a ser aplicada a cada elemento.
 *  - matrix: matriz de entrada.
 * Retorna: nova matriz com a função aplicada individualmente em todos os elementos.
 */
Matrix *applyFunctionToMatrix(float (*func)(float), Matrix *matrix);
