#pragma once

#include "core/matrix.h"

/*
 * matricesHaveSameDimensions:
 *  - matrix_a: primeira matriz.
 *  - matrix_b: segunda matriz.
 * Retorna: 1 quando as matrizes têm dimensões iguais e 0 em caso contrário.
 */
int matricesHaveSameDimensions(Matrix *matrix_a, Matrix *matrix_b);

/*
 * transposeMatrix:
 *  - matrix: matriz a ser transposta.
 * Retorna: nova matriz com linhas e colunas invertidas.
 */
Matrix *transposeMatrix(Matrix *matrix);

/*
 * multiplyMatrices:
 *  - matrix_a: matriz à esquerda no produto.
 *  - matrix_b: matriz à direita no produto.
 * Retorna: produto matricial entre matrix_a e matrix_b.
 */
Matrix *multiplyMatrices(Matrix *matrix_a, Matrix *matrix_b);

/*
 * multiplyMatrixByScalar:
 *  - matrix: matriz que será escalada.
 *  - scalar: valor escalar multiplicador.
 * Retorna: nova matriz com cada elemento multiplicado pelo escalar.
 */
Matrix *multiplyMatrixByScalar(Matrix *matrix, float scalar);

/*
 * addMatrices:
 *  - matrix_a: primeira matriz.
 *  - matrix_b: segunda matriz.
 * Retorna: matriz resultante da soma elemento a elemento.
 */
Matrix *addMatrices(Matrix *matrix_a, Matrix *matrix_b);

/*
 * subtractMatrices:
 *  - matrix_a: matriz minuendo.
 *  - matrix_b: matriz subtraendo.
 * Retorna: matriz com a diferença elemento a elemento.
 */
Matrix *subtractMatrices(Matrix *matrix_a, Matrix *matrix_b);

/*
 * hadamardProduct:
 *  - matrix_a: primeira matriz.
 *  - matrix_b: segunda matriz.
 * Retorna: produto de Hadamard (multiplicação elemento a elemento).
 * As matrizes devem ter as mesmas dimensões.
 */
Matrix *hadamardProduct(Matrix *matrix_a, Matrix *matrix_b);

/*
 * applyFunctionToMatrix:
 *  - func: função a ser aplicada a cada elemento.
 *  - matrix: matriz de entrada.
 * Retorna: nova matriz com a função aplicada individualmente em todos os elementos.
 */
Matrix *applyFunctionToMatrix(float (*func)(float), Matrix *matrix);
