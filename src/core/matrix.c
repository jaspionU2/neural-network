#include "core/matrix.h"
#include "core/helpers.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

/*
 * createMatrix:
 *  - rows: número de linhas da matriz.
 *  - cols: número de colunas da matriz.
 * Retorna: ponteiro para uma estrutura Matrix recém-alocada.
 * A matriz serve como base para armazenar valores numéricos de entrada, pesos e saídas da rede.
 */
Matrix *createMatrix(int rows, int cols)
{
    Matrix *matrix = (Matrix *)malloc(sizeof(Matrix));

    if (!matrix)
        fatalError("Error: Allocation failed for Matrix header.");

    matrix->rows = rows <= 0 ? 1 : rows;
    matrix->cols = cols <= 0 ? 1 : cols;
    matrix->data = (float *)calloc(matrix->rows * matrix->cols, sizeof(float));

    if (!matrix->data)
        fatalError("Error: allocation from malloc get wrong.");

    return matrix;
}

/*
 * fillMatrixWithValue:
 *  - matrix: matriz a ser preenchida.
 *  - value: valor constante a ser atribuído a todos os elementos.
 * Preenche toda a matriz com um único valor, útil para inicializações e reset de dados.
 */
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

/*
 * setMatrixValue:
 *  - value: valor a ser atribuído.
 *  - matrix: matriz alvo.
 *  - row: índice da linha.
 *  - col: índice da coluna.
 * Define um valor específico em uma posição da matriz, permitindo atualização de elementos.
 */
void setMatrixValue(float value, Matrix *matrix, int row, int col)
{
    matrix->data[idxMatrix(row, col, matrix->cols)] = value;
}

/*
 * getMatrixValue:
 *  - matrix: matriz de origem.
 *  - row: índice da linha.
 *  - col: índice da coluna.
 * Retorna: o valor armazenado na posição indicada.
 * Esse acesso é usado em operações matemáticas e inspeção de dados da rede.
 */
float getMatrixValue(Matrix *matrix, int row, int col)
{
    return matrix->data[idxMatrix(row, col, matrix->cols)];
}

/*
 * randomUniformFloat:
 *  - min: limite inferior do intervalo.
 *  - max: limite superior do intervalo.
 * Retorna: número aleatório em ponto flutuante no intervalo informado.
 * É usado para gerar pesos e bias iniciais com distribuição uniforme.
 */
float randomUniformFloat(float min, float max)
{
    double scale = (double)rand() / RAND_MAX;
    return min + scale * (max - min);
}

/*
 * fillMatrixRandom:
 *  - matrix: matriz a receber valores aleatórios.
 * Preenche cada elemento com um valor aleatório para criar uma inicialização inicial da rede.
 */
void fillMatrixRandom(Matrix *matrix, float min, float max)
{
    for (int i = 0; i < matrix->rows; i++)
    {
        for (int j = 0; j < matrix->cols; j++)
        {
            setMatrixValue(randomUniformFloat(min, max), matrix, i, j);
        }
    }
}

void fillMatrixXavier(Matrix *matrix, int fan_in, int fan_out)
{
    float limit = sqrtf(6.0f / (fan_in + fan_out));

    for (int i = 0; i < matrix->rows; i++)
    {
        for (int j = 0; j < matrix->cols; j++)
        {
            setMatrixValue(randomUniformFloat(-limit, limit), matrix, i, j);
        }
    }
}

/*
 * cloneMatrix:
 *  - matrix: matriz original a ser copiada.
 * Retorna: nova matriz com os mesmos valores da original.
 * A cópia profunda evita que duas estruturas compartilhem o mesmo buffer de memória.
 */
Matrix *cloneMatrix(Matrix *matrix)
{
    if (!matrix || !matrix->data) return NULL;

    Matrix *dupMatrix = createMatrix(matrix->rows, matrix->cols);
    memcpy(dupMatrix->data, matrix->data, matrix->rows * matrix->cols * sizeof(float));
    return dupMatrix;
}

Matrix *flatMatrix(Matrix *matrix)
{
    if (!matrix || !matrix->data)
        fatalError("Error: Null matrix passed to flatMatrix.");

    int totalElements = matrix->rows * matrix->cols;
    Matrix *flattenedMatrix = createMatrix(1, totalElements);

    memcpy(flattenedMatrix->data, matrix->data, totalElements * sizeof(float));

    return flattenedMatrix;
}

Matrix *getBatchMatrix(Matrix *src, int startIdx, int batchSize)
{
    if (!src || !src->data)
        fatalError("Error: Invalid source matrix in getBatchView.");

    int actualBatchSize = batchSize;
    if (startIdx + batchSize > src->rows)
    {
        actualBatchSize = src->rows - startIdx;
    }

    Matrix *batch = createMatrix(actualBatchSize, src->cols);

    int offset = startIdx * src->cols;

    memcpy(batch->data, &src->data[offset], actualBatchSize * src->cols * sizeof(float));

    return batch;
}

/*
 * printMatrixFormatted:
 *  - matrix: matriz a ser exibida no terminal.
 * Exibe os valores em um formato legível para debugging e visualização de pesos e saídas.
 */
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

/*
 * freeMatrixData:
 *  - matrix: matriz cujo buffer será liberado.
 * Libera a memória interna do vetor de dados e reinicializa as dimensões da matriz.
 */
void freeMatrixData(Matrix *matrix)
{
    if (matrix == NULL)
        return;

    free(matrix->data);
    matrix->data = NULL;

    matrix->rows = 0;
    matrix->cols = 0;
}