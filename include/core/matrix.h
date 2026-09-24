#pragma once

typedef struct
{
    int rows;
    int cols;
    float *data;
} Matrix;

#define idxMatrix(row, col, totalCols) (((row) * (totalCols) + (col)))

/*
 * createMatrix:
 *  - rows: número de linhas da matriz.
 *  - cols: número de colunas da matriz.
 * Retorna: ponteiro para uma estrutura Matrix alocada dinamicamente.
 */
Matrix *createMatrix(int rows, int cols);

/*
 * setMatrixValue:
 *  - value: valor a ser armazenado.
 *  - matrix: matriz alvo.
 *  - row: índice da linha.
 *  - col: índice da coluna.
 * Define um valor específico em uma posição da matriz.
 */
void setMatrixValue(float value, Matrix *matrix, int row, int col);

/*
 * getMatrixValue:
 *  - matrix: matriz de origem.
 *  - row: índice da linha.
 *  - col: índice da coluna.
 * Retorna: o valor armazenado na posição indicada.
 */
float getMatrixValue(Matrix *matrix, int row, int col);

/*
 * fillMatrixWithValue:
 *  - matrix: matriz a ser preenchida.
 *  - value: valor constante para todos os elementos.
 * Preenche todos os elementos da matriz com um único valor.
 */
void fillMatrixWithValue(Matrix *matrix, float value);

/*
 * fillMatrixRandom:
 *  - matrix: matriz a receber valores aleatórios.
 * Preenche a estrutura com valores aleatórios em um intervalo uniforme.
 */
void fillMatrixRandom(Matrix *matrix, float min, float max);

/*
 * fillMatrixXavier:
 *  - matrix: matriz a receber os pesos inicializados.
 *  - fan_in: quantidade de entradas da camada de origem.
 *  - fan_out: quantidade de saídas da camada de destino.
 * Inicializa os valores da matriz usando a distribuição Xavier para manter a escala adequada.
 */
void fillMatrixXavier(Matrix *matrix, int fan_in, int fan_out);

/*
 * cloneMatrix:
 *  - matrix: matriz original a ser copiada.
 * Retorna: uma nova matriz com os mesmos dados da original.
 */
Matrix *cloneMatrix(Matrix *matrix);

/*
 * getBatchMatrix:
 *  - src: matriz contendo o dataset completo.
 *  - startIdx: índice da primeira linha do lote.
 *  - batchSize: quantidade de linhas a serem recortadas.
 * Extrai um bloco de linhas contíguas do dataset para formar o mini-batch.
 */
Matrix *getBatchMatrix(Matrix *src, int startIdx, int batchSize);

/*
 * flatMatrix:
 * Reshape da matriz original para uma única linha (1 x N).
 */
Matrix *flatMatrix(Matrix *matrix);

/*
 * randomUniformFloat:
 *  - min: limite inferior do intervalo.
 *  - max: limite superior do intervalo.
 * Retorna: número aleatório em ponto flutuante no intervalo informado.
 */
float randomUniformFloat(float min, float max);

/*
 * printMatrixFormatted:
 *  - matrix: matriz a ser impressa no terminal.
 * Exibe a matriz em formato legível para inspeção visual de valores.
 */
void printMatrixFormatted(Matrix *matrix);

/*
 * freeMatrixData:
 *  - matrix: matriz cujo buffer será liberado.
 * Libera os dados internos da matriz e zera suas dimensões.
 */
void freeMatrixData(Matrix *matrix);
