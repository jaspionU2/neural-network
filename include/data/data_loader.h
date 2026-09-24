#pragma once

#include "core/matrix.h"
#include <stddef.h>

typedef struct
{
    int label;
    Matrix *imgMatrix;
} Image;

/*
 * newImage:
 *  - label: rótulo ou classe associada à imagem.
 *  - rows: quantidade de linhas da matriz da imagem.
 *  - cols: quantidade de colunas da matriz da imagem.
 * Retorna: estrutura Image inicializada com uma matriz de pixels.
 */
Image *newImage(int label, int rows, int cols);

/*
 * loadImageDataset:
 *  - filename: caminho do arquivo CSV contendo as amostras.
 *  - n_sample: quantidade máxima de imagens a carregar.
 * Retorna: array dinâmico de imagens carregadas do dataset.
 */
Image **loadImageDataset(const char *filename, int n_sample, int startAt);

/*
 * cloneImage:
 *  - image: imagem original a ser duplicada.
 * Retorna: cópia profunda da imagem, incluindo a matriz de pixels.
 */
Image *cloneImage(Image *image);

/*
 * appendImageToDataset:
 *  - imgDataset: ponteiro para o array de imagens do dataset.
 *  - currentSize: ponteiro com o tamanho atual do dataset.
 *  - img: imagem a ser adicionada.
 * Adiciona uma nova imagem ao final do dataset redimensionando o vetor.
 */
void appendImageToDataset(Image ***imgDataset, int *currentSize, Image *img);

/*
 * datasetToMatrix:
 *  - imgDataset: conjunto de imagens a serem convertidas.
 *  - datasetSize: quantidade de imagens no conjunto.
 * Retorna: matriz unidimensional contendo os dados de todas as imagens em um formato linearizado.
 */
Matrix *datasetToMatrix(Image **imgDataset, int datasetSize);

/*
 * printImageMatrix:
 *  - imgMatrix: matriz da imagem a ser exibida.
 * Exibe os valores da imagem em formato legível para diagnóstico visual.
 */
void printImageMatrix(Matrix *imgMatrix);

/*
 * freeImageDataset:
 *  - dataset: ponteiro para o array de imagens a ser liberado.
 *  - datasetSize: número de elementos do dataset.
 * Libera toda a memória alocada para as imagens e para o vetor de ponteiros.
 */
void freeImageDataset(Image ***dataset, int datasetSize);