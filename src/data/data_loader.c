#include "core/matrix.h"
#include "math/math_utils.h"
#include "data/data_loader.h"
#include "core/helpers.h"
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/*
 * newImage:
 *  - label: rótulo ou classe associada à imagem.
 *  - rows: número de linhas da matriz de pixels.
 *  - cols: número de colunas da matriz de pixels.
 * Retorna: ponteiro para uma Image recém-alocada e pronta para receber os dados da imagem.
 * Essa estrutura encapsula a classe e a matriz que representa os valores do pixel.
 */
Image *newImage(int label, int rows, int cols)
{
    Image *img = (Image *)malloc(sizeof(Image));

    if (!img)
        fatalError("Error: Fail to allocate memory to Image struct.");

    img->label = label;
    img->imgMatrix = createMatrix(rows, cols);

    if (!img->imgMatrix)
        fatalError("Error: Allocation fail. It was not possible to create a new image matrix.");

    return img;
}

/*
 * cloneImage:
 *  - image: imagem original a ser duplicada.
 * Retorna: nova imagem com cópia profunda da matriz de pixels.
 * A cópia evita que alterações em uma instância afetem a outra.
 */
Image *cloneImage(Image *image)
{
    Image *dupImage = newImage(image->label, 0, 0);
    dupImage->imgMatrix = cloneMatrix(image->imgMatrix);

    return dupImage;
}

/*
 * appendImageToDataset:
 *  - imgDataset: ponteiro para o array de imagens.
 *  - currentSize: ponteiro com o tamanho atual do dataset.
 *  - img: imagem a ser adicionada.
 * Adiciona uma imagem ao final do dataset, realocando o vetor conforme necessário.
 */
void appendImageToDataset(Image ***imgDataset, int *currentSize, Image *img)
{
    int newSize = *currentSize + 1;

    Image **temp = realloc(*imgDataset, newSize * sizeof(Image *));

    if (!temp)
        fatalError("Error: realloc to a imageDataset fail.");

    *imgDataset = temp;

    (*imgDataset)[*currentSize] = cloneImage(img);
    *currentSize = newSize;
}

/*
 * loadImageDataset:
 *  - filename: caminho do arquivo CSV que contém as imagens.
 *  - n_sample: número máximo de amostras a carregar.
 * Retorna: array de ponteiros para imagens lidas do arquivo.
 * A função percorre o CSV, extrai rótulo e pixels e monta um dataset para treino ou teste.
 */
Image **loadImageDataset(const char *filename, int n_sample, int startAt)
{
    FILE *fp = fopen(filename, "r");

    if (!fp)
        fatalError("Error: fopen cannot open the file, filename does not exist or it's wrong");

    char buffer[8912];
    int count_samples = 0;
    Image **ds = NULL;

    int isHeader = 1;
    int currentDsSize = 0;

    for (int i = 0; i < startAt; i++)
    {
        if (fgets(buffer, sizeof(buffer), fp) == NULL) 
            break;
    }

    while (fgets(buffer, sizeof(buffer), fp) != NULL && count_samples < n_sample)
    {
        if (isHeader)
        {
            isHeader = 0;
            continue;
        }

        buffer[strcspn(buffer, "\r\n")] = '\0';

        char *token = strtok(buffer, ",");

        if (!token)
            fatalError("Error: strtok, there are no tokens to read.");

        int label_val = atoi(token);
        Image *img = newImage(label_val, 28, 28);

        for (int i = 0; i < img->imgMatrix->rows; i++)
        {
            for (int j = 0; j < img->imgMatrix->cols; j++)
            {
                token = strtok(NULL, ",");
                if (token)
                {
                    float value = atoi(token) / 255.0f;
                    setMatrixValue(value, img->imgMatrix, i, j);
                }
            }
        }

        count_samples++;
        appendImageToDataset(&ds, &currentDsSize, img);
    }

    fclose(fp);

    return ds;
}

Matrix *datasetToMatrix(Image **imgDataset, int datasetSize)
{
    if (!imgDataset || datasetSize <= 0)
        fatalError("Error: Invalid dataset passed to flatDataset.");

    if (!imgDataset[0] || !imgDataset[0]->imgMatrix)
        fatalError("Error: First image in dataset is uninitialized.");

    int pixels_per_img = imgDataset[0]->imgMatrix->rows * imgDataset[0]->imgMatrix->cols;

    Matrix *flatData = createMatrix(datasetSize, pixels_per_img);

    for (int i = 0; i < datasetSize; i++)
    {
        if (!imgDataset[i] || !imgDataset[i]->imgMatrix || !imgDataset[i]->imgMatrix->data)
            fatalError("Error: Found null image during row dataset flattening.");

        float *src = imgDataset[i]->imgMatrix->data;

        for (int j = 0; j < pixels_per_img; j++)
        {
            flatData->data[idxMatrix(i, j, flatData->cols)] = src[j];
        }
    }

    return flatData;
}

void printImageMatrix(Matrix *imgMatrix)
{
    for (int i = 0; i < imgMatrix->rows; i++)
    {
        for (int j = 0; j < imgMatrix->cols; j++)
        {
            float v = getMatrixValue(imgMatrix, i, j);

            if (v > 0.8f)
                printf("##");
            else if (v > 0.5f)
                printf("  ");
            else if (v > 0.2f)
                printf("..");
            else
                printf("  ");
        }
        printf("\n");
    }
}

/*
 * freeImageDataset:
 *  - dataset: ponteiro para o array de imagens a ser liberado.
 *  - datasetSize: número de imagens no dataset.
 * Libera todas as imagens e o vetor de ponteiros, evitando vazamento de memória.
 */
void freeImageDataset(Image ***dataset, int datasetSize)
{
    if (*dataset)
    {
        for (int i = 0; i < datasetSize; i++)
        {
            freeMatrixData((*dataset)[i]->imgMatrix);
            free((*dataset)[i]);
        }
    }

    free(*dataset);
}
