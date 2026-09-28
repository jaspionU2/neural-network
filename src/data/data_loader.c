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
    Image *image = (Image *)malloc(sizeof(Image));

    if (!image)
        fatalError("Error: Fail to allocate memory to Image struct.");

    image->label = label;
    image->imgMatrix = createMatrix(rows, cols);

    if (!image->imgMatrix)
        fatalError("Error: Allocation fail. It was not possible to create a new image matrix.");

    return image;
}

/*
 * cloneImage:
 *  - image: imagem original a ser duplicada.
 * Retorna: nova imagem com cópia profunda da matriz de pixels.
 * A cópia evita que alterações em uma instância afetem a outra.
 */
Image *cloneImage(Image *image)
{
    Image *duplicate_image = newImage(image->label, 0, 0);
    duplicate_image->imgMatrix = cloneMatrix(image->imgMatrix);

    return duplicate_image;
}

/*
 * appendImageToDataset:
 *  - image_dataset: ponteiro para o array de imagens.
 *  - current_size: ponteiro com o tamanho atual do dataset.
 *  - image: imagem a ser adicionada.
 * Adiciona uma imagem ao final do dataset, realocando o vetor conforme necessário.
 */
void appendImageToDataset(Image ***image_dataset, int *current_size, Image *image)
{
    int new_size = *current_size + 1;
    Image **resized_dataset = realloc(*image_dataset, new_size * sizeof(Image *));

    if (!resized_dataset)
        fatalError("Error: realloc to a imageDataset fail.");

    *image_dataset = resized_dataset;
    (*image_dataset)[*current_size] = cloneImage(image);
    *current_size = new_size;
}

/*
 * loadImageDataset:
 *  - filename: caminho do arquivo CSV que contém as imagens.
 *  - sample_limit: número máximo de amostras a carregar.
 * Retorna: array de ponteiros para imagens lidas do arquivo.
 * A função percorre o CSV, extrai rótulo e pixels e monta um dataset para treino ou teste.
 */
Image **loadImageDataset(const char *filename, int sample_limit, int start_offset)
{
    FILE *file = fopen(filename, "r");

    if (!file)
        fatalError("Error: fopen cannot open the file, filename does not exist or it's wrong");

    char buffer[8912];
    int loaded_samples = 0;
    Image **dataset = NULL;
    int has_header = 1;
    int dataset_size = 0;

    for (int i = 0; i < start_offset; i++)
    {
        if (fgets(buffer, sizeof(buffer), file) == NULL)
            break;
    }

    while (fgets(buffer, sizeof(buffer), file) != NULL && loaded_samples < sample_limit)
    {
        if (has_header)
        {
            has_header = 0;
            continue;
        }

        buffer[strcspn(buffer, "\r\n")] = '\0';
        char *token = strtok(buffer, ",");

        if (!token)
            fatalError("Error: strtok, there are no tokens to read.");

        int label_value = atoi(token);
        Image *image = newImage(label_value, 28, 28);

        for (int row_index = 0; row_index < image->imgMatrix->rows; row_index++)
        {
            for (int col_index = 0; col_index < image->imgMatrix->cols; col_index++)
            {
                token = strtok(NULL, ",");
                if (token)
                {
                    float value = atoi(token) / 255.0f;
                    image->imgMatrix->data[idxMatrix(row_index, col_index, image->imgMatrix->cols)] = value;
                }
            }
        }

        loaded_samples++;
        appendImageToDataset(&dataset, &dataset_size, image);
    }

    fclose(file);

    return dataset;
}

Matrix *datasetToMatrix(Image **image_dataset, int dataset_size)
{
    if (!image_dataset || dataset_size <= 0)
        fatalError("Error: Invalid dataset passed to flatDataset.");

    if (!image_dataset[0] || !image_dataset[0]->imgMatrix)
        fatalError("Error: First image in dataset is uninitialized.");

    int pixels_per_image = image_dataset[0]->imgMatrix->rows * image_dataset[0]->imgMatrix->cols;
    Matrix *flat_data = createMatrix(dataset_size, pixels_per_image);

    for (int image_index = 0; image_index < dataset_size; image_index++)
    {
        if (!image_dataset[image_index] || !image_dataset[image_index]->imgMatrix || !image_dataset[image_index]->imgMatrix->data)
            fatalError("Error: Found null image during row dataset flattening.");

        float *source_data = image_dataset[image_index]->imgMatrix->data;

        for (int pixel_index = 0; pixel_index < pixels_per_image; pixel_index++)
        {
            flat_data->data[idxMatrix(image_index, pixel_index, flat_data->cols)] = source_data[pixel_index];
        }
    }

    return flat_data;
}

void printImageMatrix(Matrix *img_matrix)
{
    for (int row_index = 0; row_index < img_matrix->rows; row_index++)
    {
        for (int col_index = 0; col_index < img_matrix->cols; col_index++)
        {
            float pixel_value = img_matrix->data[idxMatrix(row_index, col_index, img_matrix->cols)];

            if (pixel_value > 0.8f)
                printf("##");
            else if (pixel_value > 0.5f)
                printf("  ");
            else if (pixel_value > 0.2f)
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
 *  - dataset_size: número de imagens no dataset.
 * Libera todas as imagens e o vetor de ponteiros, evitando vazamento de memória.
 */
void freeImageDataset(Image ***dataset, int dataset_size)
{
    if (*dataset)
    {
        for (int image_index = 0; image_index < dataset_size; image_index++)
        {
            freeMatrixData((*dataset)[image_index]->imgMatrix);
            free((*dataset)[image_index]);
        }
    }

    free(*dataset);
}
