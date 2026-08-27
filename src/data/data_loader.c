#include "core/matrix.h"
#include "math/math_utils.h"
#include "data/data_loader.h"
#include "core/helpers.h"
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

Image* newImage(int label, int rows, int cols)
{
    Image *img = (Image*) malloc(sizeof(Image));

    if (!img)
        fatalError("Error: Fail to allocate memory to Image struct.");

    img->label = label;
    img->imgMatrix = createMatrix(rows, cols);

    if (!img->imgMatrix)
        fatalError("Error: Allocation fail. It was not possible to create a new image matrix.");

    return img;
}

Image *cloneImage(Image *image)
{
    Image *dupImage = newImage(image->label, 0, 0);
    dupImage->imgMatrix = cloneMatrix(image->imgMatrix);

    return dupImage;
}

void appendImageToDataset(Image*** imgDataset, int* currentSize, Image* img)
{
    int newSize = *currentSize + 1;

    Image **temp = realloc(*imgDataset, newSize * sizeof(Image*));

    if (!temp)
        fatalError("Error: realloc to a imageDataset fail.");

    *imgDataset = temp;

    (*imgDataset)[*currentSize] = cloneImage(img);
    *currentSize = newSize;
}


Image** loadImageDataset(const char *filename, int n_sample)
{
    FILE *fp = fopen(filename, "r");

    if (!fp)
        fatalError("Error: fopen cannot open the file, filename does not exist or it's wrong");

    char buffer[8912];
    int count_samples = 0;
    Image **ds = NULL;

    int isHeader = 1;
    int currentDsSize = 0;

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

void freeImageDataset(Image*** dataset, int datasetSize)
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
