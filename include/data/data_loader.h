#pragma once

#include "core/matrix.h"
#include <stddef.h>

typedef struct
{
    int label;
    Matrix *imgMatrix;
} Image;

// typedef struct
// {
//     Image *images;
//     size_t n_images;
// } ImageDataset;

Image* newImage(int label, int rows, int cols);
Image** loadImageDataset(const char *filename, int n_sample);
Image *cloneImage(Image *image);
void appendImageToDataset(Image ***imgDataset, int *currentSize, Image *img);
void freeImageDataset(Image ***dataset, int datasetSize);