#include "data/data_loader.h"
#include "core/matrix.h"
#include "math/matrix_operations.h"
#include "math/math_utils.h"
#include "network/neural_net.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char const *argv[])
{
    srand(time(NULL));

    const char *filename = argc > 1 ? argv[1] : "./data/mnist_test.csv";
    int sampleSize = 10;

    Image **imgs = loadImageDataset(filename, sampleSize);

    for (int i = 0; i < sampleSize; i++)
    {
        printf("Image %d | label: %d\n", i, imgs[i]->label);
        printMatrixFormatted(imgs[i]->imgMatrix);
        printf("\n");
    }

    freeImageDataset(&imgs, sampleSize);

    return 0;
}
