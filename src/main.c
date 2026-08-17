#include "matrix.h"
#include "matrix_operations.h"
#include "math_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char const *argv[])
{
    srand(time(NULL));

    Matrix2D matrixA = newMatrix(5, 5);
    Matrix2D matrixB = newMatrix(5, 5);
    randMatrix(matrixA);
    randMatrix(matrixB);
    printMatrix(matrixA);
    printMatrix(matrixB);
    printMatrix(applyFunc(sigmoid, sumMatrix(matrixA, matrixB)));

    return 0;
}
