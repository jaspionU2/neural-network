#pragma once

#include "core/matrix.h"

int matricesHaveSameDimensions(Matrix *matrixA, Matrix *matrixB);
Matrix *transposeMatrix(Matrix *matrix);
Matrix *multiplyMatrices(Matrix *matrixA, Matrix *matrixB);
Matrix *multiplyMatrixByScalar(Matrix *matrix, float scalar);
Matrix *addMatrices(Matrix *matrixA, Matrix *matrixB);
Matrix *subtractMatrices(Matrix *matrixA, Matrix *matrixB);
Matrix *applyFunctionToMatrix(float (*func)(float), Matrix *matrix);
