#pragma once

#include "matrix.h"

int checkDimensions(Matrix2D matrixA, Matrix2D matrixB);
Matrix2D transposeMatrix(Matrix2D matrix);
Matrix2D multiplyMatrix(Matrix2D matrixA, Matrix2D matrixB);
Matrix2D multiplyMatrixScalar(Matrix2D matrix, float scalar);
Matrix2D sumMatrix(Matrix2D matrixA, Matrix2D matrixB);
Matrix2D subMatrix(Matrix2D matrixA, Matrix2D matrixB);
Matrix2D applyFunc(float (*func)(float), Matrix2D matrix);
