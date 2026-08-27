#pragma once

typedef struct
{
    int rows;
    int cols;
    float *data;
} Matrix;

Matrix *createMatrix(int rows, int cols);
void setMatrixValue(float value, Matrix *matrix, int row, int col);
float getMatrixValue(Matrix *matrix, int row, int col);
void fillMatrixWithValue(Matrix *matrix, float value);
void fillMatrixRandom(Matrix *matrix);
Matrix* cloneMatrix(Matrix *matrix);
float randomUniformFloat(float min, float max);
void printMatrixFormatted(Matrix *matrix);
void freeMatrixData(Matrix *matrix);
