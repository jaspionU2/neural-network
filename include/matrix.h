#pragma once

typedef struct 
{
    int rows;
    int cols;
    float *data;
} Matrix2D;

Matrix2D newMatrix(int rows, int cols);
void addToMatrix(float value, Matrix2D matrix, int row, int col);
void fillMatrix(Matrix2D matrix, float value);
float randomUniform(float min, float max);
void randMatrix(Matrix2D matrix);
float getFromMatrix(Matrix2D matrix, int row, int col);
void printMatrix(Matrix2D matrix);
void freeMatrix(Matrix2D matrix);

