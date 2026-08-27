#include "math/math_utils.h"
#include "core/matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

float sigmoid(float input)
{
    return 1 / (1 + expf(-input));
}

// float weightedSum(const float *weights, int wSize, const float *inputs, int iSize, float bias)
// {
//     float output = 0.0f;

//     for (int i = 0; i < wSize; i++)
//     {
//         output += weights[i] * inputs[i];
//     }

//     return output + bias;
// }
