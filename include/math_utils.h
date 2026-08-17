#pragma once

float sigmoid(float value);
float linearCombination(const float* weights, int wSize, const float* inputs, int iSize, float bias);