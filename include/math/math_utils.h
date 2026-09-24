#pragma once

#include "core/matrix.h"
#include "network/neural_net.h"

/*
 * sigmoid:
 *  - value: entrada da função.
 * Retorna: valor transformado pela função sigmoide, usado para ativação não linear.
 */
void sigmoid(Matrix *matrix);

/*
 * softmax:
 *  - output: matriz de entradas que será convertida em probabilidades.
 * Normaliza os valores da matriz em uma distribuição de probabilidade usando a função softmax.
 */
void softmax(Matrix *output);

/*
 * weightedSum:
 *  - weights: vetor de pesos sinápticos.
 *  - wSize: tamanho do vetor de pesos.
 *  - inputs: vetor de entradas da camada ou neurônio.
 *  - iSize: tamanho do vetor de entradas.
 *  - bias: termo constante somado ao cálculo.
 * Retorna: soma ponderada das entradas com o bias.
 */
Matrix *weightedSum(Matrix *inputs, Matrix *weights, Matrix *bias);

/*
 * crossEntropy:
 *  - output: vetor de saídas da rede após a camada final.
 *  - idxExpectedClass: índice da classe correta esperada.
 * Retorna: perda de entropia cruzada para a classificação correta.
 */
double crossEntropy(Matrix *output, int idxExpectedClass);

/*
 * sigmoidDerivative:
 *  - preActivation: matriz com valores pós-ativação sigmóide.
 * Retorna: nova matriz contendo as derivadas sigmóides.
 * Calcula a derivada da função sigmóide para cada elemento: σ'(x) = x(1-x).
 */
Matrix *sigmoidDerivative(Matrix *preActivation);

/*
 * outputDelta:
 *  - output: saída calculada pela rede.
 *  - target: alvo esperado da amostra.
 * Retorna: matriz com a diferença entre saída e alvo (delta da camada de saída).
 * Usada no backpropagation para calcular o erro da camada final.
 */
Matrix *outputDelta(Matrix *output, Matrix *target);

/*
 * hiddenDelta:
 *  - output: ativação da camada oculta.
 *  - nextWeights: pesos que conectam a camada atual à próxima.
 *  - nextDelta: delta retropropagado da próxima camada.
 * Retorna: delta calculado para a camada oculta.
 * Propaga o erro para trás através das camadas ocultas durante backpropagation.
 */
Matrix *hiddenDelta(Matrix *output, Matrix *nextWeights, Matrix *nextDelta);

/*
 * updateLayerParameters:
 *  - layer: camada cujos pesos e bias serão atualizados.
 *  - prevInput: ativação da camada anterior.
 *  - delta: gradiente de erro calculado para esta camada.
 *  - batchSize: número de amostras no lote.
 *  - learningRate: taxa de aprendizado aplicada nas atualizações.
 * Atualiza os pesos e bias da camada usando gradient descent.
 */
void updateLayerParameters(NetworkLayer *layer, Matrix *prevInput, Matrix *delta, int batchSize, float learningRate);