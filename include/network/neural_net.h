#pragma once

#include "core/matrix.h"
#include "data/data_loader.h"
#include <stdarg.h>
#include <stdbool.h>

typedef struct
{
    int neuron_count;
    Matrix *weights;
    Matrix *bias;
    Matrix *output;
} NetworkLayer;

typedef struct
{
    NetworkLayer *layers;
    int layer_count;
} NeuralNetModel;

/*
 * newNetworkLayer:
 *  - batch_size: quantidade de amostras processadas no lote.
 *  - neuron_count: quantidade de neurônios presentes nessa camada.
 *  - previous_layer_neuron_count: quantidade de neurônios da camada anterior.
 * Retorna: camada inicializada com pesos, bias e saída da ativação.
 */
NetworkLayer newNetworkLayer(int batch_size, int neuron_count, int previous_layer_neuron_count);

/*
 * newNeuralNet:
 *  - batch_size: quantidade de amostras por lote.
 *  - layer_count: número de camadas da rede, incluindo a saída.
 *  - hidden_layer_neuron_count: quantidade de neurônios para as camadas ocultas.
 *  - input_neuron_count: tamanho da camada de entrada.
 *  - output_neuron_count: tamanho da camada de saída.
 * Retorna: modelo neural com as camadas alocadas e inicializadas.
 */
NeuralNetModel newNeuralNet(int batch_size, int layer_count, int hidden_layer_neuron_count, int input_neuron_count, int output_neuron_count);

/*
 * appendLayerToNeuralNet:
 *  - net: rede neural que receberá a nova camada.
 *  - layer: camada a ser anexada.
 * Acrescenta uma camada ao final do modelo neural.
 */
void appendLayerToNeuralNet(NeuralNetModel *net, NetworkLayer layer);

/*
 * propagateForward:
 *  - samples: matriz de entradas do lote (B x N_in).
 *  - net: ponteiro para o modelo da rede neural.
 * Realiza a passagem direta dos valores por todas as camadas da rede.
 */
void propagateForward(Matrix *samples, NeuralNetModel *net);

/*
 * propagateBackward:
 * Calcula os gradientes retropropagados para ajustar os pesos da rede.
 */
void propagateBackward(NeuralNetModel *net, Matrix *samples, Matrix *labels, float learning_rate);

/*
 * trainNeuralNetOnImages:
 *  - net: modelo neural a ser treinado.
 *  - dataset: conjunto de imagens de treinamento.
 *  - batch_size: tamanho do lote usado no treinamento.
 *  - dataset_size: número de amostras no dataset.
 *  - epochs: quantidade de épocas de treinamento.
 *  - learning_rate: taxa de aprendizado usada na atualização dos pesos.
 *  - validation_split: fração do dataset reservada para validação.
 * Executa o processo de treinamento da rede usando as imagens informadas.
 */
void trainingNeuralNetOnImages(NeuralNetModel *net, Image **dataset, int batch_size, int dataset_size, int epochs, float learning_rate, float validation_split);

float testNeuralNet(NeuralNetModel *net, Image **dataset, int dataset_size, int batch_size);

float validateNeuralNet(NeuralNetModel *net, Image **dataset, int dataset_size, int batch_size);

void saveParametersOnCsv(char *filename, char *mode, char *csv_header, char *fmt, ...);

/*
 * printNeuralNet:
 *  - net: rede neural a ser exibida.
 * Imprime as camadas, pesos, bias e saídas da rede para inspeção.
 */
void printNeuralNet(NeuralNetModel *net);

/*
 * freeNeuralNet:
 *  - neural_net: modelo a ser liberado.
 * Libera a memória alocada para as matrizes e limpa o vetor de camadas.
 */
void freeNeuralNet(NeuralNetModel *neural_net);