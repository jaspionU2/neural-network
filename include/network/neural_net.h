#pragma once

#include "core/matrix.h"
#include "data/data_loader.h"

typedef struct
{
    int n_neurons;
    Matrix *weights;
    Matrix *bias;
    Matrix *output;
} NetworkLayer;

typedef struct
{
    NetworkLayer *layers;
    int layerCount;
} NeuralNetModel;

/*
 * newNetworkLayer:
 *  - n_samples: quantidade de amostras.
 *  - n_neurons: quantidade de neurônios presentes nessa camada.
 *  - n_neuronsLast: quantidade de neurônios da camada anterior.
 * Retorna: camada inicializada com pesos, bias e saída da ativação.
 */
NetworkLayer newNetworkLayer(int n_samples, int n_neurons, int n_neuronsLast);

/*
 * newNeuralNet:
 *  - n_Layers: número de camadas da rede.
 *  - n_neurons: quantidade de neurônios para as camadas ocultas.
 *  - inputNeurons: tamanho da camada de entrada.
 *  - outuputNeurons: tamanho da camada de saída.
 * Retorna: modelo neural com as camadas alocadas e inicializadas.
 */
NeuralNetModel newNeuralNet(int n_samples, int n_Layers, int n_neurons, int inputNeurons, int outuputNeurons);

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
void propagateBackward(NeuralNetModel *net, Matrix *samples, Matrix *labels, float learningRate);

/*
 * trainNeuralNetOnImages:
 *  - net: modelo neural a ser treinado.
 *  - dataset: conjunto de imagens de treinamento.
 *  - batchSize: tamanho do lote usado no treinamento.
 *  - datasetSize: número de amostras no dataset.
 *  - epochs: quantidade de épocas de treinamento.
 *  - learningRate: taxa de aprendizado usada na atualização dos pesos.
 * Executa o processo de treinamento da rede usando as imagens informadas.
 */
void trainNeuralNetOnImages(NeuralNetModel *net, Image **dataset, int batchSize, int datasetSize, int epochs, float learningRate);

/*
 * printNeuralNet:
 *  - net: rede neural a ser exibida.
 * Imprime as camadas, pesos, bias e saídas da rede para inspeção.
 */
void printNeuralNet(NeuralNetModel *net);

/*
 * freeNeuralNet:
 *  - neuralNet: modelo a ser liberado.
 * Libera a memória alocada para as matrizes e limpa o vetor de camadas.
 */
void freeNeuralNet(NeuralNetModel *neuralNet);