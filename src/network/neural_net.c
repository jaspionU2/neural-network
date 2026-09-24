#include "core/matrix.h"
#include "math/matrix_operations.h"
#include "core/helpers.h"
#include "network/neural_net.h"
#include "math/math_utils.h"
#include "data/data_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * newNetworkLayer:
 *  - n_samples: quantidade de amostras.
 *  - n_neurons: número de neurônios presentes nessa camada.
 *  - n_neuronsLast: quantidade de neurônios da camada anterior.
 * Retorna: uma NetworkLayer inicializada com pesos, bias e saída.
 * Esse construtor cria o estado interno de uma camada para que a rede possa processar dados.
 */
NetworkLayer newNetworkLayer(int n_samples, int n_neurons, int n_neuronsLast)
{
    Matrix *biases = createMatrix(1, n_neurons);
    Matrix *weights = createMatrix(n_neuronsLast, n_neurons);
    Matrix *output = createMatrix(n_samples, n_neurons);

    if (biases == NULL || biases->data == NULL ||
        weights == NULL || weights->data == NULL ||
        output == NULL || output->data == NULL)
    {
        if (biases != NULL)
        {
            freeMatrixData(biases);
            free(biases);
            biases = NULL;
        }
        if (weights != NULL)
        {
            freeMatrixData(weights);
            free(weights);
            weights = NULL;
        }
        if (output != NULL)
        {
            freeMatrixData(output);
            free(output);
            output = NULL;
        }

        fatalError("Erro: Allocation fail. Don't possible to create new NetworkLayer.");
    }

    fillMatrixXavier(weights, n_neuronsLast, n_neurons);
    fillMatrixRandom(biases, 0.0f, 0.0f);

    NetworkLayer layer = {
        .n_neurons = n_neurons,
        .bias = biases,
        .weights = weights,
        .output = output};

    return layer;
}

/*
 * appendLayerToNeuralNet:
 *  - net: ponteiro para o modelo neural que receberá a nova camada.
 *  - layer: camada a ser anexada ao final da rede.
 * Acrescenta uma camada ao vetor de camadas do modelo, expandindo a arquitetura da rede.
 */
void appendLayerToNeuralNet(NeuralNetModel *net, NetworkLayer layer)
{
    NetworkLayer *temp = realloc(net->layers, (net->layerCount + 1) * sizeof(NetworkLayer));

    if (!temp)
        fatalError("Error: Allocation fail. It was not possible to append an layer to the net.");

    net->layers = temp;

    net->layers[net->layerCount] = layer;

    net->layerCount++;
}

/*
 * newNeuralNet:
 *  - n_samples: quantidade de amostras.
 *  - n_Layers: número de camadas da rede, incluindo a saída.
 *  - n_neurons: número de neurônios das camadas intermediárias.
 *  - inputNeurons: número de neurônios da camada de entrada.
 *  - outputNeurons: número de neurônios da camada de saída.
 * Retorna: um NeuralNetModel com as camadas inicializadas e prontas para treinamento.
 * Essa função monta a estrutura da arquitetura da rede neural.
 */
NeuralNetModel newNeuralNet(int n_samples, int n_Layers, int n_neurons, int inputNeurons, int outputNeurons)
{
    NeuralNetModel neuralNet = {.layerCount = 0, .layers = NULL};

    int n_lastLayer = inputNeurons;

    for (int i = 0; i < n_Layers; i++)
    {
        NetworkLayer currentLayer;

        if (i == n_Layers - 1)
        {
            currentLayer = newNetworkLayer(n_samples, outputNeurons, n_lastLayer);
        }
        else
        {
            currentLayer = newNetworkLayer(n_samples, n_neurons, n_lastLayer);
        }

        appendLayerToNeuralNet(&neuralNet, currentLayer);

        n_lastLayer = neuralNet.layers[i].n_neurons;
    }

    return neuralNet;
}

void propagateForward(Matrix *samples, NeuralNetModel *net)
{
    if (!samples || !samples->data)
        fatalError("Error: Passed a invalid sample as argument");

    if (!net || !net->layers || net->layerCount <= 0)
        fatalError("Error: Passed a invalid network as argument");

    Matrix *currentInput = samples;

    for (int i = 0; i < net->layerCount; i++)
    {
        NetworkLayer *layer = &net->layers[i];

        Matrix *z = weightedSum(currentInput, layer->weights, layer->bias);

        if (i == net->layerCount - 1)
        {
            softmax(z);
        }
        else
        {
            sigmoid(z);
        }

        memcpy(layer->output->data, z->data, z->rows * z->cols * sizeof(float));

        freeMatrixData(z);
        free(z);

        currentInput = net->layers[i].output;
    }
}

void propagateBackward(NeuralNetModel *net, Matrix *samples, Matrix *labels, float learningRate)
{
    if (!net || !net->layers || net->layerCount <= 0)
        fatalError("Error: Passed a invalid network as argument");

    if (!samples || !samples->data)
        fatalError("Error: Passed a invalid sample as argument");

    if (!labels || !labels->data)
        fatalError("Error: Passed a invalid label as argument");

    int batchSize = samples->rows;
    Matrix *deltas[net->layerCount];

    for (int i = net->layerCount - 1; i >= 0; i--)
    {
        NetworkLayer *layer = &net->layers[i];

        if (i == net->layerCount - 1)
        {
            deltas[i] = outputDelta(layer->output, labels);
        }
        else
        {
            NetworkLayer *nextLayer = &net->layers[i + 1];
            deltas[i] = hiddenDelta(layer->output, nextLayer->weights, deltas[i + 1]);
        }
    }

    for (int i = 0; i < net->layerCount; i++)
    {
        Matrix *prevInput = (i == 0) ? samples : net->layers[i - 1].output;

        updateLayerParameters(&net->layers[i], prevInput, deltas[i], batchSize, learningRate);

        freeMatrixData(deltas[i]);
        free(deltas[i]);
    }
}

/*
 * trainNeuralNetOnImages:
 *  - net: ponteiro para o modelo neural a ser treinado.
 *  - dataset: array de imagens de treinamento.
 *  - batchSize: tamanho do lote usado em cada atualização de pesos.
 *  - datasetSize: número total de imagens do dataset.
 *  - epochs: número de épocas de treinamento.
 *  - learningRate: taxa de aprendizado aplicada na otimização da rede.
 * Essa função ainda será implementada para realizar o loop de treinamento sobre as imagens.
 */
void trainNeuralNetOnImages(NeuralNetModel *net, Image **dataset, int batchSize, int datasetSize, int epochs, float learningRate)
{
    Matrix *inputs = datasetToMatrix(dataset, datasetSize);
    int numClasses = net->layers[net->layerCount - 1].n_neurons;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        for (int k = 0; k < datasetSize; k += batchSize)
        {
            Matrix *samples = getBatchMatrix(inputs, k, batchSize);
            Matrix *classLabels = createMatrix(batchSize, numClasses);

            for (int b = 0; b < samples->rows; b++)
            {
                Image *img = dataset[k + b];
                setMatrixValue(1.0f, classLabels, b, img->label);
            }

            propagateForward(samples, net);

            Matrix *output = net->layers[net->layerCount - 1].output;

            printf("\n=== Processando Mini-Batch [Imagens %d a %d] ===\n", k, k + samples->rows - 1);

            for (int b = 0; b < samples->rows; b++)
            {
                Image *img = dataset[k + b];
                int trueLabel = img->label;

                int predictedClass = 0;
                float maxProb = getMatrixValue(output, b, 0);

                for (int c = 1; c < output->cols; c++)
                {
                    float prob = getMatrixValue(output, b, c);
                    if (prob > maxProb)
                    {
                        maxProb = prob;
                        predictedClass = c;
                    }
                }

                printf("  [Lote Item %d | Img #%d] Label Real: %d | Previsão: %d (Confiança: %.2f%%)\n",
                       b + 1, k + b, trueLabel, predictedClass, maxProb * 100.0f);
            }

            printf("==================================================\n");

            propagateBackward(net, samples, classLabels, learningRate);

            freeMatrixData(samples);
            free(samples);

            freeMatrixData(classLabels);
            free(classLabels);
        }
    }

    freeMatrixData(inputs);
    free(inputs);
}

float testNeuralNet(NeuralNetModel *net, Image **dataset, int datasetSize, int batchSize)
{
    Matrix *inputs = datasetToMatrix(dataset, datasetSize);

    int countCorrectHits = 0;
    int numBatchs = 0;

    for (int i = 0; i < datasetSize; i += batchSize)
    {
        int hitsOnBatch = 0;

        Matrix *samples = getBatchMatrix(inputs, i, batchSize);

        propagateForward(samples, net);

        Matrix *output = net->layers[net->layerCount - 1].output;

        for (int b = 0; b < samples->rows; b++)
        {
            Image *img = dataset[i + b];
            int trueLabel = img->label;

            int predictedClass = 0;
            float maxProb = getMatrixValue(output, b, 0);

            for (int c = 1; c < output->cols; c++)
            {
                float prob = getMatrixValue(output, b, c);
                if (prob > maxProb)
                {
                    maxProb = prob;
                    predictedClass = c;
                }
            }

            printf("  [Lote Item %d | Img #%d] Label Real: %d | Previsão: %d (Confiança: %.2f%%)\n",
                   b + 1, i + b, trueLabel, predictedClass, maxProb * 100.0f);

            printImageMatrix(img->imgMatrix);

            if (predictedClass == trueLabel)
                hitsOnBatch++;
        }

        float percentOnBatch = ((float)hitsOnBatch / samples->rows) * 100.0f;
        printf(" Porcentagem de acerto no Lote n.%d: %.2f%% \n", numBatchs++, percentOnBatch);

        countCorrectHits += hitsOnBatch;

        freeMatrixData(samples);
        free(samples);
    }

    freeMatrixData(inputs);
    free(inputs);

    return (float)countCorrectHits / datasetSize * 100.0f;
}

void saveNeuralNet(NeuralNetModel *net, char *filename)
{
    if (!net || !net->layerCount || !net->layers)
        fatalError("Error: Error: Passed a invalid network as argument");

    FILE *fp = fopen(filename, "wb");

    if (!fp)
        fatalError("Error: fopen fail to open the file to write.");

    int numLayer = net->layerCount;

    if (fwrite(&numLayer, sizeof(int), 1, fp) != 1)
        fatalError("Error: fwrite fail to write in file.");

    for (int i = 0; i < numLayer; i++)
    {
        NetworkLayer *layer = &net->layers[i];

        fwrite(&layer->n_neurons, sizeof(int), 1, fp);

        Matrix *weights = layer->weights;
        fwrite(&layer->weights->rows, sizeof(int), 1, fp);
        fwrite(&layer->weights->cols, sizeof(int), 1, fp);
        fwrite(layer->weights->data, sizeof(float), weights->rows * weights->cols, fp);

        Matrix *bias = layer->bias;
        fwrite(&layer->bias->rows, sizeof(int), 1, fp);
        fwrite(&layer->bias->cols, sizeof(int), 1, fp);
        fwrite(layer->bias->data, sizeof(float), bias->rows * bias->cols, fp);
    }

    fclose(fp);
}

NeuralNetModel loadNeuralNet(char *filename, int batchSize)
{
    FILE *fp = fopen(filename, "rb");

    if (!fp)
        fatalError("Error: fopen fail to open the file to read.");

    int layerCount = 0;

    if (fread(&layerCount, sizeof(int), 1, fp) != 1)
    {
        fclose(fp);
        fatalError("Error: fread fail to read layerCount from file.");
    }

    NeuralNetModel net = {.layerCount = 0, .layers = NULL};

    for (int i = 0; i < layerCount; i++)
    {
        int n_neurons = 0;
        int wRows = 0, wCols = 0;
        int bRows = 0, bCols = 0;

        if (fread(&n_neurons, sizeof(int), 1, fp) != 1)
            fatalError("Error: fread fail to read n_neurons.");

        if (fread(&wRows, sizeof(int), 1, fp) != 1)
            fatalError("Error: fread fail to read weights rows.");
        if (fread(&wCols, sizeof(int), 1, fp) != 1)
            fatalError("Error: fread fail to read weights cols.");

        NetworkLayer layer = newNetworkLayer(batchSize, n_neurons, wRows);

        if (fread(layer.weights->data, sizeof(float), wRows * wCols, fp) != (size_t)(wRows * wCols))
            fatalError("Error: fread failed to read weights data.");

        if (fread(&bRows, sizeof(int), 1, fp) != 1)
            fatalError("Error: fread fail to read bias rows.");
        if (fread(&bCols, sizeof(int), 1, fp) != 1)
            fatalError("Error: fread fail to read bias cols.");

        if (fread(layer.bias->data, sizeof(float), bRows * bCols, fp) != (size_t)(bRows * bCols))
            fatalError("Error: fread failed to read bias data.");

        appendLayerToNeuralNet(&net, layer);
    }

    fclose(fp);

    return net;
}

/*
 * freeNeuralNet:
 *  - neuralNet: ponteiro para o modelo neural a ser liberado.
 * Libera a memória alocada para cada camada, incluindo pesos, bias e saídas da rede.
 */
void freeNeuralNet(NeuralNetModel *neuralNet)
{
    if (neuralNet == NULL)
        return;

    for (int i = 0; i < neuralNet->layerCount; i++)
    {
        freeMatrixData(neuralNet->layers[i].weights);
        free(neuralNet->layers[i].weights);

        freeMatrixData(neuralNet->layers[i].bias);
        free(neuralNet->layers[i].bias);

        freeMatrixData(neuralNet->layers[i].output);
        free(neuralNet->layers[i].output);
    }

    free(neuralNet->layers);
    neuralNet->layers = NULL;
    neuralNet->layerCount = 0;
}

/*
 * printNeuralNet:
 *  - net: ponteiro para o modelo a ser impresso.
 * Exibe em stdout detalhes da arquitetura da rede, incluindo pesos, bias e saídas de cada camada.
 */
void printNeuralNet(NeuralNetModel *net)
{
    if (net == NULL)
    {
        printf("NeuralNet: NULL\n");
        return;
    }

    printf("NeuralNet: %d layers\n", net->layerCount);

    for (int i = 0; i < net->layerCount; i++)
    {
        NetworkLayer *layer = &net->layers[i];
        printf("\nLayer %d:\n", i);
        printf("  neurons: %d\n", layer->n_neurons);

        printf("  weights:\n");
        printMatrixFormatted(layer->weights);

        printf("  bias:\n");
        printMatrixFormatted(layer->bias);

        printf("  output:\n");
        printMatrixFormatted(layer->output);
    }
}
