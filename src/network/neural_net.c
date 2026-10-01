#include "core/matrix.h"
#include "math/matrix_operations.h"
#include "core/helpers.h"
#include "network/neural_net.h"
#include "math/math_utils.h"
#include "data/data_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <float.h>
#include <time.h>

/*
 * newNetworkLayer:
 *  - batch_size: quantidade de amostras.
 *  - neuron_count: número de neurônios presentes nessa camada.
 *  - previous_layer_neuron_count: quantidade de neurônios da camada anterior.
 * Retorna: uma NetworkLayer inicializada com pesos, bias e saída.
 * Esse construtor cria o estado interno de uma camada para que a rede possa processar dados.
 */
NetworkLayer newNetworkLayer(int batch_size, int neuron_count, int previous_layer_neuron_count)
{
    Matrix *bias_matrix = createMatrix(1, neuron_count);
    Matrix *weight_matrix = createMatrix(previous_layer_neuron_count, neuron_count);
    Matrix *output_matrix = createMatrix(batch_size, neuron_count);

    if (bias_matrix == NULL || bias_matrix->data == NULL ||
        weight_matrix == NULL || weight_matrix->data == NULL ||
        output_matrix == NULL || output_matrix->data == NULL)
    {
        if (bias_matrix != NULL)
        {
            freeMatrixData(bias_matrix);
            free(bias_matrix);
            bias_matrix = NULL;
        }
        if (weight_matrix != NULL)
        {
            freeMatrixData(weight_matrix);
            free(weight_matrix);
            weight_matrix = NULL;
        }
        if (output_matrix != NULL)
        {
            freeMatrixData(output_matrix);
            free(output_matrix);
            output_matrix = NULL;
        }

        fatalError("Erro: Allocation fail. Don't possible to create new NetworkLayer.");
    }

    fillMatrixXavier(weight_matrix, previous_layer_neuron_count, neuron_count);
    fillMatrixRandom(bias_matrix, 0.0f, 0.0f);

    NetworkLayer layer = {
        .neuron_count = neuron_count,
        .bias = bias_matrix,
        .weights = weight_matrix,
        .output = output_matrix};

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
    NetworkLayer *temp = realloc(net->layers, (net->layer_count + 1) * sizeof(NetworkLayer));

    if (!temp)
        fatalError("Error: Allocation fail. It was not possible to append an layer to the net.");

    net->layers = temp;
    net->layers[net->layer_count] = layer;
    net->layer_count++;
}

/*
 * newNeuralNet:
 *  - batch_size: quantidade de amostras.
 *  - layer_count: número de camadas da rede, incluindo a saída.
 *  - hidden_layer_neuron_count: número de neurônios das camadas intermediárias.
 *  - input_neuron_count: número de neurônios da camada de entrada.
 *  - output_neuron_count: número de neurônios da camada de saída.
 * Retorna: um NeuralNetModel com as camadas inicializadas e prontas para treinamento.
 * Essa função monta a estrutura da arquitetura da rede neural.
 */
NeuralNetModel newNeuralNet(int batch_size, int layer_count, int hidden_layer_neuron_count, int input_neuron_count, int output_neuron_count)
{
    NeuralNetModel neural_net = {.layer_count = 0, .layers = NULL};
    int previous_layer_neuron_count = input_neuron_count;

    for (int i = 0; i < layer_count; i++)
    {
        NetworkLayer current_layer;

        if (i == layer_count - 1)
        {
            current_layer = newNetworkLayer(batch_size, output_neuron_count, previous_layer_neuron_count);
        }
        else
        {
            current_layer = newNetworkLayer(batch_size, hidden_layer_neuron_count, previous_layer_neuron_count);
        }

        appendLayerToNeuralNet(&neural_net, current_layer);
        previous_layer_neuron_count = neural_net.layers[i].neuron_count;
    }

    return neural_net;
}



void propagateForward(Matrix *samples, NeuralNetModel *net)
{
    if (!samples || !samples->data)
        fatalError("Error: Passed a invalid sample as argument");

    if (!net || !net->layers || net->layer_count <= 0)
        fatalError("Error: Passed a invalid network as argument");

    Matrix *current_input = samples;

    for (int i = 0; i < net->layer_count; i++)
    {
        NetworkLayer *layer = &net->layers[i];
        Matrix *activation = weightedSum(current_input, layer->weights, layer->bias);

        if (i == net->layer_count - 1)
        {
            softmax(activation);
        }
        else
        {
            sigmoid(activation);
        }

        memcpy(layer->output->data, activation->data, activation->rows * activation->cols * sizeof(float));

        freeMatrixData(activation);
        free(activation);

        current_input = net->layers[i].output;
    }
}

void propagateBackward(NeuralNetModel *net, Matrix *samples, Matrix *labels, float learning_rate)
{
    if (!net || !net->layers || net->layer_count <= 0)
        fatalError("Error: Passed a invalid network as argument");

    if (!samples || !samples->data)
        fatalError("Error: Passed a invalid sample as argument");

    if (!labels || !labels->data)
        fatalError("Error: Passed a invalid label as argument");

    int batch_size = samples->rows;
    Matrix *deltas[net->layer_count];

    for (int i = net->layer_count - 1; i >= 0; i--)
    {
        NetworkLayer *layer = &net->layers[i];

        if (i == net->layer_count - 1)
        {
            deltas[i] = outputDelta(layer->output, labels);
        }
        else
        {
            NetworkLayer *next_layer = &net->layers[i + 1];
            deltas[i] = hiddenDelta(layer->output, next_layer->weights, deltas[i + 1]);
        }
    }

    for (int i = 0; i < net->layer_count; i++)
    {
        Matrix *previous_input = (i == 0) ? samples : net->layers[i - 1].output;
        updateLayerParameters(&net->layers[i], previous_input, deltas[i], batch_size, learning_rate);

        freeMatrixData(deltas[i]);
        free(deltas[i]);
    }
}

static int predictClassFromOutput(Matrix *output, int row, float *confidence)
{
    int predicted_class = 0;
    float max_probability = output->data[idxMatrix(row, 0, output->cols)];

    for (int class_index = 1; class_index < output->cols; class_index++)
    {
        float probability = output->data[idxMatrix(row, class_index, output->cols)];
        if (probability > max_probability)
        {
            max_probability = probability;
            predicted_class = class_index;
        }
    }

    if (confidence != NULL)
        *confidence = max_probability;

    return predicted_class;
}

static int evaluateBatch(NeuralNetModel *net,
                         Image **dataset,
                         Matrix *samples,
                         int offset,
                         bool print_image_matrix)
{
    Matrix *output = net->layers[net->layer_count - 1].output;
    int correct_hits_in_batch = 0;

    for (int batch_item = 0; batch_item < samples->rows; batch_item++)
    {
        Image *image = dataset[offset + batch_item];
        float confidence = 0.0f;
        int predicted_class = predictClassFromOutput(output, batch_item, &confidence);

        if (predicted_class == image->label)
            correct_hits_in_batch++;

        printf("  [Lote Item %d | Img #%d] Label Real: %d | Previsão: %d (Confiança: %.2f%%)\n",
               batch_item + 1,
               offset + batch_item,
               image->label,
               predicted_class,
               confidence * 100.0f);

        if (print_image_matrix)
            printImageMatrix(image->imgMatrix);
    }

    return correct_hits_in_batch;
}

/*
 * trainNeuralNetOnImages:
 *  - net: ponteiro para o modelo neural a ser treinado.
 *  - dataset: array de imagens de treinamento.
 *  - batch_size: tamanho do lote usado em cada atualização de pesos.
 *  - dataset_size: número total de imagens do dataset.
 *  - epochs: número de épocas de treinamento.
 *  - learning_rate: taxa de aprendizado aplicada na otimização da rede.
 *  - validation_split: fração do dataset reservada para validação.
 * Essa função ainda será implementada para realizar o loop de treinamento sobre as imagens.
 */
void trainingNeuralNetOnImages(NeuralNetModel *net, Image **dataset, int batch_size, int dataset_size, int epochs, float learning_rate, float validation_split)
{
    if (validation_split < 0.0f || validation_split >= 1.0f)
        fatalError("Error: validation_split must be in range [0.0, 1.0).");

    Matrix *inputs = datasetToMatrix(dataset, dataset_size);

    int validation_size = (int)(validation_split * dataset_size);
    int training_size = dataset_size - validation_size;

    if (training_size <= 0)
        fatalError("Error: training split produced zero training samples.");

    int class_count = net->layers[net->layer_count - 1].neuron_count;
    int batch_index = 0;

    int patience = 5;
    int stagnation_patience = patience;
    int overfitting_patience = patience;
    float best_loss = FLT_MAX;
    float best_val_loss = best_loss;
    float min_delta = 1e-4f;

    char filename[256];
    time_t timestamp = time(NULL);

    snprintf(filename, sizeof(filename), "plot/net_metric_%ld.csv", (long)timestamp);

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float loss_dataset = 0.0f;
        int batches_processed = 0;

        for (int k = 0; k < training_size; k += batch_size)
        {
            Matrix *samples = getBatchMatrix(inputs, k, batch_size);
            Matrix *class_labels = createMatrix(samples->rows, class_count);

            for (int batch_item = 0; batch_item < samples->rows; batch_item++)
            {
                Image *image = dataset[k + batch_item];
                class_labels->data[idxMatrix(batch_item, image->label, class_labels->cols)] = 1.0f;
            }

            propagateForward(samples, net);

            Matrix *output = net->layers[net->layer_count - 1].output;

            printf("\n=== Processando Mini-Batch [Imagens %d a %d] ===\n", k, k + samples->rows - 1);

            int correct_hits_in_batch = evaluateBatch(net, dataset, samples, k, false);
            float batch_accuracy = (float)correct_hits_in_batch / samples->rows;
            float avg_loss_batch = CategoricalCrossEntropy(output, class_labels);

            loss_dataset += avg_loss_batch;
            batches_processed++;

            printf("\n  [Métrica do Mini-Batch] Acurácia: %.2f%% | Média Erro: %.6f\n",
                   batch_accuracy * 100.0f, avg_loss_batch);

            printf("==================================================\n");

            saveParametersOnCsv(filename, "a", "batch,accuracy,learning_rate", "%d,%.2f,%.2f\n", batch_index++, batch_accuracy * 100.0f, learning_rate);

            propagateBackward(net, samples, class_labels, learning_rate);

            freeMatrixData(samples);
            free(samples);

            freeMatrixData(class_labels);
            free(class_labels);
        }

        float loss_epoch = loss_dataset / batches_processed;

        printf("[Debug Época %d] Perda Atual: %.6f | Melhor Perda: %.6f | Paciência: %d\n",
               epoch + 1, loss_epoch, best_loss, stagnation_patience);

        if (loss_epoch < (best_loss - min_delta))
        {
            best_loss = loss_epoch;
            stagnation_patience = patience; 
        }
        else
        {
            stagnation_patience--; 

            if (stagnation_patience == 0)
            {
                printf("\n[Early Stopping] Treinamento interrompido na época %d devido a estagnação da perda.\n", epoch + 1);
                break; 
            }
        }

        if (validation_size > 0)
        {
            float val_loss = validateNeuralNet(net, &dataset[training_size], validation_size, batch_size);
            printf("[Epoch %d] Validation Err: %.2f%%\n", epoch + 1, val_loss);

            if (val_loss < (best_val_loss - min_delta))
            {
                best_val_loss = val_loss;
                overfitting_patience = patience;
            }
            else
            {
                overfitting_patience--;

                if (overfitting_patience == 0)
                {
                    printf("\n[Early Stopping por Overfitting] A rede esta memorizando, portanto a perda durante validação parou de melhor.\n");
                    break;
                }
            }
        }
   
    }

    freeMatrixData(inputs);
    free(inputs);
}

float testNeuralNet(NeuralNetModel *net, Image **dataset, int dataset_size, int batch_size)
{
    Matrix *inputs = datasetToMatrix(dataset, dataset_size);
    int correct_hits = 0;
    int batch_index = 0;

    for (int i = 0; i < dataset_size; i += batch_size)
    {
        Matrix *samples = getBatchMatrix(inputs, i, batch_size);

        propagateForward(samples, net);

        int correct_hits_in_batch = evaluateBatch(net, dataset, samples, i, true);
        float batch_accuracy = ((float)correct_hits_in_batch / samples->rows) * 100.0f;

        printf(" Porcentagem de acerto no Lote: %.2f%% \n", batch_accuracy);
        printf(" Porcentagem de acerto no Lote n.%d: %.2f%% \n", batch_index++, batch_accuracy);

        correct_hits += correct_hits_in_batch;

        freeMatrixData(samples);
        free(samples);
    }

    freeMatrixData(inputs);
    free(inputs);

    return (float)correct_hits / dataset_size * 100.0f;
}

float validateNeuralNet(NeuralNetModel *net, Image **dataset, int dataset_size, int batch_size)
{
    Matrix *inputs = datasetToMatrix(dataset, dataset_size);
    float loss_dataset = 0.0f;
    int total_batches = 0;

    int class_count = net->layers[net->layer_count - 1].neuron_count;

    for (int i = 0; i < dataset_size; i += batch_size)
    {
        Matrix *samples = getBatchMatrix(inputs, i, batch_size);
        Matrix *class_labels = createMatrix(samples->rows, class_count);

        for (int batch_item = 0; batch_item < samples->rows; batch_item++)
        {
            Image *image = dataset[i + batch_item];
            class_labels->data[idxMatrix(batch_item, image->label, class_labels->cols)] = 1.0f;
        }

        propagateForward(samples, net);

        Matrix *output = net->layers[net->layer_count - 1].output;

        float avg_loss_batch = CategoricalCrossEntropy(output, class_labels);

        loss_dataset += avg_loss_batch;
        total_batches++;

        freeMatrixData(samples);
        free(samples);

        freeMatrixData(class_labels);
        free(class_labels);
    }

    freeMatrixData(inputs);
    free(inputs);

    return loss_dataset / total_batches;
}

void saveNeuralNet(NeuralNetModel *net, char *filename)
{
    if (!net || !net->layer_count || !net->layers)
        fatalError("Error: Error: Passed a invalid network as argument");

    FILE *file = fopen(filename, "wb");

    if (!file)
        fatalError("Error: fopen fail to open the file to write.");

    int layer_count = net->layer_count;

    if (fwrite(&layer_count, sizeof(int), 1, file) != 1)
        fatalError("Error: fwrite fail to write in file.");

    for (int i = 0; i < layer_count; i++)
    {
        NetworkLayer *layer = &net->layers[i];

        fwrite(&layer->neuron_count, sizeof(int), 1, file);

        Matrix *weights = layer->weights;
        fwrite(&layer->weights->rows, sizeof(int), 1, file);
        fwrite(&layer->weights->cols, sizeof(int), 1, file);
        fwrite(layer->weights->data, sizeof(float), weights->rows * weights->cols, file);

        Matrix *bias = layer->bias;
        fwrite(&layer->bias->rows, sizeof(int), 1, file);
        fwrite(&layer->bias->cols, sizeof(int), 1, file);
        fwrite(layer->bias->data, sizeof(float), bias->rows * bias->cols, file);
    }

    fclose(file);
}

NeuralNetModel loadNeuralNet(char *filename, int batch_size)
{
    FILE *file = fopen(filename, "rb");

    if (!file)
        fatalError("Error: fopen fail to open the file to read.");

    int layer_count = 0;

    if (fread(&layer_count, sizeof(int), 1, file) != 1)
    {
        fclose(file);
        fatalError("Error: fread fail to read layerCount from file.");
    }

    NeuralNetModel net = {.layer_count = 0, .layers = NULL};

    for (int i = 0; i < layer_count; i++)
    {
        int neuron_count = 0;
        int weight_rows = 0, weight_cols = 0;
        int bias_rows = 0, bias_cols = 0;

        if (fread(&neuron_count, sizeof(int), 1, file) != 1)
            fatalError("Error: fread fail to read neuron_count.");

        if (fread(&weight_rows, sizeof(int), 1, file) != 1)
            fatalError("Error: fread fail to read weights rows.");
        if (fread(&weight_cols, sizeof(int), 1, file) != 1)
            fatalError("Error: fread fail to read weights cols.");

        NetworkLayer layer = newNetworkLayer(batch_size, neuron_count, weight_rows);

        if (fread(layer.weights->data, sizeof(float), weight_rows * weight_cols, file) != (size_t)(weight_rows * weight_cols))
            fatalError("Error: fread failed to read weights data.");

        if (fread(&bias_rows, sizeof(int), 1, file) != 1)
            fatalError("Error: fread fail to read bias rows.");
        if (fread(&bias_cols, sizeof(int), 1, file) != 1)
            fatalError("Error: fread fail to read bias cols.");

        if (fread(layer.bias->data, sizeof(float), bias_rows * bias_cols, file) != (size_t)(bias_rows * bias_cols))
            fatalError("Error: fread failed to read bias data.");

        appendLayerToNeuralNet(&net, layer);
    }

    fclose(file);

    return net;
}

void saveParametersOnCsv(char *filename, char *mode, char *csv_header, char *fmt, ...)
{
    FILE *file = fopen(filename, mode);

    if (!file)
        fatalError("Error: fopen fail to open the file to write or append.");

    if (strcmp(mode, "w") == 0 || (strcmp(mode, "a") == 0 && ftell(file) == 0))
    {
        if (fputs(csv_header, file) == EOF)
        {
            fclose(file);
            fatalError("Error: fputs failed to write csv header.");
        }

        if (csv_header[strlen(csv_header) - 1] != '\n')
            fputc('\n', file);
    }

    va_list arguments;
    va_start(arguments, fmt);
    vfprintf(file, fmt, arguments);
    va_end(arguments);

    if (fmt[strlen(fmt) - 1] != '\n')
        fputc('\n', file);

    fclose(file);
}

/*
 * freeNeuralNet:
 *  - neural_net: ponteiro para o modelo neural a ser liberado.
 * Libera a memória alocada para cada camada, incluindo pesos, bias e saídas da rede.
 */
void freeNeuralNet(NeuralNetModel *neural_net)
{
    if (neural_net == NULL)
        return;

    for (int i = 0; i < neural_net->layer_count; i++)
    {
        freeMatrixData(neural_net->layers[i].weights);
        free(neural_net->layers[i].weights);

        freeMatrixData(neural_net->layers[i].bias);
        free(neural_net->layers[i].bias);

        freeMatrixData(neural_net->layers[i].output);
        free(neural_net->layers[i].output);
    }

    free(neural_net->layers);
    neural_net->layers = NULL;
    neural_net->layer_count = 0;
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

    printf("NeuralNet: %d layers\n", net->layer_count);

    for (int i = 0; i < net->layer_count; i++)
    {
        NetworkLayer *layer = &net->layers[i];
        printf("\nLayer %d:\n", i);
        printf("  neurons: %d\n", layer->neuron_count);

        printf("  weights:\n");
        printMatrixFormatted(layer->weights);

        printf("  bias:\n");
        printMatrixFormatted(layer->bias);

        printf("  output:\n");
        printMatrixFormatted(layer->output);
    }
}
