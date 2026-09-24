#include "data/data_loader.h"
#include "core/matrix.h"
#include "math/matrix_operations.h"
#include "math/math_utils.h"
#include "network/neural_net.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEFAULT_DATASET_PATH "./data/mnist_test.csv"
#define DEFAULT_TRAIN_SIZE 100
#define DEFAULT_TEST_SIZE 25
#define DEFAULT_BATCH_SIZE 5
#define DEFAULT_EPOCHS 3
#define DEFAULT_LEARNING_RATE 0.05f

enum DemoMode
{
    DEMO_TRAIN = 0,
    DEMO_TEST = 1,
    DEMO_ALL = 2,
    DEMO_INVALID = 3
};

static void printHeader(const char *title)
{
    printf("\n==================================================\n");
    printf("%s\n", title);
    printf("==================================================\n");
}

static void printUsage(const char *programName)
{
    printf("Uso:\n");
    printf(" -modo treino- %s [dataset] [train_size] [batch_size] [epochs] [learning_rate]\n", programName);
    printf(" -modo teste- %s [dataset] [test_size] [batch_size]\n", programName);
    printf(" -modo treino/teste- %s [dataset] [train_size] [test_size] [batch_size] [epochs] [learning_rate]\n", programName);
    printf("\nDescricao dos argumentos:\n");
    printf("  dataset      = caminho do arquivo CSV do conjunto de dados\n");
    printf("  train_size   = quantidade de imagens usadas no treinamento\n");
    printf("  test_size    = quantidade de imagens usadas no teste\n");
    printf("  batch_size   = numero de imagens por lote\n");
    printf("  epochs       = numero de epocas de treinamento\n");
    printf("  learning_rate= taxa de aprendizado da rede\n");
    printf("\nModos de uso:\n");
    printf("  train = treina a rede e salva os pesos em neural_net_cp.bin\n");
    printf("  test  = carrega a rede salva e avalia a acuracia no conjunto de teste\n");
    printf("  all   = executa treinamento seguido de teste em uma unica execucao\n");
    printf("\nExemplos:\n");
    printf("  %s train ./data/mnist_test.csv 100 5 3 0.05\n", programName);
    printf("  %s test  ./data/mnist_test.csv 25 5\n", programName);
    printf("  %s all   ./data/mnist_test.csv 100 25 5 3 0.05\n", programName);
}

static enum DemoMode parseMode(const char *mode)
{
    if (mode == NULL)
        return DEMO_INVALID;

    if (strcmp(mode, "train") == 0)
        return DEMO_TRAIN;
    if (strcmp(mode, "test") == 0)
        return DEMO_TEST;
    if (strcmp(mode, "all") == 0)
        return DEMO_ALL;

    return DEMO_INVALID;
}

static void trainDemo(const char *filename,
                      int trainSamples,
                      int batchSize,
                      int epochs,
                      float learningRate)
{
    Image **trainDataset = loadImageDataset(filename, trainSamples, 0);

    if (trainDataset == NULL)
    {
        fprintf(stderr, "Erro: nao foi possivel carregar o dataset de treinamento: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    printHeader("TREINAMENTO DA REDE NEURAL");
    printf("Arquivo: %s\n", filename);
    printf("Amostras de treino: %d\n", trainSamples);
    printf("Batch size: %d\n", batchSize);
    printf("Epocas: %d\n", epochs);
    printf("Taxa de aprendizado: %.3f\n", learningRate);

    NeuralNetModel net = newNeuralNet(batchSize, 4, 128, 784, 10);
    trainNeuralNetOnImages(&net, trainDataset, batchSize, trainSamples, epochs, learningRate);

    char modelFile[] = "neural_net_cp.bin";
    saveNeuralNet(&net, modelFile);

    printf("\nRede treinada salva em: %s\n", modelFile);

    freeImageDataset(&trainDataset, trainSamples);
    freeNeuralNet(&net);
}

static float testDemo(const char *filename,
                      int testSamples,
                      int startOffset,
                      int batchSize)
{
    printHeader("TESTE DA REDE NEURAL");

    char modelFile[] = "neural_net_cp.bin";
    NeuralNetModel loadedNet = loadNeuralNet(modelFile, batchSize);
    Image **testDataset = loadImageDataset(filename, testSamples, startOffset);

    if (testDataset == NULL)
    {
        fprintf(stderr, "Erro: nao foi possivel carregar o dataset de teste: %s\n", filename);
        freeNeuralNet(&loadedNet);
        exit(EXIT_FAILURE);
    }

    float accuracy = testNeuralNet(&loadedNet, testDataset, testSamples, batchSize);

    printf("\nAcuracia final no conjunto de teste: %.2f%%\n", accuracy);

    freeImageDataset(&testDataset, testSamples);
    freeNeuralNet(&loadedNet);

    return accuracy;
}

int main(int argc, char const *argv[])
{
    srand((unsigned int)time(NULL));

    const char *filename = DEFAULT_DATASET_PATH;
    int trainSamples = DEFAULT_TRAIN_SIZE;
    int testSamples = DEFAULT_TEST_SIZE;
    int batchSize = DEFAULT_BATCH_SIZE;
    int epochs = DEFAULT_EPOCHS;
    float learningRate = DEFAULT_LEARNING_RATE;

    enum DemoMode mode = DEMO_ALL;

    if (argc == 1)
    {
        printUsage(argv[0]);
        printf("===\nDIGITE (Y/y) SE DESEJAR SEGUIR COM A EXECUCAO DO CODIGO: ");

        char input[1];
        scanf("%1s", input);
        if (strcasecmp(input, "y") != 0)
            return EXIT_SUCCESS;
    }

    printf("\n==================================================\n");
    printf("SELECAO DE MODO DA DEMONSTRACAO\n");
    printf("==================================================\n");
    printf("  [1] Treinamento      -> digite: train\n");
    printf("  [2] Teste            -> digite: test\n");
    printf("  [3] Treinamento + Teste -> digite: all\n");
    printf("==================================================\n");
    printf("Escolha: ");

    char input[16];
    scanf("%15s", input);
    mode = parseMode(input);

    if (mode == DEMO_INVALID)
    {
        printf("Modo invalido: %s\n\n", argv[1]);
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1)
        filename = argv[1];
    if (argc > 2)
        trainSamples = atoi(argv[2]);
    if (argc > 3)
        testSamples = atoi(argv[3]);
    if (argc > 4)
        batchSize = atoi(argv[4]);
    if (argc > 5)
        epochs = atoi(argv[5]);
    if (argc > 6)
        learningRate = (float)atof(argv[6]);

    if (trainSamples <= 0)
        trainSamples = DEFAULT_TRAIN_SIZE;
    if (testSamples <= 0)
        testSamples = DEFAULT_TEST_SIZE;
    if (batchSize <= 0)
        batchSize = DEFAULT_BATCH_SIZE;
    if (epochs <= 0)
        epochs = DEFAULT_EPOCHS;
    if (learningRate <= 0.0f)
        learningRate = DEFAULT_LEARNING_RATE;

    printHeader("DEMONSTRACAO DE TREINAMENTO E TESTE DE REDE NEURAL");
    printf("Modo selecionado: %s\n", input);
    printf("Dataset: %s\n", filename);
    printf("Treinamento: %d imagens | Teste: %d imagens\n", trainSamples, testSamples);

    if (mode == DEMO_TRAIN)
    {
        trainDemo(filename, trainSamples, batchSize, epochs, learningRate);
    }
    else if (mode == DEMO_TEST)
    {
        testDemo(filename, testSamples, trainSamples, batchSize);
    }
    else
    {
        trainDemo(filename, trainSamples, batchSize, epochs, learningRate);
        testDemo(filename, testSamples, trainSamples, batchSize);
    }

    printHeader("FIM DA DEMONSTRACAO");
    return EXIT_SUCCESS;
}