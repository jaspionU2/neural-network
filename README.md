# Rede Neural em C: classificador de dígitos MNIST do zero

Rede neural totalmente conectada (MLP) escrita em C puro, sem bibliotecas de álgebra linear ou de machine learning. Matrizes, propagação, backpropagation, treino, validação e persistência do modelo foram implementados à mão, usando apenas a biblioteca padrão (`stdio.h`, `stdlib.h`, `string.h`, `math.h`).

O objetivo é didático: ver cada multiplicação de matriz, cada gradiente e cada `malloc`/`free` que frameworks como PyTorch e TensorFlow escondem.

---

## Sumário

1. [Funcionalidades](#funcionalidades)
2. [Estrutura do projeto](#estrutura-do-projeto)
3. [Requisitos](#requisitos)
4. [Compilação](#compilação)
5. [Como usar](#como-usar)
6. [Visualização das métricas (Python)](#visualização-das-métricas-python)
7. [Formato do dataset](#formato-do-dataset)
8. [Arquitetura padrão da rede](#arquitetura-padrão-da-rede)
9. [Teoria aplicada](#teoria-aplicada)
10. [Formato do arquivo de modelo (.bin)](#formato-do-arquivo-de-modelo-bin)
11. [Limitações conhecidas e próximos passos](#limitações-conhecidas-e-próximos-passos)

---

## Funcionalidades

- **Biblioteca de matrizes própria**: criação, clonagem, achatamento, fatiamento em lotes, transposição, multiplicação matricial, produto de Hadamard, soma com *broadcasting*, subtração e multiplicação por escalar.
- **Rede totalmente conectada configurável**: número de camadas, neurônios por camada oculta, entradas e saídas definidos na criação do modelo.
- **Ativações**: sigmoide nas camadas ocultas (com proteção contra overflow) e softmax estável numericamente na saída.
- **Função de perda**: entropia cruzada categórica com *clipping* para evitar `log(0)`.
- **Inicialização Xavier/Glorot** dos pesos e bias inicializados em zero.
- **Backpropagation** implementado manualmente, camada por camada.
- **Treino em mini-batches** com gradiente descendente.
- **Split de validação** (20% do conjunto de treino por padrão).
- **Early stopping duplo**: por estagnação da perda de treino e por overfitting (perda de validação parou de melhorar).
- **Salvar e carregar o modelo** em arquivo binário `.bin`.
- **Leitor de CSV** para o MNIST, com normalização dos pixels para `[0, 1]`.
- **Log de métricas** por mini-batch em `plot/net_metric.csv` (acurácia e taxa de aprendizado), útil para gerar gráficos.
- **Menu interativo** no terminal com três modos: `train`, `test` e `all`.
- **Build único (*unity build*)**: um só arquivo (`unity_build.c`) inclui todos os `.c`.

---

## Estrutura do projeto

A organização abaixo foi inferida a partir dos `#include` do código. Ajuste se o seu repositório for diferente.

```
.
├── data/
│   └── mnist_test.csv          # dataset (não incluso no repositório)
├── plot/                       # saída de métricas (net_metric.csv)
├── include/
│   ├── core/       matrix.h, helpers.h
│   ├── math/       matrix_operations.h, math_utils.h
│   ├── data/       data_loader.h
│   └── network/    neural_net.h
└── src/
    ├── unity_build.c           # ponto único de compilação
    ├── core/
    │   ├── matrix.c            # estrutura Matrix, alocação, inicialização, batches
    │   └── helpers.c           # fatalError, barra de progresso, extensão de arquivo
    ├── math/
    │   ├── matrix_operations.c # álgebra linear básica
    │   └── math_utils.c        # sigmoide, softmax, deltas, atualização, perda
    ├── data/
    │   └── data_loader.c       # leitura do CSV, estrutura Image, dataset -> matriz
    ├── network/
    │   └── neural_net.c        # camadas, forward, backward, treino, teste, save/load
    └── app/
        └── main.c              # menu interativo
```

| Módulo                   | Responsabilidade                                                                                                                     |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| `core/matrix`            | Tipo `Matrix` (linhas, colunas, vetor `float` linearizado), criação, cópia, fatiamento de lotes e inicialização (aleatória e Xavier) |
| `core/helpers`           | Tratamento de erro fatal (`perror` + `exit`), barra de progresso e extração de extensão                                              |
| `math/matrix_operations` | Transposição, produto matricial, Hadamard, soma com broadcasting, subtração                                                          |
| `math/math_utils`        | Sigmoide, softmax, derivada da sigmoide, deltas de saída e ocultos, atualização de parâmetros, entropia cruzada                      |
| `data/data_loader`       | Lê o CSV, normaliza pixels, monta `Image` e converte o dataset em uma matriz `N x 784`                                               |
| `network/neural_net`     | `NetworkLayer`, `NeuralNetModel`, `propagateForward`, `propagateBackward`, treino, validação, teste, persistência                    |
| `app/main`               | Interface de linha de comando                                                                                                        |

---

## Requisitos

- Compilador C com suporte a C99 ou superior (GCC ou Clang).
- Linux, macOS ou WSL. O código usa `strcasecmp` e `unistd.h`, que não existem no MSVC nativo.
- Dataset MNIST em formato CSV (ver [Formato do dataset](#formato-do-dataset)).

---

## Compilação

Com a estrutura acima, a partir da raiz do projeto:

```bash
gcc -O2 -Wall -Iinclude src/unity_build.c -o neural_net -lm
```

- `-Iinclude` permite resolver includes como `"core/matrix.h"`.
- `-lm` liga a biblioteca matemática (`expf`, `logf`, `sqrtf`).
- `-O2` faz diferença grande no tempo de treino, já que a multiplicação de matrizes é ingênua (três laços aninhados).

Antes de treinar, crie a pasta de métricas, caso ela não exista:

```bash
mkdir -p plot
```

---

## Como usar

Execute:

```bash
./neural_net
```

O programa é **interativo**: ele não lê argumentos de linha de comando, tudo é perguntado no terminal.

### Fluxo

1. Pergunta o modo de execução:

| Modo    | O que faz                                                      |
| ------- | -------------------------------------------------------------- |
| `train` | Treina a rede e salva os pesos em um arquivo `.bin`            |
| `test`  | Carrega um modelo salvo e mede a acurácia no conjunto de teste |
| `all`   | Treina e depois testa, na mesma execução                       |

2. Pede os parâmetros relevantes ao modo escolhido:

| Parâmetro                             | Descrição                                     | Padrão                  |
| ------------------------------------- | --------------------------------------------- | ----------------------- |
| Caminho do CSV                        | Arquivo do dataset                            | `./data/mnist_test.csv` |
| Imagens por lote (`batch_size`)       | Tamanho do mini-batch                         | 5                       |
| Imagens de treino (`train_size`)      | Quantidade de amostras de treino              | 100                     |
| Épocas (`epochs`)                     | Passadas completas sobre o conjunto de treino | 3                       |
| Taxa de aprendizado (`learning_rate`) | Tamanho do passo do gradiente                 | 0.05                    |
| Imagens de teste (`test_size`)        | Quantidade de amostras de teste               | 25                      |

Valores menores ou iguais a zero são substituídos pelos padrões.

3. No fim do treino, o programa pede o nome do arquivo de saída. A extensão precisa ser `.bin`.
4. No modo `test`, ele pede o arquivo `.bin` do modelo treinado e imprime a acurácia final.

### Exemplo de sessão (`all`)

```
Escolha: all
Caminho do arquivo CSV: ./data/mnist_test.csv
Numero de imagens por lote: 32
Quant. de imagens usadas no treinamento: 5000
Numero de epocas de treinamento: 20
Taxa de aprendizado da rede: 0.1
Quant. de imagens usadas no teste: 1000
...
Nome do arquivo para salvar a rede treinada (extensão deve ser .bin): modelo.bin
Acuracia final no conjunto de teste: XX.XX%
```

> **Sobre o conjunto de teste:** as imagens de teste são lidas **a partir da linha `train_size`** do mesmo CSV. Assim, treino e teste não se sobrepõem, desde que o arquivo tenha pelo menos `train_size + test_size` amostras.

---

## Visualização das métricas (Python)
 
Durante o treino, o programa em C anexa uma linha por mini-batch em `plot/net_metric.csv`. O script Python transforma esse arquivo em gráficos.
 
### Colunas do CSV de métricas
 
| Coluna          | Conteúdo                                                                                    |
| --------------- | ------------------------------------------------------------------------------------------- |
| `batch`         | Contador de mini-batches. Não reinicia a cada época, então cresce ao longo de todo o treino |
| `accuracy`      | Acurácia do mini-batch, em porcentagem (0 a 100)                                            |
| `learning_rate` | Taxa de aprendizado usada                                                                   |
 
### Instalação das dependências
 
```bash
pip install pandas matplotlib numpy
```
 
Se preferir isolar o ambiente:
 
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas matplotlib numpy
```
 
### Uso
 
Execute o script a partir da **raiz do projeto**, pois os caminhos são resolvidos a partir do diretório atual:
 
```bash
python3 app.py --x batch --y accuracy
```
 
| Argumento | Obrigatório | Padrão                | Descrição                                             |
| --------- | ----------- | --------------------- | ----------------------------------------------------- |
| `--x`     | Sim         |                       | Nome da coluna do CSV para o eixo X (ex.: `batch`)    |
| `--y`     | Sim         |                       | Nome da coluna do CSV para o eixo Y (ex.: `accuracy`) |
| `--csv`   | Não         | `plot/net_metric.csv` | Caminho do CSV de métricas                            |
 
Exemplos:
 
```bash
# Acurácia por mini-batch (uso principal)
python3 app.py --x batch --y accuracy
 
# Usando um CSV salvo em outro lugar
python3 app.py --csv resultados/run_01.csv --x batch --y accuracy
```
 
### O que o script gera
 
Um gráfico salvo em `plot/<x>_vs_<y>.png` (por exemplo, `plot/batch_vs_accuracy.png`), com resolução de 200 dpi e duas curvas:
 
- **Bruta** (azul, translúcida): o valor de cada mini-batch, que oscila bastante.
- **Tendência** (vermelha): média móvel de 50 lotes (`window_size = 50` em `plot_metrics.py`), que deixa visível se a rede está aprendendo.
A pasta `plot/` é criada automaticamente caso não exista. O terminal imprime o caminho do arquivo gerado.
 
### Mensagens de erro
 
| Situação                         | Resultado                                            |
| -------------------------------- | ---------------------------------------------------- |
| CSV não encontrado               | `FileNotFoundError` com o caminho completo procurado |
| `--x` ou `--y` não existe no CSV | `ValueError` listando as colunas disponíveis         |
 
### Como funciona
 
1. `app.py` lê os argumentos com `argparse` e chama `PlotMetric(csv, x, y)`.
2. `PlotMetric` carrega o CSV com `pandas` e valida os nomes das colunas.
3. Calcula a média móvel com `rolling(window=50, min_periods=1).mean()`. Com `min_periods=1`, os primeiros pontos usam menos de 50 amostras em vez de ficarem vazios.
4. Desenha as duas curvas com `matplotlib` e salva o PNG.
> **Dica:** o programa em C abre o CSV em modo de anexar (`"a"`). Rodar o treino várias vezes acumula as linhas no mesmo arquivo, e o contador `batch` recomeça do zero a cada execução, o que bagunça o gráfico. Apague ou renomeie `plot/net_metric.csv` antes de cada novo treino (por exemplo, `mv plot/net_metric.csv plot/run_01.csv`) e use `--csv` para plotar execuções antigas.
 
---

## Formato do dataset

CSV no padrão MNIST: uma amostra por linha, o **rótulo (0 a 9)** na primeira coluna, seguido de **784 valores de pixel (0 a 255)** de uma imagem 28x28. A primeira linha lida é tratada como cabeçalho e descartada.

```
label,1x1,1x2,...,28x28
7,0,0,0,...,253,0
2,0,0,0,...,18,0
```

Os pixels são divididos por 255, resultando em valores entre 0 e 1.

---

## Arquitetura padrão da rede

A rede criada pelo menu é `newNeuralNet(batchSize, 4, 128, 784, 10)`:

```
Entrada     Oculta 1    Oculta 2    Oculta 3    Saída
 784    ->    128    ->   128    ->   128    ->   10
(pixels)    sigmoide    sigmoide    sigmoide    softmax
```

| Camada         | Pesos     | Bias | Parâmetros  |
| -------------- | --------- | ---- | ----------- |
| 1 (784 -> 128) | 784 x 128 | 128  | 100.480     |
| 2 (128 -> 128) | 128 x 128 | 128  | 16.512      |
| 3 (128 -> 128) | 128 x 128 | 128  | 16.512      |
| 4 (128 -> 10)  | 128 x 10  | 10   | 1.290       |
| **Total**      |           |      | **134.794** |

---

## Teoria aplicada

### 1. Neurônio e camada densa

Cada neurônio calcula uma soma ponderada das entradas mais um viés (*bias*) e aplica uma função de ativação. Para uma camada inteira, com um lote de `B` amostras, isso vira uma operação matricial:

```
Z = X · W + b
A = f(Z)
```

- `X` tem dimensão `B x n_in` (cada linha é uma amostra).
- `W` tem dimensão `n_in x n_out`.
- `b` tem dimensão `1 x n_out` e é somado a todas as linhas por *broadcasting*. É exatamente o que `addMatrices` implementa quando uma das matrizes tem 1 linha.
- `f` é a função de ativação.

Em código, isso é `weightedSum(inputs, weights, bias)` seguido de `sigmoid` ou `softmax`.

### 2. Propagação direta (*forward pass*)

A saída de cada camada vira a entrada da seguinte:

```
A0 = X
A1 = sigmoide(A0 · W1 + b1)
A2 = sigmoide(A1 · W2 + b2)
A3 = sigmoide(A2 · W3 + b3)
A4 = softmax (A3 · W4 + b4)      <- probabilidades das 10 classes
```

Cada camada guarda sua própria saída (`layer->output`), pois o backpropagation precisa dessas ativações.

### 3. Funções de ativação

**Sigmoide** (camadas ocultas):

```
σ(z) = 1 / (1 + e^(-z))        σ'(z) = σ(z) · (1 - σ(z))
```

Comprime qualquer valor em `(0, 1)` e introduz a não linearidade, sem a qual várias camadas equivaleriam a uma só transformação linear. A implementação limita a entrada em ±88 para evitar overflow em `expf` com `float`. A derivada é calculada a partir da **saída já ativada** (`a · (1 - a)`), sem recalcular a exponencial.

**Softmax** (camada de saída):

```
softmax(z_i) = e^(z_i) / Σ_j e^(z_j)
```

Transforma os 10 valores de saída em uma distribuição de probabilidade (positivos, soma 1). A classe prevista é a de maior probabilidade. O código subtrai o valor máximo da linha antes de exponenciar (*log-sum-exp trick*), o que não altera o resultado matemático mas evita overflow.

### 4. Codificação *one-hot* dos rótulos

O rótulo `3` vira o vetor `[0,0,0,1,0,0,0,0,0,0]`. É a matriz `target` que se compara com a saída da softmax.

### 5. Função de perda: entropia cruzada categórica

```
L = -(1/B) · Σ_amostras Σ_classes  y_c · log(ŷ_c)
```

Como `y` é one-hot, só conta o logaritmo da probabilidade atribuída à classe correta. Se a rede dá probabilidade alta à classe certa, a perda é próxima de 0. Se dá probabilidade baixa, a perda cresce muito. A predição é limitada a `[1e-7, 1 - 1e-7]` antes do `logf` para evitar `-inf`.

### 6. Backpropagation

O objetivo é calcular o quanto cada peso contribuiu para o erro, usando a regra da cadeia. Define-se o **delta** de cada camada, `δ = ∂L/∂Z`.

**Camada de saída.** A combinação softmax + entropia cruzada tem uma derivada que se simplifica de forma conhecida:

```
δ_saída = ŷ - y
```

É o que `outputDelta` faz (`subtractMatrices(output, target)`).

**Camadas ocultas.** O erro é propagado para trás pelos pesos da camada seguinte e multiplicado, elemento a elemento, pela derivada da ativação:

```
δ_l = (δ_(l+1) · W_(l+1)ᵀ) ⊙ a_l ⊙ (1 - a_l)
```

Em código, é `hiddenDelta`: transpõe os pesos seguintes, multiplica pelo delta seguinte, e aplica o produto de Hadamard com a derivada da sigmoide.

**Gradientes dos parâmetros:**

```
∂L/∂W_l = A_(l-1)ᵀ · δ_l
∂L/∂b_l = soma das linhas de δ_l
```

`propagateBackward` primeiro calcula **todos** os deltas, da última camada à primeira, e só depois atualiza os parâmetros. Assim todos os deltas usam os pesos antigos, como a teoria exige.

### 7. Gradiente descendente em mini-batches

Em vez de atualizar com uma amostra (ruidoso) ou com o dataset inteiro (lento), usa-se um lote de `B` amostras:

```
W ← W - (η / B) · ∂L/∂W
b ← b - (η / B) · ∂L/∂b
```

- `η` é a taxa de aprendizado.
- Dividir por `B` equivale a usar a **média** dos gradientes do lote (`scale = learning_rate / batch_size` em `updateLayerParameters`).

Uma **época** é uma passada completa por todos os lotes de treino.

### 8. Inicialização Xavier (Glorot)

Começar com pesos todos iguais faz todos os neurônios aprenderem a mesma coisa. Começar com pesos grandes demais satura a sigmoide. A inicialização Xavier sorteia, de uma distribuição uniforme:

```
W ~ U(-limite, +limite),   limite = sqrt(6 / (n_in + n_out))
```

Isso mantém a variância das ativações e dos gradientes aproximadamente estável entre camadas. Os bias começam em zero.

### 9. Normalização dos dados

Pixels de 0 a 255 são divididos por 255. Entradas na faixa `[0, 1]` evitam somas ponderadas muito grandes e mantêm a sigmoide fora da região de saturação nas primeiras iterações.

### 10. Validação e early stopping

Parte do conjunto de treino (20% por padrão, as últimas amostras) é separada para **validação**. A rede não aprende com ela, apenas é avaliada a cada época. Isso detecta *overfitting*: a perda de treino continua caindo enquanto a de validação para de melhorar, sinal de que a rede está memorizando.

Dois critérios interrompem o treino, ambos com **paciência de 5 épocas** e melhoria mínima (`min_delta`) de `1e-4`:

| Critério    | Condição                                                  |
| ----------- | --------------------------------------------------------- |
| Estagnação  | A perda média de treino não melhora por 5 épocas seguidas |
| Overfitting | A perda de validação não melhora por 5 épocas seguidas    |

### 11. Métrica de avaliação

A acurácia é a fração de amostras em que `argmax(saída)` coincide com o rótulo verdadeiro. A função de predição também retorna a **confiança** (a probabilidade da classe escolhida).

---

## Formato do arquivo de modelo (.bin)

Escrito com `fwrite`, em ordem, sem cabeçalho de versão:

```
int   layer_count
para cada camada:
    int    neuron_count
    int    weights.rows
    int    weights.cols
    float  weights.data[rows * cols]
    int    bias.rows
    int    bias.cols
    float  bias.data[rows * cols]
```

O arquivo guarda apenas pesos e bias. As saídas intermediárias são recriadas ao carregar, com o `batch_size` informado no momento do teste. O formato depende do tamanho de `int`/`float` e da ordem de bytes (*endianness*) da máquina, então não é portável entre arquiteturas diferentes.

---

## Limitações conhecidas e próximos passos

Pontos que valem atenção, para quem for estudar ou evoluir o código:

- **Último lote incompleto:** quando `train_size` não é múltiplo de `batch_size`, o último lote tem menos linhas que o buffer de saída das camadas (dimensionado com `batch_size`). Prefira valores de `train_size` divisíveis por `batch_size`, ou trate o lote final de forma explícita.
- **Sem embaralhamento:** as amostras são percorridas sempre na mesma ordem a cada época. Embaralhar (*shuffle*) costuma melhorar a convergência.
- **Validação nas últimas amostras:** o split é sequencial, não aleatório. Se o CSV estiver ordenado por classe, a validação ficará enviesada.
- **Multiplicação de matrizes ingênua:** três laços `O(n³)`, sem otimização de cache (por exemplo, percorrer `B` transposta ou usar *blocking*). É o maior gargalo de desempenho.
- **Sigmoide em camadas profundas:** sofre com *vanishing gradient*. ReLU (com inicialização He) é a alternativa usual.
- **Interface 100% interativa:** não há *parsing* de argumentos (`argc/argv`) para uso em scripts.
- **`plot/net_metric.csv`:** o diretório `plot/` precisa existir antes do treino.
- **Portabilidade:** `strcasecmp` e `unistd.h` limitam a compilação nativa no Windows.

Ideias de evolução: ReLU + He init, momentum/Adam, shuffle, otimização de `multiplyMatrices`, argumentos de CLI, versionamento do formato `.bin` e testes unitários para as operações de matriz.

---