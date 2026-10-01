import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def PlotMetric(csvFile: str, x_param: str, y_param: str):
    """
    Gera um gráfico dinâmico a partir do CSV de métricas da rede neural.

    :str csvFile: Caminho relativo ou absoluto do arquivo CSV.
    :str x_param: Nome da coluna do CSV para o eixo X.
    :str y_param: Nome da coluna do CSV para o eixo Y.
    """
    if csvFile.startswith("/"):
        csvFile = csvFile[1:] 

    currentDir = os.getcwd()
    fullPathCsv = os.path.join(currentDir, csvFile)

    if not os.path.exists(fullPathCsv):
        raise FileNotFoundError(f"Arquivo CSV não encontrado em: {fullPathCsv}")

    df = pd.read_csv(fullPathCsv)

    if x_param not in df.columns or y_param not in df.columns:
        raise ValueError(
            f"Parâmetro inválido! Colunas disponíveis no CSV: {list(df.columns)}"
        )

    x_data = df[x_param]
    y_data = df[y_param]

    window_size = 50
    y_smoothed = y_data.rolling(window=window_size, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained") 

    ax.plot(x_data, y_data, linewidth=0.5, alpha=0.15, color='tab:blue', label='Bruto (Mini-batch)')
    
    ax.plot(x_data, y_smoothed, linewidth=2, color='tab:red', label=f'Tendência (Média Móvel {window_size} lotes)')
    
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_title(f"Evolução de {y_param.capitalize()} por {x_param.capitalize()} (Suavizado)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower right')
    
    plotDir = os.path.join(currentDir, "plot")
    os.makedirs(plotDir, exist_ok=True)

    pathSaveFig = os.path.join(plotDir, f"{x_param}_vs_{y_param}.png")
    fig.savefig(pathSaveFig, dpi=200)
    plt.close(fig)

    print(f"Gráfico gerado com sucesso: {pathSaveFig}")
