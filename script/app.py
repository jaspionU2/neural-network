import argparse
from plot_metrics import PlotMetric

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gera gráficos dinâmicos das métricas da rede neural."
    )

    # Argumentos do terminal
    parser.add_argument(
        "--csv",
        type=str,
        default="plot/net_metric.csv",
        help="Caminho do arquivo CSV de métricas",
    )
    parser.add_argument(
        "--x", type=str, required=True, help="Nome da coluna para o eixo X (ex: batch)"
    )
    parser.add_argument(
        "--y",
        type=str,
        required=True,
        help="Nome da coluna para o eixo Y (ex: accuracy)",
    )

    args = parser.parse_args()
    
    PlotMetric(args.csv, args.x, args.y)
