import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


planilhas = ['/content/dados_1.xlsx','/content/dados_2.xlsx','/content/dados_3.xlsx','/content/dados_4.xlsx','/content/dados_5.xlsx'] # Caminhos pegos pelo GoogleColab

for planilha in planilhas:
    
    data = pd.read_excel(planilha)

    # Removendo espaços em branco no início e fim dos nomes das colunas
    data.columns = data.columns.str.strip()

    # Substituindo valores NaN por 0
    data.fillna(0, inplace=True)

    # Obtendo os valores das colunas de interesse
    X = data[['VENCIMENTOS', 'ENCARGOS', 'BENEFÍCIOS']].values

    # Definindo o número de clusters
    num_clusters = 3

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)

    kmeans.fit(X)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Fazendo o gráfico de pontos com X no meio de seus clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=150, c='red')
    plt.xlabel('VENCIMENTOS')
    plt.ylabel('ENCARGOS')
    plt.title('K-Means - Clusterização de Vencimentos, Encargos e Benefícios')
    plt.show()