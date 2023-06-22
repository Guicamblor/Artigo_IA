#K-Means em 3D com uma nova biblioteca de gráficos

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import plotly.graph_objects as go

# Lendo os dados da planilha
data = pd.read_excel('/content/folhahspm2018-xlsx.xlsx')

# Obtendo os valores das colunas de interesse
X = data[[' VENCIMENTOS ', ' ENCARGOS ', ' BENEFÍCIOS ']].values

# Definindo o número de clusters
num_clusters = 3

# Criando o modelo K-Means
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# Realizando o treinamento do modelo
kmeans.fit(X)

# Obtendo os rótulos dos clusters e os centróides
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Criando os traces dos pontos de dados
data_trace = go.Scatter3d(
    x=X[:, 0],
    y=X[:, 1],
    z=X[:, 2],
    mode='markers',
    marker=dict(
        color=labels,
        colorscale='Viridis',
        size=4,
        line=dict(color='black', width=0.5)
    ),
    name='Pontos de Dados'
)

# Criando o trace dos centróides dos clusters
centroid_trace = go.Scatter3d(
    x=centroids[:, 0],
    y=centroids[:, 1],
    z=centroids[:, 2],
    mode='markers',
    marker=dict(
        color='red',
        symbol='x',
        size=8,
        line=dict(color='black', width=1)
    ),
    name='Centróides'
)

# Criando a lista de dados
data_list = [data_trace, centroid_trace]

# Criando o layout do gráfico
layout = go.Layout(
    scene=dict(
        xaxis=dict(title='VENCIMENTOS'),
        yaxis=dict(title='ENCARGOS'),
        zaxis=dict(title='BENEFÍCIOS')
    ),
    title='K-Means - Clusterização de Vencimentos, Encargos e Benefícios'
)

# Criando a figura do gráfico
fig = go.Figure(data=data_list, layout=layout)

# Exibindo o gráfico
fig.show()
