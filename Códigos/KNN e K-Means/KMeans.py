import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Lendo os dados da planilha
data = pd.read_excel('/content/folhahspm2018-xlsx.xlsx') # Caminhos pegos pelo GoogleColab

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

# Plotando os pontos de dados e os centróides dos clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=150, c='red')
plt.xlabel('VENCIMENTOS')
plt.ylabel('ENCARGOS')
plt.title('K-Means - Clusterização de Vencimentos, Encargos e Benefícios')
plt.show()