#Regressão Logistica
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

# Lista de caminhos para as quatro planilhas
planilhas = ['/content/dados_1.xlsx']

# Definir limites para agrupar a variável dependente em classes
limite1 = 1000
limite2 = 2000

# Configurações do gráfico
plt.figure(figsize=(8, 6))
colors = ['blue', 'green', 'red']
markers = ['o', '^', 's']

# Para cada planilha
for planilha in planilhas:
    # Lendo os dados da planilha
    data = pd.read_excel(planilha)

    # Fazendo trim no nome da coluna "BENEFÍCIOS"
    data = data.rename(columns=lambda x: x.strip() if isinstance(x, str) else x)

    # Substituindo valores NaN por 0 na coluna "BENEFÍCIOS"
    data['ENCARGOS'].fillna(0, inplace=True)

    # Convertendo a variável dependente contínua em classes
    data['CLASSE'] = pd.cut(data['ENCARGOS'], bins=[-float('inf'), limite1, limite2, float('inf')], labels=[0, 1, 2])

    # Obtendo os valores da coluna "BENEFÍCIOS"
    X = data['ENCARGOS'].values.reshape(-1, 1)

    # Obtendo os valores do alvo (variável dependente)
    y = data['CLASSE'].values

    # Criação do modelo de regressão logística
    logistic_model = LogisticRegression()

    # Treinamento do modelo
    logistic_model.fit(X, y)

    # Realizando previsões
    y_pred = logistic_model.predict(X)

    # Plotando os pontos de dados
    plt.scatter(X, y, c=y, cmap='viridis', marker=markers[0], label=planilha)

    # Plotando a fronteira de decisão
    x_boundary = np.linspace(min(X), max(X), 100)
    y_boundary = logistic_model.predict(x_boundary.reshape(-1, 1))
    plt.plot(x_boundary, y_boundary, c=colors[0])

# Configurações do gráfico
plt.xlabel('BENEFÍCIOS')
plt.ylabel('CLASSE')
plt.title('Regressão Logística - Classificação de BENEFÍCIOS')
plt.show()