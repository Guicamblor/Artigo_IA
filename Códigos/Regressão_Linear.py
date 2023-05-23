#Regressão Linear
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

#Este código pode-se colocar mais de uma planilha e caso queira pode alterar a Coluna (Testes foram feitos nas colunas dessas planilhas "BENEFÍCIOS", "ENCARGOS" e "VENCIMENTOS")

planilhas = ['/content/dados_5.xlsx'] # Caminho pego pelo GoogleColab

plt.figure(figsize=(8, 6))

for planilha in planilhas:
    data = pd.read_excel(planilha)

    data = data.rename(columns=lambda x: x.strip() if isinstance(x, str) else x)

    data['ENCARGOS'].fillna(0, inplace=True)

    X = data['ENCARGOS'].values.reshape(-1, 1)

    data['VENCIMENTOS'].fillna(0, inplace=True)

    y = data['VENCIMENTOS'].values

    # Criação do modelo de regressão linear
    linear_model = LinearRegression()

    # Treinamento do modelo
    linear_model.fit(X, y)

    # Realizando previsões
    y_pred = linear_model.predict(X)

    # Plotando os pontos de dados e a linha de regressão
    plt.scatter(X, y, label=planilha)
    plt.plot(X, y_pred, c='red')

plt.xlabel('ENCARGOS')
plt.ylabel('VENCIMENTOS')
plt.title('Regressão Linear - Relação entre ENCARGOS e VENCIMENTOS')
#plt.legend("Valores da planilha")
plt.show()