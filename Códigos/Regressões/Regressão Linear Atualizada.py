#Regressão Linear Atualizada

import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Este código pode-se colocar mais de uma planilha e caso queira pode alterar a Coluna (Testes foram feitos nas colunas dessas planilhas "BENEFÍCIOS", "ENCARGOS" e "VENCIMENTOS")

planilhas = ['/content/dados_5.xlsx']

# Criando uma lista para armazenar os dados
data_list = []

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

    # Criando o trace de dispersão dos pontos de dados
    scatter_trace = go.Scatter(
        x=X.flatten(),
        y=y,
        mode='markers',
        name=planilha
    )

    # Criando o trace da linha de regressão
    regression_trace = go.Scatter(
        x=X.flatten(),
        y=y_pred,
        mode='lines',
        name='Regressão - ' + planilha
    )

    # Adicionando os traces à lista de dados
    data_list.append(scatter_trace)
    data_list.append(regression_trace)

# Criando o layout do gráfico
layout = go.Layout(
    xaxis=dict(title='ENCARGOS'),
    yaxis=dict(title='VENCIMENTOS'),
    title='Regressão Linear - Relação entre ENCARGOS e VENCIMENTOS',
    showlegend=False,
)

# Criando a figura do gráfico
fig = go.Figure(data=data_list, layout=layout)

# Exibindo o gráfico
fig.show()
