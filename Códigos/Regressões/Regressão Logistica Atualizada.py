#Regressão Logistica Atualizada
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

# Carregar a base de dados de câncer de mama
data = load_breast_cancer()
X = data.data[:, 0]  # Utilizaremos apenas uma característica para facilitar a visualização
y = data.target

# Ajustar a escala dos dados para a regressão logística
X = (X - np.mean(X)) / np.std(X)

# Criar o modelo de regressão logística
model = LogisticRegression()
model.fit(X.reshape(-1, 1), y)

# Gerar pontos para a curva sigmoide
x_values = np.linspace(np.min(X), np.max(X), 100)
y_values = model.predict_proba(x_values.reshape(-1, 1))[:, 1]

# Criar o gráfico de dispersão dos pontos de dados
scatter = go.Scatter(x=X, y=y, mode='markers', marker=dict(colorscale='RdBu', line=dict(color='black')), name="Tipos de Cancer")
layout = go.Layout(title=dict(text="Regressão Logística", xref='paper', x=0.5, xanchor='center'),
                   xaxis=dict(title="Característica"),
                   yaxis=dict(title="Diagnóstico (0: Benigno, 1: Maligno)"))
data = [scatter]
fig = go.Figure(data=data, layout=layout)

# Adicionar a curva sigmoide
sigmoide = go.Scatter(x=x_values, y=y_values, mode='lines', name='Curva Sigmoide', line=dict(color='red'))
fig.add_trace(sigmoide)

# Adicionar uma linha de decisão em 0.5
linha_decisao = go.Scatter(x=[np.min(X), np.max(X)], y=[0.5, 0.5], mode='lines', name='Limiar de Decisão', line=dict(color='green', dash='dash'))
fig.add_trace(linha_decisao)

# Exibir o gráfico
fig.show()
