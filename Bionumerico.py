# Importando bibliotecas
import numpy as np #Numeros, a gente viu em aula
import matplotlib.pyplot as plt #Biblioteca que faz gráfico etc etc
from sklearn.linear_model import LinearRegression #Literalmente uma biblioteca com o tema desse trabalho

# Dados usados de exemplo: densidade (kg/L) e teor de água (% em massa) do biodiesel
densidade = np.array([0.88, 0.86, 0.90, 0.87, 0.91, 0.89, 0.85, 0.88, 0.92, 0.86]) #Exemplos, da pra alterar
teor_agua = np.array([0.25, 0.22, 0.27, 0.23, 0.28, 0.26, 0.21, 0.24, 0.29, 0.22]) #Exemplos, da pra alterar

# Visualizando os dados em um gráfico de dispersão
plt.scatter(densidade, teor_agua)
plt.xlabel('Densidade (kg/L)') #Dando nome pro eixo X
plt.ylabel('Teor de água (% em massa)') #Dando nome pro eixo Y
plt.show()

# Criando o modelo de regressão linear simples
modelo = LinearRegression()

# Ajustando o modelo aos dados de treinamento
modelo.fit(densidade.reshape(-1, 1), teor_agua) #Literalmente pega os dados das linhas 7 e 8

# Visualizando a linha de regressão no gráfico
x = np.linspace(0.85, 0.92, 100) 
y = modelo.predict(x.reshape(-1, 1))
plt.scatter(densidade, teor_agua)
plt.plot(x, y, color='red')
plt.xlabel('Densidade (kg/L)')
plt.ylabel('Teor de água (% em massa)')
plt.show()

# Avaliando o modelo usando o coeficiente de determinação (R²)
r2 = modelo.score(densidade.reshape(-1, 1), teor_agua)
print('Coeficiente de determinação (R²):', r2)
