from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5]]
y = [2, 4, 6, 8, 10]

# Criar o modelo:
modelo = LinearRegression()

# Treinar o modelo:
modelo.fit(X, y)

saida = modelo.predict([[50]])

print(saida[0])