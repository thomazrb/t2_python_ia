from sklearn.ensemble import RandomForestClassifier
import pandas as pd

dados = pd.DataFrame(
    {
        'Temperatura': [30, 25, 20, 15, 32, 28],
        'Umidade': [60, 70, 80, 90, 55, 75],
        'Choveu': [1, 1, 0, 0, 1, 1]
    }
)
print(dados)

X = dados[['Temperatura', 'Umidade']]
y = dados['Choveu']

print(X)
print(y)

modelo = RandomForestClassifier(n_estimators=1000, random_state=42)

modelo.fit(X, y) # <- TREINO (APRENDEU)

novo_dado = pd.DataFrame(
    {
        'Temperatura': [22.5],
        'Umidade': [75]
    }
)

saida = modelo.predict(novo_dado)

print(saida)