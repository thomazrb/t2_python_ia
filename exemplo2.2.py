from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

dados = pd.DataFrame(
    {
        'Temperatura': [30, 25, 20, 15, 32, 28, 22, 18, 26, 29],
        'Umidade': [60, 70, 80, 90, 55, 75, 78, 68, 72, 62],
        'Choveu': [1, 1, 0, 0, 1, 1, 1, 0, 1, 1]
    }
)

X = dados[['Temperatura', 'Umidade']]
y = dados['Choveu']

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)


print(X_teste)
print(y_teste)


modelo = RandomForestClassifier(n_estimators=1000, random_state=42)

modelo.fit(X_treino, y_treino) # <- TREINO (APRENDEU)

acuracia = modelo.score(X_teste, y_teste)

print(acuracia)

saida = modelo.predict(X_teste)

print(saida)