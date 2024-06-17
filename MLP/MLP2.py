import numpy as np
import random as rd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from prettytable import PrettyTable

def inicializa_pesos(entradas, neur, faixa):
    return np.random.uniform(-faixa, faixa, (entradas, neur)), np.random.uniform(-faixa, faixa, (1, neur))

def calcula_saida(x, v, v0, w, w0):
    zin_j = np.dot(x, v) + v0
    z_j = np.tanh(zin_j)
    yin = np.dot(z_j, w) + w0
    y = np.tanh(yin)
    return y, z_j

def atualiza_pesos(x, t, z_j, y, alfa, v, v0, w, w0):
    deltinha_k = (t - y) * (1 + y) * (1 - y)
    deltaw = alfa * np.dot(z_j.T, deltinha_k)
    deltaw0 = alfa * deltinha_k.sum(axis=0)
    deltinha_j = np.dot(deltinha_k, w.T) * (1 + z_j) * (1 - z_j)
    deltav = alfa * np.dot(x.T, deltinha_j)
    deltav0 = alfa * deltinha_j.sum(axis=0)
    return v + deltav, v0 + deltav0, w + deltaw, w0 + deltaw0

def treinar_rede(x, t, alfa, errotolerado, faixa_v, faixa_w, ciclo_maximo, neur):
    entradas = x.shape[1]
    vsai = t.shape[1]

    v, v0 = inicializa_pesos(entradas, neur, faixa_v)
    w, w0 = inicializa_pesos(neur, vsai, faixa_w)

    ciclo = 0
    errototal = float('inf')

    while errototal > errotolerado and ciclo < ciclo_maximo:
        errototal = 0
        for padrao in range(x.shape[0]):
            y, z_j = calcula_saida(x[padrao:padrao+1, :], v, v0, w, w0)
            errototal += np.sum((t[padrao:padrao+1, :] - y) ** 2) / 2
            v, v0, w, w0 = atualiza_pesos(x[padrao:padrao+1, :], t[padrao:padrao+1, :], z_j, y, alfa, v, v0, w, w0)
        
        ciclo += 1
    
    return v, v0, w, w0, ciclo, errototal

def avaliar_rede(x, t, v, v0, w, w0):
    y, _ = calcula_saida(x, v, v0, w, w0)
    rmse = np.sqrt(mean_squared_error(t, y))
    return y, rmse

# Parâmetros de teste
alfas = [0.001, 0.005, 0.01]
erros_tolerados = [0.05, 0.02, 0.01]
faixa_aleatoria_v = [1, 1.1]
faixa_aleatoria_w = [0.1, 0.2, 0.3]
ciclos_maximos = [500, 1000, 1500]
neurs = [50, 100, 150, 200]

# Gerando entradas e targets
xmin, xmax, npontos = -1, 1, 50
x = np.linspace(xmin, xmax, npontos).reshape(-1, 1)
t = (np.sin(x) * np.sin(2 * x)).reshape(-1, 1)

resultados = []

for alfa in alfas:
    for errotolerado in erros_tolerados:
        for faixa_v in faixa_aleatoria_v:
            for faixa_w in faixa_aleatoria_w:
                for ciclo_maximo in ciclos_maximos:
                    for neur in neurs:
                        v, v0, w, w0, ciclos, errototal = treinar_rede(x, t, alfa, errotolerado, faixa_v, faixa_w, ciclo_maximo, neur)
                        y, rmse = avaliar_rede(x, t, v, v0, w, w0)
                        resultados.append({
                            "alfa": alfa, "erro_tolerado": errotolerado,
                            "faixa_v": faixa_v, "faixa_w": faixa_w,
                            "ciclo_maximo": ciclo_maximo, "neur": neur,
                            "rmse": rmse
                        })

# Cria uma tabela PrettyTable
tabela = PrettyTable()
tabela.field_names = ["Alfa", "Erro Tolerado", "Faixa V", "Faixa W", "Ciclo Máximo", "Neurônios", "RMSE"]

for resultado in resultados:
    tabela.add_row([
        resultado["alfa"],
        resultado["erro_tolerado"],
        resultado["faixa_v"],
        resultado["faixa_w"],
        resultado["ciclo_maximo"],
        resultado["neur"],
        resultado["rmse"]
    ])

print(tabela)

# Encontra o menor RMSE
melhor_resultado = min(resultados, key=lambda x: x["rmse"])

# Gráfico com as variáveis utilizadas no menor RMSE
v, v0, w, w0, _, _ = treinar_rede(x, t, melhor_resultado["alfa"], melhor_resultado["erro_tolerado"],
                                  melhor_resultado["faixa_v"], melhor_resultado["faixa_w"],
                                  melhor_resultado["ciclo_maximo"], melhor_resultado["neur"])

y, _ = avaliar_rede(x, t, v, v0, w, w0)

plt.plot(x, t, color='red', label='Target')
plt.plot(x, y, color='blue', label='Previsão')
plt.title(f'Gráfico com as variáveis do menor RMSE\nAlfa: {melhor_resultado["alfa"]} '
          f'Erro Tolerado: {melhor_resultado["erro_tolerado"]} Faixa V: {melhor_resultado["faixa_v"]} '
          f'Faixa W: {melhor_resultado["faixa_w"]} Ciclo Máximo: {melhor_resultado["ciclo_maximo"]} '
          f'Neurônios: {melhor_resultado["neur"]}')
plt.legend()
plt.show()
