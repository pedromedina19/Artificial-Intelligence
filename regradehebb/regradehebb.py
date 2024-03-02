# Importando bibliotecas necessárias
import numpy as np


deltaW = np.zeros(64)
b = 0


def treinamentoHebb(entrada1, entrada2):
    global b, deltaW

    # Inicializando variáveis
    entrada = np.zeros((2, 64))
    y = np.array([1, -1])

    # Loop para inicialização
    for cont2 in range(64):
        entrada[0][cont2] = entrada1[cont2]
        entrada[1][cont2] = entrada2[cont2]

    # Aplicação da regra
    for cont1 in range(2):
        for cont2 in range(64):
            deltaW[cont2] += entrada[cont1][cont2] * y[cont1]
        b += y[cont1]

    return ("Treinamento concluído.")


def testeHebb(entrada):
    global b, deltaW

    deltaTeste = 0
    for cont2 in range(64):
        deltaTeste += (deltaW[cont2] * entrada[cont2])
    deltaTeste += b

    if deltaTeste >= 0:
        return ("Resultado == 1 (Tabela 1)")
    else:
        return ("Resultado == -1 (Tabela 2)")
