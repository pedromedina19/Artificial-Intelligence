import numpy as np
import random as rd
import matplotlib.pyplot as plt
import math
from collections import Counter

# Funções do algoritmo genético

# Gera n elementos aleatórios dentro dos limites fornecidos
def gerarElementos(n, limite_inferior=0, limite_superior=512):
    return [rd.uniform(limite_inferior, limite_superior) for _ in range(n)]

# Converte os elementos gerados em uma representação binária, incluindo parte fracionária
def gerarElementosBinarios(vetor_aleatorio, tamanho_cromossomo, precisao):
    vetor_binario = []
    for numero in vetor_aleatorio:
        inteiro = int(numero)
        fracao = numero - inteiro
        inteiro_binario = f"{inteiro:0>{tamanho_cromossomo}b}"
        fracao_binaria = ''.join(['1' if (fracao := fracao * 2) >= 1 else '0' for _ in range(precisao)])
        vetor_binario.append(f"{inteiro_binario}.{fracao_binaria}")
    return vetor_binario

# Calcula a imagem (aptidão) dos elementos
def gerarImagem(vetor_aleatorio):
    return [-abs(x * math.sin(math.sqrt(abs(x)))) for x in vetor_aleatorio]

# Gera as probabilidades de seleção com base na aptidão dos elementos
def gerarProbabilidades(vetor_imagem):
    soma_imagem = sum(vetor_imagem)
    return [(x / soma_imagem) * 100 for x in vetor_imagem]

# Realiza a seleção por torneio
def selecao_torneio(vetor_prob, vetor_binario, tamanho_torneio):
    """Realiza a seleção por torneio."""
    indices_torneio = np.random.choice(len(vetor_prob), tamanho_torneio, replace=False)
    melhores_torneio = sorted([(vetor_binario[i], vetor_prob[i]) for i in indices_torneio], key=lambda x: x[1], reverse=True)
    return melhores_torneio[0][0]

# Separa os n melhores elementos com base nas probabilidades
def separarMelhores(vetor_prob, vetor_binario, n):
    """Separa os n melhores elementos."""
    melhores_indices = sorted(range(len(vetor_prob)), key=lambda i: vetor_prob[i], reverse=True)[:n]
    return [vetor_binario[i] for i in melhores_indices]

# Sorteia casais para o cruzamento
def sortearCasais(qtd_melhores):
    """Sorteia casais para cruzamento."""
    indices = np.random.permutation(qtd_melhores)
    return [(indices[i], indices[i+1]) for i in range(0, qtd_melhores, 2)]

# Realiza o cruzamento entre os pais, gerando filhos
def recombinar(ponto_corte, casais_sorteados, vetor_melhores):
    """Realiza o cruzamento entre os pais."""
    filhos = []
    for pai1_idx, pai2_idx in casais_sorteados:
        pai1, pai2 = vetor_melhores[pai1_idx], vetor_melhores[pai2_idx]
        pai1_partes, pai2_partes = pai1.split('.'), pai2.split('.')
        filho1 = f"{pai1_partes[0][:ponto_corte]}{pai2_partes[0][ponto_corte:]}.{pai1_partes[1]}"
        filho2 = f"{pai2_partes[0][:ponto_corte]}{pai1_partes[0][ponto_corte:]}.{pai2_partes[1]}"
        filhos.extend([filho1, filho2])
    return filhos

# Realiza o cruzamento de dois pontos entre os pais, gerando filhos
def recombinar2(pontos_corte, casais_sorteados, vetor_melhores):
    """Realiza o cruzamento de dois pontos entre os pais."""
    filhos = []
    corte1, corte2 = sorted(pontos_corte)
    for pai1_idx, pai2_idx in casais_sorteados:
        pai1, pai2 = vetor_melhores[pai1_idx], vetor_melhores[pai2_idx]
        pai1_partes, pai2_partes = pai1.split('.'), pai2.split('.')
        filho1 = f"{pai1_partes[0][:corte1]}{pai2_partes[0][corte1:corte2]}{pai1_partes[0][corte2:]}.{pai1_partes[1]}"
        filho2 = f"{pai2_partes[0][:corte1]}{pai1_partes[0][corte1:corte2]}{pai2_partes[0][corte2:]}.{pai2_partes[1]}"
        filhos.extend([filho1, filho2])
    return filhos

# Realiza a mutação dos filhos
def gerarMutacao(filhos, prob_mutacao, tamanho_cromossomo):
    """Realiza a mutação dos filhos."""
    num_mutacao = int(len(filhos) * prob_mutacao / 100)
    for _ in range(num_mutacao):
        indice = rd.randint(0, len(filhos) - 1)
        posicao = rd.randint(0, tamanho_cromossomo - 1)
        lista_binario = list(filhos[indice])
        lista_binario[posicao] = '0' if lista_binario[posicao] == '1' else '1'
        filhos[indice] = ''.join(lista_binario)
    return filhos

# Separa os n piores elementos com base nas probabilidades
def separarPiores(vetor_prob, vetor_binario, n):
    """Separa os n piores elementos."""
    piores_indices = sorted(range(len(vetor_prob)), key=lambda i: vetor_prob[i])[:n]
    return [vetor_binario[i] for i in piores_indices]

# Substitui os piores elementos pelos filhos gerados
def substituirPioresPorFilhos(populacao, piores_elementos, filhos_melhores):
    """Substitui os piores elementos pelos filhos."""
    indices_piores = [populacao.index(pior) for pior in piores_elementos]
    for i, indice in enumerate(indices_piores):
        populacao[indice] = filhos_melhores[i]
    return populacao

# Converte números binários para decimais
def novosValoresDecimais(vetor_binarios):
    """Converte números binários para decimais."""
    def binario_para_decimal(binario):
        inteiro, fracao = binario.split('.')
        return int(inteiro, 2) + sum(int(b) * 2**-(i+1) for i, b in enumerate(fracao))
    return [binario_para_decimal(binario) for binario in vetor_binarios]

# Seleciona os n_elites melhores indivíduos com base em sua aptidão
def elitism(vetor_prob, vetor_binario, n_elites):
    """Seleciona os n_elites melhores indivíduos com base em sua aptidão."""
    pop_com_fitness = list(zip(vetor_binario, vetor_prob))
    pop_com_fitness.sort(key=lambda x: x[1], reverse=True)
    return [ind for ind, fit in pop_com_fitness[:n_elites]]

# Recebendo os valores do usuário via input()
tamanho_cromossomo = int(input("Tamanho do Cromossomo (parte inteira): "))
tamanho_populacao = int(input("Tamanho da População: "))
qtd_melhores = int(input("Quantidade de Melhores por Geração: "))
prob_mutacao = int(input("Probabilidade de Mutação (em porcentagem): "))
qtd_geracoes = int(input("Quantidade de Gerações: "))
precisao_fracao = int(input("Precisão da Parte Fracionária (bits): "))
n_elites = int(input("Número de Elites: "))

# População inicial
vetor_aleatorio = gerarElementos(tamanho_populacao)
vetor_binario = gerarElementosBinarios(vetor_aleatorio, tamanho_cromossomo, precisao_fracao)
vetor_imagem = gerarImagem(vetor_aleatorio)
vetor_prob = gerarProbabilidades(vetor_imagem)

print("População inicial:", vetor_binario)

# Execução das gerações
for geracao in range(qtd_geracoes):
    print(f"Geração {geracao + 1}")

    # Seleção dos melhores
    melhores = separarMelhores(vetor_prob, vetor_binario, qtd_melhores)

    # Seleção dos casais
    casais_sorteados = sortearCasais(qtd_melhores)
    
    # Recombinação com probabilidade de cruzamento
    filhos_melhores = []
    for casal in casais_sorteados:
        if rd.random() < 0.8:  # Probabilidade de cruzamento (80% por exemplo)
            filhos_melhores.extend(recombinar(rd.randint(1, tamanho_cromossomo-1), [casal], melhores))
        else:
            filhos_melhores.extend([melhores[casal[0]], melhores[casal[1]]])

    # Mutação
    filhos_melhores = gerarMutacao(filhos_melhores, prob_mutacao, tamanho_cromossomo)
    
    # Elitismo
    elites = elitism(vetor_prob, vetor_binario, n_elites)
    
    # Substituição dos piores pelos melhores filhos
    piores = separarPiores(vetor_prob, vetor_binario, len(filhos_melhores) - n_elites)
    vetor_binario = substituirPioresPorFilhos(vetor_binario, piores, filhos_melhores[n_elites:])

    # Reintroduzir os elites
    vetor_binario = vetor_binario[:len(vetor_binario) - n_elites] + elites

    # Atualização dos valores
    vetor_aleatorio = novosValoresDecimais(vetor_binario)
    vetor_imagem = gerarImagem(vetor_aleatorio)
    vetor_prob = gerarProbabilidades(vetor_imagem)

    print("Melhores da Geração:", melhores)
    print("Filhos Gerados:", filhos_melhores)
    print("População Atualizada:", vetor_binario)

# Exibição dos resultados
melhor_indice = np.argmax(vetor_prob)
melhor_individuo = vetor_aleatorio[melhor_indice]
melhor_aptidao = vetor_prob[melhor_indice]

print(f"Melhor indivíduo: {melhor_individuo}, com aptidão: {melhor_aptidao}")

# Gráfico dos resultados
fig, ax = plt.subplots()
ax.plot(vetor_aleatorio, vetor_imagem, 'bo')
ax.plot(melhor_individuo, melhor_aptidao, 'ro')
ax.set_title('Gráfico da Função')
ax.set_xlabel('X')
ax.set_ylabel('f(X)')
plt.show()
