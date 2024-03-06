import numpy as np

def treinar_perceptron(entrada, target, w, b, alfa, limiar):
    # Inicializa a condição de erro e o contador de ciclos
    condErro = 1
    contCiclo = 0
    # Entra no loop enquanto houver erro
    while condErro == 1:
        # Reinicia a condição de erro para cada ciclo
        condErro = 0
        # Loop através de cada linha de entrada
        for lin in range(len(entrada)):
            # Calcula a soma ponderada das entradas
            yLiq = np.dot(entrada[lin], w) + b
            # Aplica a função de ativação
            y = 1 if yLiq >= limiar else 0
            # Imprime a saída e o alvo
            print(f"\n yLiq: {yLiq:.2f} - y: {y:.2f} - target: {target[lin]:.2f}")
            # Se a saída é diferente do alvo, ajusta os pesos e o bias
            if y != target[lin]:
                # Define a condição de erro como 1
                condErro = 1
                # Ajusta os pesos
                w += alfa * (target[lin] - y) * entrada[lin]
                # Ajusta o bias
                b += alfa * (target[lin] - y)
        # Imprime o número de ciclos
        print(f"\n Ciclo: {contCiclo} \n")
        # Incrementa o contador de ciclos
        contCiclo += 1
    # Retorna os pesos e o bias ajustados
    return w, b

def testar_perceptron(entrada, w, b, limiar):
    print("\n\t --->Testando a rede treinada")
    print("\n\n Teste com as entradas do treinamento")
    for lin in range(len(entrada)):
        teste = np.dot(entrada[lin], w) + b
        yTeste = 1 if teste >= limiar else 0
        print(f"\n Entrada: {entrada[lin]} - Saída da rede: {yTeste}")

def operacao_perceptron(entrada, w, b, limiar):
    yLiq = np.dot(entrada, w) + b #soma ponderada das entradas
    y = 1 if yLiq >= limiar else 0 #aplica a função de ativação
    return y

# Inicialização
entradas = {
    'AND': np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
    'OR': np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
    'NAND': np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
    'NOR': np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
    'XOR': np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
}

targets = {
    'AND': np.array([0, 0, 0, 1]),
    'OR': np.array([0, 1, 1, 1]),
    'NAND': np.array([1, 1, 1, 0]),
    'NOR': np.array([1, 0, 0, 0]),
    'XOR': np.array([0, 1, 1, 0])  
}

pesos = {
    'AND': np.array([0.5, -0.1]),
    'OR': np.array([0.5, -0.1]),
    'NAND': np.array([0.5, -0.1]),
    'NOR': np.array([0.5, -0.1]),
    'XOR': np.array([0.5, -0.1])  
}

bias = {
    'AND': 0.6,
    'OR': 0.6,
    'NAND': 0.6,
    'NOR': 0.6,
    'XOR': 0.6  
}

alfa = 0.5  # taxa de aprendizagem.
limiar = 0
opc = 0

while opc != 11:
    print("\n\n ************ Programa Perceptron ************")
    print("\n\n Digite 1 para treinar a rede AND")
    print("\n Digite 2 para treinar a rede OR")
    print("\n Digite 3 para treinar a rede NAND")
    print("\n Digite 4 para treinar a rede NOR")
    print("\n Digite 5 para treinar a rede XOR")
    print("\n Digite 6 para testar a rede AND")
    print("\n Digite 7 para testar a rede OR")
    print("\n Digite 8 para testar a rede NAND")
    print("\n Digite 9 para testar a rede NOR")
    print("\n Digite 10 para testar a rede XOR")
    print("\n Digite 11 para Sair\n ->")
    opc = int(input())
    if opc in [1, 2, 3, 4, 5]:
        porta = ['AND', 'OR', 'NAND', 'NOR', 'XOR'][opc - 1]
        entrada = entradas[porta]
        target = targets[porta]
        w = pesos[porta]
        b = bias[porta]
        w, b = treinar_perceptron(entrada, target, w, b, alfa, limiar)
        pesos[porta] = w
        bias[porta] = b
    elif opc in [6, 7, 8, 9, 10]:
        porta = ['AND', 'OR', 'NAND', 'NOR', 'XOR'][opc - 6]
        entrada = entradas[porta]
        w = pesos[porta]
        b = bias[porta]
        if porta == 'XOR':
            # XOR = (A AND (NOT B)) OR ((NOT A) AND B)
            for i in range(4):
                entrada1 = operacao_perceptron([entrada[i][0], 1 - entrada[i][1]], pesos['AND'], bias['AND'], limiar)
                entrada2 = operacao_perceptron([1 - entrada[i][0], entrada[i][1]], pesos['AND'], bias['AND'], limiar)
                saida = operacao_perceptron([entrada1, entrada2], pesos['OR'], bias['OR'], limiar)
                print(f"\n Entrada: {entrada[i]} - Saída da rede: {saida}")
        else:
            testar_perceptron(entrada, w, b, limiar)
