import numpy as np

class MLP:
    def __init__(self):
        # Inicialização da classe MLP
        # Definindo a taxa de aprendizado e os dados de entrada e saída
        self.alpha = 0.001
        self.inputs = np.array([[[1, 0.5, -1]], [[0, 0.5, 1]], [[1, -0.5, -1]]])
        self.targets = np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, 1]])

    def forward_propagation(self, input_data, v, w, b_hidden, b_output):
        # Propagação para frente: calcula as saídas das camadas oculta e de saída
        z = np.tanh((np.dot(input_data, v) + b_hidden))
        y = np.tanh((np.dot(z, w) + b_output))
        return z, y

    def back_propagation(self, targets, y, yin, z, zin, w):
        # Propagação para trás: calcula o erro e os deltas
        deltinha_yin = targets - y
        erro = 0.5 * np.sum(deltinha_yin**2)
        deltinha_y = deltinha_yin * (1.0 - np.tanh(yin)**2)
        deltinha_zin = deltinha_y.dot(w.T)
        deltinha_z = deltinha_zin * (1.0 - np.tanh(zin)**2)
        return erro, deltinha_y, deltinha_z

    def update_weights(self, deltinha_y, deltinha_z, z, inputs, v, w, b_hidden, b_output):
        # Atualização dos pesos e bias
        delta_w = self.alpha * np.dot(deltinha_y.T, z)
        delta_b_W = self.alpha * deltinha_y
        delta_v = self.alpha * np.dot(deltinha_z.T, inputs)
        delta_b_V = self.alpha * deltinha_z
        w = w + delta_w.T
        b_output = b_output + delta_b_W
        v = v + delta_v.T
        b_hidden = b_hidden + delta_b_V
        return v, w, b_hidden, b_output

    def treinamento(self):
        # Inicialização dos pesos e bias
        v, w, b_hidden, b_output = self.initialize_weights()
        erro = np.Infinity
        epoch = 0

        # Loop de treinamento
        while epoch < 100000 and erro > 0.1:
            epoch += 1
            erro = 0
            for i in range(len(self.inputs)):
                # Propagação para frente
                z, y = self.forward_propagation(self.inputs[i], v, w, b_hidden, b_output)
                # Propagação para trás
                erro_i, deltinha_y, deltinha_z = self.back_propagation(self.targets[i], y, y, z, z, w)
                erro += erro_i
                # Atualização dos pesos e bias
                v, w, b_hidden, b_output = self.update_weights(deltinha_y, deltinha_z, z, self.inputs[i], v, w, b_hidden, b_output)

            print("Epoch:", epoch, "Error:", erro)

        # Retorno dos pesos e bias treinados
        return v, w, b_hidden, b_output

    def teste(self, inputs, v, w, b_hidden, b_output):
        # Teste da MLP
        outputs = []
        for input_data in inputs:
            _, y = self.forward_propagation(input_data, v, w, b_hidden, b_output)
            outputs.append(y)

        return outputs

    def initialize_weights(self):
        # Inicialização dos pesos e bias
        v = np.random.uniform(size=(3, 100), low=-0.5, high=0.5)
        w = np.random.uniform(size=(100, 3), low=-0.5, high=0.5)
        b_hidden = np.random.uniform(size=(1, 100), low=-0.5, high=0.5)
        b_output = np.random.uniform(size=(1, 3), low=-0.5, high=0.5)
        return v, w, b_hidden, b_output


# Criação e treinamento da MLP
mlp = MLP()
v, w, b_hidden, b_output = mlp.treinamento()

# Dados de teste
test_inputs = [[1, 0.5, -1], [0, 0.5, 1], [1, -0.5, -1]]

# Teste da MLP
test_outputs = mlp.teste(test_inputs, v, w, b_hidden, b_output)

print("Outputs após o teste:")
for output in test_outputs:
    print(output)
