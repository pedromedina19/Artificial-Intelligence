from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image

# Inicializando a Rede Neural Convolucional (CNN)
classificador = Sequential()

# Primeira camada de convolução e normalização em lote, seguida de uma camada de pooling
classificador.add(Conv2D(64, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

# Segunda camada de convolução e normalização em lote, seguida de uma camada de pooling
classificador.add(Conv2D(64, (3,3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

# Conversão dos mapas de características em um vetor único para alimentar as camadas densas
classificador.add(Flatten())

# Camadas densas com regularização por dropout para reduzir overfitting
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))

# Camada de saída com uma unidade para classificação binária
classificador.add(Dense(units = 1, activation = 'sigmoid'))

# Compilando o modelo com otimizador Adam e função de perda para classificação binária
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['accuracy'])

# Configuração dos geradores de imagem com pré-processamento e aumento de dados
gerador_treinamento = ImageDataGenerator(rescale = 1./255,
                                         rotation_range = 7,
                                         horizontal_flip = True,
                                         shear_range = 0.2,
                                         height_shift_range = 0.07,
                                         zoom_range = 0.2)
gerador_teste = ImageDataGenerator(rescale = 1./255)

# Preparação das bases de dados de treinamento e teste com as imagens processadas pelos geradores
base_treinamento = gerador_treinamento.flow_from_directory('dataset/training_set',
                                                           target_size = (64, 64),
                                                           batch_size = 32,
                                                           class_mode = 'binary')
base_teste = gerador_teste.flow_from_directory('dataset/test_set',
                                               target_size = (64, 64),
                                               batch_size = 32,
                                               class_mode = 'binary')

# Treinamento do modelo usando os dados preparados anteriormente
classificador.fit(base_treinamento, steps_per_epoch = 4000 // 24,
                  epochs = 20, validation_data = base_teste,
                  validation_steps = 1000 // 24)

# Carregamento e preparação da imagem para teste do modelo treinado
imagem_teste = image.load_img('dataset/test_set/gato/cat.3502.jpg',
                              target_size = (64,64))
imagem_teste = image.img_to_array(imagem_teste)
imagem_teste /= 255
imagem_teste = np.expand_dims(imagem_teste, axis = 0)

# Realização da previsão usando o modelo treinado
previsao = classificador.predict(imagem_teste)
previsao = (previsao > 0.5)

# Exibição do resultado da previsão
print(previsao)

# Exibição dos índices das classes utilizadas pelo modelo
base_treinamento.class_indices
