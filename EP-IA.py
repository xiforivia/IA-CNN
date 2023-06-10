from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.datasets import fashion_mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from sklearn.model_selection import KFold
import numpy as np
import sys

file = open('resultado.txt', 'w')
sys.stdout = file

def pre_processamento():
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images.reshape((60000,28,28,1))
    train_images = train_images.astype('float32')/255 # Modificar os valores de cada pixel para que eles variem de 0 a 1 melhorará a taxa de aprendizado do nosso modelo.

    test_images = test_images.reshape((10000,28,28,1))
    test_images = test_images.astype('float32')/255 # Modificar os valores de cada pixel para que eles variem de 0 a 1 melhorará a taxa de aprendizado do nosso modelo.

    train_labels = to_categorical(train_labels) # Nosso modelo não pode trabalhar com dados categóricos diretamente. Portanto, devemos usar uma codificação quente. Em uma codificação ativa, os dígitos de 0 a 9 são representados como um conjunto de nove zeros e um único. O dígito é determinado pela localização do número 1. Por exemplo, você representaria um 3 como [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    test_labels = to_categorical(test_labels) # one hot encoding

    return train_images, train_labels, test_images, test_labels

# Definir a arquitetura da CNN
def criar_modelo(conv_layers, filters, dense_size):
    model = models.Sequential()

    # Adicionar camadas de convolução-pooling
    for i in range(conv_layers):
        model.add(layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape = (28,28,1)))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))


    model.add(layers.Flatten())
    model.add(layers.Dense(dense_size, activation='relu'))
    model.add(layers.Dense(10, activation='softmax')) # 10 é o número de classes
    
    model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

### Main

train_images, train_labels, test_images, test_labels = pre_processamento()

# Definir os valores fixos dos parâmetros
filters = 32
dense_size = 64
k_folds = 5
epochs = 5
batch_size = 64

cv = KFold(n_splits=k_folds, shuffle=True, random_state=42) # vamos embaralhá-los antes de dividi-lo, seed 42

# --------- Testar diferentes quantidades de camadas de convolução-pooling ------------
conv_layers_list = [1, 2, 3]

dct_layer = {}
melhor_acc = 0
melhor_layer = 0
for layer in conv_layers_list:
    fold_no = 1 #contador
    acc_per_fold = [] #acurácia de cada fold
    loss_per_fold = [] #acurácia de cada fold
    for train, test in cv.split(train_images, train_labels): #pra cada fold
        print(f"Treinando fold {fold_no}, com {layer} camada(s) de convolução-pooling" )
        train_X = train_images[train]
        test_X = train_images[test]
        model = criar_modelo(layer, filters, dense_size)
        model.fit(train_X, train_labels[train], epochs=epochs, batch_size=batch_size, verbose=2)
        test_loss, test_acc = model.evaluate(test_X, train_labels[test], verbose=2)
        acc_per_fold.append(test_acc * 100)
        loss_per_fold.append(test_loss * 100)
        fold_no = fold_no + 1

    media_acc_layer = sum(acc_per_fold)/len(acc_per_fold)
    media_loss_layer = sum(loss_per_fold)/len(loss_per_fold)

    dct_layer.update({layer: {"acuracia": media_acc_layer, "loss": media_loss_layer}})

    print(f"Média acurácia dos 5 folds pra {layer} camada(s) de convolução-pooling:", media_acc_layer)
    print(f"Essa acurácia significa que o modelo usando {layer} camada(s) de convolução-pooling, filter: {filters} e tamanho da camada densa: {dense_size} é capaz de classificar corretamente em média {round(media_acc_layer, 1)}% das imagens")
    if media_acc_layer > melhor_acc:
        melhor_acc = media_acc_layer
        melhor_layer = layer
print(f"Portanto, a melhor quantidade de camada(s) de convolução-pooling é {melhor_layer}, que possui {round(melhor_acc, 1)} de acurácia.")

# --------- Testar diferentes quantidades de feature maps (filters) ------------

filters_list = [16, 32, 64]

dct_filters = {}
melhor_acc = 0
melhor_filter = 0
for filters in filters_list:
    fold_no = 1 #contador
    acc_per_fold = [] #acurácia de cada fold
    loss_per_fold = []
    for train, test in cv.split(train_images, train_labels):
        print(f"Treinando fold {fold_no}, com {filters} filters" )
        train_X = train_images[train]
        test_X = train_images[test]
        model = criar_modelo(melhor_layer, filters, dense_size)
        model.fit(train_X, train_labels[train], epochs=epochs, batch_size=batch_size, verbose=2)
        test_loss, test_acc = model.evaluate(test_X, train_labels[test], verbose=2)
        acc_per_fold.append(test_acc * 100)
        loss_per_fold.append(test_loss * 100)
        fold_no = fold_no + 1

    media_acc_filters = sum(acc_per_fold)/len(acc_per_fold)
    media_loss_filters = sum(loss_per_fold)/len(loss_per_fold)

    dct_filters.update({filters: {"acuracia": media_acc_filters, "loss": media_loss_filters}})

    print(f"Média acurácia dos 5 folds pra {filters} filters:", media_acc_filters)
    print(f"Essa acurácia significa que o modelo usando {melhor_layer} camada(s) de convolução-pooling, filter: {filters} e tamanho da camada densa: {dense_size} é capaz de classificar corretamente em média {round(media_acc_filters, 1)}% das imagens")
    if media_acc_filters > melhor_acc:
        melhor_acc = media_acc_filters
        melhor_filter = filters
print(f"Portanto, a melhor quantidade de filters é {melhor_filter}, que possui {round(melhor_acc, 1)} de acurácia.")

# --------- Testar diferentes tamanhos de camada densa ------------

dense_size_list = [64, 128, 256]

dct_dense = {}
melhor_acc = 0
melhor_dense = 0
for dense_size in dense_size_list:
    fold_no = 1 #contador
    acc_per_fold = [] #acurácia de cada fold
    loss_per_fold = []
    for train, test in cv.split(train_images, train_labels): #pra cada fold
        print(f"Treinando fold {fold_no}, com tamanho da camada densa: {dense_size}" )
        train_X = train_images[train]
        test_X = train_images[test]
        model = criar_modelo(melhor_layer, melhor_filter, dense_size)
        model.fit(train_X, train_labels[train], epochs=epochs, batch_size=batch_size, verbose=2)
        test_loss, test_acc = model.evaluate(test_X, train_labels[test], verbose=2)
        acc_per_fold.append(test_acc * 100)
        loss_per_fold.append(test_loss * 100)
        fold_no = fold_no + 1

    media_acc_dense = sum(acc_per_fold)/len(acc_per_fold)
    media_loss_dense = sum(loss_per_fold)/len(loss_per_fold)

    dct_dense.update({dense_size: {"acuracia": media_acc_dense, "loss": media_loss_dense}})

    print(f"Média acurácia dos 5 folds pra tamanho da camada densa: {dense_size}:", media_acc_dense)
    print(f"Essa acurácia significa que o modelo usando {melhor_layer} camada(s) de convolução-pooling, filter: {melhor_filter} e tamanho da camada densa: {dense_size} é capaz de classificar corretamente em média {round(media_acc_dense, 1)}% das imagens")
    if media_acc_dense > melhor_acc:
        melhor_acc = media_acc_dense
        melhor_dense = dense_size
print(f"Portanto, o melhor tamanho da camada densa é {melhor_dense}, que possui {round(melhor_acc, 1)} de acurácia.")

# Resumo
print(f"Ao final, a melhor combinação foi: \n{melhor_layer} camada(s) de convolução-pooling, {melhor_filter} filter(s) e tamanho da camada densa: {melhor_dense}")

# ----- Dropout -----

# O Dropout é uma técnica de regularização utilizada para reduzir o overfitting em redes neurais.
# Durante o treinamento, uma proporção dos neurônios é aleatoriamente "desligada" (dropout) em cada atualização do gradiente,
# o que força a rede a aprender recursos mais robustos e evita a dependência excessiva de neurônios específicos.

# Vamos modificar a função create_cnn_model_with_dense_size para adicionar uma camada Dropout antes da camada densa:
def criar_modelo_com_dropout(conv_layers, filters, dense_size, dropout_rate):
    model = models.Sequential()

    for i in range(conv_layers):
        model.add(layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape = (28,28,1)))
        print("Conv2D")
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        print("MaxPooling2D")

    model.add(layers.Flatten())
    model.add(layers.Dense(dense_size, activation='relu'))
    model.add(layers.Dropout(dropout_rate)) #Dropout adicionado
    model.add(layers.Dense(10, activation='softmax'))
    
    model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

# --------- Testar diferentes porcentagens de dropout ------------

dropout_rates = [0.1, 0.3, 0.5, 0.7]

dct_dropout = {}
melhor_acc = 0
melhor_dropout = 0
for dropout_rate in dropout_rates:
    fold_no = 1 #contador
    acc_per_fold = [] #acurácia de cada fold
    loss_per_fold = []
    for train, test in cv.split(train_images, train_labels): #pra cada fold
        print(f"Treinando fold {fold_no}, com {dropout_rate} de dropout_rate" )
        train_X = train_images[train]
        test_X = train_images[test]
        model = criar_modelo_com_dropout(melhor_layer, melhor_filter, melhor_dense, dropout_rate)
        model.fit(train_X, train_labels[train], epochs=epochs, batch_size=batch_size, verbose=2)
        test_loss, test_acc = model.evaluate(test_X, train_labels[test], verbose=2, )
        acc_per_fold.append(test_acc * 100)
        loss_per_fold.append(test_loss * 100)
        fold_no = fold_no + 1

    media_acc_dropout = sum(acc_per_fold)/len(acc_per_fold)
    media_loss_dropout = sum(loss_per_fold)/len(loss_per_fold)

    dct_dropout.update({dropout_rate: {"acuracia": media_acc_dropout, "loss": media_loss_dropout}})

    print(f"Média acurácia dos 5 folds pra {dropout_rate} de dropout_rate:", media_acc_dropout)
    print(f"Essa acurácia significa que o modelo usando layer: {melhor_layer}, filter: {melhor_filter}, tamanho da camada densa: {melhor_dense} e dropout: {dropout_rate} é capaz de classificar corretamente em média {round(media_acc_dropout, 1)}% das imagens")
    if media_acc_dropout > melhor_acc:
        melhor_acc = media_acc_dropout
        melhor_dropout = dropout_rate
print(f"Portanto, o melhor dropout é {melhor_dropout}, que possui {round(melhor_acc, 1)} de acurácia.")

# ----- Batch Normalization -----

# Batch Normalization é uma técnica usada para acelerar o treinamento de redes neurais e estabilizar o processo de aprendizado.
# Ela normaliza a ativação de cada camada, aplicando uma transformação que mantém a média próxima de zero e o desvio padrão próximo de um.
# Isso ajuda a reduzir a covariância de ativação entre as camadas e torna o treinamento mais rápido e estável.

# Vamos modificar a função create_cnn_model_with_dense_dropout para adicionar uma camada Batch Normalization antes da camada densa:
def criar_modelo_com_batchnorm(conv_layers, filters, dense_size, dropout_rate):
    model = models.Sequential()

    for i in range(conv_layers):
        model.add(layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape = (28,28,1)))
        print("Conv2D")
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        print("MaxPooling2D")

    model.add(layers.Flatten())
    model.add(layers.Dense(dense_size, activation='relu'))
    model.add(layers.BatchNormalization()) # Batch Normalization adicionado
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(10, activation='softmax')) # 10 é o número de classes
    
    model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

dct_batchnorm = {}
fold_no = 1 #contador
acc_per_fold = [] #acurácia de cada fold
loss_per_fold = []
for train, test in cv.split(train_images, train_labels): #pra cada fold
    print(f"Treinando fold {fold_no}, com batch normalization" )
    train_X = train_images[train]
    test_X = train_images[test]
    model = criar_modelo_com_batchnorm(melhor_layer, melhor_filter, melhor_dense, melhor_dropout)
    model.fit(train_X, train_labels[train], epochs=epochs, batch_size=batch_size, verbose=2)
    test_loss, test_acc = model.evaluate(test_X, train_labels[test], verbose=2)
    acc_per_fold.append(test_acc * 100)
    loss_per_fold.append(test_loss * 100)
    fold_no = fold_no + 1

media_acc_batchnorm = sum(acc_per_fold)/len(acc_per_fold)
media_loss_batchnorm  = sum(loss_per_fold)/len(loss_per_fold)

dct_batchnorm.update({"acuracia": media_acc_batchnorm , "loss": media_loss_batchnorm })

print(f"Média acurácia dos 5 folds com batch normalization:", media_acc_batchnorm)
print(f"Essa acurácia significa que o modelo usando layer: {melhor_layer}, filter: {melhor_filter}, tamanho da camada densa: {melhor_dense}, dropout: {melhor_dropout} e com batch normalization é capaz de classificar corretamente em média {round(media_acc_batchnorm, 1)}% das imagens")

print(f"Portanto, possui em média {round(media_acc_batchnorm, 1)} de acurácia com o batch normalization.")

# ----- Data Augmentation -----

# Data Augmentation é uma técnica usada para expandir o conjunto de dados de treinamento, aplicando transformações aleatórias nos dados existentes,
# como rotação, zoom, espelhamento, deslocamento, entre outros. Essa técnica é útil quando o conjunto de dados de treinamento é limitado,
# pois permite aumentar a diversidade dos exemplos apresentados ao modelo.


# Criar um gerador de data augmentation
augmenter = ImageDataGenerator(
    rotation_range=20, # podem ser rotacionadas aleatoriamente em um ângulo de -20 a 20 graus
    width_shift_range=0.2, # as imagens podem ser deslocadas horizontalmente em até 20% da largura da imagem
    height_shift_range=0.2,  # as imagens podem ser deslocadas verticalmente em até 20% da largura da imagem
    shear_range=0.2, # as imagens podem ser distorcidas com um valor de cisalhamento aleatório entre -0.2 e 0.2
    zoom_range=0.2, # as imagens podem ser ampliadas ou reduzidas em até 20% aleatoriamente.
    horizontal_flip=True # imagens podem ser invertidas horizontalmente durante o data augmentation.
)

# Ajustar o gerador aos dados de treinamento
# datagen.fit(train_images.reshape((-1, 28, 28, 1)))

dct_dataaug = {}
fold_no = 1 #contador
acc_per_fold = [] #acurácia de cada fold
loss_per_fold = [] #loss de cada fold
for train, test in cv.split(train_images, train_labels): #pra cada fold
    print(f"Treinando fold {fold_no}, com data augmentation")
    
    # Obter os conjuntos de treinamento e teste para o fold atual
    train_X, test_X = train_images[train], train_images[test]
    train_y, test_y = train_labels[train], train_labels[test]
    
    # Aplicar o data augmentation aos dados de treinamento
    augmenter.fit(train_X)
    augmented_train_X = augmenter.flow(train_X, train_y, batch_size=32)

    model = criar_modelo_com_batchnorm(melhor_layer, melhor_filter, melhor_dense, melhor_dropout)

    model.fit(augmented_train_X, epochs=5, validation_data=(test_X, test_y), verbose=2)

    test_loss, test_acc = model.evaluate(test_X, train_labels[test], verbose=2)
    acc_per_fold.append(test_acc * 100)
    loss_per_fold.append(test_loss * 100)
    fold_no = fold_no + 1

media_acc_dataaug = sum(acc_per_fold)/len(acc_per_fold)
media_loss_dataaug  = sum(loss_per_fold)/len(loss_per_fold)

dct_dataaug.update({"acuracia": media_acc_dataaug , "loss": media_loss_dataaug })

print(f"Média acurácia dos 5 folds com data augmentation:", media_acc_dataaug)
print(f"Essa acurácia significa que o modelo usando layer: {melhor_layer}, filter: {melhor_filter}, tamanho da camada densa: {melhor_dense}, dropout: {melhor_dropout}, com batch normalization e com data augmentation é capaz de classificar corretamente em média {round(media_acc_dataaug, 1)}% das imagens")

print(f"Portanto, possui em média {round(media_acc_dataaug, 1)} de acurácia.")

sys.stdout = sys.__stdout__
file.close()