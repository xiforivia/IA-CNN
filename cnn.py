import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from keras.datasets import fashion_mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

def pre_processamento():
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images.reshape((60000,28,28,1))
    train_images = train_images.astype('float32')/255

    test_images = test_images.reshape((10000,28,28,1))
    test_images = test_images.astype('float32')/255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return train_images, train_labels, test_images, test_labels

# Definir a arquitetura da CNN
def criar_modelo(conv_layers, filters, dense_size):
    model = models.Sequential()

    # Adicionar camadas de convolução-pooling
    for i in range(conv_layers):
        model.add(layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape = (28,28,1)))
        print("Conv2D")
        if i != conv_layers - 1 or conv_layers == 1: #se não for a última camada de conv, faz MaxPooling2D, com exceção se tiver somente 1 camada de conv, nesse caso faz MaxPooling2D (mesmo ela sendo a última também)
            print("MaxPooling2D")
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))


    model.add(layers.Flatten())
    model.add(layers.Dense(dense_size, activation='relu'))
    model.add(layers.Dense(10, activation='softmax')) # 10 é o número de classes
    
    model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

train_images, train_labels, test_images, test_labels = pre_processamento()
# Definir os valores fixos dos parâmetros
filters = 32
dense_size = 64

# Testar diferentes quantidades de camadas de convolução-pooling
conv_layers_list = [1, 2, 3]

melhor_acc = 0
melhor_layer = 0
for layer in conv_layers_list:
    model = criar_modelo(layer, filters, dense_size)
    model.fit(train_images, train_labels, epochs=5, batch_size = 64)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("test_loss:", test_loss, "\ntest_acc:", test_acc)
    print(f"Essa acurácia significa que o modelo usando layer: {layer}, filter: {filters} e tamanho da camada densa: {dense_size} é  capaz de classificar corretamente {round(test_acc*100, 1)}%  das imagens")
    if test_acc > melhor_acc:
        melhor_acc = test_acc
        melhor_layer = layer
print(f"Portanto, a melhor layer é a {melhor_layer}, que possui {round(melhor_acc*100, 1)} de acurácia.")


# Testar diferentes quantidades filters
filters_list = [16, 32, 64]

melhor_acc = 0
melhor_filter = 0
for filters in filters_list:
    model = criar_modelo(melhor_layer, filters, dense_size)
    model.fit(train_images, train_labels, epochs=5, batch_size = 64)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("test_loss:", test_loss, "\ntest_acc:", test_acc)
    print(f"Essa acurácia significa que o modelo usando layer: {melhor_layer}, filter: {filters} e tamanho da camada densa: {dense_size} é capaz de classificar corretamente {round(test_acc*100, 1)}%  das imagens")
    if test_acc > melhor_acc:
        melhor_acc = test_acc
        melhor_filter = filters
print(f"Portanto, o melhor filter é o {melhor_filter}, que possui {round(melhor_acc*100, 1)} de acurácia.")


# Testar diferentes tamanhos da camada densa
dense_size_list = [64, 128, 256]

melhor_acc = 0
melhor_dense = 0
for dense_size in dense_size_list:
    model = criar_modelo(melhor_layer, melhor_filter, dense_size)
    model.fit(train_images, train_labels, epochs=5, batch_size = 64)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("test_loss:", test_loss, "\ntest_acc:", test_acc)
    print(f"Essa acurácia significa que o modelo usando layer: {melhor_layer}, filter: {melhor_filter} e tamanho da camada densa: {dense_size} é capaz de classificar corretamente {round(test_acc*100, 1)}%  das imagens")
    if test_acc > melhor_acc:
        melhor_acc = test_acc
        melhor_dense = dense_size
print(f"Portanto, o melhor tamanho da camada densa é {melhor_dense}, que possui {round(melhor_acc*100, 1)} de acurácia.")

print(f"Ao final, a melhor combinação foi: \nlayer: {melhor_layer}, filter: {melhor_filter} e tamanho da camada densa: {melhor_dense}")


# Definir a arquitetura da CNN
def criar_modelo_com_dropout(conv_layers, filters, dense_size, dropout_rate):
    model = models.Sequential()

    # Adicionar camadas de convolução-pooling
    for i in range(conv_layers):
        model.add(layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape = (28,28,1)))
        print("Conv2D")
        if i != conv_layers - 1 or conv_layers == 1: #se não for a última camada de conv, faz MaxPooling2D, com exceção se tiver somente 1 camada de conv, nesse caso faz MaxPooling2D (mesmo ela sendo a última também)
            print("MaxPooling2D")
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))


    model.add(layers.Flatten())
    model.add(layers.Dense(dense_size, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(10, activation='softmax')) # 10 é o número de classes
    
    model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

# Testar diferentes dropout
dropout_rates = [0.1, 0.3, 0.5, 0.7]

melhor_acc = 0
melhor_dropout = 0
for dropout_rate in dropout_rates:
    model = criar_modelo_com_dropout(melhor_layer, melhor_filter, melhor_dense, dropout_rate)
    model.fit(train_images, train_labels, epochs=5, batch_size = 64)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("test_loss:", test_loss, "\ntest_acc:", test_acc)
    print(f"Essa acurácia significa que o modelo usando layer: {melhor_layer}, filter: {melhor_filter}, tamanho da camada densa: {melhor_dense} e dropout: {dropout_rate} é capaz de classificar corretamente {round(test_acc*100, 1)}%  das imagens")
    if test_acc > melhor_acc:
        melhor_acc = test_acc
        melhor_dropout = dropout_rate
print(f"Portanto, o melhor dropout é o {melhor_dropout}, que possui {round(melhor_acc*100, 1)} de acurácia.")


def criar_modelo_com_batchnorm(conv_layers, filters, dense_size, dropout_rate):
    model = models.Sequential()

    # Adicionar camadas de convolução-pooling
    for i in range(conv_layers):
        model.add(layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape = (28,28,1)))
        print("Conv2D")
        if i != conv_layers - 1 or conv_layers == 1: #se não for a última camada de conv, faz MaxPooling2D, com exceção se tiver somente 1 camada de conv, nesse caso faz MaxPooling2D (mesmo ela sendo a última também)
            print("MaxPooling2D")
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))


    model.add(layers.Flatten())
    model.add(layers.Dense(dense_size, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(10, activation='softmax')) # 10 é o número de classes
    
    model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

#Testar com batchnorm
model = criar_modelo_com_batchnorm(melhor_layer, melhor_filter, melhor_dense, melhor_dropout)
model.fit(train_images, train_labels, epochs=5, batch_size = 64)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("test_loss:", test_loss, "\ntest_acc:", test_acc)
print(f"Essa acurácia significa que o modelo usando layer: {melhor_layer}, filter: {melhor_filter}, tamanho da camada densa: {melhor_dense}, dropout: {melhor_dropout} e com batch normalization é capaz de classificar corretamente {round(test_acc*100, 1)}%  das imagens")

print(f"Portanto, possui {round(test_acc*100, 1)} de acurácia.")

#Testar com data augmentation