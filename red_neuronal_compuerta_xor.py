#Importacion de modulos nesesarios
import numpy as np 
from keras.models import Sequential
from keras.layers.core import Dense
import matplotlib.pyplot as plt

#Definicion de las entradas y salidas de la red
entradas = np.array([[0,0],[0,1],[1,0],[1,1]])
salida = np.array([[0],[1],[1],[0]])

#Creacion de la arquitectura de la red
modelo = Sequential()

#agregar las capas de la red --> 22 capas ocultas, 2 neuronas de entrada
modelo.add(Dense(14,input_dim=2,activation='sigmoid'))
modelo.add(Dense(8,activation='tanh'))
#capa de salida
modelo.add(Dense(1,activation='relu'))

#ajustes del modelo
#definir el tipo de perdida(error), el optimizador,tipo de metrica --> %
modelo.compile(loss='mse',optimizer='adam',metrics=['binary_accuracy'])

#entremaniento
modelo.fit(entradas,salida,epochs=1200)
print(entradas)
print (modelo.predict(entradas).round(),end=' ')












