# -*- coding: utf-8 -*-
"""
## Задание

Распознайте рукописную цифру, написанную на листе от руки.
"""

# Ваше решение
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import plot_model
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split

# Распаковываю свой датасет.
patoolib.extract_archive("start_model.rar", outdir="/content")

# Задаю размеры импортируемых фото по заданию
img_height = 28  # Высота изобажения
img_width = 28   # Ширина изображения

# Записываю датасет в переменные
base_dir = 'content/data_numbers'
X = []  # Датасет изображений
Y = []  # Целевые значения цифр
img_height = 28
img_weight = 28
for fold in os.listdir(base_dir):
    for img in os.listdir(base_dir + '/' + fold):
        X.append(image.img_to_array(image.load_img(base_dir+'/'+fold+'/'+img, 
                                                         target_size=(img_height,img_width),
                                                         color_mode='grayscale')))
        Y.append(int(fold[0]))
X = np.array(X)
Y = np.array(Y)
X = X.reshape(X.shape[0],28,28)
print(X.shape)

# Инвертирую цвет изображений и увеличиваю их яркость
for i in range(X.shape[0]):
  for j in range(X.shape[1]):
    for z in range(X.shape[2]):
      X[i][j][z] = 255 - X[i][j][z]        
      if (X[i][j][z] < 20):
        X[i][j][z] = 0
      X[i][j][z] = X[i][j][z]*1.25

# Вывожу пример каждой цифры
for i in range(10):
  for j in range(Y.shape[0]):
    if (Y[j] == i):
      plt.imshow(X[j].reshape(28,28),cmap='gray')
      plt.show()
      break

# Делаю разбивку на Тренеровочный сет и Тестовый
x_train,x_test, y_train, y_test = train_test_split(X,Y,
                                                   test_size=0.2,
                                                   random_state=42,
                                                   shuffle = True)
x_train = (x_train.reshape(x_train.shape[0],28,28,1) )/255
x_test = (x_test.reshape(x_test.shape[0],28,28,1) )/255
y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)

# Вывод размерностей БД
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# Создание нейросетисети
model= Sequential()
model.add (Conv2D(32,3,padding = 'same',input_shape = (x_train.shape[1:]),activation='relu'))
model.add (MaxPooling2D(2))
model.add (Conv2D(32,3,padding = 'same',input_shape = (x_train.shape[1:]),activation='relu'))
model.add (MaxPooling2D(2))
model.add (Flatten())
model.add (Dense(16,activation='relu'))
model.add (Dense(num_classes,activation='softmax'))

# Компиляция нейросети
model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.0001),metrics=['accuracy'])
model.summary()

# Обучение сети
history = model.fit(x_train,y_train,
                    validation_data=(x_test,y_test),
                    batch_size = 32,
                    epochs = 60,
                    verbose = 1)

# Вывод результатов обучения
acc  = history.history['accuracy']
val_acc  = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot (acc,label='acc')
plt.plot (val_acc,label='val_acc')
plt.legend()
plt.title("History accuracy")
plt.show()

plt.plot (loss,label='loss')
plt.plot (val_loss,label='val_loss')
plt.legend()
plt.title("History Loss")
plt.show()

