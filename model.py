# coding=utf-8
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,UpSampling2D
"""
定义alexnet网络模型
"""
class model_set():

    def alexnet(self):
        model = Sequential()
        model.add(UpSampling2D(input_shape=(32, 32, 3),size=(6,6)))
        model.add(Conv2D(96, (11, 11), strides=(4, 4), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))

        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))

        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
        print(model.summary())

        return model
    def Lenet(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same', activation=tf.nn.relu,
                                   input_shape=(32, 32, 3)),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
            tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation=tf.nn.relu),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=120, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=84, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=10, activation=tf.nn.softmax),
        ])
        return model
