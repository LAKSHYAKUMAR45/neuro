from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense


class ConvolutionalNeuralNetwork:

    def __init__(self, input_shape, action_space):
        self.model = Sequential()
        self.model.add(Conv2D(32,
                              4,
                              strides=(4, 4),
                              padding="same",
                              activation="relu",
                              input_shape=input_shape,
                              data_format="channels_last"))
        self.model.add(Conv2D(64,
                              2,
                              strides=(2, 2),
                              padding="same",
                              activation="relu",
                              input_shape=input_shape,
                              data_format="channels_last"))
        self.model.add(Conv2D(64,
                              1,
                              strides=(1, 1),
                              padding="same",
                              activation="relu",
                              input_shape=input_shape,
                              data_format="channels_last"))
        self.model.add(Flatten())
        self.model.add(Dense(98, activation="relu"))
        self.model.add(Dense(action_space))
        self.model.compile(loss="mean_squared_error",
                           optimizer=RMSprop(lr=0.005,
                                             rho=0.95,
                                             epsilon=0.01),
                           metrics=["accuracy"])
        self.model.summary()
