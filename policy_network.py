import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, BatchNormalization, Dense, Activation
from keras.optimizers import sgd


class PolicyNetwork:
    def __init__(self, input_dim=0, output_dim=0, lr=0.01):
        self.input_dim = input_dim
        self.lr = lr

        # LSTM(Long Short Term Memory) networks
        self.model = Sequential()

        self.model.add(LSTM(256, input_shape=(1, input_dim),
                            return_sequences=True, stateful=False, dropout=0.5))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(256, return_sequences=True, stateful=False, dropout=0.5))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(256, return_sequences=True, stateful=False, dropout=0.5))
        self.model.add(BatchNormalization())
        self.model.add(Dense(output_dim))
        self.model.add(Activation('sigmoid'))

        self.model.compile(optimizer=sgd(lr=lr), loss='mse')
        self.prob = None

    def reset(self):  # Initialize prob variable
        self.prob = None

    def predict(self, sample):  # Compute probabilities of BUY or SELL by combining 17 dim inputs
        self.prob = self.model.predict(np.array(sample).reshape(1, -1, self.input_dim))[0]
        return self.prob

    def train_on_batch(self, x, y):  # x: learning data, y: label
        return self.model.train_on_batch(x, y)

    def save_model(self, model_path):  # Save as a file
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True)

    def load_model(self, model_path):  # Load model
        if model_path is not None:
            self.model.load_weights(model_path)
