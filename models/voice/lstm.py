import keras
from keras.layers import Dense, LSTM, Dropout


def lstm(signal_len, n_mfcc, class_dim, optimizer='adam'):
    model = keras.Sequential([
        LSTM(128, input_shape=(signal_len // 512 + 1, n_mfcc)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(class_dim, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    return model
