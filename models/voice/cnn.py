import keras
from keras.layers import Dense, Conv1D, BatchNormalization, Activation, Dropout, MaxPooling1D, Flatten


def cnn(signal_len, n_mfcc, class_dim, optimizer='adam'):
    model = keras.Sequential([
        Conv1D(32, 5, input_shape=(signal_len // 512 + 1, n_mfcc)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(),

        Conv1D(16, 5),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(),

        Conv1D(8, 3),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(),
        Dropout(0.3),

        Conv1D(8, 3),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(),
        Dropout(0.3),

        Flatten(),
        Dense(128),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        Dense(class_dim, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    return model
