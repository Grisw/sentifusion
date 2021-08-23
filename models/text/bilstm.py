import keras
from keras.layers import Embedding, Bidirectional, Dense, LSTM, Dropout


def bilstm(seq_len, dict_dim, class_dim, emb_dim=128, hid_dim=32, hid_dim2=128, optimizer='adam'):
    model = keras.Sequential([
        Embedding(dict_dim, emb_dim, input_length=seq_len),
        Bidirectional(LSTM(hid_dim * 4)),
        Dense(hid_dim2, activation='tanh'),
        Dropout(0.3),
        Dense(class_dim, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    return model
