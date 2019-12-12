import keras
import numpy as np
from keras.layers import Embedding, Bidirectional, Dense, LSTM, Dropout
import os
import csv
from keras.utils import to_categorical
import argparse
import jieba


def load_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line not in vocab:
                vocab[line] = len(vocab)
    return vocab


def load_class(class_path):
    classes = []
    with open(class_path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            classes.append(line)
    return classes


def prepare_data(name, seq_len):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir, os.path.pardir, 'data', 'text', name)
    train_path = os.path.join(data_path, 'train.csv')
    test_path = os.path.join(data_path, 'test.csv')
    class_path = os.path.join(data_path, 'class.txt')
    vocab_path = os.path.join(data_path, 'vocab.txt')
    assert os.path.exists(data_path), 'The given dataset "%s" dose not exist.' % name
    assert os.path.exists(train_path), 'train.csv dose not exist in given dataset "%s".' % name
    assert os.path.exists(test_path), 'test.csv dose not exist in given dataset "%s".' % name
    assert os.path.exists(class_path), 'class.txt dose not exist in given dataset "%s".' % name

    classes = load_class(class_path)

    building_vocab = not os.path.exists(vocab_path)
    if not building_vocab:
        vocab = load_vocab(vocab_path)
    else:
        vocab = {'<UNK>': 0, '<PAD>': 1}

    def load_dataset(path):
        x = []
        y = []
        with open(path, 'r', encoding='utf8') as f:
            reader = csv.DictReader(f)
            for item in reader:
                label = int(item['label'])
                text = jieba.cut(item['text'])
                encoded = []
                for word in text:
                    word = word.strip()
                    if len(word) > 0 and len(encoded) < seq_len:
                        if building_vocab:
                            idx = vocab.setdefault(word, len(vocab))
                        else:
                            idx = vocab.get(word, vocab['<UNK>'])
                        encoded.append(idx)
                if len(encoded) > 0:
                    for _ in range(len(encoded), seq_len):
                        encoded.append(vocab['<PAD>'])
                    x.append(encoded)
                    y.append(to_categorical(label, num_classes=len(classes)))
        return x, y

    train_x, train_y = load_dataset(train_path)
    test_x, test_y = load_dataset(test_path)

    if building_vocab:
        with open(vocab_path, 'w', encoding='utf8') as f:
            for w in vocab:
                f.write(w)
                f.write('\n')

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y), vocab


def bilstm(seq_len, dict_dim, emb_dim=128, hid_dim=32, hid_dim2=128, class_dim=2, optimizer='adam'):
    model = keras.Sequential([
        Embedding(dict_dim, emb_dim, input_length=seq_len),
        Bidirectional(LSTM(hid_dim * 4)),
        Dropout(0.3),
        Dense(hid_dim2, activation='tanh'),
        Dropout(0.3),
        Dense(class_dim, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    return model


if __name__ == '__main__':
    train_x, train_y, test_x, test_y, vocab = prepare_data('senta', 128)
    model = bilstm(128, len(vocab), class_dim=3)
    model.fit(train_x, train_y, batch_size=128, epochs=3, validation_split=0.1)

    while True:
        s = input('>>> ')
        encoded = [vocab.get(w, vocab['<UNK>']) for w in jieba.cut(s)]
        for _ in range(len(encoded), 128):
            encoded.append(vocab['<PAD>'])
        y = model.predict(np.array([encoded]))
        print(y[0])
