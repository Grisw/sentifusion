import os
import csv
from keras.utils import to_categorical
import jieba
import numpy as np
import librosa


def load_vocab(name):
    vocab_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'text', name, 'vocab.txt')
    vocab = {}
    with open(vocab_path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line not in vocab:
                vocab[line] = len(vocab)
    return vocab


def load_class(name, dataset):
    class_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', dataset, name, 'class.txt')
    assert os.path.exists(class_path), 'class.txt dose not exist in given dataset "%s".' % name
    classes = []
    with open(class_path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            classes.append(line)
    return classes


def prepare_text_data(name, seq_len, mode='train'):
    vocab_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'text', name, 'vocab.txt')
    building_vocab = not os.path.exists(vocab_path)

    if mode == 'train':
        data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'text', name, 'train.csv')
    elif mode == 'test':
        if building_vocab:
            raise Exception('vocab.txt must exists when argument "mode" is "test".')
        data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'text', name, 'test.csv')
    else:
        raise Exception('Argument "mode" should be "train" or "test".')
    assert os.path.exists(data_path), 'The given dataset "%s" dose not exist.' % name

    classes = load_class(name, 'text')

    if not building_vocab:
        vocab = load_vocab(name)
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

    x, y = load_dataset(data_path)

    if building_vocab:
        with open(vocab_path, 'w', encoding='utf8') as f:
            for w in vocab:
                f.write(w)
                f.write('\n')

    return np.array(x), np.array(y), vocab, classes


def prepare_facial_data(name, image_shape, mode='train'):
    if mode == 'train':
        data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'facial', name, 'train.csv')
    elif mode == 'test':
        data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'facial', name, 'test.csv')
    else:
        raise Exception('Argument "mode" should be "train" or "test".')

    classes = load_class(name, 'facial')

    faces = []
    y = []
    with open(data_path, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for item in reader:
            face = [int(pixel) / 255.0 for pixel in item['pixels'].split(' ')]
            face = np.reshape(face, image_shape)
            faces.append(face)
            y.append(to_categorical(int(item['emotion']), num_classes=len(classes)))

    return np.array(faces), np.array(y), classes


def prepare_voice_data(name, signal_len, n_mfcc, mode='train'):
    wavs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'voice', name, 'wav')
    if mode == 'train':
        csv_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'voice', name, 'train.csv')
    elif mode == 'test':
        csv_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'voice', name, 'test.csv')
    else:
        raise Exception('Argument "mode" should be "train" or "test".')

    classes = load_class(name, 'voice')

    mfccs = []
    y = []

    with open(csv_path, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for item in reader:
            wav_file = os.path.join(wavs_path, item['wav'])
            signal, sr = librosa.load(wav_file, sr=None)
            s_len = len(signal)
            if s_len < signal_len:
                pad_len = signal_len - s_len
                pad_rem = pad_len % 2
                pad_len //= 2
                signal = np.pad(signal, (pad_len, pad_len + pad_rem),
                                'constant', constant_values=0)
            else:
                pad_len = s_len - signal_len
                pad_len //= 2
                signal = signal[pad_len:pad_len + signal_len]
            mfcc = librosa.feature.mfcc(signal, sr, n_mfcc=n_mfcc).T
            mfccs.append(mfcc)
            y.append(to_categorical(int(item['label']), num_classes=len(classes)))

    return np.array(mfccs), np.array(y), classes
