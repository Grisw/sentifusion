import argparse
import importlib
from utils import prepare_text_data, load_vocab, load_class
import os
import jieba
import numpy as np


def get_model(model, dataset, ckpt, max_seq_len):
    model_module = importlib.import_module('models.text.%s' % model)
    model_func = getattr(model_module, model)
    vocab = load_vocab(dataset)
    classes = load_class(dataset, 'text')
    model = model_func(max_seq_len, len(vocab), len(classes))
    model.load_weights(os.path.join('results', 'text', model, '%s.ckpt' % ckpt))
    return model, vocab, classes


def infer(model, vocab, classes, max_seq_len, text):
    encoded = [vocab.get(w, vocab['<UNK>']) for w in jieba.cut(text)]
    for _ in range(len(encoded), max_seq_len):
        encoded.append(vocab['<PAD>'])
    y = model.predict(np.array([encoded]))
    return zip(classes, y[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True, choices=['infer', 'train', 'test'])
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    model_module = importlib.import_module('models.text.%s' % args.model)
    model_func = getattr(model_module, args.model)

    if args.mode == 'infer':
        assert args.ckpt, 'Missing argument "--ckpt".'
        vocab = load_vocab(args.dataset)
        classes = load_class(args.dataset, 'text')
        model = model_func(args.max_seq_len, len(vocab), len(classes))
        model.load_weights(os.path.join('results', 'text', args.model, '%s.ckpt' % args.ckpt))

        while True:
            s = input('>>> ')
            encoded = [vocab.get(w, vocab['<UNK>']) for w in jieba.cut(s)]
            for _ in range(len(encoded), args.max_seq_len):
                encoded.append(vocab['<PAD>'])
            y = model.predict(np.array([encoded]))
            print(list(zip(classes, y[0])))
    elif args.mode == 'train':
        train_x, train_y, vocab, classes = prepare_text_data(args.dataset, args.max_seq_len, 'train')
        model = model_func(args.max_seq_len, len(vocab), len(classes))
        if args.ckpt:
            model.load_weights(os.path.join('results', 'text', args.model, '%s.ckpt' % args.ckpt))

        from keras.callbacks import EarlyStopping

        try:
            model.fit(train_x, train_y, batch_size=args.batch_size, epochs=10000, validation_split=0.1, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
        except KeyboardInterrupt:
            pass
        output_path = os.path.join('results', 'text', args.model)
        os.makedirs(output_path, exist_ok=True)
        num = 0
        for filename in os.listdir(output_path):
            name, suffix = filename.split('.')
            if suffix == 'ckpt' and name.isdigit():
                idx = int(name)
                if idx > num:
                    num = idx
        num += 1
        model.save_weights(os.path.join(output_path, '%d.ckpt' % num))
        print('Save to %d.ckpt' % num)
    elif args.mode == 'test':
        assert args.ckpt, 'Missing argument "--ckpt".'
        test_x, test_y, vocab, classes = prepare_text_data(args.dataset, args.max_seq_len, 'test')
        model = model_func(args.max_seq_len, len(vocab), len(classes))
        model.load_weights(os.path.join('results', 'text', args.model, '%s.ckpt' % args.ckpt))
        loss, acc = model.evaluate(test_x, test_y, batch_size=args.batch_size)
        print('loss: %f, acc: %f' % (loss, acc))


if __name__ == '__main__':
    main()
