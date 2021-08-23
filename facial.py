import argparse
import importlib
from utils import prepare_facial_data, load_class
import os
import numpy as np


def get_model(model, dataset, ckpt, image_shape):
    model_module = importlib.import_module('models.facial.%s' % model)
    model_func = getattr(model_module, model)
    classes = load_class(dataset, 'facial')
    model = model_func(image_shape, len(classes))
    model.load_weights(os.path.join('results', 'facial', model, '%s.ckpt' % ckpt))
    return model, classes


def infer(model, classes, image_shape, img):
    face = img / 255.0
    face = np.reshape(face, image_shape)
    y = model.predict(np.array([face]))
    return zip(classes, y[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True, choices=['infer', 'train', 'test'])
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--image_shape', type=int, nargs=3, default=[48, 48, 1])
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    model_module = importlib.import_module('models.facial.%s' % args.model)
    model_func = getattr(model_module, args.model)

    if args.mode == 'infer':
        assert args.ckpt, 'Missing argument "--ckpt".'
        classes = load_class(args.dataset, 'facial')
        model = model_func(args.image_shape, len(classes))
        model.load_weights(os.path.join('results', 'facial', args.model, '%s.ckpt' % args.ckpt))

        # import numpy as np
        # while True:
        #     s = input('>>> ')
        #     encoded = [vocab.get(w, vocab['<UNK>']) for w in jieba.cut(s)]
        #     for _ in range(len(encoded), args.max_seq_len):
        #         encoded.append(vocab['<PAD>'])
        #     y = model.predict(np.array([encoded]))
        #     print(list(zip(classes, y[0])))
    elif args.mode == 'train':
        train_x, train_y, classes = prepare_facial_data(args.dataset, args.image_shape, 'train')
        model = model_func(args.image_shape, len(classes))
        if args.ckpt:
            model.load_weights(os.path.join('results', 'facial', args.model, '%s.ckpt' % args.ckpt))

        from keras.callbacks import EarlyStopping

        try:
            model.fit(train_x, train_y, batch_size=args.batch_size, epochs=10000, validation_split=0.1, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
        except KeyboardInterrupt:
            pass
        output_path = os.path.join('results', 'facial', args.model)
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
        test_x, test_y, classes = prepare_facial_data(args.dataset, args.image_shape, 'test')
        model = model_func(args.image_shape, len(classes))
        model.load_weights(os.path.join('results', 'facial', args.model, '%s.ckpt' % args.ckpt))
        loss, acc = model.evaluate(test_x, test_y, batch_size=args.batch_size)
        print('loss: %f, acc: %f' % (loss, acc))


if __name__ == '__main__':
    main()
