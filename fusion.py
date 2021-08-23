import text
import facial
import voice
import speech_recognition as sr
import argparse
import os

r = sr.Recognizer()


def speech_to_text(audio_path=None):
    if audio_path is None:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=1)
            audio = r.listen(source, timeout=3, phrase_time_limit=2)
    else:
        with sr.AudioFile(audio_path) as source:
            audio = r.record(source)

    try:
        return r.recognize_sphinx(audio, language='zh-CN')
    except sr.UnknownValueError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fm', '--facial_model', type=str, required=True)
    parser.add_argument('-fd', '--facial_dataset', type=str, required=True)
    parser.add_argument('-fc', '--facial_ckpt', type=str, required=True)
    parser.add_argument('--image_shape', type=int, nargs=3, default=[48, 48, 1])

    parser.add_argument('-vm', '--voice_model', type=str, required=True)
    parser.add_argument('-vd', '--voice_dataset', type=str, required=True)
    parser.add_argument('-vc', '--voice_ckpt', type=str, required=True)
    parser.add_argument('--signal_len', type=int, default=262143)
    parser.add_argument('--n_mfcc', type=int, default=64)

    parser.add_argument('-tm', '--text_model', type=str, required=True)
    parser.add_argument('-td', '--text_dataset', type=str, required=True)
    parser.add_argument('-tc', '--text_ckpt', type=str, required=True)
    parser.add_argument('--max_seq_len', type=int, default=128)
    args = parser.parse_args()

    text_model = text.get_model(args.text_model, args.text_dataset, args.text_ckpt, args.max_seq_len)
    facial_model = facial.get_model(args.facial_model, args.facial_dataset, args.facial_ckpt, args.image_shape)
    voice_model = voice.get_model(args.voice_model, args.voice_dataset, args.voice_ckpt, args.signal_len, args.n_mfcc)


if __name__ == '__main__':
    print(speech_to_text())
