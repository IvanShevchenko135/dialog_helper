# Для примера работы алгоритма в презентации по диплому
import argparse
import time

import streamlit as st
from next_word_prediction import GPT2

from translator import Translator


@st.cache(hash_funcs={GPT2: lambda _: None})
def load_prediction_model():
    return GPT2()


def load_translator():
    return Translator('English', 'Russian')


from nltk.corpus import stopwords

gpt2 = load_prediction_model()
translator = load_translator()

stop_words = set(stopwords.words('english'))


def get_predictions(text, word_count=3):
    if not text:
        return ['-' for x in range(word_count)]

    predictions = gpt2.predict_next(text, 100)
    predictions = [word for word in predictions if word not in stop_words and word.isalpha()]

    return list(map(
        lambda word: {'word': word, 'translation': translator(word).lower()},
        predictions[:word_count]),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Next word prediction')
    parser.add_argument('--text', type=str, help='Text for prediction', default='this is my favourite')
    parser.add_argument('--n', type=int, help='Number to predict', default=20)
    args = parser.parse_args()
    start_time = time.time()
    predictions = get_predictions(args.text, args.n)
    print('Text: ', args.text, '...', sep='')
    for index, word in enumerate(predictions):
        print(index, ') ', word, sep='')

    print("--- %s seconds ---" % (time.time() - start_time))
