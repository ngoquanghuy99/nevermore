from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import keras
import json
from keras.preprocessing.text import  tokenizer_from_json, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import io
import nltk
nltk.download('punkt')
from nltk import word_tokenize
from config import *
from postag_model import create_model

from utils import process

app = Flask(__name__)


@app.before_first_request
def load_model_to_app():
    # app.predictor = load_model('./static/POS_BiLSTM_CRF_WSJ_new.h5')
    with open('static/POS/tokenizer.json') as f1:
        data1 = json.load(f1)
        tokenizer = tokenizer_from_json(data1)
    app.tokenizer = tokenizer
    with open('static/POS/tag_tokenizer.json') as f2:
        data2 = json.load(f2)
        tag_tokenizer = tokenizer_from_json(data2)
    app.tag_tokenizer = tag_tokenizer

    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    tag_index = tag_tokenizer.word_index
    app.index_tag = {i: t for t, i in tag_index.items()}
    tag_size = len(tag_index) + 1

    model = create_model(vocab_size, max_length, embedding_dim, word_index, tag_index)
    model.load_weights('static/POS/POS_BiLSTM_CRF_WSJ_new.h5')
    app.pos_tagger = model

    # sentiment analysis model
    with open('static/SA/tokenizer.json') as f3:
        data3 = json.load(f3)
        tokenizer3 = tokenizer_from_json(data3)
    app.sa_tokenizer = tokenizer3
    model3 = load_model('static/SA/model.h5')
    app.sa_model = model3


@app.route("/")
def index():
    return render_template('index.html', pred=0)


@app.route('/predict', methods=['GET', 'POST'])
def predict():

    ## my own code
    data = request.form['text1']
    if request.form.get("POS", None) == 'POS Tagging':
        tokens = word_tokenize(data)
        encoded_sent = app.tokenizer.texts_to_sequences([tokens])[0]
        encoded_sent = pad_sequences([encoded_sent], maxlen=max_length, padding='post')

        pred = app.pos_tagger.predict(encoded_sent)
        sequence_tags = []
        for sequence in pred:
            sequence_tag = []
            for categorical in sequence:
                sequence_tag.append(app.index_tag.get(np.argmax(categorical)))
            sequence_tags.append(sequence_tag)
        res1 = sequence_tags[0][:len(tokens)]
        res2 = []
        for tok, tag in zip(tokens, res1):
            res2.append((tok, tag))
        class_ = res2
        # return render_template('index.html', pred=class_)

    elif request.form.get("NER", None) == 'Named Entity Recognition':
        class_ = None
    elif request.form.get("SENTIMENT", None) == "Sentiment Analysis":

        processed_review = process(data)
        encoded_review = app.sa_tokenizer.texts_to_sequences([processed_review])[0]
        encoded_review = pad_sequences([encoded_review], maxlen=150, padding='post', truncating='post')
        pre = app.sa_model.predict(encoded_review)

        if pre[0][0] > 0.6:
            # print('Positive with {}%'.format(pred[0][0] * 100))
            prcnt = str(pre[0][0] * 100)
            class_ = 'Positive ' + prcnt
        else:
            # print('Negative with {}%'.format(100 - pred[0][0] * 100))
            prcnt = str(100 - pre[0][0] * 100)
            class_ = 'Negative ' + prcnt
    elif request.form.get("CLASSIFICATION", None) == "Text Classification":
        class_ = None
    elif request.form.get("SUMMARIZATION", None) == "Text Summarization":
        class_ = None
    else:
        return None

    return render_template('index.html', pred=class_)
    ##

def main():
    """Run the app."""
    app.run(host='0.0.0.0', port=8000, debug=False)  # nosec


if __name__ == '__main__':
    main()