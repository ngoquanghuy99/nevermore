from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import keras
import json
from keras.preprocessing.text import  tokenizer_from_json, Tokenizer
from keras.preprocessing.sequence import pad_sequences
import io
import nltk
nltk.download('punkt')
from nltk import word_tokenize
from config import *
from postag_model import create_model


app = Flask(__name__)


@app.before_first_request
def load_model_to_app():
    # app.predictor = load_model('./static/POS_BiLSTM_CRF_WSJ_new.h5')
    with open('static/tokenizer.json') as f1:
        data1 = json.load(f1)
        tokenizer = tokenizer_from_json(data1)
    app.tokenizer = tokenizer
    with open('static/tag_tokenizer.json') as f2:
        data2 = json.load(f2)
        tag_tokenizer = tokenizer_from_json(data2)
    app.tag_tokenizer = tag_tokenizer

    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    tag_index = tag_tokenizer.word_index
    app.index_tag = {i: t for t, i in tag_index.items()}
    tag_size = len(tag_index) + 1

    model = create_model(vocab_size, max_length, embedding_dim, word_index, tag_index)
    model.load_weights('static/POS_BiLSTM_CRF_WSJ_new.h5')
    app.predictor = model

@app.route("/")
def index():
    return render_template('index.html', pred=0)


@app.route('/predict', methods=['POST'])
def predict():

    ## my own code
    data = request.form['text1']
    tokens = word_tokenize(data)
    encoded_sent = app.tokenizer.texts_to_sequences([tokens])[0]
    encoded_sent = pad_sequences([encoded_sent], maxlen=max_length, padding='post')

    pred = app.predictor.predict(encoded_sent)
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
    return render_template('index.html', pred=class_)
    ##

def main():
    """Run the app."""
    app.run(host='0.0.0.0', port=8000, debug=False)  # nosec


if __name__ == '__main__':
    main()