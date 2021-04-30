import os
import pickle
import numpy as np
from PIL import Image
import re
import gc
import urllib
from flask import Flask, render_template,  url_for, request
import tensorflow as tf
from model_ import CNN_Encoder, RNN_Decoder
tf.compat.v1.enable_eager_execution()

app = Flask(__name__)

embedding_dim = 256
units = 512
vocab_size = 6000 + 1
max_caption_length = 80

with open('weights/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predictCaption():
    print('here')
    count = 0
    url = request.form['imageSource']
    imageName = "./imageData/"+str(count)+".jpg"
    urllib.request.urlretrieve(url, imageName)

    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet', input_shape=(299,299,3))

    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)

    encoder.load_weights('weights/weights_encoder')
    decoder.load_weights('weights/weights_decoder')


    hidden = decoder.reset_state(batch_size=1)


    temp_input = tf.expand_dims(load_image(imageName)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                -1,
                                                img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_caption_length):
        predictions, hidden, attention_weights = decoder(dec_input,
                                                        features,
                                                        hidden)


        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            break

        dec_input = tf.expand_dims([predicted_id], 0)

    predict = ' '.join(result)

    gc.collect()
    
    return render_template('./result.html',prediction = predict, urlImg = url)

if __name__ == '__main__':
    app.run()
