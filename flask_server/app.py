import base64
import json
from io import BytesIO

import numpy as np 
import requests
from flask import Flask, request, jsonify
from keras.applications import inception_v3
from keras.preprocessing import image


app = Flask(__name__)


@app.route('/hello/', methods=['GET', 'POST'])
def hello_world():
    return 'Hello, world!'



@app.route('/imageclassifier/predict/', methods=['POST'])
def image_classifier():
    # Decoding and pre-processing base64 image
    img = image.img_to_array(image.load_img(
        BytesIO(base64.b64decode(request.form['b64'])), 
        target_size= (244, 244))) / 255. 

    img = img.astype('float16')

    payload = {
    'instances' : [{'input_image': img.tolist()}]
    }


    # sending post request to TensorFlow Serving server
    r = requests.post('http://192.168.115.26:1001/v1/models/image-serving:predict', json=payload)
    pred = json.loads(r.content.decode('utf-8'))


    # Decoding the response
    # decode_predictions(preds, top=5) by default gives top 5 results
    # You can pass "top=10" to get top 10 predicitons
    return jsonify(inception_v3.decode_predictions(np.array(pred['predictions']))[0])