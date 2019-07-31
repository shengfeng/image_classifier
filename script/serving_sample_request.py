import argparse
import json

import numpy as np 
import requests
from keras.applications import inception_v3
from keras.preprocessing import image

# Argument parser for giving input image_path from command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path of image")

args = vars(ap.parse_args())

image_path = args['image']
# Processing our input image
img = image.img_to_array(image.load_img(image_path, target_size=(224, 224))) / 255.

# this is added because of a bug in tf_serving
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
print(json.dumps(inception_v3.decode_predictions(np.array(pred['predictions']))[0]))