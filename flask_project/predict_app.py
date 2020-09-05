import numpy as np
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator,img_to_array
from flask import Flask
from flask import jsonify
from flask import request
import io
import base64

app=Flask(__name__)

def get_model():
    global model
    model=load_model("/Users/liubo/PycharmProjects/tensorflow_ml/model1.h5")
    print(' * Model loaded')

def preprocess_image(image,target_size):
    if image.mode != "RGB":
        image=image.convert('RGB')
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image,axis=0)
    return image

print(' * Loading Keras Model')
get_model()

@app.route('/predict',methods=['POST'])
def predict():
    msg=request.get_json(force=True)
    encoded = msg['image']
    decoded=base64.b64decode(encoded)
    image=Image.open(io.BytesIO(decoded))
    preproced_image = preprocess_image(image,target_size=(224,224))
    predict = model.predict(preproced_image).tolist()
    response={
            'prediction':{
                'dog':predict[0][0],
                'cat':predict[0][1]
            }
        }
    return jsonify(response)