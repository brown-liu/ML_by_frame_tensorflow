from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import shutil
import random
file=r'D:\PycharmProjects\tensorflow_ml\build_nn\1599295854'
model=keras.models.load_model(file)

#get some image

def prepare_image(path):
    img=image.load_img(path)
    img_array=image.img_to_array(img)
    img_array_expand_dims=np.expand_dims(img_array,axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expand_dims)


for filename in os.listdir(r"D:\PycharmProjects\tensorflow_ml\build_nn\test\mix"):

    predict=model.predict(prepare_image(f"D:\\PycharmProjects\\tensorflow_ml\\build_nn\\test\\mix\\{filename}" ))
    print(filename)
    if predict.argmax()==0:
        os.rename(f"D:\\PycharmProjects\\tensorflow_ml\\build_nn\\test\\mix\\{filename}",
                  f"D:\\PycharmProjects\\tensorflow_ml\\build_nn\\test\\mix\\cat_{filename}"
                  )
    else:
        os.rename(f"D:\\PycharmProjects\\tensorflow_ml\\build_nn\\test\\mix\\{filename}",
                  f"D:\\PycharmProjects\\tensorflow_ml\\build_nn\\test\\mix\\dog_{filename}"
                  )
