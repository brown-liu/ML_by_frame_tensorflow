import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras

class Load_data:
    def __init__(self):

        self.datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)

        self.test_path=r'D:\PycharmProjects\tensorflow_ml\build_nn\test'
        self.train_path=r'D:\PycharmProjects\tensorflow_ml\build_nn\train'
        self.valid_path=r'D:\PycharmProjects\tensorflow_ml\build_nn\valid'

        self.train_batches=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory\
            (self.train_path,target_size=(224,224),batch_size=10)

        self.valid_batches=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(
            self.valid_path,target_size=(224,224),batch_size=10
        )

        self.test_batches=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(
            self.test_path,target_size=(224,224),batch_size=10
        )

if __name__=="__main__":
    data=Load_data()
    print(data.train_batches.class_indices)