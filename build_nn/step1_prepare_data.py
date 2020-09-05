


#####################################################  INITIAL SEPERATION FILE  ##########################################################

import os
import random
import shutil

# main_path is the source file for images of dog and cat
class Prepare_data:
    def __init__(self):
        self.main_path=r'D:\DataSets\kagglecatsanddogs_3367a\PetImages'
        print(os.listdir(self.main_path))
        self.seperate_files(self.main_path)

    # check if GPU is available
    # print(tf.config.list_physical_devices('GPU'))
    # use makedirs
    def seperate_files(self,main_path):
        if not os.path.isdir(r'D:\PycharmProjects\tensorflow_ml\build_nn\train\dog'):
            os.makedirs(r'train\dog')
            os.makedirs(r'train\cat')
            os.makedirs(r'valid\dog')
            os.makedirs(r'valid\cat')
            os.makedirs(r'test\dog')
            os.makedirs(r'test\cat')
            print('Folders created!')

        ## sample : will not pick the same one twice. but choice will do.

            for dog in random.sample(os.listdir(main_path+r'\dog'),1000):
                shutil.move(main_path+'\\dog\\'+dog, r'D:\PycharmProjects\tensorflow_ml\build_nn\train\dog')

            for cat in random.sample(os.listdir(main_path + r'\cat'),1000):
                shutil.move(main_path+'\\cat\\'+cat,r'D:\PycharmProjects\tensorflow_ml\build_nn\train\cat')

            print('training set completed')

            for dog in random.sample(os.listdir(main_path+r'\dog'),200):
                shutil.move(main_path+'\\dog\\'+dog,r'D:\PycharmProjects\tensorflow_ml\build_nn\valid\dog')
            for cat in random.sample(os.listdir(main_path + r'\cat'),200):
                shutil.move(main_path+'\\cat\\'+cat,r'D:\PycharmProjects\tensorflow_ml\build_nn\valid\cat')
            print("validation set completed")

            for dog in random.sample(os.listdir(main_path + r'\dog'),100):
                shutil.move(main_path+'\\dog\\'+dog, r'D:\PycharmProjects\tensorflow_ml\build_nn\test\dog')
            for cat in random.sample(os.listdir(main_path + r'\cat'),100):
                shutil.move(main_path+'\\cat\\'+cat,r'D:\PycharmProjects\tensorflow_ml\build_nn\test\cat')
            print("test set completed")
        else:
            print("Raw dataset ready")


if __name__=="__main__":
    Prepare_data()


