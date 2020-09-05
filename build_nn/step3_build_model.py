
import tensorflow
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from build_nn.step2_load_data import Load_data
from build_nn.step1_prepare_data import Prepare_data
import time
Prepare_data()
step2_data=Load_data()
# download mobilenet

mobileNet=keras.applications.mobilenet.MobileNet()

# modify layers

reduced_mobile_net= mobileNet.layers[-6].output
final_layer_added=Dense(2,activation='softmax')(reduced_mobile_net)
new_model=Model(inputs=mobileNet.input,outputs=final_layer_added)

for layer in new_model.layers[:-5]:
    layer.trainable=False


new_model.compile(Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

new_model.fit(step2_data.train_batches,steps_per_epoch=100,validation_data=step2_data.valid_batches,validation_steps=20,epochs=10,verbose=2)

new_model.save(f'D:\\PycharmProjects\\tensorflow_ml\\build_nn\\model_{round(time.time())}')
