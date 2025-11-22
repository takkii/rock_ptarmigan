import glob
import os

import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from tensorflow import keras
from tensorflow.keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

after = cv2.imread('../images/Sample/after/face.gif')
after_rgb = cv2.cvtColor(after, cv2.COLOR_RGB2BGR)
after_rgb_sh = after_rgb.shape  # (262, 350, 3)

train_dir = 'train'
file_type = 'gif'

train_list_path = glob.glob('../images/' + train_dir + '/*.' + file_type)

train_list = []
train_shape = after_rgb_sh

for img in train_list_path:
    temp_img = load_img(img, target_size=train_shape)
    temp_img_array = img_to_array(temp_img) / 255
    train_list.append(temp_img_array)

x_train = np.array(train_list)

model = keras.Sequential(
    [
        keras.Input(shape=after_rgb_sh),
        layers.Conv2D(32, 5, name='Layer_0', strides=2, activation="relu", trainable=False),
        layers.Conv2D(32, 3, name='Layer_1', activation="relu", trainable=False),
        layers.Conv2D(32, 3, name='Layer_2', activation="relu", trainable=False),
    ]
)

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0, name='accuracy')])

feature_extractor = keras.Model(
    inputs=model.inputs,
    outputs=model.get_layer(name="Layer_1").output,
)

x = tf.ones((1, 262, 350, 3))
features = feature_extractor(x)

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(features)

print(feature_batch_average.shape[1])
# 32
