import cv2
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

after = cv2.imread('../images/Sample/after/face.gif')
after_rgb = cv2.cvtColor(after, cv2.COLOR_RGB2BGR)
after_rgb_sh = after_rgb.shape  # (262, 350, 3)

model = keras.Sequential(
    [
        keras.Input(shape=after_rgb_sh),
        layers.Conv2D(32, 5, name='Layer_0', strides=2, activation="relu", trainable=False),
        layers.Conv2D(32, 3, name='Layer_1', activation="relu", trainable=False),
        layers.Conv2D(32, 3, name='Layer_2', activation="relu", trainable=False),
    ]
)

feature_extractor = keras.Model(
    inputs=model.inputs,
    outputs=model.get_layer(name="Layer_1").output,
)
# Call feature extractor on test input.
x = tf.ones((1, 262, 350, 3))
features = feature_extractor(x)

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(features)

print(len(feature_batch_average.shape))
# 2
