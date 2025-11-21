import cv2
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 2025-11-21 07:53:50.287100: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# train data
after = cv2.imread('../images/Sample/after/face.gif')
after_rgb = cv2.cvtColor(after, cv2.COLOR_RGB2BGR)
# after_dt = after.dtype
# after_sh = after.shape  # (262, 350, 3)
after_rgb_sh = after_rgb.shape  # (262, 350, 3)
# after_rgb_np = np.array(after_rgb_sh).reshape(-1, 1)
# after_np = np.array(after_sh).reshape(-1, 1)
# py
# [[262][350][3]]
# [[262][350][3]]

before = cv2.imread('../images/Sample/before/face.gif')
before_rgb = cv2.cvtColor(before, cv2.COLOR_RGB2BGR)
# before_dt = before.dtype
# before_sh = before.shape  # (262, 350, 3)
before_rgb_sh = before_rgb.shape  # (262, 350, 3)

# https://www.tensorflow.org/guide/keras/sequential_model?hl=ja
# model = keras.Sequential()
# model.add(keras.Input(shape=after_rgb_sh))  # 250x250 RGB images
# model.add(layers.Conv2D(32, 5, strides=2, activation="relu"))
# model.add(layers.Conv2D(32, 3, activation="relu"))
# model.add(layers.MaxPooling2D(3))
#
# # Can you guess what the current output shape is at this point? Probably not.
# # Let's just print it:
# model.summary()
#
# model.add(layers.Conv2D(32, 3, activation="relu"))
# model.add(layers.Conv2D(32, 3, activation="relu"))
# model.add(layers.MaxPooling2D(3))
# model.add(layers.Conv2D(32, 3, activation="relu"))
# model.add(layers.Conv2D(32, 3, activation="relu"))
# model.add(layers.MaxPooling2D(2))
#
# # And now?
# model.summary()
#
# # Now that we have 4x4 feature maps, time to apply global max pooling.
# model.add(layers.GlobalMaxPooling2D())
#
# # Finally, we add a classification layer.
# model.add(layers.Dense(10))

# Model: "sequential"
# ┌─────────────────────────────────┬────────────────────────┬───────────────┐
# │ Layer (type)                    │ Output Shape           │       Param # │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ conv2d (Conv2D)                 │ (None, 129, 173, 32)   │         2,432 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ conv2d_1 (Conv2D)               │ (None, 127, 171, 32)   │         9,248 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ max_pooling2d (MaxPooling2D)    │ (None, 42, 57, 32)     │             0 │
# └─────────────────────────────────┴────────────────────────┴───────────────┘
#  Total params: 11,680 (45.62 KB)
#  Trainable params: 11,680 (45.62 KB)
#  Non-trainable params: 0 (0.00 B)
# Model: "sequential"
# ┌─────────────────────────────────┬────────────────────────┬───────────────┐
# │ Layer (type)                    │ Output Shape           │       Param # │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ conv2d (Conv2D)                 │ (None, 129, 173, 32)   │         2,432 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ conv2d_1 (Conv2D)               │ (None, 127, 171, 32)   │         9,248 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ max_pooling2d (MaxPooling2D)    │ (None, 42, 57, 32)     │             0 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ conv2d_2 (Conv2D)               │ (None, 40, 55, 32)     │         9,248 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ conv2d_3 (Conv2D)               │ (None, 38, 53, 32)     │         9,248 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ max_pooling2d_1 (MaxPooling2D)  │ (None, 12, 17, 32)     │             0 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ conv2d_4 (Conv2D)               │ (None, 10, 15, 32)     │         9,248 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ conv2d_5 (Conv2D)               │ (None, 8, 13, 32)      │         9,248 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ max_pooling2d_2 (MaxPooling2D)  │ (None, 4, 6, 32)       │             0 │
# └─────────────────────────────────┴────────────────────────┴───────────────┘
#  Total params: 48,672 (190.12 KB)
#  Trainable params: 48,672 (190.12 KB)
#  Non-trainable params: 0 (0.00 B)

initial_model = keras.Sequential(
    [
        keras.Input(shape=before_rgb.shape),
        layers.Conv2D(32, 5, strides=2, activation="relu"),
        layers.Conv2D(32, 3, activation="relu", name="my_intermediate_layer"),
        layers.Conv2D(32, 3, activation="relu"),
    ]
)
feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    outputs=initial_model.get_layer(name="my_intermediate_layer").output,
)
# Call feature extractor on test input.
x = tf.ones((1, 262, 350, 3))
features = feature_extractor(x)
# print('before_rgb :')
# print(features)

# before_rgb :
# tf.Tensor(
# [[[[0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]
#    [0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]
#    [0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]
#    ...
#    [0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]
#    [0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]
#    [0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]]
#
#   [[0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]
#    [0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]
#    [0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]
#    ...
#    [0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]
#    [0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]
#    [0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]]
#
#   [[0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]
#    [0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]
#    [0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]
#    ...
#    [0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]
#    [0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]
#    [0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]]
#
#   ...
#
#   [[0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]
#    [0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]
#    [0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]
#    ...
#    [0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]
#    [0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]
#    [0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]]
#
#   [[0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]
#    [0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]
#    [0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]
#    ...
#    [0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]
#    [0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]
#    [0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]]
#
#   [[0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]
#    [0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]
#    [0.03606447 0.         0.         ... 0.04603489 0.06376393
#     0.        ]
#    ...
#    [0.03606451 0.         0.         ... 0.04603494 0.06376392
#     0.        ]
#    [0.03606452 0.         0.         ... 0.04603488 0.06376393
#     0.        ]
#    [0.03606451 0.         0.         ... 0.04603484 0.06376395
#     0.        ]]]], shape=(1, 127, 171, 32), dtype=float32)

# initial_model = keras.Sequential(
#     [
#         keras.Input(shape=before_rgb_sh),
#         layers.Conv2D(32, 5, strides=2, activation="relu"),
#         layers.Conv2D(32, 3, activation="relu"),
#         layers.Conv2D(32, 3, activation="relu"),
#     ]
# )
# feature_extractor = keras.Model(
#     inputs=initial_model.inputs,
#     outputs=[layer.output for layer in initial_model.layers],
# )
#
# # Call feature extractor on test input.
# x = tf.ones((1, 262, 350, 3))
# features = feature_extractor(x)

initial_model = keras.Sequential(
    [
        keras.Input(shape=after_rgb_sh),
        layers.Conv2D(32, 5, strides=2, activation="relu"),
        layers.Conv2D(32, 3, activation="relu", name="my_intermediate_layer"),
        layers.Conv2D(32, 3, activation="relu"),
    ]
)
feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    outputs=initial_model.get_layer(name="my_intermediate_layer").output,
)
# Call feature extractor on test input.
x = tf.ones((1, 262, 350, 3))
features = feature_extractor(x)
# print('after_rgb :')
# print(features)

# after_rgb :
# tf.Tensor(
# [[[[0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]
#    [0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]
#    [0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]
#    ...
#    [0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]
#    [0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]
#    [0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]]
#
#   [[0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]
#    [0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]
#    [0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]
#    ...
#    [0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]
#    [0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]
#    [0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]]
#
#   [[0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]
#    [0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]
#    [0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]
#    ...
#    [0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]
#    [0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]
#    [0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]]
#
#   ...
#
#   [[0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]
#    [0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]
#    [0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]
#    ...
#    [0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]
#    [0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]
#    [0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]]
#
#   [[0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]
#    [0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]
#    [0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]
#    ...
#    [0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]
#    [0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]
#    [0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]]
#
#   [[0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]
#    [0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]
#    [0.6807088  0.         0.         ... 0.30171645 0.40353853
#     0.28995144]
#    ...
#    [0.68070894 0.         0.         ... 0.30171645 0.40353853
#     0.2899514 ]
#    [0.6807089  0.         0.         ... 0.30171645 0.40353855
#     0.28995138]
#    [0.68070894 0.         0.         ... 0.30171645 0.40353853
#     0.28995138]]]], shape=(1, 127, 171, 32), dtype=float32)

# initial_model = keras.Sequential(
#     [
#         keras.Input(shape=after_rgb_sh),
#         layers.Conv2D(32, 5, strides=2, activation="relu"),
#         layers.Conv2D(32, 3, activation="relu"),
#         layers.Conv2D(32, 3, activation="relu"),
#     ]
# )
# feature_extractor = keras.Model(
#     inputs=initial_model.inputs,
#     outputs=[layer.output for layer in initial_model.layers],
# )
#
# # Call feature extractor on test input.
# x = tf.ones((1, 262, 350, 3))
# features = feature_extractor(x)
