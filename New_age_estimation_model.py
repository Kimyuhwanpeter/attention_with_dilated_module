# -*- coding:utf-8 -*-
import tensorflow as tf

l1_l2 = tf.keras.regularizers.L1L2(0.00001, 0.000001)
l1 = tf.keras.regularizers.l1(0.00001)

def attention_dilated_conv_block(input, filters, dilation_rate):

    s = tf.reduce_mean(input, axis=-1, keepdims=True)
    s = tf.nn.sigmoid(s)    # attenion map !!!!!

    h = tf.keras.layers.ReLU()(input)
    h = tf.pad(h, [[0,0],[dilation_rate,dilation_rate],[dilation_rate,dilation_rate],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=filters // 4, kernel_size=3, strides=1, padding="valid", use_bias=False,
                               kernel_regularizer=l1_l2, activity_regularizer=l1, dilation_rate=dilation_rate)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1, use_bias=False,
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.pad(h, [[0,0],[dilation_rate,dilation_rate],[dilation_rate,dilation_rate],[0,0]], "REFLECT")
    h = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="valid", use_bias=False,
                                        depthwise_regularizer=l1_l2, activity_regularizer=l1, dilation_rate=dilation_rate)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1, use_bias=False,
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.DepthwiseConv2D(kernel_size=1, strides=1, use_bias=False,
                                        depthwise_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = tf.keras.layers.BatchNormalization()(h)

    return (h * s) + input

def age_model(input_shape=(224, 224, 3), num_classes=54):

    h = inputs = tf.keras.Input(input_shape)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="valid", use_bias=False,
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding="valid")(h)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="valid", use_bias=False,
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding="valid")(h)

    for i in range(2):
        h = attention_dilated_conv_block(h, 128, (i+1)*2)   # 2, 4

    h = tf.keras.layers.ReLU()(h)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="valid", use_bias=False,
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding="valid")(h)

    for i in range(2):
        h = attention_dilated_conv_block(h, 256, (i+2)*2)   # 4, 6

    h = tf.keras.layers.ReLU()(h)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="valid", use_bias=False,
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding="valid")(h)

    for i in range(2):
        h = attention_dilated_conv_block(h, 512, (i+3)*2)   # 6, 8

    h = tf.keras.layers.ReLU()(h)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, strides=1, padding="valid", use_bias=False,
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding="valid")(h)

    for i in range(2):
        h = attention_dilated_conv_block(h, 1024, (i+4)*2)  # 8, 10

    h = tf.keras.layers.ReLU()(h)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, strides=1, padding="valid", use_bias=False,
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.GlobalAveragePooling2D()(h)

    h = tf.keras.layers.Dense(num_classes)(h)

    return tf.keras.Model(inputs=inputs, outputs=h)

model = age_model()
model.summary()