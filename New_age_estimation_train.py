# -*- coding:utf-8 -*-
from New_age_estimation_model import *
from random import shuffle, random

import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"tr_txt_path": "D:/[1]DB/[1]second_paper_DB/original_MORPH/train80_test20/train_1.txt",
                           
                           "tr_img_path": "D:/[1]DB/[1]second_paper_DB/original_MORPH/Crop_dlib/",
                           
                           "te_txt_path": "D:/[1]DB/[1]second_paper_DB/original_MORPH/train80_test20/test_1.txt",
                           
                           "te_img_path": "D:/[1]DB/[1]second_paper_DB/original_MORPH/Crop_dlib/",
                           
                           "img_size": 224,
                           
                           "n_classes": 60,
                           
                           "batch_size": 16,

                           "epochs": 200,

                           "lr": 0.0001,

                           "train": True,
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "",
                           
                           "save_checkpoint": ""})

optim = tf.keras.optimizers.Adam(FLAGS.lr)

def input_func(img_list, lab_list):

    img = tf.io.read_file(img_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])

    if random() > 0.5:
        img = tf.image.flip_left_right(img)

    img = tf.image.per_image_standardization(img)

    if lab_list == 74:
        lab_list = 72
        lab_list = lab_list - 16
        lab = tf.one_hot(lab_list, FLAGS.n_classes)
    elif lab_list == 75:
        lab_list = 73
        lab_list = lab_list - 16
        lab = tf.one_hot(lab_list, FLAGS.n_classes)
    elif lab_list == 76:
        lab_list = 74
        lab_list = lab_list - 16
        lab = tf.one_hot(lab_list, FLAGS.n_classes)
    elif lab_list == 77:
        lab_list = 75
        lab_list = lab_list - 16
        lab = tf.one_hot(lab_list, FLAGS.n_classes)
    else:
        lab_list = lab_list - 16
        lab = tf.one_hot(lab_list, FLAGS.n_classes)

    return img, lab

def test_func(img_list, lab_list):

    img = tf.io.read_file(img_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.image.per_image_standardization(img)

    if lab_list == 74:
        lab_list = 72
        lab = lab_list - 16
    elif lab_list == 75:
        lab_list = 73
        lab = lab_list - 16
    elif lab_list == 76:
        lab_list = 74
        lab = lab_list - 16
    elif lab_list == 77:
        lab_list = 75
        lab = lab_list - 16
    else:
        lab = lab_list - 16

    return img, lab

@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def cal_loss(model, images, labels):

    with tf.GradientTape() as tape: # 이거하고 style transfer 다시 연구 지금 돌리고있는것으로 만족하지 말자!

        logits = run_model(model, images, True)

        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(labels, logits)

    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    return loss

def measure_AE(model, images, labels):

    logits = run_model(model, images, False)
    logits = tf.nn.softmax(logits, -1)
    logits = tf.squeeze(logits, 0)
    predict = tf.argmax(logits, -1, output_type=tf.int32)

    AE = tf.cast(tf.abs(predict - labels[0]), tf.float32)

    return AE

def main():
    
    model = age_model(input_shape=(FLAGS.img_size, FLAGS.img_size, 3), num_classes=FLAGS.n_classes)
    model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)

    if FLAGS.train:
        count = 0
        train_data = np.loadtxt(FLAGS.tr_txt_path, dtype="<U200", skiprows=0, usecols=0) 
        train_data = [FLAGS.tr_img_path + data for data in train_data]
        train_label = np.loadtxt(FLAGS.tr_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        test_data = np.loadtxt(FLAGS.te_txt_path, dtype="<U200", skiprows=0, usecols=0)
        test_data = [FLAGS.te_img_path + data for data in test_data]
        test_label = np.loadtxt(FLAGS.te_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        te_gener = tf.data.Dataset.from_tensor_slices((test_data, test_label))
        te_gener = te_gener.map(test_func)
        te_gener = te_gener.batch(1)
        te_gener = te_gener.prefetch(tf.data.experimental.AUTOTUNE)

        for epoch in range(FLAGS.epochs):

            A = list(zip(train_data, train_label))
            shuffle(A)
            train_data, train_label = zip(*A)
            train_data, train_label = np.array(train_data), np.array(train_label)

            tr_gener = tf.data.Dataset.from_tensor_slices((train_data, train_label))
            tr_gener = tr_gener.shuffle(len(train_data))
            tr_gener = tr_gener.map(input_func)
            tr_gener = tr_gener.batch(FLAGS.batch_size)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)
        
            tr_iter = iter(tr_gener)
            tr_idx = len(train_data) // FLAGS.batch_size
            for step in range(tr_idx):

                batch_images, batch_labels = next(tr_iter)

                total_loss = cal_loss(model, batch_images, batch_labels)

                if count % 10 == 0:
                    print("Epoch: {} [{}/{}] Total loss = {}".format(epoch, step + 1, tr_idx, total_loss))

                if count % 100 == 0 and count != 0:
                    te_iter = iter(te_gener)
                    te_idx = len(test_data) // 1
                    AE = 0.
                    for i in range(te_idx):
                        images, labels = next(te_iter)

                        AE += measure_AE(model, images, labels)

                    MAE = AE / len(test_data)
                    print("MAE = {}".format(MAE))

                count += 1




if __name__ == "__main__":
    main()