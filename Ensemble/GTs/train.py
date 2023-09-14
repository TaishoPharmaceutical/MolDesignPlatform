from .Models.GatedGraphTransformer import create_masks, Encoder
from .RunModel.RunModel import PredictModel
from .Utils.DataUtils import FeatMol, encode_to_array, read_csv, MAX_LEN
from datetime import date
from rdkit import Chem
from glob import glob

import numpy as np
import pandas as pd
import tensorflow as tf
import time
import argparse
import os

tf.random.set_seed(123)

num_layers = 6
d_model = 320
num_heads = 20
dff = 2048
input_vocab_size=None
pe_input = MAX_LEN
rate = 0.1


def sep_train_and_test_data(file, save_file_name, test_rate=0.1):
    df = pd.read_csv(file)
    tasks = list(df.columns)[1:]

    file_path = os.path.dirname(save_file_name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    reg_tasks = []
    cls_tasks = []
    for task in tasks:
        p = set(df[task].dropna())
        if len(p)<=3:
            cls_tasks.append(task)
        else:
            reg_tasks.append(task)

    for t in reg_tasks:
        df[t + "z"] = (df[t] - df[t].mean())/df[t].std()

    #予測後にz_scoreから値に直すための平均値と標準偏差を記録する
    d={}
    d["task_name"] = reg_tasks
    d["mean"] = [df[x].mean() for x in reg_tasks]
    d["std"] = [df[x].std() for x in reg_tasks]

    dfx = pd.DataFrame(d)
    sv = save_file_name.split(".")
    dfx.to_csv(sv[0] + "mean_std.csv")

    df=df.sample(frac=1)
    num_train = int(len(df)*0.9)
    train_df=df[:num_train]
    test_df=df[num_train:]

    return train_df, test_df, reg_tasks, cls_tasks


def train_model(save_folder, reg_tasks, cls_tasks, encoder_type="normal", batch_size=200, epochs=100):
    #save_path = save_folder + "/%s/pred/ggt_model"%str(date.today())
    #folders = glob(save_folder+"/*")
    #folders = sorted(folders)
    #save_folder = folders[-1]

    if encoder_type!="normal":
        save_path = save_folder + "/result/ggt_model_pooling"
    else:
        save_path = save_folder + "/result/ggt_model"

    train_file = save_folder + "/data/train.csv"
    test_file = save_folder + "/data/test.csv"
    tasks = reg_tasks + cls_tasks

    train_data = read_csv(train_file, tasks, "SMILES")

    gen = tf.data.Dataset.from_tensor_slices(tuple(train_data))
    gen = gen.shuffle(buffer_size=train_data[0].shape[0])
    gen = gen.batch(batch_size)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    gen = gen.prefetch(buffer_size=AUTOTUNE)

    test_data = read_csv(test_file, tasks, "SMILES")

    test_gen = tf.data.Dataset.from_tensor_slices(tuple(test_data))
    test_gen = test_gen.batch(batch_size)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    test_gen = test_gen.prefetch(buffer_size=AUTOTUNE)

    transformer = PredictModel(num_layers, d_model, num_heads, dff,
                        input_vocab_size, pe_input, len(reg_tasks), len(cls_tasks), rate, encoder_type=encoder_type)
    learning_rate = 0.0001
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    test_loss = []
    min_loss = 100
    count = 0
    for epoch in range(epochs):
        start = time.time()

        train_loss = []
        for batch, data in enumerate(gen):
            loss=train_step(transformer, optimizer, data[:3], data[3:], reg_tasks, cls_tasks)
            train_loss.append(loss)
        train_loss = tf.stack(train_loss)
        train_loss = tf.reduce_mean(train_loss)

        losses = []
        for test_data in test_gen:
            inp = test_data[:3]
            labels = test_data[3:]
            dec_inp=tf.range(1,71)
            mask=tf.cast(tf.math.less(dec_inp,len(reg_tasks)+len(cls_tasks)+1), dtype=tf.int32)
            dec_inp *=mask
            dec_inp = tf.tile([dec_inp], [tf.shape(inp[2])[0],1])
            dec_mask, _, _ = create_masks(dec_inp, dec_inp)
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp[2], inp[2])
            vloss = transformer.training(inp[0], dec_inp, False, enc_padding_mask, dec_mask,
                                inp[1], labels[:len(reg_tasks)],
                                labels[len(reg_tasks):len(reg_tasks)+len(cls_tasks)])
            losses.append(vloss)
        losses = tf.stack(losses)
        losses = tf.reduce_mean(losses)
        print("Epoch: %d"%epoch, train_loss, losses)

        if epoch==0:
            min_loss =losses
        else:
            if min_loss > losses:
                min_loss = losses
                transformer.save_weights(save_path)
                count = 0
            else:
                count+=1

        if count>=10:
            return transformer

    return transformer

@tf.function
def train_step(transformer, optimizer, inp, labels, reg_tasks, cls_tasks):
    dec_inp=tf.range(1,71)
    mask=tf.cast(tf.math.less(dec_inp,len(reg_tasks)+len(cls_tasks)+1), dtype=tf.int32)
    dec_inp *=mask
    dec_inp = tf.tile([dec_inp], [tf.shape(inp[2])[0],1])
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp[2], inp[2])
    dec_mask, _, _ = create_masks(dec_inp, dec_inp)

    with tf.GradientTape() as tape:
        loss = transformer.training(inp[0], dec_inp, True, enc_padding_mask, dec_mask,
                                    inp[1], labels[:len(reg_tasks)],
                                    labels[len(reg_tasks):len(reg_tasks)+len(cls_tasks)])

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    return loss
