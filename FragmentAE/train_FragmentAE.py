#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import pandas as pd
import random
import pickle
import time
import argparse

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger

import os

from ChemUtils.PreProcessing import read_file, smiles_to_fp_array, CHAR_LEN, array_to_smiles

from GraphTransformers.Models.GatedGraphTransformer import GatedGraphTransformerS, CustomSchedule, loss_function
from GraphTransformers.Utils.DataUtils import FeatMol, encode_to_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--df", help="path: data file", type=str)
    parser.add_argument("--save", help="folder path: save train model", type=str)
    parser.add_argument("--gpu", help="using gpu number", type=str)
    parser.add_argument("--batch", help="batch size of a training", type=int, default=200)
    parser.add_argument("--epoch", help="epoch", type=int, default=100)

    args = parser.parse_args()

    fp_radius = 2
    fp_bits = 2048
    MAX_SMILES_LEN=90
    BATCH_SIZE = args.batch
    EPOCHS=args.epoch
    checkpoint_path = args.save+"/"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    save_weights = os.path.join(args.save, "FragmentAE")


    # preprocessing
    smiles_array, fp_array, fps = read_file(args.df)
    true_smiles = array_to_smiles(smiles_array, CHAR_LEN+2)

    mols = [Chem.MolFromSmiles(x) for x in true_smiles]
    cmols = [FeatMol(x) for x in mols]

    inputs = encode_to_array(cmols)

    gen = tf.data.Dataset.from_tensor_slices((inputs[0], inputs[1], inputs[2], smiles_array))
    gen = gen.shuffle(buffer_size=smiles_array.shape[0])
    gen = gen.batch(BATCH_SIZE)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    gen = gen.prefetch(buffer_size=AUTOTUNE)


    # define transformer
    num_layers = 4
    d_model = 64
    num_heads = 8
    dff = 512
    input_vocab_size=fp_bits+3
    target_vocab_size = CHAR_LEN+3
    pe_input = inputs[2].shape[1]
    pe_target = MAX_SMILES_LEN
    rate=0.1
    transformer = GatedGraphTransformerS(num_layers,d_model,num_heads,
                                         dff, input_vocab_size, target_vocab_size,
                                         pe_input, pe_target, rate)


    # training
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')

    learning_rate = 0.0001
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')


    loss_object = tf.keras.losses.MeanSquaredError(reduction='none')

    def loss_function2(real, pred, array):
        mask = tf.math.logical_not(tf.math.equal(array, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)


    @tf.function
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar_inp)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(tar_real, predictions)


    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (data)) in enumerate(gen):
            train_step(data[:-1], data[-1])
            if batch % 100 == 0:
                print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                 ckpt_save_path))

        print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                             train_loss.result(),
                                                             train_accuracy.result()))

        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    transformer.save_weights(save_weights)
