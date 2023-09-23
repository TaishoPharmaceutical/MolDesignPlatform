#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import time
import pickle
import numpy as np

from VQVAEChem import VQVAE, create_masks, CustomSchedule
from VQVAEChem import MAX_LEN, MAX_NB, MAX_RING_SIZE, SMILES_LENGTH
from VQVAEChem import encode_size, hidden_size, num_embeddings, commitment_cost
from ChemUtils import batch_gen_gcn_mol_feat, ATOM_LEN, BOND_LEN, evaluate_vec

import os
import argparse
from glob import glob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--df", help="path: data file", type=str)
    parser.add_argument("--save", help="folder path: save train model", type=str)
    parser.add_argument("--gpu", help="using gpu number", type=str)
    parser.add_argument("--batch", help="batch size of a training", type=int, default=200)
    parser.add_argument("--VQTrainEpoch", help="epoch", type=int, default=500000)
    parser.add_argument("--VAETrainEpoch", help="epoch", type=int, default=250000)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    save_path = args.save
    batch_size = args.batch
    train_file_name = args.df
    max_epoch = args.VQTrainEpoch
    vae_train_epoch = args.VAETrainEpoch
    os.makedirs(f"{save_path}", exist_ok=True)

    model = VQVAE(encode_size, hidden_size, MAX_LEN, num_embeddings=num_embeddings, commitment_cost =commitment_cost)
    if len(glob(f"{save_path}/VQVAE_decoder*")) !=0:
        model.load_weights(f"{save_path}/VQVAE_decoder")
    if len(glob(f"{save_path}/vae*")) !=0:
        model.load_weights(f"{save_path}/vae")

    gen = tf.data.Dataset.from_generator(lambda: batch_gen_gcn_mol_feat(train_file_name, "SMILES", MAX_LEN, batch_size,
                                                                        False, f"{save_path}/train-param.pkl", test=False),
                                        (tf.float32, tf.int32, tf.int32, tf.int32,tf.int32,
                                         tf.float32, tf.bool, tf.bool, tf.int32))

    learning_rate = CustomSchedule(encode_size)
    print_per = 500

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    train_loss = tf.keras.metrics.Mean(name='train_loss')


    @tf.function(experimental_relax_shapes=True)
    def train_step(model, masks, data, optimizer, max_len):
        with tf.GradientTape() as tape:
            r_loss, vq_loss, atom_feat, bond_feat = model(data, masks)
            loss = r_loss + vq_loss

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_loss(r_loss)

        return atom_feat, bond_feat

    min_loss = 1
    r_loss = 1
    start=time.time()
    for i, data in enumerate(gen):
        inp_len = tf.cast(tf.not_equal(data[2], 0), dtype=tf.int32)
        inp_len = tf.reduce_sum(inp_len, -1)+2
        tar_len = data[-1]+1
        masks = create_masks(inp_len, tar_len, batch_size)
        atom_feat, bond_feat = train_step(model, masks, data, optimizer, MAX_LEN)

        if i%print_per==0:
            atom_feat = tf.argmax(atom_feat, -1)
            atom_feat = tf.one_hot(atom_feat, ATOM_LEN, dtype=tf.float32)
            atom_feat = tf.reshape(atom_feat, [-1, MAX_LEN*ATOM_LEN])

            bond_feat = tf.argmax(bond_feat, -1)
            bond_feat = tf.one_hot(bond_feat, BOND_LEN, dtype=tf.float32)
            bond_feat = tf.reshape(bond_feat, [-1, MAX_LEN * MAX_LEN* BOND_LEN])

            p = tf.concat([atom_feat, bond_feat], 1)
            p = p.numpy()
            d = data[5].numpy()
            r_loss = train_loss.result().numpy()

            print(evaluate_vec(d, p, data[6].numpy(), data[7].numpy(), MAX_LEN))
            print("restoration_loss", r_loss)
            print("learning_late = %f, epoch=%d, time=%f"%(learning_rate(tf.cast(tf.range(i+1,i+2),
                                                                                 dtype=tf.float32)), i,time.time()-start))
            start=time.time()
            train_loss.reset_states()

        if i%1000 == 1000-1:
            model.save_weights(f'{save_path}/VQVAE_decoder')

            if  r_loss < min_loss:
                model.save_weights(f'{save_path}/VQVAE_decoder')
                min_loss = r_loss

        if i == max_epoch:
            break

    batch_size = 200

    model = VQVAE(encode_size, hidden_size, MAX_LEN, num_embeddings=num_embeddings, commitment_cost =commitment_cost)
    if len(glob(f"{save_path}/VQVAE_decoder*")) !=0:
        model.load_weights(f"{save_path}/VQVAE_decoder")
    if len(glob(f"{save_path}/vae*")) !=0:
        model.load_weights(f"{save_path}/vae")

    train_file_name = f"{save_path}/train-param.pkl"
    gen = tf.data.Dataset.from_generator(lambda: batch_gen_gcn_mol_feat(train_file_name, "SMILES", MAX_LEN, batch_size,
                                                                        True, f"{save_path}/train-param.pkl", test=True),
                                        (tf.float32, tf.int32, tf.int32, tf.int32,tf.int32,
                                         tf.float32, tf.bool, tf.bool, tf.int32))

    encodes = []
    for i, data in enumerate(gen):
        ret,_ = model.encode(data)
        encodes.append(ret.numpy())

    ret = np.vstack(encodes)
    with open(f"{save_path}/vq_code.pkl", 'wb') as pf:
        pickle.dump(ret, pf)


    with open(f"{save_path}/vq_code.pkl",  'rb') as pf:
        encodes = pickle.load(pf)

    BATCH_SIZE = 200

    new_gen = tf.data.Dataset.from_tensor_slices(encodes)
    new_gen = new_gen.shuffle(buffer_size=encodes.shape[0])
    new_gen = new_gen.repeat()
    new_gen = new_gen.batch(BATCH_SIZE)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    new_gen = new_gen.prefetch(buffer_size=AUTOTUNE)

    beta = 0.00005
    step_beta = 0.00005
    max_beta = 0.0005

    print_per = 1000
    beta_update_per = 25000

    learning_rate = 0.0001
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    @tf.function
    def vae_train_step(model, data, optimizer, beta=0.1):
        with tf.GradientTape() as tape:
            r_loss, kl_loss, re_indices = model.vae_train(data)
            loss = r_loss + kl_loss*beta

        grads = tape.gradient(loss, model.vae.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.vae.trainable_variables))

        return r_loss, kl_loss, re_indices

    re_mols = []

    for i, data in enumerate(new_gen):
        r_loss, kl_loss, re_indices = vae_train_step(model, data, optimizer, beta)

        if i%print_per==0:
            ds = data.numpy()
            indices = re_indices.numpy()

            count = 0
            for j,d in enumerate(ds):
                if (d == indices[j]).all():
                    count += 1

            print(r_loss, kl_loss)
            print("%d/%d"%(count,len(ds)))

        if i%beta_update_per == beta_update_per-1:
            beta +=step_beta


        if i%20000 == 20000-1:
            model.save_weights(f'{save_path}/vae')


        if i%vae_train_epoch == vae_train_epoch-1:
            break

    model.input_smiles_base_sampling("COCC[C@@H](C)C(=O)N(C)Cc1ccc(O)cc1")
    model.restoration(["COCC[C@@H](C)C(=O)N(C)Cc1ccc(O)cc1"])
    model.sampling(50,0.2)
    model.save_weights(f"{save_path}/VQVAE")

