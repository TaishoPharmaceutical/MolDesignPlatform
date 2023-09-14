import sys
sys.path.append("../")

from GTs.Models.GatedGraphTransformer import create_masks, Encoder
from GTs.RunModel.RunModel import PredictModel
from GTs.Utils.DataUtils import FeatMol, encode_to_array, read_csv, MAX_LEN

from sklearn.metrics import roc_auc_score
from glob import glob
from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps
from tqdm import tqdm
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import r2_score, roc_auc_score

import pandas as pd
import numpy as np
import tensorflow as tf
import argparse

from GTs.train import num_layers,d_model,num_heads,dff,input_vocab_size,pe_input,rate

result_name = ["_Esemble", "_normal", "_pooling"]


def DrawMolWeightMap(mol, weight):
    num_atoms = mol.GetNumAtoms()
    return SimilarityMaps.GetSimilarityMapFromWeights(mol, weight[:num_atoms])

def GetMolsWeightMapImages(mols, weights):
    images = [DrawMolWeightMap(mol, weights[i]) for i,mol in enumerate(mols)]
    return images


class EsembleModel():
    def __init__(self, ggt_models, data_frame, reg_tasks, ggt_max_len=MAX_LEN):
        self.ggt = ggt_models
        self.ggt_max_len=ggt_max_len
        self.reg_tasks = len(reg_tasks)
        self.class_tasks = len(ggt_models[0].class_output)
        self.max_len=ggt_max_len

        self.z_mean = []
        self.z_std = []

        for x in reg_tasks:
            self.z_mean.append(np.mean(data_frame[x]))
            self.z_std.append(np.std(data_frame[x]))

        self.z_mean = tf.convert_to_tensor(np.array(self.z_mean, dtype=np.float32))[:,tf.newaxis]
        self.z_std = tf.convert_to_tensor(np.array(self.z_std, dtype=np.float32))[:,tf.newaxis]

    def call(self, smiles, sub_pred = False):
        ggt_feat, mols, flags = self.get_feats(smiles)
        batch_size=len(mols)
        mean = tf.concat([self.z_mean]*batch_size, -1)
        std = tf.concat([self.z_std]*batch_size, -1)

        dec_inp=tf.range(1,71)
        mask=tf.cast(tf.math.less(dec_inp,self.reg_tasks+self.class_tasks+1), dtype=tf.int32)
        dec_inp *=mask
        dec_inp = tf.tile([dec_inp], [batch_size,1])
        dec_mask, _, _ = create_masks(dec_inp, dec_inp)
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(ggt_feat[2], ggt_feat[2])
        prediction = [x(ggt_feat[0], dec_inp, False, enc_padding_mask, dec_mask, ggt_feat[1]) for x in self.ggt]
        reg_all = [x[0]*std+mean for x in prediction]
        reg = tf.stack(reg_all)
        reg = tf.reduce_mean(reg, 0)

        cls_all = [x[1][:,:,1] for x in prediction]
        cls = tf.stack(cls_all)
        cls = tf.reduce_mean(cls, 0)

        ret1 = tf.concat([reg,cls], 0)
        ret_all = [tf.concat([x,cls_all[i]], 0) for i,x in enumerate(reg_all)]

        if sub_pred:
            return [ret1]+ret_all, flags
        else:
            return [ret1], flags

    def get_mol_feature(smiles):
        ggt_feat, mols, flags = self.get_feats(smiles)

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(ggt_feat[2], ggt_feat[2])
        molf = [x.get_mol_feature(ggt_feat[0], False, enc_padding_mask, ggt_feat[1]) for x in self.ggt]

        molf = tf.concat(molf, -1)
        return molf

    def get_feats(self, smiles):
        mols = [Chem.MolFromSmiles(x) for x in smiles]

        flags = []
        fmols = []
        re_mols = []
        for x in mols:
            if x is None:
                flags.append(False)
                continue

            if x.GetNumAtoms()>self.max_len:
                flags.append(False)
                continue

            try:
                fmol = FeatMol(x)
                fmols.append(fmol)
                re_mols.append(x)
                flags.append(True)

            except:
                flags.append(False)
                continue

        ggt_feat = encode_to_array(fmols, self.ggt_max_len)

        return ggt_feat, re_mols, flags


    def predict(self, smiles, sub_pred = False):
        batch_size = 500
        num_smiles = len(smiles)
        times = int(np.ceil(num_smiles/batch_size))

        res = []
        flags = []
        for i in range(times):
            batch = smiles[i*batch_size:(i+1)*batch_size]
            r, f = self.call(batch, sub_pred)
            res.append(r)
            flags += f
            #print("prediction progress: %d/%d"%(i+1,times))

        res = list(map(list, zip(*res)))
        res = [tf.concat(res[i], -1) for i in range(len(res))]


        gather = []
        count=1
        for f in flags:
            if f:
                gather.append(count)
                count+=1
            else:
                gather.append(0)

        n_tasks = res[0].shape[0]
        gather=tf.convert_to_tensor(np.array(gather, dtype=np.int32))

        nans=tf.convert_to_tensor(np.array([np.nan]*n_tasks, dtype=np.float32))[:, tf.newaxis]
        res = [tf.concat([nans, res[i]], -1) for i in range(len(res))]
        res = [tf.gather(res[i], gather, axis=-1) for i in range(len(res))]

        return res


    def get_weights(self, smiles):
        ggt_feat, mols, flags = self.get_feats(smiles)
        batch_size=len(mols)

        dec_inp=tf.range(1,71)
        mask=tf.cast(tf.math.less(dec_inp,self.reg_tasks+self.class_tasks+1), dtype=tf.int32)
        dec_inp *=mask
        dec_inp = tf.tile([dec_inp], [batch_size,1])
        dec_mask, _, _ = create_masks(dec_inp, dec_inp)
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(ggt_feat[2], ggt_feat[2])
        weight = self.ggt[1].get_weight(ggt_feat[0], dec_inp, False, enc_padding_mask, dec_mask, ggt_feat[1], self.max_len)
        weight_ = tf.stack([weight[i]*self.z_std[i]+self.z_mean[i] for i in range(len(self.z_mean))])
        weight = weight[len(self.z_mean):]

        weight = tf.concat([weight_,weight],0)
        smiles = [Chem.MolToSmiles(x) for x in mols]

        return weight, smiles



def get_esemble_model(ggt_path, data_path, reg_tasks,
                      class_tasks, ggt_max_len=MAX_LEN):

    """
    data_path : The data is needed for converting z_score to original value
    """
    n_tasks1=len(reg_tasks)
    n_tasks2=len(class_tasks)
    path = ["","_pooling"]
    encoder_type = ["normal", "pooling"]
    ggt_path = [ggt_path + path[i] for i in range(2)]

    transformer = [PredictModel(num_layers, d_model, num_heads, dff,
                   input_vocab_size, pe_input, n_tasks1,n_tasks2, rate, encoder_type[i]) for i in range(2)]

    for i in range(2):
        transformer[i].load_weights(ggt_path[i])

    df = pd.read_csv(data_path)
    esemble_model = EsembleModel(transformer, df, reg_tasks)

    return esemble_model


def categological_eval(y_true, y_pred):
    p_ = np.round(y_pred)
    eval_met = list(y_true*2 + p_)

    tp = eval_met.count(3)
    fn = eval_met.count(2)
    fp = eval_met.count(1)
    tn = eval_met.count(0)

    num = len(eval_met)
    y_ = list(y_true)
    pos = y_.count(1)
    neg = y_.count(0)

    return ((tp+tn)/num, tp/pos, tn/neg, fp/neg, fn/pos)

def make_result_tables(pred_df, save_folder, reg_tasks, class_tasks):

    for i, x in enumerate(reg_tasks):
        df = pd.DataFrame()
        df["true value"] = pred_df[x]
        df["pred value"] = pred_df[x+result_name[0]]
        df.dropna().to_csv(save_folder + "/score" + x + "reg.csv", index=False)

        y_true=np.array(pred_df[x])
        y_pred=np.array(pred_df[x+result_name[0]])
        f = np.isnan(y_true) | np.isnan(y_pred)
        y_true = y_true[~f]
        y_pred = y_pred[~f]

        clf_score = r2_score(y_true.reshape(-1,1), y_pred.reshape(-1,1))
        print(reg_tasks[i],clf_score)

    scores = []
    for i, x in enumerate(class_tasks):
        y_true = np.array(pred_df[x])
        y_pred = np.array(pred_df[x+result_name[0]])

        f = np.isnan(y_true) | np.isnan(y_pred)

        y_true = y_true[~f]
        y_pred = y_pred[~f]

        score=roc_auc_score(y_true, y_pred)
        print(class_tasks[i],score)

        evaluations = categological_eval(y_true, y_pred>0.5)

        scores.append([score]+list(evaluations))

    scores = list(map(list,zip(*scores)))

    df = pd.DataFrame()
    df["tasks"] = class_tasks
    df["AUC"] = scores[0]
    df["Accuracy"] = scores[1]
    df["True Positive"] = scores[2]
    df["True Negative"] = scores[3]
    df["False Positive"] = scores[4]
    df["False Negative"] = scores[5]

    df.to_csv(save_folder + "/scoreclass.csv")

