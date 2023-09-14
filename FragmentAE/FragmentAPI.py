#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import pandas as pd
import random
import pickle
import time

from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import rdMolDescriptors

from flask import Flask, jsonify, abort, make_response, Response
from flask import request
import json
from json import JSONEncoder
import itertools

fp_radius = 2
fp_bits = 2048
MAX_SMILES_LEN=90
MAX_INPUT_LEN=70

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import os

for dev in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(dev, True)


from ChemUtils.PreProcessing import read_file, smiles_to_fp_array, CHAR_LEN, array_to_smiles
from ChemUtils.ChemUtils import FragmentConnector

from GraphTransformers.Models.GatedGraphTransformer import GatedGraphTransformerS, CustomSchedule, loss_function
from GraphTransformers.Utils.DataUtils import FeatMol, encode_to_array

root_path = os.path.dirname(os.path.dirname(__file__))

# Define transformer model
def get_model(weight_path):
    num_layers = 4
    d_model = 64
    num_heads = 8
    dff = 512
    input_vocab_size=fp_bits+3
    target_vocab_size = CHAR_LEN+3
    pe_input = MAX_INPUT_LEN
    pe_target = MAX_SMILES_LEN
    rate=0.1
    transformer = GatedGraphTransformerS(num_layers,d_model,num_heads,
                                       dff, input_vocab_size, target_vocab_size,
                                       pe_input, pe_target, rate)
    transformer.load_weights(weight_path)
    return transformer

def get_fragments(smiles):
    origin = [smiles]
    mols = [Chem.MolFromSmiles(x) for x in origin]
    cmols = [FeatMol(x) for x in mols]
    inputs = encode_to_array(cmols)

    inp = [np.stack([inputs[0][0]]*2000),
           np.stack([inputs[1][0]]*2000),
           np.stack([inputs[2][0]]*2000)]
    res, res_score = transformer.reconstruct(inp, CHAR_LEN+1, MAX_INPUT_LEN, 2000, noise=0.25)

    smiles = array_to_smiles(res, CHAR_LEN+2)
    smiles = list(set(smiles))
    smiles = [x for x in smiles if x.count("*")==origin[0].count("*")]
    #smiles = get_filtered_smiles(smiles)
    pred_mols = [Chem.MolFromSmiles(x) for x in smiles if x is not None]
    pred_mols = [x for x in pred_mols if x is not None]
    return [Chem.MolToSmiles(x) for x in pred_mols]

def get_smiles_val(smiles):
    wc = smiles.count("*")
    if wc==1:
        smiles = smiles.replace("*", "[1*]")
        return [smiles]

    rand = itertools.permutations(range(1,wc+1), wc)

    smis = []
    count = 0
    for idx in rand:
        smis.append(get_smiles_val_core(smiles, idx))

    return smis


def get_smiles_val_core(smiles, idx):
    smi = ""
    count = 0
    for x in smiles:
        if x == "*":
            smi =smi + f"[{idx[count]}*]"
            count +=1
        else:
            smi =smi + x

    return smi


api = Flask(__name__)
api.config['JSON_AS_ASCII'] = False

@api.route('/generate', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        smiles1 = request.form.get("smi1", "")
        smiles2 = request.form.get("smi2", "")
    else:
        smiles1 = request.args.get("smi1", "")
        smiles2 = request.args.get("smi2", "")
    if smiles1 =="" or smiles2=="":
        return ""

    frags = get_fragments(smiles1)
    frags = [get_smiles_val(x) for x in frags]
    frags = sum(frags, [])

    frags_ = []
    for x in frags:
        try:
            frags_.append(Chem.MolFromSmiles(x))
        except:
            frags_.append(None)
    frags = [x for x in frags_ if x is not None]

    wc = smiles2.count("*")
    core = get_smiles_val_core(smiles2, range(1,1+wc))
    core = Chem.MolFromSmiles(core)

    mols = [FragmentConnector([x, core]) for x in frags]

    smi = []
    for x in mols:
        try:
            smi.append(Chem.MolToSmiles(x))
        except:
            continue

    return jsonify(smi)


if __name__ == '__main__':
    config_path = os.path.join(root_path, "Settings", "config.json")
    with open(config_path, "rt") as fp:
        config = json.loads(fp.read())

    os.environ["CUDA_VISIBLE_DEVICES"]=config["use_GPU_No"]["FragmentAE"]

    weight_path = os.path.join(config["FragmentAE_train"]["save_path"], "FragmentAE")
    transformer = get_model(weight_path)

    if config["ssl"]["key"]=="":
        api.run(host=config["ip"]["FragmentAE"],
                port=config["port"]["FragmentAE"])
    else:
        api.run(host=config["ip"]["FragmentAE"],
                port=config["port"]["FragmentAE"],
                ssl_context=(config["ssl"]["crt"], config["ssl"]["key"]))
