#!/usr/bin/env python
# coding: utf-8

from rdkit import Chem
from rdkit.Chem import Draw
from flask import Flask, jsonify, request
#import joblib

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from VQVAEChemGen import VQVAEGen

import os
import json

api = Flask(__name__)

@api.route('/vae', methods=['GET', 'POST'])
def vae():
    if request.method == 'POST':
        data = request.get_json()
        smiles = data["smiles"]

    elif request.method == 'GET':
        smiles =  request.args.get("smiles", "Not defined")

    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles),
                              isomericSmiles=False)
    mols, indices = chemgen.input_smiles_base_sampling(smiles, n = 200, e = 0.32)
    smiles = [Chem.MolToSmiles(x) for x in mols]

    return jsonify(smiles)


if __name__ == '__main__':
    root_path=os.path.dirname(os.path.dirname(__file__))
    vqvae_path = os.path.join(root_path, "VQVAE")
    config_path = os.path.join(root_path,"Settings","config.json")
    with open(config_path, "rt") as pf:
        config = json.load(pf)

    os.environ["CUDA_VISIBLE_DEVICES"]=config["use_GPU_No"]["VQVAE"]

    chemgen = VQVAEGen(os.path.join(config["VQVAE_train"]["save_path"], "VQVAE"))

    if config["ssl"]["key"]=="":
        api.run(host=config["ip"]["VQVAE"],
                port=config["port"]["VQVAE"])
    else:
        api.run(host=config["ip"]["VQVAE"],
                port=config["port"]["VQVAE"],
                ssl_context=(config["ssl"]["crt"], config["ssl"]["key"]))

