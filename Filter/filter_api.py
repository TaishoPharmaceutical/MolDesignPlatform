#!/usr/bin/env python
# coding: utf-8

from rdkit import Chem
from rdkit.Chem import Descriptors, MolStandardize
import pandas as pd
import numpy as np
import os
import json
from flask import Flask, jsonify, request
from Filter import Filter

root_path = os.path.dirname(os.path.dirname(__file__))
filter_path = os.path.join(root_path, "Filter")

api = Flask(__name__)

@api.route('/filt', methods=['GET', 'POST'])
def filt():
    if request.method == 'POST':
        data = request.get_json()
        smiles = data["smiles"]
        level = data["level"]
        if level is None:
            level="rough"
    else:
        smiles = request.args.get("smiles", "Not_defined")
        level = request.args.get("level", "rough")
        smiles = smiles.split(",")

    vt_mols = [Chem.MolFromSmiles(x) for x in smiles]
    vt_mols = [x for x in vt_mols if x is not None]

    myfilt = Filter(f"{filter_path}/filter.csv")
    myfilt.setLevel(level)

    mols = myfilt.getFilteredMols(vt_mols)
    smiles = [Chem.MolToSmiles(x) for x in mols]

    return jsonify(smiles)


if __name__ == '__main__':
    config_path = os.path.join(root_path,"Settings","config.json")
    with open(config_path, "rt") as pf:
        config = json.load(pf)

    if config["ssl"]["key"]=="":
        api.run(host=config["ip"]["Filter"],
                port=config["port"]["Filter"])
    else:
        api.run(host=config["ip"]["Filter"],
                port=config["port"]["Filter"],
                ssl_context=(config["ssl"]["crt"], config["ssl"]["key"]))

