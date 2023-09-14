#!/usr/bin/env python
# coding: utf-8

from glob import glob
import os
import pandas as pd
from rdkit.Chem import PandasTools as pt
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
from flask import Flask, jsonify, request
import json

suzuki_coupling = AllChem.ReactionFromSmarts("[c:1][c:2]([Cl,Br,I])[c:3].[c:4](B(O)O)>>[c:1][c:2]([c:4])[c:3]")
suzuki_coupling.Initialize()

condensation = AllChem.ReactionFromSmarts('[C:1](=[O:2])O.[N!0H:3][C:4]>>[C:1](=[O:2])[N:3][C:4]')
condensation.Initialize()

reductive_amination = AllChem.ReactionFromSmarts('[c,CX4!R:1][C:2](=[O])[c,CX4!R,#1:3].[c,CX4!R:4][NH1!R:5][c,CX4!R,#1:6]>>[c,CX4!R:1][C:2]([N:5]([c,CX4!R:4])[c,CX4!R,#1:6])[c,CX4!R,#1:3]')
reductive_amination.Initialize()

buchwald_amination = AllChem.ReactionFromSmarts('[c:1][c:2]([Cl,Br,I])[c:3].[c,CX4!R:4][NH1!R:5][c,CX4!R,#1:6]>>[c:1][c:2]([N:5]([c,CX4!R:4])[c,CX4!R,#1:6])[c:3]')
buchwald_amination.Initialize()

root_path = os.path.dirname(os.path.dirname(__file__))
reactor_path = os.path.join(root_path, "Reactor")

arylhalides = pd.read_csv(os.path.join(reactor_path,"reactants","arylhalide.csv"))
borons = pd.read_csv(os.path.join(reactor_path,"reactants","boron.csv"))
carboxyls = pd.read_csv(os.path.join(reactor_path,"reactants","carboxyl.csv"))
amines = pd.read_csv(os.path.join(reactor_path,"reactants","amine.csv"))
ra_ketones = pd.read_csv(os.path.join(reactor_path,"reactants","ketone.csv"))
ra_amines = pd.read_csv(os.path.join(reactor_path,"reactants","ra_amine.csv"))

# __name__は現在のファイルのモジュール名
api = Flask(__name__)
#api.config['JSON_AS_ASCII'] = False
#CORS(api, allow_headers = 'Content-Type')

@api.route('/react', methods=['GET', 'POST'])
def react():
    if request.method == 'POST':
        data = request.get_json()
        smiles = data["smiles"]
        react = data["reaction"]
        min_wt = data["MolWt_min"]
        max_wt = data["MolWt_max"]
        min_ar = data["ArNum_min"]
        max_ar = data["ArNum_max"]
    else:
        smiles = request.args.get("smiles", "Not_defined")
        react = request.args.get("reaction", "Not_defined")
        min_wt=None
        max_wt=None
        min_ar=None
        max_ar=None

    max_wt = 65535 if max_wt is None else max_wt
    max_ar = 65535 if max_ar is None else max_ar
    min_wt = -1 if min_wt is None else min_wt
    min_ar = -1 if min_ar is None else min_ar

    if react == "condensation-with-amine":
        reactants = amines
        reaction = condensation
    elif react == "condensation-with-carboxylic":
        reactants = carboxyls
        reaction = condensation
    elif react == "suzuki-with-boron":
        reactants = borons
        reaction = suzuki_coupling
    elif react == "suzuki-with-arylhalide":
        reactants = arylhalides
        reaction = suzuki_coupling
    elif react == "reductive-amination-with-amine":
        reactants = ra_amines
        reaction = reductive_amination
    elif react == "reductive-amination-with-aldehyde-or-ketone":
        reactants = ra_ketones
        reaction = reductive_amination
    elif react == "buchwald-amination-with-amine":
        reactants = ra_amines
        reaction = buchwald_amination
    elif react == "buchwald-amination-with-arylhalide":
        reactants = arylhalide
        reaction = buchwald_amination


    mol = Chem.MolFromSmiles(smiles)
    if not reaction.IsMoleculeReactant(mol):
        return ""

    reactants = reactants[reactants["NumAr"] <= max_ar]
    reactants = reactants[reactants["NumAr"] >= min_ar]
    reactants = reactants[reactants["mol_weight"] <= max_wt]
    reactants = reactants[reactants["mol_weight"] >= min_wt]
    reactants = reactants.sample(frac=1)

    mols = [Chem.MolFromSmiles(x) for x in reactants[:1000]["smiles"]]
    results = []
    ids = []

    first_x = ["condensation-with-carboxylic","suzuki-with-arylhalide",
               "reductive-amination-with-aldehyde-or-ketone","buchwald-amination-with-arylhalide"]
    second_x = ["condensation-with-amine", "suzuki-with-boron",
                "reductive-amination-with-amine","buchwald-amination-with-amine"]

    if react in first_x:
        for i,x in enumerate(mols):
            try:
                results.append(reaction.RunReactants([x,mol]))
                ids.append(reactants.iloc[i]["reagent-info"])
            except:
                continue
    elif react in second_x:
        for i,x in enumerate(mols):
            try:
                results.append(reaction.RunReactants([mol,x]))
                ids.append(reactants.iloc[i]["reagent-info"])
            except:
                continue

    if len(results)==0:
        return ""

    results = [{"smiles":Chem.MolToSmiles(z), "reagent-info":ids[i]} for i,x in enumerate(results) for y in x for z in y]
    df = pd.DataFrame(results)
    df = df.drop_duplicates()
    results = df.to_dict("records")

    return jsonify(results)

if __name__ == '__main__':
    config_path = os.path.join(root_path,"Settings","config.json")
    with open(config_path, "rt") as pf:
        config = json.load(pf)

    if config["ssl"]["key"]=="":
        api.run(host=config["ip"]["Reactor"],
                port=config["port"]["Reactor"])
    else:
        api.run(host=config["ip"]["Reactor"],
                port=config["port"]["Reactor"],
                ssl_context=(config["ssl"]["crt"], config["ssl"]["key"]))
