#!/usr/bin/env python
# coding: utf-8

from glob import glob
import os
import pandas as pd
import argparse

from rdkit.Chem import PandasTools as pt
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors

root_path = os.path.dirname(os.path.dirname(__file__))
reactor_path = os.path.join(root_path, "Reactor")

import sys
sys.path.append(root_path)
from Filter.Filter import Filter

def get_df(mols):
    df = pd.DataFrame()
    df["smiles"]=[Chem.MolToSmiles(x) for x in mols]
    df["mol_weight"]=[Descriptors.ExactMolWt(x) for x in mols]
    df["NumAr"]=[Descriptors.NumAromaticRings(x) for x in mols]

    return df

def get_reactants(file_name):
    df = pd.read_csv(file_name)
    mols = [Chem.MolFromSmiles(x) for x in df["smiles"]]
    df["smiles"] = [Chem.MolToSmiles(x) for x in mols]

    suzuki_arylhalide = Chem.MolFromSmarts("[c:1][c:2]([Cl,Br,I])[c:3]")
    suzuki_boron = Chem.MolFromSmarts("[c:4](B(O)O)")
    arylhalides = [x for x in mols if len(x.GetSubstructMatches(suzuki_arylhalide)) == 1]
    borons = [x for x in mols if len(x.GetSubstructMatches(suzuki_boron)) == 1]

    cond_carboxyls = Chem.MolFromSmarts("[C:1](=[O:2])O")
    cond_amine = Chem.MolFromSmarts("[N!0H:3][C:4]")
    carboxyls = [x for x in mols if len(x.GetSubstructMatches(cond_carboxyls)) == 1]
    amines = [x for x in mols if len(x.GetSubstructMatches(cond_amine)) == 1]

    ra_ketone = Chem.MolFromSmarts("[c,CX4!R:1][C:2](=[O])[c,CX4!R,#1:3]")
    ra_amine = Chem.MolFromSmarts("[c,CX4!R:4][NH1!R:5][c,CX4!R,#1:6]")
    ra_ketones = [x for x in mols if len(x.GetSubstructMatches(ra_ketone)) == 1]
    ra_amines = [x for x in mols if len(x.GetSubstructMatches(ra_amine)) == 1]

    filt = Filter(f"{root_path}/Filter/filter.csv")
    filt.setLevel("basic")
    arylhalides = filt.getFilteredMols(arylhalides)
    borons = filt.getFilteredMols(borons)
    carboxyls = filt.getFilteredMols(carboxyls)
    amines = filt.getFilteredMols(amines)
    ra_ketones = filt.getFilteredMols(ra_ketones)
    ra_amines = filt.getFilteredMols(ra_amines)

    arylhalides = get_df(arylhalides)
    arylhalides = pd.merge(arylhalides, df, how="left")
    arylhalides.to_csv(f"{reactor_path}/reactants/arylhalide.csv", index=False)
    borons = get_df(borons)
    borons = pd.merge(borons, df, how="left")
    borons.to_csv(f"{reactor_path}/reactants/boron.csv", index=False)
    carboxyls = get_df(carboxyls)
    carboxyls = pd.merge(carboxyls, df, how="left")
    carboxyls.to_csv(f"{reactor_path}/reactants/carboxyl.csv", index=False)
    amines = get_df(amines)
    amines = pd.merge(amines, df, how="left")
    amines.to_csv(f"{reactor_path}/reactants/amine.csv", index=False)
    ra_ketones = get_df(ra_ketones)
    ra_ketones = pd.merge(ra_ketones, df, how="left")
    ra_ketones.to_csv(f"{reactor_path}/reactants/ketone.csv", index=False)
    ra_amines = get_df(ra_amines)
    ra_amines = pd.merge(ra_amines, df, how="left")
    ra_amines.to_csv(f"{reactor_path}/reactants/ra_amine.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--df", help="path: data file", type=str)
    args = parser.parse_args()

    get_reactants(args.df)
