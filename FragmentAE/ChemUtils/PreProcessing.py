from .SmilesEncoding import smiles_to_hot, hot_to_smiles, array_to_smiles, CHAR_LEN
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem.SaltRemover import SaltRemover
from multiprocessing import Pool
from time import time

fp_bits = 2048
fp_radius=2

def removeSalt(mol):
    remover=SaltRemover()
    return remover.StripMol(mol, dontRemoveEverything=True)

def getMol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        return Chem.MolFromSmiles(smiles)
    except:
        return None

def getSmiles(mol):
    return Chem.MolToSmiles(mol, isomericSmiles=False)

def getFP(mol):
    return np.array(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, fp_radius, fp_bits), dtype=np.float32)


def read_file_base(file_name, smiles_column="SMILES", remove_salt=True,
                   isomeric_smiles=False, fp_radius = 2, fp_bits=2048, useCount=False):
    df = pd.read_csv(file_name)
    mols = [Chem.MolFromSmiles(x) for x in df[smiles_column]]
    mols = [x for x in mols if x is not None]

    if remove_salt:
        remover=SaltRemover()
        mols = [remover.StripMol(mol, dontRemoveEverything=True) for mol in mols]

    smiles = [Chem.MolToSmiles(x, isomericSmiles=False) for x in mols]

    hot_smiles = [smiles_to_hot(x) for x in smiles]
    mols = [x for i, x in enumerate(mols) if hot_smiles[i] is not None]
    hot_smiles = [x for x in hot_smiles if x is not None]

    if useCount==False:
        fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(x, fp_radius, fp_bits) for x in mols]
        fp_array = [fp_bits_to_array(x) for x in fps]
    else:
        fp_array = [GetHashedMorganFingerprintArray(x, fp_radius, fp_bits) for x in mols]

    mw = [Descriptors.ExactMolWt(x) for x in mols]
    return hot_smiles, fp_array, mw


def read_file_base_multi(file_name, file_type="csv", smiles_column="SMILES", remove_salt=False,
                   useCount=False, num_proc=16):
    start = time()
    if file_type=="csv":
        df = pd.read_csv(file_name)
        smiles = list(df[smiles_column])
    elif file_type=="txt":
        with open(file_name, "rt") as pf:
            smiles = pf.read().split("\n")

    p = Pool(num_proc)
    mols = p.map(getMol,smiles)
    mols = [x for x in mols if x is not None]

    if remove_salt:
        mols = p.map(removeSalt, mols)

    smiles = p.map(getSmiles, mols)
    hot_smiles = p.map(smiles_to_hot, smiles)
    mols = [x for i, x in enumerate(mols) if hot_smiles[i] is not None]
    hot_smiles = [x for x in hot_smiles if x is not None]

    fps = p.map(getFP, mols)
    print(len(fps))

    fps = np.stack(fps)
    fp_array = [fp_bits_to_array(x) for x in fps]

    return hot_smiles, fp_array, fps


def read_file(file_name, file_type="csv", charlen=CHAR_LEN,
              smiles_column="SMILES", remove_salt=False,
              useCount=False, num_proc=16):
    hot_smiles, fp_array, fps = read_file_base_multi(file_name, file_type, smiles_column, remove_salt,
                               useCount, num_proc=16)

    smiles_array = encode_to_array(hot_smiles, charlen)
    fp_array = encode_to_array(fp_array, fp_bits)

    return smiles_array, fp_array, fps


def read_file_base_mmp(file_name, transformation="Transformation", target_smiles="Target_Mol", source_smiles="Source_Mol", remove_salt=True,
                   isomeric_smiles=False, fp_radius = 2, fp_bits=2048, useCount=False):
    df = pd.read_csv(file_name)
    fs = list(df["Transformation"])
    fs = [x.split(">>") for x in fs]
    fs = list(map(list,zip(*fs)))

    f1 = [Chem.MolFromSmiles(x) for x in fs[0]]
    f2 = [Chem.MolFromSmiles(x) for x in fs[1]]

    fp1 = np.array([list(Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(x,2)) for x in f1], dtype=np.int32)
    fp2 = np.array([list(Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(x,2)) for x in f2], dtype=np.int32)
    mols = [Chem.MolFromSmiles(x) for x in df[source_smiles]]
    if remove_salt:
        remover=SaltRemover()
        mols = [remover.StripMol(mol, dontRemoveEverything=True) for mol in mols]

    smiles = [Chem.MolToSmiles(x, isomericSmiles=False) for x in mols]
    mols = [Chem.MolFromSmiles(x) for x in smiles]

    sfp = np.array([list(Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(x,2)) for x in mols])
    tfp = sfp - fp1 -fp2
    tfp = [np.where(x==1)[0] for x in tfp]

    mols = [Chem.MolFromSmiles(x) for x in df[target_smiles]]
    if remove_salt:
        remover=SaltRemover()
        mols = [remover.StripMol(mol, dontRemoveEverything=True) for mol in mols]

    smiles = [Chem.MolToSmiles(x, isomericSmiles=False) for x in mols]

    hot_smiles = [smiles_to_hot(x) for x in smiles]
    mols = [x for i, x in enumerate(mols) if hot_smiles[i] is not None]
    hot_smiles = [x for x in hot_smiles if x is not None]

    mw = [Descriptors.ExactMolWt(x) for x in mols]
    return hot_smiles, tfp, mw


def read_file_mmp(file_name, charlen=CHAR_LEN,
                  transformation="Transformation", target_smiles="Target_Mol", source_smiles="Source_Mol", remove_salt=True,
                   isomeric_smiles=False, fp_radius = 2, fp_bits=2048, useCount=False, mw_calc=False, step=20, max_weight=500):
    hot_smiles, fp_array, mw = read_file_base_mmp(file_name, transformation, target_smiles, source_smiles, remove_salt,
                               isomeric_smiles, fp_radius, fp_bits, useCount)

    smiles_array = encode_to_array(hot_smiles, charlen)

    if mw_calc:
        fp_array, vocab_size = encode_to_array_plus_mw(fp_array, fp_bits, mw, step, max_weight)
        return smiles_array, fp_array, vocab_size
    else:
        fp_array = encode_to_array(fp_array, fp_bits)
        return smiles_array, fp_array


def encode(lang1, vocab_size):
    lang1 = list(lang1)
    lang1 = [vocab_size] + lang1 + [vocab_size+1]
    return lang1


def encode_to_array(lang1, vocab_size):
    lang1 = [encode(x, vocab_size) for x in lang1]
    max_length = max([len(x) for x in lang1])

    code = np.zeros([len(lang1), max_length], dtype=np.int32)

    for i,x in enumerate(lang1):
        code[i][:len(x)] = np.array(x, dtype=np.int32)+1

    return code


def encode_to_array_plus_mw(lang1, vocab_size, mw, step=20, max_weight=500):
    vocab_size_mw = vocab_size+int(max_weight/step)+1

    lang1 = [[vocab_size_mw]+[vocab_size+int(mw[i]/step)]+list(x)+[vocab_size_mw+1] for i,x in enumerate(lang1)]
    max_length = max([len(x) for x in lang1])

    code = np.zeros([len(lang1), max_length], dtype=np.int32)

    for i,x in enumerate(lang1):
        code[i][:len(x)] = np.array(x, dtype=np.int32)+1

    return code, vocab_size_mw


def fp_bits_to_array(fp):
    fp_array = list(np.where(fp == 1)[0])
    return fp_array


def smiles_to_fp(smiles, remove_salt=True,
                       isomeric_smiles=False, fp_radius = 2, fp_bits=2048):

    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False)
    mol = Chem.MolFromSmiles(smiles)

    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, fp_radius, fp_bits)
    fp = fp_bits_to_array(fp)
    fp = encode_to_array([fp], fp_bits)

    return fp


def smiles_to_fp_array(smiles, remove_salt=True,
                       isomeric_smiles=False, fp_radius = 2, fp_bits=2048):

    fp = [smiles_to_fp(x, remove_salt, isomeric_smiles, fp_radius, fp_bits) for x in smiles]
    max_len = max([len(x) for x in fp])
    fp_array = np.zeros([len(fp), max_len], dtype=np.int32)

    for i,x in enumerate(fp):
        fp_array[i][:len(x)] = x

    return fp_array


def smiles_to_mwfp_array(smiles, remove_salt=True,
                       isomeric_smiles=False, fp_radius = 2, fp_bits=2048,
                         step=20, max_weight=500):

    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False)
    mol = Chem.MolFromSmiles(smiles)
    mw = Descriptors.ExactMolWt(mol)

    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, fp_radius, fp_bits)
    fp = fp_bits_to_array(fp)
    fp = encode_to_array_plus_mw([fp], fp_bits, mw, step, max_weight)

    return fp


def read_file_for_descriptors(file_name, charlen=CHAR_LEN, smiles_column="SMILES", remove_salt=True,
                              isomeric_smiles=False):
    df = pd.read_csv(file_name)
    mols = [Chem.MolFromSmiles(x) for x in df[smiles_column]]
    mols = [x for x in mols if x is not None]

    if remove_salt:
        remover=SaltRemover()
        mols = [remover.StripMol(mol, dontRemoveEverything=True) for mol in mols]

    smiles = [Chem.MolToSmiles(x, isomericSmiles=False) for x in mols]

    hot_smiles = [smiles_to_hot(x) for x in smiles]
    smiles = [x for i, x in enumerate(smiles) if hot_smiles[i] is not None]
    hot_smiles = [x for x in hot_smiles if x is not None]
    smiles_array = encode_to_array(hot_smiles, charlen)

    descs = [get_all_descriptors(x) for x in smiles]
    descs, vocab_size = descs_to_array(descs)

    return smiles_array, descs, vocab_size


def get_all_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    desc = ["%s_%.2g"%(y[0],y[1](mol)) for y in Descriptors.descList]
    return desc


def descs_to_array(descs):
    d={}
    descs_set = list(set([e for row in descs for e in row]))
    vocab_size = len(descs_set)
    for i,x in enumerate(descs_set):
        d[x] = i

    descs = [[d[y] for y in x] for x in descs]
    descs = [encode(x, vocab_size) for x in descs]
    descs_len = [len(x) for x in descs]
    max_len = max(descs_len)

    dcode = np.zeros([len(descs), max_len], dtype=np.int32)
    for i,x in enumerate(descs):
        dcode[i][:len(x)] = np.array(x, dtype=np.int32)+1

    return dcode, vocab_size


def GetHashedMorganFingerprintArray(mol, radian=2, fp_bits=2048):
    fps = rdMolDescriptors.GetHashedMorganFingerprint(mol,radian, fp_bits).GetNonzeroElements()
    keys = list(fps.keys())
    values = list(fps.values())
    fps = [np.array([keys[i]]*values[i], dtype=np.int32) for i in range(len(keys))]
    fps = np.hstack(fps)

    return  fps
