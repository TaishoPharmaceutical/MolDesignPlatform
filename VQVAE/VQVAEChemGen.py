from VQVAEChem import VQVAE
from rdkit import Chem
from tqdm import tqdm
import numpy as np
import os.path as op

from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import DataStructs
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem import Descriptors, MolStandardize
from rdkit.Chem import QED
import pandas as pd
import copy
import gc


MAX_LEN=61
MAX_NB=9
MAX_RING_SIZE=9
SMILES_LENGTH = 150

encode_size = 128
hidden_size = 256
num_embeddings=4096
commitment_cost=0.25


class VQVAEGen():
    def __init__(self, model_weights_path, encode_size=encode_size, hidden_size=hidden_size, num_embeddings=num_embeddings, commitment_cost=commitment_cost):
        self.model = VQVAE(encode_size, hidden_size, MAX_LEN, num_embeddings=num_embeddings, commitment_cost =commitment_cost)
        self.model.load_weights(model_weights_path)

    def sampling(self, n=100, width=1):
        mols, indices = self.model.sampling(n, width, zero_code = None, random='normal', z_mean=False, max_size=None, fix_indices=None)
        re_smiles = [Chem.MolToSmiles(x, canonical=False) for x in mols if x is not None]
        mols = [Chem.MolFromSmiles(x) for x in re_smiles]
        mols = [x for x in mols if x is not None]
        return mols

    def get_latent_vecotr(self, smiles):
        indices, x, pre_latent_vector, latent_vector = self.model.encode_from_smiles(smiles)
        return indices, x, pre_latent_vector, latent_vector

    def restoration(self, smiles):
        mols, indices = self.model.restoration(smiles)
        return mols

    def input_smiles_base_sampling(self, smiles, n=1, e=0.3,  max_size= None, fix_indices = None, input_type='smiles', unique=True):
        mols, indices = self.model.input_smiles_base_sampling(smiles, n, e, max_size, fix_indices, input_type, unique)
        re_smiles = [Chem.MolToSmiles(x, canonical=False) for x in mols if x is not None]
        mols = [Chem.MolFromSmiles(x) for x in re_smiles]
        mols = [x for x in mols if x is not None]
        return mols, indices
