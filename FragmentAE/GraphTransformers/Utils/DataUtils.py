import numpy as np
import pandas as pd

from rdkit import Chem
from tqdm import tqdm


MAX_LEN=70

def encode_to_array(cmols, max_len=MAX_LEN):
    num_mols = len(cmols)

    feats = np.zeros([num_mols, max_len, cmols[0].features.shape[-1]], dtype=np.float32)
    adj_lists = np.zeros([num_mols, max_len, max_len], dtype=np.float32)
    feat_lens = np.zeros([num_mols, max_len], dtype=np.float32)

    for i in range(num_mols):
        feat_len=len(cmols[i].features)

        feats[i][:feat_len, :] = cmols[i].features
        adj_lists[i][:feat_len, :feat_len] = cmols[i].adj_list
        feat_lens[i][:feat_len]=np.ones([feat_len])

    return [np.array(feats, dtype=np.float32), np.array(adj_lists, dtype=np.float32), np.array(feat_lens, dtype=np.int32)]


class FeatMol():
    """
    For featurization of mols
    conv_mol_featurize is featurizing process for Graphconvolution
    return:list of [features,adj_list,membership,deg_slice]
    These values is used in Graphconv,Graphpool,GraphGather class
    feature is consisted of 51 atomfeatures,so you need to use placeholder of [None,51] shape in first Graphoonvolution
    """
    def __init__(self,mol):
        cmol = self.conv_mol(mol)
        self.features = cmol[0]
        self.adj_list=cmol[1]

    def one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            raise Exception(
                    "input {0} not in allowable set{1}:".format(x, allowable_set))
        return list(map(lambda s: x==s, allowable_set))

    def one_of_k_encoding_unk(self, x , allowable_set):
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x ==s, allowable_set))

    def atom_encoding(self, x):
        allowable_set = ["","C","N","B","O","P","S","F","Cl","Br","I", "*"]
        if x not in allowable_set:
            x = allowable_set[-1]
        ret = list(map(lambda s: x ==s, allowable_set))

        return ret

    def atom_features(self, atom):
        results=self.atom_encoding(
            atom.GetSymbol()
        ) + self.one_of_k_encoding(atom.GetDegree(),
                                       [0,1,2,3,4,5,6,7,8]) + \
        self.one_of_k_encoding_unk(atom.GetImplicitValence(),
                                       [0,1,2,3,4,5,6]) +\
        [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]+\
        self.one_of_k_encoding_unk(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2])+\
        [atom.GetIsAromatic()] +\
        self.one_of_k_encoding_unk(atom.GetTotalNumHs(),[0,1,2,3,4])

        return np.array(results)

    def atom_to_id(self, atom):
        features = get_feature_list(atom)
        return features_to_id(features, intervals)

    def get_bond_id(self,bondtype):
        bond_types=[1.0,1.5,2.0,3.0]
        bid=4
        if bondtype in bond_types:
            bid=bond_types.index(bondtype)
        return bid

    def conv_mol(self, mol):
        idx_nodes=[(a.GetIdx(),self.atom_features(a)) for a in mol.GetAtoms()]
        atom_len=len(idx_nodes)
        idx_nodes.sort() #Sort by ind to ensure same order as rd_kit
        idx,nodes=list(zip(*idx_nodes))

        adj_matrix = Chem.GetAdjacencyMatrix(mol)

        return [np.array(nodes, dtype=np.float32), np.array(list(adj_matrix), dtype=np.float32)]


class FeatMol_Scaf():
    """
    For featurization of mols
    conv_mol_featurize is featurizing process for Graphconvolution
    return:list of [features,adj_list,membership,deg_slice]
    These values is used in Graphconv,Graphpool,GraphGather class
    feature is consisted of 51 atomfeatures,so you need to use placeholder of [None,51] shape in first Graphoonvolution
    """
    def __init__(self,mol):
        cmol = self.conv_mol(mol)
        self.features = cmol[0]
        self.adj_list=cmol[1]

    def one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            raise Exception(
                    "input {0} not in allowable set{1}:".format(x, allowable_set))
        return list(map(lambda s: x==s, allowable_set))

    def one_of_k_encoding_unk(self, x , allowable_set):
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x ==s, allowable_set))

    def atom_encoding(self, x):
        if x == "*":
            return [0,1]
        else:
            return [1,0]

    def atom_features(self, atom):
        results=self.atom_encoding(
            atom.GetSymbol()
        ) + self.one_of_k_encoding(atom.GetDegree(),
                                       [0,1,2,3,4,5,6,7,8]) + \
        self.one_of_k_encoding_unk(atom.GetImplicitValence(),
                                       [0,1,2,3,4,5,6]) +\
        [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]+\
        self.one_of_k_encoding_unk(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2])+\
        [atom.GetIsAromatic()] +\
        self.one_of_k_encoding_unk(atom.GetTotalNumHs(),[0,1,2,3,4])

        return np.array(results)

    def atom_to_id(self, atom):
        features = get_feature_list(atom)
        return features_to_id(features, intervals)

    def get_bond_id(self,bondtype):
        bond_types=[1.0,1.5,2.0,3.0]
        bid=4
        if bondtype in bond_types:
            bid=bond_types.index(bondtype)
        return bid

    def conv_mol(self, mol):
        idx_nodes=[(a.GetIdx(),self.atom_features(a)) for a in mol.GetAtoms()]
        atom_len=len(idx_nodes)
        idx_nodes.sort() #Sort by ind to ensure same order as rd_kit
        idx,nodes=list(zip(*idx_nodes))

        adj_matrix = Chem.GetAdjacencyMatrix(mol)

        return [np.array(nodes, dtype=np.float32), np.array(list(adj_matrix), dtype=np.float32)]


def read_csv(file_name, tasks, smiles, max_len=MAX_LEN, save_file=None):
    df=pd.read_csv(file_name)
    #tasks = reg_tasks + class_tasks
    nans = [~np.isnan(df[t]) for t in tasks]
    na = nans[0]
    if len(tasks) > 1:
        for i in range(1, len(nans)):
            na = na | nans[i]

    df = df[na]
    df.fillna(np.nan)

    mols=[Chem.MolFromSmiles(x) for x in df[smiles]]
    labels=[df[x] for x in tasks]
    labels=list(map(list,zip(*labels)))

    res_labels=[]
    res_mols=[]
    rmols = []
    for i in tqdm(range(len(mols))):
        if mols[i] is None:
            #print("cannot convert %s to mol:"%df[smiles][i])
            continue

        if mols[i].GetNumAtoms()>MAX_LEN:
            continue

        try:
            res_mols.append(FeatMol(mols[i]))
            res_labels.append(labels[i])
            rmols.append(mols[i])
        except:
            continue

    res_mols = encode_to_array(res_mols, max_len)
    res_labels=list(map(list,zip(*res_labels)))
    res_labels = [np.array(x, dtype=np.float32) for x in res_labels]
    res = res_mols + res_labels

    if save_file is not None:
        df2 = pd.DataFrame()
        df2["SMILES"]=[Chem.MolToSmiles(x) for x in rmols]

        for i,x in enumerate(tasks):
            df2[x] = res_labels[i]

        df2.to_csv(save_file, index=False)


    print("read file complete")
    print("total: %d smiles"%len(res_mols[0]))

    return res
