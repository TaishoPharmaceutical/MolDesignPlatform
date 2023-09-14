from rdkit import Chem
from tqdm import tqdm
import numpy as np
import os.path as op
import pickle
import random

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

atom_list = ["","C","N","[N+]","B","O","[O-]","P","S","F","Cl","Br","I", "*"]
ATOM_LEN = len(atom_list)
bond_types = [1.0, 0, 2.0, 3.0]
BOND_LEN = len(bond_types)

char_list= [
    "H","C","N","O","F","P","S","Cl","Br","I",
    "n","c","o","s",
    "1","2","3","4","5","6","7","8","9",
    "(",")","[","]",
    "-","=","#","/","\\","+","@","X"
]

char_len = len(char_list)

char_dict={
    'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'P': 5,
    'S': 6, 'Cl': 7, 'Br': 8, 'I': 9,
    'n': 10, 'c': 11, 'o': 12, 's': 13,
    '1': 14, '2': 15, '3': 16, '4': 17, '5': 18, '6': 19, '7': 20, '8': 21, '9':22,
    '(': 23, ')': 24, '[': 25, ']': 26, '-': 27, '=': 28, '#': 29,
    '/': 30, '\\': 31, '+': 32, '@': 33, 'X': 34
}

bType = [Chem.rdchem.BondType.SINGLE,
         Chem.rdchem.BondType.AROMATIC,
         Chem.rdchem.BondType.DOUBLE,
         Chem.rdchem.BondType.TRIPLE]

class data_reader():
    def __init__(self):
        return
    def read_csv(self,file_name,tasks,smiles,remove_salt=True,mode="classification",kekulize=False):
        if mode=="classification":
            df=pd.read_csv(fiIe_name).fi11na(2)
        else:
            df=pd.read_csv(fiIe_name).dropna()
        mols=[Chem.MolFromsmiles(x) for x in df[smiles]]
        tasks=[df[x] for x in tasks]
        tasks=list(map(list,zip(*tasks)))
        res_tasks=[]
        res_mols=[]
        remover=SaltRemover()
        for i in range(len(mols)):
            if mols[i]==None:
                print(df[smiles][i])
            else:
                res_tasks.append(tasks[i])
            if remove_salt:
                mols[i]=remover.StripMol(mols[i],dontRemoveEverything=True)
            if kekulize:
                Chem.Kekulize(mols[i])
            res_mols.append(mols[i])
        res_tasks=list(map(list,zip(*res_tasks)))
        print("Read file complete")
        return res_mols,res_tasks

    def read_csv_per_task(self,file_name,tasks,smiles,remove_salt=True,kekulize=False):
        df=pd.read_csv(file_name).fillna("#VALUE!")
        mols=[Chem.MolFromSmiles(x) for x in df[smiles]]
        tasks=[df[x] for x in tasks]
        tasks=list(map(list,zip(*tasks)))
        res_tasks=[]
        res_mols=[]
        remover=SaltRemover()
        for i in range(len(mols)):
            if mols[i]==None:
                print(df[smiles][i])
            else:
                res_tasks.append(tasks[i])

            if remove_salt:
                mols[i]=remover.StripMol(mols[i],dontRemoveEverything=True)
            if kekulize:
                Chem.Kekulize(mols[i])
            res_mols.append(mols[i])

        res_tasks=list(map(list,zip(*res_tasks)))
        ret_mols=[[] for x in range(len(res_tasks))]
        ret_y=[[] for x in range(len(res_tasks))]
        for i,mol in enumerate(res_mols):
            for j in range(len(res_tasks)):
                if res_tasks[j][i]!="#VALUE!":
                    cmol = ConvMol(mol)
                    ret_mols[j].append(cmol)
                    ret_y[j].append(float(res_tasks[j][i]))

        print("Read file complete")
        return ret_mols,ret_y

    def read_csvs(self,file_names,tasks,smiles,remove_salt=True,kekulize=False):
        mols=None
        ys=None
        for i,file_name in enumerate(file_names):
            if i==0:
                mols,ys=self.read_csv_per_task(file_name,[tasks[i]],smiles,remove_salt=remove_salt,kekulize=kekulize)
            else:
                x,y=self.read_csv_per_task(file_name,[tasks[i]],smiles,remove_salt=remove_salt,kekulize=kekulize)
                mols.extend(x)
                ys.extend(y)
        return mols,ys

class ConvMol_Sssr():
    """
    For featurization of mols
    conv_mol_featurize is featurizing process for Graphconvolution
    return:list of [features,adj_list,membership,deg_slice]
    These values is used in Graphconv,Graphpool,GraphGather class
    feature is consisted of 51 atomfeatures,so you need to use placeholder of [None,51] shape in first Graphoonvolution
    """
    def __init__(self,mol):
        mol_values = self.conv_mol(mol)
        self.features = mol_values[0]
        self.adj_list = mol_values[1]
        self.bond_type = mol_values[2]
        self.value_flag = mol_values[3]
        self.num_nodes = mol_values[4]
        self.adj_len = mol_values[5]
        self.sssr = mol_values[6]
        self.sssr_len= mol_values[7]
        self.sssr_adj= mol_values[8]
        self.sssr_bond= mol_values[9]
        self.adj_list_for_sssr= mol_values[10]
        self.bond_type_for_sssr = mol_values[11]
        self.sssr_flag = mol_values[12]
        self.adj_flag = mol_values[13]
        self.sssr_flag2 = mol_values[14]
        self.member = mol_values[15]
        self.member_adj_flag = mol_values[16]
        self.member_sssr_flag = mol_values[17]
        self.sssr_flag3 = mol_values[18]
        self.adj_flag2 = mol_values[19]

    def one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
        return list(map(lambda s: x==s, allowable_set))

    def one_of_k_encoding_unk(self, x , allowable_set):
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x ==s, allowable_set))

    def atom_encoding(self, x, smarts):
        allowable_set = ["","C","N","N+","B","O","O-","P","S","F","Cl","Br","I", "*"]
        if x not in allowable_set:
            x = allowable_set[-1]
        if x == "N" and "+" in smarts:
            x = allowable_set[3]
        if x == "O" and "-" in smarts:
            x = allowable_set[6]

        ret = list(map(lambda s: x ==s, allowable_set))

        return ret

    def atom_features(self, atom, sssr, bool_id_feat=False, explicit_H=False):
        if bool_id_feat:
            return np.array([self.atom_to_id(atom)])
        else:
            results=self.atom_encoding(
                atom.GetSymbol(), atom.GetSmarts()
            ) + self.one_of_k_encoding(atom.GetDegree(),
                                       [0,1,2,3,4,5,6,7,8]) +\
            self.one_of_k_encoding_unk(atom.GetImplicitValence(),
                                       [0,1,2,3,4,5,6]) +\
            [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]+\
            self.one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2])+\
            [atom.GetIsAromatic()]

            if not explicit_H:
                results = results + self.one_of_k_encoding_unk(atom.GetTotalNumHs(),[0,1,2,3,4])

            count=0
            if len(sssr)>0:
                atom_id = atom.GetIdx()
                for s in sssr:
                    if atom_id in sssr:
                        count+=1

            if count==0:
                count=1

            return np.array(results)/count

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
        sssr = Chem.GetSymmSSSR(mol)
        sssr_len = len(sssr)
        sssr = [sorted(list(s)) for s in sssr]
        sssr = sorted(sssr)

        idx_nodes=[(a.GetIdx(),self.atom_features(a,sssr)) for a in mol.GetAtoms()]
        atom_len=len(idx_nodes)
        idx_nodes.sort() #Sort by ind to ensure same order as rd_kit
        idx,nodes=list(zip(*idx_nodes))


        if sssr_len > 0:
            sssr_count = 0
            use_idx = []
            nodes_ = []

            for i in range(atom_len):
                if i in use_idx:
                    continue
                else:
                    use_idx.append(i)
                    nodes_.append(nodes[i])

                if sssr_count < sssr_len:
                    if i in sssr[sssr_count]:
                        for s in sssr[sssr_count]:
                            if s in use_idx: continue

                            use_idx.append(s)
                            nodes_.append(nodes[s])
                        sssr_count +=1
        else:
            use_idx = idx
            nodes_ = nodes

        idx = [use_idx, list(range(atom_len))]
        idx = list(map(list, zip(*idx)))

        d = {}
        for x in idx:
            d[x[0]] = x[1]

        for i in range(len(sssr)):
            for j in range(len(sssr[i])):
                sssr[i][j] = d[sssr[i][j]]

        #Stack nodes in to an array
        nodes=np.vstack(nodes_)
        num_nodes = len(nodes)

        #Get bond lists with reverse edges included
        edge_list=[(d[b.GetBeginAtomIdx()],d[b.GetEndAtomIdx()],self.get_bond_id(b.GetBondTypeAsDouble())) for b in mol.GetBonds()]

        #Get canonical adjacency list
        canon_bond_type = [[] for mol_id in range(num_nodes)]
        canon_adj_list=[[] for mol_id in range(num_nodes)]
        for edge in edge_list:
            canon_adj_list[edge[0]].append(edge[1]+1)
            canon_adj_list[edge[1]].append(edge[0]+1)
            canon_bond_type[edge[0]].append(edge[2])
            canon_bond_type[edge[1]].append(edge[2])

        adj_list=[]
        bond_type=[]
        adj_len=[]
        for i in range(len(nodes)):
            a = [0]*9
            b = [0]*9
            a[0:len(canon_adj_list[i])]=canon_adj_list[i]
            b[0:len(canon_adj_list[i])]=canon_bond_type[i]
            adj_list.append(a)
            bond_type.append(b)
            adj_len.append(len(canon_adj_list[i]))

        adj_list = np.array(adj_list)
        bond_type = np.array(bond_type)
        value_flag = adj_list > 0

        if sssr_len != 0:
            sssr_ = np.zeros([sssr_len, MAX_RING_SIZE], dtype=np.int32)
            for i in range(sssr_len):
                sssr_[i][0:len(sssr[i])] = np.array(sorted(sssr[i]), dtype=np.int32) +1
            sssr = sssr_

            sssr_adj = [[] for x in range(sssr_len)]
            sssr_bond = [[] for x in range(sssr_len)]
            adj_list_for_sssr = copy.deepcopy(canon_adj_list)
            bond_type_for_sssr = copy.deepcopy(canon_bond_type)
            for i, al in enumerate(canon_adj_list):
                x = np.where(sssr == i+1)
                for j,a in enumerate(al):
                    if a == 0: continue
                    b = np.where(sssr == a)
                    for l,k in enumerate(b[0]):
                        if len(x[0])!=0:
                            if k in list(x[0]): continue
                            sssr_adj[k].extend(list(num_nodes + x[0] +1))
                            sssr_bond[k].extend([bond_type[i][j]])
                        else:
                            sssr_adj[k].append(i+1)
                            sssr_bond[k].append(bond_type[i][j])

                        if l == 0:
                            adj_list_for_sssr[i][j] = num_nodes + k + 1
                        else:
                            adj_list_for_sssr[i].append(num_nodes+k+1)
                            bond_type_for_sssr.append(bond_type[i][j])

            sssr_adj_ = []
            sssr_bond_ = []
            adj_list_for_sssr_ = []
            bond_type_for_sssr_ = []
            for i, sa in enumerate(sssr_adj):
                a = [0]*MAX_NB
                b = [0]*MAX_NB
                a[0:len(sa)]=sa
                b[0:len(sssr_bond[i])] = sssr_bond[i]
                sssr_adj_.append(a)
                sssr_bond_.append(b)

            for i, adj in enumerate(adj_list_for_sssr):
                a = [0]*MAX_NB
                b = [0]*MAX_NB
                a[0:len(adj)]=adj
                b[0:len(bond_type_for_sssr[i])] = bond_type_for_sssr[i]
                adj_list_for_sssr_.append(a)
                bond_type_for_sssr_.append(b)

            sssr_adj_ = np.array(sssr_adj_, dtype = np.int32)
            sssr_bond_ = np.array(sssr_bond_, dtype = np.int32)
            adj_list_for_sssr_ = np.array(adj_list_for_sssr_, dtype = np.int32)
            bond_type_for_sssr_ = np.array(bond_type_for_sssr_, dtype = np.int32)

            use_idx = []
            member_ = list(set(range(1,num_nodes+1))-set(np.hstack([s for s in sssr])))
            sssr_member = list(range(num_nodes + 1, num_nodes+sssr_len+1))

            memberx = []

            for i in range(1, num_nodes+1):
                if i in member_:
                    use_idx.append(i)
                    memberx.append(i)
                else:
                    if i in use_idx: continue

                    x = np.where(sssr==i)
                    for j in x[0]:
                        use_idx.extend(sssr[j])
                        memberx.append(sssr_member[j])
            member_ = memberx
            sssr_flag2 = sssr>0
            sssr_flag3 = sssr_adj_ > num_nodes
            adj_flag2 = (sssr_adj_ > 0) & ~sssr_flag3

        else:
            sssr_adj_ = None
            sssr_bond_ = None
            adj_list_for_sssr_=[]
            bond_type_for_sssr_=[]
            for i in range(len(nodes)):
                a = [0] * MAX_NB
                b = [0] * MAX_NB
                a[0: len(canon_adj_list[i])] = canon_adj_list[i]
                b[0: len(canon_adj_list[i])] = canon_bond_type[i]
                adj_list_for_sssr_.append(a)
                bond_type_for_sssr_.append(b)
            adj_list_for_sssr_ = np.array(adj_list_for_sssr_, dtype = np.int32)
            bond_type_for_sssr_ = np.array(bond_type_for_sssr_, dtype = np.int32)
            sssr = None
            sssr_flag2 = None
            sssr_flag3 = None
            adj_flag2 = None

            member_ = range(1, num_nodes+1)

        sssr_flag = adj_list_for_sssr_ > num_nodes
        adj_flag = (adj_list_for_sssr_ > 0) & ~sssr_flag

        member = [0] * MAX_LEN
        member[0:len(member_)] = member_
        member = np.array(member, dtype=np.int32)

        member_sssr_flag = member > num_nodes
        member_adj_flag = (member>0) & ~member_sssr_flag

        return (nodes, adj_list, bond_type, value_flag, num_nodes, adj_len,
                sssr, sssr_len, sssr_adj_, sssr_bond_, adj_list_for_sssr_,
                bond_type_for_sssr_, sssr_flag, adj_flag, sssr_flag2, member,
                member_adj_flag, member_sssr_flag, sssr_flag3, adj_flag2)


def create_batch_sssr(cmols, max_len=40):

    nodes = []
    adj_list = []
    bond_type = []
    members = []
    num_atoms = 0

    sssr = []
    num_sssr = 0

    total_atoms = sum([cmol.num_nodes for cmol in cmols])
    omembers = np.zeros([len(cmols), max_len], dtype=np.int32)
    for i,cmol in enumerate(cmols):
        nodes.append(cmol.features)
        adj_list.append(cmol.adj_list+cmol.value_flag*num_atoms)

        bond_type.append(cmol.bond_type)

        member = cmol.member + cmol.member_adj_flag*num_atoms + cmol.member_sssr_flag*(total_atoms - cmol.num_nodes + num_sssr)
        members.append(member)
        omembers[i][:cmol.num_nodes] = np.arange(num_atoms+1, num_atoms + cmol.num_nodes+1, dtype=np.int32)

        if cmol.sssr_len > 0:
            sr = cmol.sssr+cmol.sssr_flag2*num_atoms
            sssr.append(sr)
            num_sssr += cmol.sssr_len
        else:
            sssr.append(np.zeros([1, MAX_RING_SIZE], dtype=np.int32))
            num_sssr += 1

        num_atoms += cmol.num_nodes

    nodes = np.vstack(nodes)
    adj_list = np.vstack(adj_list)
    bond_type = np.vstack(bond_type)
    members = np.vstack(members)
    sssr = np.vstack(sssr)

    adj_list_c = adj_list + bond_type*len(nodes)


    return (np.array(nodes, np.float32), np.array(adj_list_c, np.int32), np.array(members, np.int32),
            np.array(sssr, np.int32), omembers)


def create_batch(cmols, use_gru=False, max_len=None):

    nodes = []
    adj_list = []
    bond_type = []
    members = []
    num_atoms = 0
    for i,cmol in enumerate(cmols):
        nodes.append(cmol.features)
        adj_list.append(cmol.adj_list+cmol.value_flag*num_atoms)
        bond_type.append(cmol.bond_type)
        members.append([i]*cmol.num_nodes)
        num_atoms += cmol.num_nodes

    nodes = np.vstack(nodes)
    adj_list = np.vstack(adj_list)
    bond_type = np.vstack(bond_type)
    members = np.hstack(members)

    if use_gru:
        members = members_to_gather_list(list(members), max_len)

    adj_list_c = adj_list + bond_type*len(nodes)

    return (nodes, adj_list_c, members, adj_list)

def create_deg_batch(cmols, use_gru=False, max_len=None):

    aid = 1
    member = 0
    total_list=[]
    al=[]
    ai=[]
    node=[]
    adj=[]
    by=[]
    mb=[]
    for i,cmol in enumerate(cmols):
        adj_list_c = cmol.adj_list+cmol.value_flag*(aid-1)
        al.append(cmol.adj_len)
        ai.append(range(aid,aid+cmol.num_nodes))
        node.append(cmol.features)
        adj.append(adj_list_c)
        by.append(cmol.bond_type)
        mb.append([i]*cmol.num_nodes)
        aid+=cmol.num_nodes

    al = np.hstack(al)
    ai = np.hstack(ai)
    node = np.vstack(node)
    adj = np.vstack(adj)
    by = np.vstack(by)
    mb = np.hstack(mb)
    total_list =[list(al), list(ai), list(node), list(adj), list(by), list(mb)]
    total_list = list(zip(*total_list))

    total_list.sort()
    deg,aid,nodeses,adj_list,bond_type,membership=list(zip(*total_list))
    adj_list = np.array(adj_list).reshape(-1)
    aid = np.array(aid)

    adj_list = np.vectorize(lambda x: np.where(aid==x)[0]+1 if x!=0 else 0)(adj_list)
    adj_list = adj_list.reshape([-1,9])

    deg_slice=[]
    deg_max=max(deg)
    start=0
    for x in range(0,9):
        c=deg.count(x)
        deg_slice.append([[start,0],[c,0]])
        start+=c

    adj_list_b = np.array(adj_list)+np.array(bond_type)*len(nodeses)

    nodeses=np.vstack(nodeses)

    if use_gru:
        membership=members_to_gather_list(membership, max_len)

    return[nodeses,np.array(adj_list_b),np.array(membership),np.array(adj_list),np.array(deg_slice)]


def members_to_gather_list(members, max_len=None):
    len_mols = max(members)+1
    nm = np.array(members, dtype=np.int32)

    elsize = [members.count(x) for x in range(len_mols)]
    if max_len is None:
        max_size = max(elsize)
    else:
        max_size = max_len

    mg = [np.pad(np.where(nm==x)[0]+1, [0, max_size-elsize[x]], 'constant') for x in range(len_mols)]
    mg = np.vstack(mg)

    return mg


def cmol_to_onehotvec(cmol, max_len=100):
    atom_feats = np.concatenate([np.ones([max_len, 1]), np.zeros([max_len, ATOM_LEN-1])], -1)
    atom_feats[0:cmol.num_nodes] = cmol.features[:, 0:ATOM_LEN]

    bond_feats = np.concatenate([np.zeros([max_len, max_len, 1]),
                                np.ones([max_len, max_len,1]),
                                np.zeros([max_len, max_len, 2])], -1)

    for i, alit in enumerate(cmol.adj_list):
        for j, adj in enumerate(alit):
            if adj == 0:
                continue
            btype = cmol.bond_type[i][j]
            if adj-1<=i:
                bond_feats[i][adj-1]=np.identity(BOND_LEN)[btype]

    length = cmol.num_nodes
    length_one_hot = np.identity(max_len)[length-1]
    atom_feats = np.reshape(atom_feats, -1)
    bond_feats = np.reshape(bond_feats, -1)

    feat = np.hstack([atom_feats, bond_feats])

    atom_mask = make_atom_mask(length, max_len)
    bond_mask = make_bond_mask(length, max_len)

    return [feat, atom_mask, bond_mask, length]


def make_atom_mask(length, max_len=100):
    return np.tri(length, max_len, dtype=np.bool)[-1]


def make_bond_mask(length, max_len=100):
    mask = np.zeros([max_len, max_len], dtype=np.bool)
    mask[0:length]=np.tri(length, max_len, dtype = np.bool)
    return mask


def onehotvec_to_mol(onehot_vec, max_len, return_mol = True):
    length = onehot_vec[0:max_len]
    length = np.argmax(length)+1
    atom_feats = onehot_vec[max_len: max_len + max_len*ATOM_LEN]
    bond_feats = onehot_vec[max_len+max_len*ATOM_LEN:]

    atom_mask = make_atom_mask(length, max_len)
    bond_mask = make_bond_mask(length, max_len)

    atom_feats = np.reshape(atom_feats, [max_len, ATOM_LEN])
    atom_feats = np.argmax(atom_feats, -1)
    atom_feats = atom_feats[atom_mask]

    atom_symbols = [atom_list[x] for x in atom_feats]

    bond_feats = np.reshape(bond_feats, [max_len, max_len, BOND_LEN])
    bond_feats = np.argmax(bond_feats,-1)
    bfs = np.array(bond_types)[bond_feats]
    bfs = bfs * bond_mask
    bfs = bfs[:length, :length]
    bfs = bfs+bfs.T
    bfs = np.sum(bfs, -1)
    bond_feats = bond_feats[bond_mask]

    for i, a in enumerate(atom_symbols):
        if a == 'N':
            if bfs[i] >=4:
                atom_symbols[i] = '[%s+]'%atom_symbols[i]

    atom_mols = [Chem.MolFromSmarts(x) for i,x in enumerate(atom_symbols)]

    if atom_mols[0] is None:
        return None

    mol = atom_mols[0]
    if len(atom_mols) > 1:
        for x in atom_mols[1:]:
            if x is None:
                mol = Chem.CombineMols(mol, Chem.MolFromSmarts('*'))
            else:
                mol = Chem.CombineMols(mol, x)

    row_length = length
    row = 0
    column = -1
    count = 0
    emol = Chem.EditableMol(mol)
    a = sum(range(1, row+2))

    for i, x in enumerate(bond_feats):
        column+=1
        if i>=a:
            row+=1
            column=0
            a=sum(range(1,row+2))

        if x ==1:
            continue

        try:
            emol.AddBond(row, column, bType[x])
        except:
            x=0

    if return_mol:
        return emol.GetMol()
    else:
        return Chem.MolToSmiles(emol, GetMol())


def onehotvec_to_mol_no_length(onehot_vec, max_len, return_mol = True, force_connect = True):
    atom_feats = onehot_vec[:max_len*ATOM_LEN]
    atom_feats = np.reshape(atom_feats, [max_len, ATOM_LEN])
    atom_feats = np.argmax(atom_feats, -1)
    if 0 in list(atom_feats):
        length = np.where(atom_feats==0)[0][0]
    else:
        length = max_len

    bond_feats = onehot_vec[max_len*ATOM_LEN:]

    atom_mask = make_atom_mask(length, max_len)
    bond_mask = make_bond_mask(length, max_len)

    atom_feats = atom_feats[atom_mask]

    atom_symbols = [atom_list[x] for x in atom_feats]

    bond_feats = np.reshape(bond_feats, [max_len, max_len, BOND_LEN])
    bond_feats_ = np.argmax(bond_feats,-1)
    bfs = np.array(bond_types)[bond_feats_]
    bfs = bfs * bond_mask
    bfs = bfs[:length, :length]
    bfs = bfs+bfs.T
    bfs = np.sum(bfs, -1)
    bond_feats_ = bond_feats_[bond_mask]
    bond_feats = bond_feats[bond_mask]

    for i, a in enumerate(atom_symbols):
        if a == 'N':
            if bfs[i] >=4:
                atom_symbols[i] = '[%s+]'%atom_symbols[i]

    atom_mols = [Chem.MolFromSmarts(x) for i,x in enumerate(atom_symbols)]

    if atom_mols[0] is None:
        return None

    mol = atom_mols[0]
    if len(atom_mols) > 1:
        for x in atom_mols[1:]:
            if x is None:
                mol = Chem.CombineMols(mol, Chem.MolFromSmarts('*'))
            else:
                mol = Chem.CombineMols(mol, x)

    row_length = length
    row = 0
    column = -1
    count = 0
    emol = Chem.EditableMol(mol)
    a = sum(range(1, row+2))

    for i, x in enumerate(bond_feats_):
        column+=1
        if i>=a:
            row+=1
            column=0
            a=sum(range(1,row+2))
            if force_connect:
                if sum(bond_feats_[i:a]==1) == a-i:
                    bf = bond_feats[i:a]
                    bf = bf*np.array([1.,0.,1.,1.])
                    bfw = np.where(bf==np.max(bf))
                    target = bfw[0][0]
                    bt = bfw[1][0]
                    emol.AddBond(row, int(target), bType[int(bt)])

        if x ==1:
            continue

        try:
            emol.AddBond(row, column, bType[x])
        except:
            x=0

    if return_mol:
        return emol.GetMol()
    else:
        return Chem.MolToSmiles(emol, GetMol())


def smiles_to_hot(smiles, length=120):
    nxt = False
    hot_smiles = []
    for i in range(len(smiles)):
        if nxt:
            nxt = False
            continue

        if smiles[i:i+2] in char_list:
            nxt = True
            hot_smiles.append(char_dict[smiles[i:i+2]])
        elif smiles[i] in char_list:
            hot_smiles.append(char_dict[smiles[i]])
        else:
            print('error')
            return None

    len_smiles = len(hot_smiles)
    if len_smiles < length:
        len_smiles+=1
    smiles_mask = [1]*len_smiles + [0]*(length-len_smiles)
    hot_smiles += [33]*(length-len(hot_smiles))

    return np.array(hot_smiles, dtype=np.int32), np.array(smiles_mask, dtype=np.int32)


def hot_to_smiles(hot_smiles):
    smiles = ''
    for i in hot_smiles:
        if i == 33 or i == 34:
            break
        smiles += char_list[i]
    return smiles


def read_file(file_name, column_name, remove_salt=True, rand_root=False):
    df = pd.read_csv(file_name)
    mols = [Chem.MolFromSmiles(x) for x in df[column_name]]

    if remove_salt:
        remover=SaltRemover()
        mols = [remover.StripMol(mol, dontRemoveEverything=True) for mol in mols]

    if rand_root == False:
        smiles = [Chem.MolToSmiles(x, isomericSmiles=False) for x in mols]
    else:
        atom_counts = [len(mol.GetAtoms()) for mol in mols]
        root_atoms = [np.random.randint(atom_count) for atom_count in atom_counts]
        smiles = [Chem.MolToSmiles(mols[i], rootedAtAtom=root_atoms[i], canonical=False, isomericSmiles=False) for i in range(len(mols))]

    smiles = [x for x in smiles if len(x) < SMILES_LENGTH]

    return smiles


def batch_gen_gcn_mol_feat(file_name, column_name, max_len, batch_size=20, use_pickle = False, save_pickle_file=None, test=True):
    if use_pickle:
        with open(file_name, 'rb') as pf:
            m = pickle.load(pf)
    else:
        m = read_file(file_name, column_name)
        m = [Chem.MolFromSmiles(x) for x in m]
        m = [x for x in m if x.GetNumAtoms()<MAX_LEN-1]

        for i in range(len(m)):
            Chem.Kekulize(m[i])

        m_=[]
        for x in m:
            try:
                m_.append(ConvMol_Sssr(x))
            except:
                continue

        m=m_

        if save_pickle_file is not None:
            with open(save_pickle_file, 'wb') as pf:
                pickle.dump(m, pf)

    random.shuffle(m)

    n = len(m)
    start = 0
    end=batch_size

    flag = True

    while flag:
        if start>=n:
            start = 0
            end = batch_size
            random.shuffle(m)
            batch=m[start:end]

        elif end>n:
            batch = m[start:n]
            random.shuffle(m)

            if test:
                flag = False
            else:
                batch = batch + m[:end-n]
                start = end-n
                end = start + batch_size

        else:
            batch = m[start:end]
            start+=batch_size
            end+=batch_size

        gcn_inputs = create_batch_sssr(batch, max_len=max_len)
        mf_batch = [cmol_to_onehotvec(x, max_len) for x in batch]
        mf_batch = list(map(list, zip(*mf_batch)))

        yield (gcn_inputs[0],gcn_inputs[1],gcn_inputs[2],gcn_inputs[3], gcn_inputs[4],
               mf_batch[0],mf_batch[1],mf_batch[2],mf_batch[3])


def batch_gen_gcn_mol_feat_inf(file_name, column_name, max_len, inflation =1, batch_size=20, use_pickle = False, save_pickle_file=None, test=False):
    if use_pickle:
        with open(file_name+'/m-0.pkl', 'rb') as pf:
            cm = pickle.load(pf)
        with open(file_name+'/m-0.pkl', 'rb') as pf:
            cm_ = pickle.load(pf)
    else:
        m = read_file(file_name, column_name)
        m = [Chem.MolFromSmiles(x) for x in m]

        for i in range(len(m)):
            Chem.Kekulize(m[i])

        cm = [ConvMol_Sssr(x) for x in m]
        if save_pickle_file is not None:
            with open(save_pickle_file+'/m-0.pkl', 'wb') as pf:
                pickle.dump(cm, pf)
                print("finish pkl-0")

        for j in range(inflation-1):
            m_ = [Chem.MolFromSmiles(Chem.MolToSmiles(x, doRandom=True)) for x in m]
            for i in range(len(m_)):
                Chem.Kekulize(m_[i])
            cm_ = [ConvMol_Sssr(x) for x in m_]
            if save_pickle_file is not None:
                with open(save_pickle_file+'/m-%d.pkl'%(j+1), 'wb') as pf:
                    pickle.dump(cm_, pf)
                    print("finish pkl-%d"%(j+1))

        file_name = save_pickle_file

    x = [cm_, cm]
    m = list(map(list, zip(*x)))

    random.shuffle(m)

    n = len(m)
    start = 0
    end=batch_size
    flag = True
    count = 0

    while flag:
        if start>=n:
            start = 0
            end = batch_size
            random.shuffle(m)
            batch=m[start:end]

        elif end>n:
            batch = m[start:n]
            start = 0
            end = batch_size

            count +=1
            if count > inflation-1:
                count = 0
                if test:
                    flag = False

            with open(file_name+'/m-%d.pkl'%count, 'rb') as pf:
                cm_ = pickle.load(pf)

            x = [cm_, cm]
            m = list(map(list, zip(*x)))
            random.shuffle(m)

        else:
            batch = m[start:end]
            start+=batch_size
            end+=batch_size

        batch = list(map(list, zip(*batch)))
        gcn_inputs = create_batch_sssr(batch[0], max_len=max_len)
        mf_batch = [cmol_to_onehotvec(x, max_len) for x in batch[1]]
        mf_batch = list(map(list, zip(*mf_batch)))

        yield (gcn_inputs[0],gcn_inputs[1],gcn_inputs[2],gcn_inputs[3],
               gcn_inputs[4],gcn_inputs[5],gcn_inputs[6],gcn_inputs[7],
               mf_batch[0],mf_batch[1],mf_batch[2])


def batch_gen_gcn_smiles_feat(file_name, column_name, max_len, inflation =1, batch_size=20, use_pickle = False, save_pickle_file=None, test=False):
    if use_pickle:
        with open(file_name+'/m-0.pkl', 'rb') as pf:
            cm = pickle.load(pf)
        with open(file_name+'/smiles.pkl', 'rb') as pf:
            s = pickle.load(pf)

    else:
        m = read_file(file_name, column_name)
        s = [smiles_to_hot(x, length) for x in m]

        s_ = [[m[i], x] for i,x in enumerate(s) if x is not None]
        m, s = list(map(list, zip(*s_)))

        with open(save_pickle_file+'/smiles.pkl', 'wb') as pf:
            pickle.dump(s, pf)
            print("finish smiles")

        m = [Chem.MolFromSmiles(x) for x in m]
        for i in range(len(m)):
            Chem.Kekulize(m[i])

        cm = [ConvMol_Sssr(x) for x in m]
        if save_pickle_file is not None:
            with open(save_pickle_file+'/m-0.pkl', 'wb') as pf:
                pickle.dump(cm, pf)
                print("finish pkl-0")

        for j in range(inflation-1):
            m_ = [Chem.MolFromSmiles(Chem.MolToSmiles(x, doRandom=True)) for x in m]
            for i in range(len(m_)):
                Chem.Kekulize(m_[i])
            cm_ = [ConvMol_Sssr(x) for x in m_]
            if save_pickle_file is not None:
                with open(save_pickle_file+'/m-%d.pkl'%(j+1), 'wb') as pf:
                    pickle.dump(cm_, pf)
                    print("finish pkl-%d"%(j+1))

        file_name = save_pickle_file

    x = [cm, s]
    m = list(map(list, zip(*x)))

    random.shuffle(m)

    n = len(m)
    start = 0
    end=batch_size
    flag = True
    count = 0

    while flag:
        if start>=n:
            start = 0
            end = batch_size
            random.shuffle(m)
            batch=m[start:end]

        elif end>n:
            batch = m[start:n]
            start = 0
            end = batch_size

            if inflation>1:
                count +=1
                if count > inflation-1:
                    count = 0
                    if test:
                        flag = False

                with open(file_name+'/m-%d.pkl'%count, 'rb') as pf:
                    cm = pickle.load(pf)
                    x = [cm, s]
                    m = list(map(list, zip(*x)))
            else:
                if test:
                    flag = False

            random.shuffle(m)

        else:
            batch = m[start:end]
            start+=batch_size
            end+=batch_size

        batch = list(map(list, zip(*batch)))
        gcn_inputs = create_batch_sssr(batch[0], max_len=max_len)
        mf_batch = list(map(list, zip(*batch[1])))
        hot = np.vstack(mf_batch[0])
        mask = np.vstack(mf_batch[1])

        yield (gcn_inputs[0],gcn_inputs[1],gcn_inputs[2],gcn_inputs[3],
               gcn_inputs[4],gcn_inputs[5],gcn_inputs[6],gcn_inputs[7],
               hot,mask)


def evaluate_vec(y_true, y_pred, atom_mask, bond_mask, max_len):
    true_atom_feat = y_true[:, :max_len*ATOM_LEN]
    true_bond_feat = y_true[:, max_len*ATOM_LEN:]
    true_atom_feat = np.reshape(true_atom_feat, [-1, max_len, ATOM_LEN])
    true_bond_feat = np.reshape(true_bond_feat, [-1, max_len, max_len, BOND_LEN])

    pred_atom_feat = y_pred[:, :max_len*ATOM_LEN]
    pred_bond_feat = y_pred[:, max_len*ATOM_LEN:]
    pred_atom_feat = np.reshape(pred_atom_feat, [-1, max_len, ATOM_LEN])
    pred_bond_feat = np.reshape(pred_bond_feat, [-1, max_len, max_len, BOND_LEN])

    match = 0
    match_length = 0
    match_atom = 0
    match_bond = 0
    for i in range(len(true_atom_feat)):
        ta = true_atom_feat[i][atom_mask[i]]
        tb = true_bond_feat[i][bond_mask[i]]
        ta = np.reshape(ta, -1)
        tb = np.reshape(tb, -1)
        t = np.hstack([ta, tb])

        pa = pred_atom_feat[i][atom_mask[i]]
        pb = pred_bond_feat[i][bond_mask[i]]

        pa = np.reshape(pa,-1)
        pb=np.reshape(pb,-1)
        p = np.hstack([pa, pb])

        if np.sum(np.abs((t-p)))==0:
            match+=1

        if np.sum(np.abs((ta-pa)))==0:
            match_atom+=1

        if np.sum(np.abs((tb-pb)))==0:
            match_bond+=1

    total = len(true_atom_feat)
    return ("match: %d/%d, atom match: %d, bond match: %d"%(match,
                                                             total,
                                                              match_atom,
                                                             match_bond))
