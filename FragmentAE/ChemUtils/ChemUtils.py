import pandas as pd
from rdkit import Chem


def replace_canonical_smiles(csv, out_file_name = None, smiles ="SMILES"):
    '''
       convert smiles to canonical smiles of rdkit in csv file
    '''
    df = pd.read_csv(csv)

    cano_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x in df[smiles]]
    df[smiles] = cano_smiles

    if out_file_name is None:
        df.to_csv(csv)
    else:
        df.to_csv(out_file_name)


def FragmentConnector(fragments):
    if len(fragments) ==1:
        cmol = fragments[0]
    elif len(fragments) > 1:
        cmol = fragments[0]
        for i in range(1,len(fragments)):
            cmol = Chem.CombineMols(cmol, fragments[i])
    else:
        print("No fragments")
        return None

    dumy_symbol = []
    for atom in cmol.GetAtoms():
        s = atom.GetSmarts()
        if '*' in s:
            dumy_symbol.append(s)

    dumy_symbol = list(set(dumy_symbol))

    connection_list = [[] for x in range(len(dumy_symbol))]
    dumy_atom_idx = [[] for x in range(len(dumy_symbol))]
    bond_type = [Chem.rdchem.BondType.SINGLE for x in range(len(dumy_symbol))]

    for bond in cmol.GetBonds():
        b = bond.GetBeginAtom()
        e = bond.GetEndAtom()

        if b.GetSmarts() in dumy_symbol:
            idx = dumy_symbol.index(b.GetSmarts())
            connection_list[idx].append(e.GetIdx())
            dumy_atom_idx[idx].append(b.GetIdx())
            bond_type[idx] = bond.GetBondType()

        if e.GetSmarts() in dumy_symbol:
            idx = dumy_symbol.index(e.GetSmarts())
            connection_list[idx].append(b.GetIdx())
            dumy_atom_idx[idx].append(e.GetIdx())
            bond_type[idx] = bond.GetBondType()

    emol = Chem.EditableMol(cmol)
    for i,c in enumerate(connection_list):
        if len(c) != 2:
            continue

        emol.AddBond(c[0], c[1], bond_type[i])

    remove_atom_idx = []
    for a in dumy_atom_idx:
        if len(a) ==2:
            remove_atom_idx.extend(a)

    remove_atom_idx.sort(reverse=True)

    for d in remove_atom_idx:
        emol.RemoveAtom(d)

    return emol.GetMol()

