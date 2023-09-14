from rdkit import Chem
from rdkit.Chem import AllChem

def get_border_atomIdx(mol, core_ids, subst_ids):
    border_subst_ids = []
    border_core_ids = []
    for b in mol.GetBonds():
        if b.GetBeginAtomIdx() in core_ids:
            e = b.GetEndAtomIdx()
            a = b.GetBeginAtomIdx()
        elif b.GetEndAtomIdx() in core_ids:
            e = b.GetBeginAtomIdx()
            a = b.GetEndAtomIdx()
        else:
            e=-1

        if e in subst_ids:
            border_subst_ids.append(e)
            border_core_ids.append(a)

    return border_core_ids, border_subst_ids

def get_parts(smiles, core_ids):
    mol = Chem.MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()

    subst_ids = set(list(range(num_atoms)))-set(core_ids)
    subst_ids = sorted(subst_ids, reverse=True)

    border_core_ids, border_subst_ids = get_border_atomIdx(mol, core_ids, subst_ids)

    emol = Chem.EditableMol(mol)
    for i,idx in enumerate(border_core_ids):
        emol.AddAtom(Chem.Atom(0))
        emol.AddBond(idx, num_atoms+i, Chem.rdchem.BondType.SINGLE)
    for i in subst_ids:
        emol.RemoveAtom(i)
    core = emol.GetMol()

    emol = Chem.EditableMol(mol)
    for i,idx in enumerate(border_subst_ids):
        emol.AddAtom(Chem.Atom(0))
        emol.AddBond(idx, num_atoms+i, Chem.rdchem.BondType.SINGLE)
    core_ids = sorted(core_ids, reverse=True)
    for i in core_ids:
        emol.RemoveAtom(i)
    subst= emol.GetMol()

    try:
        core_smiles = "invalid"
        core_smiles = Chem.MolToSmiles(core)
        core_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(core)))
    except:
        print(core_smiles)

    try:
        subst_smiles = "invalid"
        subst_smiles = Chem.MolToSmiles(subst)
        subst_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(subst)))
    except:
        print(subst_smiles)

    return core_smiles, subst_smiles
