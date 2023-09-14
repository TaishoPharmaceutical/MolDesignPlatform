import os
import base64
import hashlib

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from io import BytesIO

import json
import requests
from constants import FILTER_API_URL

ntol_dict = {
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}

lton_dict = {
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}

def isValidSmiles(smi):
    if Chem.MolFromSmiles(smi) is not None:
        return True
    return False


def getValidSmiles(smiles):
    if smiles is None or len(smiles)==0:
        return []

    valid_mols = [Chem.MolFromSmiles(x) for x in smiles if x != ""]
    valid_smiles = [Chem.MolToSmiles(x) for x in valid_mols if x is not None]


    if len(valid_smiles)==0:
        return []

    return list(valid_smiles)


def getImgBase64FromMol(smi):
    m = Chem.MolFromSmiles(smi)
    smi_hash = hashlib.sha256(smi.encode()).hexdigest()

    img = Draw.MolsToGridImage([m], molsPerRow=1, subImgSize=(200,200), returnPNG=False)
    img.save(f"temp/{smi_hash}.png")
    with open(f"temp/{smi_hash}.png", "rb") as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')

    os.remove(f"temp/{smi_hash}.png")
    return b64


def getSVGFromMol(smi):
    m = Chem.MolFromSmiles(smi)
    dc = Draw.rdMolDraw2D.MolDraw2DSVG(200,150)

    drawOpt = dc.drawOptions()
    #drawOpt.clearBackground=False

    dc.DrawMolecule(m)
    dc.FinishDrawing()
    encoded = base64.b64encode(dc.GetDrawingText().encode())

    return 'data:image/svg+xml;base64,{}'.format(encoded.decode())


def getWeightImgBase64(smi, prop_name, prop_weight):

    m = Chem.MolFromSmiles(smi)
    smi_hash = hashlib.sha256(smi.encode()).hexdigest()
    img = SimilarityMaps.GetSimilarityMapFromWeights(m, prop_weight, size=[112,112], returnPNG=False)

    img.savefig(f"temp/{prop_name}_{smi_hash}.png", bbox_inches='tight')
    with open(f"temp/{prop_name}_{smi_hash}.png", "rb") as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')

    os.remove(f"temp/{prop_name}_{smi_hash}.png")
    return b64

# from chainer-chemistry's visualizer_utils.py
def red_blue_cmap(x):
    """Red to Blue color map
    Args:
        x (float): value between -1 ~ 1, represents normalized saliency score
    Returns (tuple): tuple of 3 float values representing R, G, B.
    """
    if x > 0:
        # Red for positive value
        # x=0 -> 1, 1, 1 (white)
        # x=1 -> 1, 0, 0 (red)
        return 1.0, 1.0 - x/2, 1.0 - x/2
    else:
        # Blue for negative value
        x *= -1
        return 1.0 - x/2, 1.0 - x/2, 1.0


def getWeightImg(smi, prop_weight):
    import requests
    import numpy as np
    from rdkit.Chem.Draw import rdMolDraw2D

    m = Chem.MolFromSmiles(smi)
    atom_num = len(["" for a in m.GetAtoms()])

    weight_np = np.asarray(prop_weight)
    weight_np = weight_np[:atom_num]

    abs_max = max(np.abs(weight_np))
    if abs_max < 1:
        scaled_weight_np = weight_np
    else:
        scaled_weight_np = weight_np/ abs_max

    drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
    atom_color_dict = {i: red_blue_cmap(v) for i, v in enumerate(scaled_weight_np)}

    drawer.DrawMolecule(
        m,
        highlightAtoms=[i for i in range(len(weight_np))],
        highlightAtomColors=atom_color_dict,
        highlightAtomRadii={i: 0.5 for i in range(len(weight_np))},
        highlightBonds=[],
    )
    drawer.FinishDrawing()
    _img = drawer.GetDrawingText()
    b64 = base64.encodebytes(_img).decode()

    return 'data:image/png;base64,{}'.format(b64)


def getKekuleSmiles(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, kekuleSmiles=True)


def getMolCoordinateInfoFromSmiles2(smi):
    mol = Chem.MolFromSmiles(smi)
    Draw.rdMolDraw2D.PrepareMolForDrawing(mol)

    molb = Chem.MolToMolBlock(mol)

    #一桁の数値が先に変換されてしまうのを防ぐため、逆順で処理する必要ある。
    for l, n in list(lton_dict.items())[::-1]:
        molb = molb.replace(l, str(n))

    molb_list = molb.split("\n")[4:-2]

    atoms = molb_list[:len(mol.GetAtoms())]
    bonds = molb_list[len(mol.GetAtoms()):]

    return atoms, bonds


def ntol(num):
    #atom number to label
    return ntol_dict[num]

def lton(label):
    #label to atom number
    return lton_dict[label]


def get_filtered_smiles(smiles, level="rough"):
    r = requests.post(FILTER_API_URL, proxies={"http": None},
                      json={"smiles": smiles, "level": level,}, verify=False)

    return r.json()
