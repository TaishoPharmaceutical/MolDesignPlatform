from rdkit import Chem
from rdkit.Chem import Descriptors, MolStandardize
import pandas as pd

class Filter:
    def __init__(self, file):
        self.filterStructure = pd.read_csv(file)

    def setLevel(self, level):
        useFlags = self.filterStructure[level].astype(bool)
        self.filtSmarts = self.filterStructure[useFlags]["SMILES/SMARTS"]
        self.avoidMols = [Chem.MolFromSmarts(s) for s in self.filtSmarts]

    def isUnstable(self, mol):
        returnFlag = False

        for f in self.avoidMols:
            returnFlag = returnFlag | mol.HasSubstructMatch(f)

        return returnFlag

    def getFilteredMols(self, mols):
        uc = MolStandardize.charge.Uncharger()

        mols = [uc.uncharge(x) for x in mols]
        mols = [x for x in mols if Chem.rdmolops.GetFormalCharge(x) == 0]
        mols = [x for x in mols if not self.isUnstable(x)]

        return mols
