

import pandas as pd
import numpy as np
import random
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from rdkit import Chem
import sys
from deepchem.feat.smiles_tokenizer import SmilesTokenizer

import sys
from io import StringIO
import math
import os
import pickle
import json
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity

_fscores = None



# Output SMILES from the tokens
def extract_smiles(token):
    split_token = token.split(" ")
    if "START" not in split_token:
        return ""
    else:
        start = split_token.index("START")
    if "END" not in split_token:
        return ""
    else:
        end = split_token.index("END")

    valid_token = split_token[start + 1: end]
    smiles = "".join(valid_token)

    return smiles

# Calculate novelty
def check_novelty(df_generated, df_train, column_name):
    # compare the generated polymers with the train set
    for i in df_generated[column_name]:
        if df_train[column_name].eq(i).any():
            df_generated.loc[df_generated[column_name] == i, 'diversity'] = 'In the original data set'
        else:
            df_generated.loc[df_generated[column_name] == i, 'diversity'] = 'novel'
    return df_generated



# Check whether the polymer's validity
def validate_mol(mol_list, column_name):

    sio = sys.stderr = StringIO()
    for i in mol_list['mol_smiles']:

        if pd.isna(i):
            mol_list.loc[mol_list[column_name] == i, 'validity']  = "none"
        elif Chem.MolFromSmiles(i) is None:
            mol_list.loc[mol_list[column_name] == i, 'validity']  = sio.getvalue().strip()[11:]
            sio = sys.stderr = StringIO() # reset the error logger
        elif ('=[Cu]' in i) or ('[Cu]=' in i) or ('=[Au]' in i) or ('[Au]=' in i):
            mol_list.loc[mol_list[column_name] == i, 'validity']  = 'Double bond at the end point'
        elif ('#[Cu]' in i) or ('[Cu]#' in i) or ('#[Au]' in i) or ('[Au]#' in i):
            mol_list.loc[mol_list[column_name] == i, 'validity']  = 'Triple bond at the end point'
        elif (i.count("[Cu]") != 1) or (i.count("[Au]") != 1):
            mol_list.loc[mol_list[column_name] == i, 'validity']  = 'More than two ends'
        else:
            bond_flag = False
            for atom in Chem.MolFromSmiles(i).GetAtoms():
                if atom.GetSymbol() == "Cu":
                    if atom.GetDegree() > 1:
                        bond_flag = True
                elif atom.GetSymbol() == "Au":
                    if atom.GetDegree() > 1:
                        bond_flag = True
            if bond_flag:
                mol_list.loc[mol_list[column_name] == i, 'validity']  = 'More than one bonds at the end point'
            else:
                mol_list.loc[mol_list[column_name] == i, 'validity'] = 'ok'
    return mol_list


def has_two_ends(df):
    for mol in df['mol_smiles']:
        if pd.isna(mol):
            continue
        elif (mol.count('[Cu]') == 1 and mol.count('[Au]') == 1):
            df.loc[df['mol_smiles'] == mol, 'has_two_ends'] = True
        else:
            df.loc[df['mol_smiles'] == mol, 'has_two_ends'] = False
    return df


def readFragmentScores(name='fpscores'):
    global _fscores

    # Generate the full path filename
    if name == "fpscores":
        json_file_path = os.path.join(os.path.dirname(__file__), "fpscores.json")
        
        # Load JSON data
        with open(json_file_path, 'r') as f:
            data = json.load(f)

    outDict = {}
    for i in data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict


def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro


def calculateScore(m):
    if _fscores is None:
        readFragmentScores()

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(m,
                                             2)  # <- 2 is the *radius* of the circular fingerprint
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0

    return sascore


def processMols(mols):
    print('smiles\tName\tsa_score')
    for i, m in enumerate(mols):
        if m is None:
            continue

    s = calculateScore(m)

    smiles = Chem.MolToSmiles(m)
    print(smiles + "\t" + m.GetProp('_Name') + "\t%3f" % s)
    
    
    
# Similarity calculation based on TanimotoSimilarity

def calculate_morgan_fingerprint(smiles_lst):
    radius = 2  # Morgan fingerprint radius
    n_bits = 2048  # Number of bits in the fingerprint
    
    fp_lst = []
    for s in smiles_lst:
        mol = Chem.MolFromSmiles(s)
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fp_lst.append(fingerprint)
        
    return fp_lst


def calculate_diversity(smiles_lst):
    
    fp_lst = calculate_morgan_fingerprint(smiles_lst)
    diversity_lst = []
    
    for i in range(len(smiles_lst)):
        for j in range(i):
            similarity = TanimotoSimilarity(fp_lst[i], fp_lst[j])
            diversity_lst.append(1-similarity)
    return diversity_lst, np.mean(diversity_lst)
    
