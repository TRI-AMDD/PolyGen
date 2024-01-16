import pandas as pd
import numpy as np
import random
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from rdkit import Chem
import sys
from deepchem.feat.smiles_tokenizer import SmilesTokenizer


class SmilesDataset(Dataset):
    def __init__(self, smiles_file, data_index, tokenizer, input_col='mol_smiles', length=5, block_size=64, task="conditional"):
        """
        Input: input1 + input2
            where input2.shape[0] = max smile string length - 1
                  input1.shape[0] = input1_size
        """
        self.df = pd.read_csv(smiles_file, sep="\t")
        self.df.columns = ["mol_smiles", "conductivity"]
        self.df = self.df.loc[data_index]
        self.df = self.df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.df['tokenized'] = self.df[input_col].apply(tokenizer.encode)
        # length of the total sequence
        self.block_size = block_size
        # length of input label
        self.length = length
        self.max_length = self.block_size + 1 - self.length
        self.df['padded'] = self.df['tokenized'].apply(lambda x: np.pad(x, (0, self.max_length-len(x)), 'constant', constant_values=0))

        self.df['padded'] = self.df['padded'].apply(torch.tensor)
        self.output = torch.tensor(np.vstack(self.df['padded']))

        if task == "unconditional":
            self.input1 = torch.randint(0, 10, size=(len(self.df['padded']), self.length))

        elif task == "conditional":
            self.input1 = torch.tensor(self._prepare_property())


    def _prepare_property(self):
        """
        high conductivity : 1
        low conductivity : 0
        """
        prop = self.df['conductivity']
        prop = prop.apply(lambda x: [x + 8])
        return prop.apply(lambda x: x*self.length)

    def __len__(self):
        return len(self.df['padded'])

    def __getitem__(self, idx):

        overall = torch.concat((self.input1[idx], self.df['padded'][idx]))
        inp = overall[:-1]
        out = overall[1:]

        return inp, out

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

    def get_block_size(self):
        return self.block_size


