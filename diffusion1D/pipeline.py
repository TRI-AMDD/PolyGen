import numpy as np
import os
import torch
import sys
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D
from denoising_diffusion_pytorch.utils import CfgNode as CN
from denoising_diffusion_pytorch.utils import token2smiles
from metrics import *
import pandas as pd
from deepchem.feat.smiles_tokenizer import SmilesTokenizer
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, rdchem
import matplotlib.pyplot as plt
from IPython.display import Image, display
import sys
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from rdkit.DataStructs import TanimotoSimilarity


class diffusion1D():
    
    @staticmethod
    def get_default_model_config():
        C = CN()
        # hyperapameters
        C.dim = 128
        C.dim_mults = (1, 2)
        # these options must be filled in externally
        C.diffusion_step = 1000
        C.vocab_size = None
        C.block_size = None

        return C

    @staticmethod
    def get_default_data_config():
        C = CN()
        # length of the total sequence
        C.block_size = 64
        C.train_test_split = (0.8, 0.2)
        # conditional or un-conditional
        C.task = "conditional"
        
        return C
    
    @staticmethod
    def get_default_train_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # optimizer parameters
        C.num_steps = None
        C.batch_size = 64
        C.learning_rate = 5e-5
        C.gradient_accumulate_every = 1    # gradient accumulation steps
        C.ema_decay = 0.995 # only applied on matmul weights
        C.pretrain = None
        return C
    
    @staticmethod
    def get_default_generate_config():
        C = CN()
        # number of the samples to generate
        C.model_index = None
        C.num_samples = 100
        C.task = "conditional"
        return C    
    
    def __init__(self):
        """
        config: dictionary defines the configuration for the GPT model
        """
    
        super().__init__()
        
    def load_model(self, model_config):
        
        self.model = Unet1D(
            dim = model_config.dim,
            dim_mults = model_config.dim_mults,
            channels = 1
        )
        print(sum(p.numel() for p in self.model.parameters()))

        self.diffusion = GaussianDiffusion1D(
            self.model,
            seq_length = model_config.block_size,
            timesteps = model_config.diffusion_step,
            objective = 'pred_v', 
        )

        self.block_size = model_config.block_size
        self.vocab_size = model_config.vocab_size
        self.device = "cuda"

    
    def data_preprocessing(self, data_config):
        self.tokenizer = SmilesTokenizer(vocab_file=os.path.join(os.path.dirname(__file__), "vocab.txt"))
        self.df = pd.read_csv(data_config.file_path, sep="\t") 
        print(self.df)
        self.data_num = len(self.df)
        self.block_size = data_config.block_size
        # Get the data index for train and test set of the dataset
        if data_config.task == "conditional":
            mask = self.df["conductivity"] == 1
            self.df.loc[mask, "mol_smiles"] = self.df.loc[mask, "mol_smiles"] + "[Ag]"
            mask = self.df["conductivity"] == 0
            self.df.loc[mask, "mol_smiles"] = self.df.loc[mask, "mol_smiles"] + "[Ac]"
            
        train_indices, test_indices = train_test_split(list(range(self.data_num)), test_size=0.2, random_state=42)
        sequences = [self.tokenizer.encode(s) for s in self.df["mol_smiles"]]

        # Padding
        pad_sequences = [self.tokenizer.add_padding_tokens(s, length = self.block_size) for s in sequences]


        # Reindex
        flat_tokens = [element for sublist in pad_sequences for element in sublist]
        sorted_tokens = sorted(set(flat_tokens))
        num_vocab = len(set(sorted_tokens))
        self.vocab_size = num_vocab
        f = open(os.path.join(os.path.dirname(__file__), "vocab.txt"), "r")
        lines = f.readlines()
        self.token_dict = [lines[t][:-1] for t in sorted_tokens]
        print(self.token_dict)
        token_reindex = {sorted_tokens[i]: i for i in range(num_vocab)}
        print(sorted_tokens)
        print("Number of tokens in the dataset:")
        print(num_vocab)

  
        # Transfer to one-hot encoding

        categorical_sequences = []
        for s in pad_sequences:
            categorical_sequences.append([token_reindex[s[i]] for i in range(self.block_size)])


        categorical_sequences = torch.tensor(categorical_sequences, dtype=torch.float)
        categorical_sequences = torch.unsqueeze(categorical_sequences, dim=-1)
        self.train_dataset = torch.permute(categorical_sequences, (0, 2, 1))[train_indices] / torch.max(categorical_sequences)
        self.test_dataset = torch.permute(categorical_sequences, (0, 2, 1))[test_indices] / torch.max(categorical_sequences)
               
        return self.train_dataset, self.test_dataset, self.vocab_size
    
    def train(self, train_config):
        
        self.model.to(self.device)
        self.trainer = Trainer1D(
            self.diffusion,
            train_dataset = self.train_dataset,
            val_dataset = self.test_dataset,
            train_batch_size = train_config.batch_size,
            train_lr = train_config.learning_rate,
            train_num_steps = train_config.num_steps,         # total training steps
            gradient_accumulate_every = train_config.gradient_accumulate_every,    # gradient accumulation steps
            ema_decay = train_config.ema_decay,                # exponential moving average decay
            amp = True,                       # turn on mixed precision
            fp16 = True, 
            results_folder = train_config.ckpts_path
        )

        if train_config.pretrain:
            self.trainer.load(train_config.pretrain)

        
        self.trainer.train()   

        return None # Loss will be stored in loss.pt in the results folder
        
    def generate(self, generate_config):
        
        self.num_samples = generate_config.num_samples
        self.trainer.load(generate_config.model_index)
        
        
        if generate_config.task == "unconditional":
            sampled_seq_epoch = self.diffusion.sample(batch_size = self.num_samples)
            sampled_seq_epoch = torch.permute(sampled_seq_epoch, (0, 2, 1))
        
            cleaned_out = []
            
            for s in sampled_seq_epoch:
                lst = [round(t.cpu().numpy()[0]) for t in s * (self.vocab_size - 1)]
                smiles = token2smiles(lst)
                cleaned_out.append(smiles)
                
        elif generate_config.task == "conditional":
            
            sampled_seq_epoch = self.diffusion.sample(batch_size = int(3*self.num_samples))
            sampled_seq_epoch = torch.permute(sampled_seq_epoch, (0, 2, 1))
        
            cleaned_out = []
            count_high = 0
            print(self.vocab_size - 1)
            for s in sampled_seq_epoch:
                lst = [round(t.cpu().numpy()[0]) for t in s * (self.vocab_size - 1)]
                smiles = token2smiles(lst, self.token_dict)
                print(smiles)
                if "[Ag]" in smiles and count_high < self.num_samples:
                    cleaned_out.append(smiles.replace("[Ag]", ""))
                    count_high += 1
                

        self.gen_df = pd.DataFrame(cleaned_out, columns =['mol_smiles'])

        return cleaned_out           
    
    
    def evaluate(self):


        ### Novelty & Uniqueness
        def remove_label(smiles):
            return smiles.replace("[Ag]", "").replace("[Ac]", "")
        self.df["mol_smiles"] = self.df["mol_smiles"].apply(remove_label)
        self.df['duplicate'] = self.df['mol_smiles'].duplicated()
        self.df = self.df[self.df['duplicate'] == False]
        self.gen_df['duplicate'] = self.gen_df['mol_smiles'].duplicated()
        uniqueness = 1 - len(self.gen_df[self.gen_df['duplicate'] == True]) / self.num_samples
        self.gen_df_2 = check_novelty(self.gen_df, self.df, 'mol_smiles')
        count_not_novel = self.gen_df_2['mol_smiles'][self.gen_df_2['diversity'] != 'novel'].count()
        novelty = 1 - count_not_novel / self.num_samples

        ### Validity
        self.gen_df_valid = validate_mol(self.gen_df, column_name='mol_smiles')
        self.gen_df_valid = has_two_ends(self.gen_df_valid)
        df1 = self.gen_df_valid
        df_valid = df1.loc[(df1['validity'] == 'ok') & (df1['has_two_ends'] == True)]
        validity = len(df_valid) / self.num_samples

        df_clean = df1.loc[(df1['duplicate'] == False) & (df1['diversity'] == 'novel') & (df1['validity'] == 'ok') & (df1['has_two_ends'] == True) ]
        
        ### Synthesizability
        sa_scores_clean = [calculateScore(Chem.MolFromSmiles(s)) for s in df_clean["mol_smiles"]]
        synthesibility = len([x for x in sa_scores_clean if x < 5]) / len(df_clean)

        ### Similarity & Diversity
        morgan_fingerprint_generated = calculate_morgan_fingerprint(df_clean["mol_smiles"])
        print(self.df)
        morgan_fingerprint_original = calculate_morgan_fingerprint(self.df["mol_smiles"])

        tanimoto_similarity = []
        for i in range(len(df_clean["mol_smiles"])):
            f1 = morgan_fingerprint_generated[i]
            similarity_scores = [TanimotoSimilarity(f1, f2) for f2 in morgan_fingerprint_original]
            tanimoto_similarity.append(np.mean(similarity_scores))

        similarity = np.mean(tanimoto_similarity)
        diversity_lst, diversity = calculate_diversity(df_clean["mol_smiles"].to_list())

        
        return uniqueness, novelty, validity, synthesibility, similarity, diversity


