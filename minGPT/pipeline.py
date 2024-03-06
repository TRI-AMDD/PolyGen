import os
import pandas as pd
import numpy as np
import torch
from rdkit import Chem

from deepchem.feat.smiles_tokenizer import SmilesTokenizer
from mingpt.model import GPT
from mingpt.utils import CfgNode as CN
from sklearn.model_selection import train_test_split
from minGPT.trainer_custom import Trainer
from minGPT.metrics import *
from minGPT.dataset import *

class minGPT():
    
    @staticmethod
    def get_default_model_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt-nano'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    @staticmethod
    def get_default_data_config():
        C = CN()
        # column stores the SMILES
        C.input_col = "mol_smiles"
        # input length for labels
        C.length = 5
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
        # dataloder parameters
        C.num_workers = 0
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 5e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        C.model = None
        C.call_back = None
        C.pretrain = None
        return C
    
    @staticmethod
    def get_default_generate_config():
        C = CN()
        # number of the samples to generate
        C.ckpts_path = None
        C.num_samples = 100
        # tempareture of GPT2 (higher values lead to higher diversity and lower validity)
        C.temperature = 1.0
        C.task = "conditional"
        return C    
    
    def __init__(self):
        """
        config: dictionary defines the configuration for the GPT model
        """
    
        super().__init__()
        
    def load_model(self, model_config):
        
        self.model = GPT(model_config)
        self.block_size = model_config.block_size
        self.vocab_size = model_config.vocab_size
        self.device = "cuda"

    
    def data_preprocessing(self, data_config):
        self.tokenizer = SmilesTokenizer(vocab_file=os.path.join(os.path.dirname(__file__), "vocab.txt"))
        self.df = pd.read_csv(data_config.file_path, sep="\t") 
        self.data_num = len(self.df)
        self.length = data_config.length
        # Get the data index for train and test set of the dataset
        train_indices, test_indices = train_test_split(list(range(self.data_num)), test_size=0.2, random_state=42)
        
        self.train_dataset = SmilesDataset(smiles_file=data_config.file_path, 
                                      data_index=train_indices, tokenizer=self.tokenizer, 
                                      input_col=data_config.input_col, length=data_config.length, 
                                      block_size=data_config.block_size, task=data_config.task)
        
        self.test_dataset = SmilesDataset(smiles_file=data_config.file_path, 
                                      data_index=test_indices, tokenizer=self.tokenizer, 
                                      input_col=data_config.input_col, length=data_config.length, 
                                      block_size=data_config.block_size, task=data_config.task)
       

        return self.train_dataset, self.test_dataset
    
    def train(self, train_config):
        if train_config.pretrain:
            ckpt = torch.load(train_config.pretrain)
            self.model.load_state_dict(ckpt)
        
        self.model.to(self.device)
            
        self.trainer = Trainer(train_config, self.model, self.train_dataset, self.test_dataset)
        self.trainer.set_callback('on_batch_end', train_config.call_back)
        loss = self.trainer.run()
        
        return loss
        
    def generate(self, generate_config):
        self.num_samples = generate_config.num_samples
        if generate_config.ckpt_path:
            ckpt = torch.load(generate_config.ckpt_path)#, map_location=torch.device('cpu'))
            self.model.load_state_dict(ckpt)
        
        self.model.to(self.device)
        if generate_config.task == "unconditional":
            prompt = torch.randint(0, 10, size=(self.num_samples, self.length)).to(self.device)
        
        elif generate_config.task == "conditional":
            target_property_value = 9
            target_input = np.array([target_property_value for i in range(self.length)])
            target_input = np.tile(target_input, (self.num_samples,1))
            prompt = torch.tensor(target_input).to(self.device)
        

        y = self.model.generate(prompt, max_new_tokens=self.block_size, temperature=generate_config.temperature, do_sample=True, top_k=40)[:, 1:]   
        generated_out = []
        raw = []
        for i in range(self.num_samples):
            out = (self.tokenizer.decode(y[i].squeeze()))
            if "[CLS]" in out:
                start = out.index("[CLS]")
            if "[SEP]" in out:
                end = out.index("[SEP]")
            out = out[start + len("[CLS]") + 1: end]
            
            out = str(out).strip().replace(" ", "")

            generated_out.append(out)
            
        self.gen_df = pd.DataFrame(generated_out, columns=["mol_smiles"])

        return generated_out           
    
    
    def evaluate(self):


        ### Novelty & Uniqueness
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
        morgan_fingerprint_original = calculate_morgan_fingerprint(self.df["mol_smiles"])

        tanimoto_similarity = []
        for i in range(len(df_clean["mol_smiles"])):
            f1 = morgan_fingerprint_generated[i]
            similarity_scores = [TanimotoSimilarity(f1, f2) for f2 in morgan_fingerprint_original]
            tanimoto_similarity.append(np.mean(similarity_scores))

        similarity = np.mean(tanimoto_similarity)
        diversity_lst, diversity = calculate_diversity(df_clean["mol_smiles"].to_list())

        
        return uniqueness, novelty, validity, synthesibility, similarity, diversity

