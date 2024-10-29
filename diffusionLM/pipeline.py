import numpy as np
import os
import torch
import sys
from improved_diffusion.config_utils import CfgNode as CN
from improved_diffusion.config_utils import token2smiles
from diffusionLM.metrics import *
from rdkit.DataStructs import TanimotoSimilarity

class diffusionLM():
    
    @staticmethod
    def get_default_model_config():
        C = CN()
        # hyperapameters
        C.model_arch = "transformer"
        C.diff_step =  2000
        C.noise_schedule = "sqrt"
        C.in_channel = 16
        C.padding_mode = "pad"
        

        return C

    @staticmethod
    def get_default_data_config():
        C = CN()
        # length of the total sequence
        C.trainset_path = None
        # conditional or un-conditional
        C.task = "conditional"
        
        return C
    
    @staticmethod
    def get_default_train_config():
        C = CN()
        # device to train on
        # optimizer parameters
        C.lr = 0.0001
        C.lr_anneal_steps = 100000
        C.bsz = 64 # batch size
        C.pretrain = None
        C.seed = 101
        C.task = "conditional"
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
        
    
    def train_job(self, model_config, data_config, train_config):
        
        if train_config.task == "unconditional":
            vocab_size = 17
        if train_config.task == "conditional":
            vocab_size = 19
        
        command = 'python ' + os.path.dirname(__file__) + '/improved-diffusion/scripts/run_train.py '
        command += '--diff_steps ' + str(model_config.diff_step)
        command += ' --model_arch ' + model_config.model_arch 
        command += ' --lr ' + str(train_config.lr)
        command += ' --lr_anneal_steps ' + str(train_config.lr_anneal_steps)
        command += ' --seed ' + str(train_config.seed)
        command += ' --noise_schedule ' + model_config.noise_schedule
        command += ' --in_channel ' + str(model_config.in_channel) 
        command += ' --padding_mode ' + model_config.padding_mode
        command += ' --bsz ' + str(train_config.bsz)
        command += ' --modality roc --submit no'
        command += ' --notes xstart_e2e'
        app = '--predict_xstart True --training_mode e2e  --vocab_size ' + str(vocab_size) +  ' --roc_train ' + data_config.trainset_path
        command += ' --app ' 
        command += '"{}"'.format(app)
        print(command)
        
        f = open(os.path.dirname(__file__) + '/improved-diffusion/train_' + train_config.task + '.sh', "w")
        f.write(command)
        f.close()
        

        return None # Loss will be stored in loss.pt in the results folder
        
    def generate_job(self, model_config, data_config, train_config, generate_config):
        
        command = 'python ' + os.path.dirname(__file__) + '/improved-diffusion/scripts/batch_decode.py '
        model_name = "diffusion_models/diff_roc_" + model_config.padding_mode + "_rand" + str(model_config.in_channel) + "_" + model_config.model_arch + "_lr" + str(train_config.lr) + "_0.0_" + str(model_config.diff_step) + "_" + model_config.noise_schedule + "_Lsimple_h128_s2_d0.1_sd" + str(train_config.seed) + "_xstart_e2e"
        command += model_name 
        command += " -1.0 ema "
        command += str(generate_config.model_index) 
        command += " "
        command += str(generate_config.num_samples)
        
        f = open(os.path.dirname(__file__) + '/improved-diffusion/generation_' + train_config.task + '.sh', "w")
        f.write(command)
        f.close()
        
        return None
    
    
    def evaluate(self, file_path, label):
 
        poly_df = pd.read_csv(file_path, header=None)
        df = pd.read_csv(os.path.dirname(__file__) + "/htp_md.csv", sep="\t")
        poly_df.columns = ["mol_smiles"]
        if label == "conditional":
            poly_df, _ = extract_samples(poly_df)
            
        num_samples = len(poly_df)
        poly_df["mol_smiles"] = poly_df["mol_smiles"].apply(lambda x: extract_smiles(x, label))
        poly_df['duplicate'] = poly_df['mol_smiles'].duplicated()
        uniqueness = 1 - len(poly_df[poly_df['duplicate'] == True]) / num_samples
        poly_df_novel = check_novelty(poly_df, df, 'mol_smiles')
        count_not_novel = poly_df_novel['mol_smiles'][poly_df_novel['diversity'] != 'novel'].count()
        novelty = 1 - count_not_novel / num_samples
        poly_df_valid = validate_mol(poly_df, column_name='mol_smiles')
        poly_df_valid = has_two_ends(poly_df_valid)
        df1 = poly_df_valid
        df_valid = df1.loc[(df1['validity'] == 'ok') & (df1['has_two_ends'] == True)]
        validity = len(df_valid) / num_samples
    
        df_clean = df1.loc[(df1['duplicate'] == False) & (df1['diversity'] == 'novel') & (df1['validity'] == 'ok') & (df1['has_two_ends'] == True)]
        
        sa_scores_clean = [calculateScore(Chem.MolFromSmiles(s)) for s in df_clean["mol_smiles"]]
        synthesibility = len([x for x in sa_scores_clean if x < 5]) / len(df_clean)
    
        morgan_fingerprint_generated = calculate_morgan_fingerprint(df_clean["mol_smiles"])
        morgan_fingerprint_original = calculate_morgan_fingerprint(df["mol_smiles"])

        tanimoto_similarity = []
        for i in range(len(df_clean["mol_smiles"])):
            f1 = morgan_fingerprint_generated[i]
            similarity_scores = [TanimotoSimilarity(f1, f2) for f2 in morgan_fingerprint_original]
            tanimoto_similarity.append(np.mean(similarity_scores))
    
        similarity = np.mean(tanimoto_similarity)
        diversity_lst, diversity = calculate_diversity(df_clean["mol_smiles"].to_list())
        
        return uniqueness, novelty, validity, synthesibility, similarity, diversity

