# De novo designs of polymer electrolytes with high conductivities using Generative AIs
![Generated polymer electrolyte](https://github.com/TRI-AMDD/PolyGen/blob/main/molecule_grid.png)

### Installation
#### minGPT
Python version: 3.8

Install the required packages minGPT, rdkit, and deepchem:
```
git clone https://github.com/karpathy/minGPT.git
cd minGPT
pip install -e .

pip install rdkit, deepchem
```
#### diffusion1D
Python version: 3.8

Install the required packages minGPT, rdkit, and deepchem:
```
git clone https://github.com/karpathy/minGPT.git minGPT/model
cd minGPT/model
pip install -e .

pip install rdkit, deepchem
```
### Dataset
Prepare the data used for training in .csv file with two columns, the separation marker is ```"\t"```
- 1st column: "mol_smiles" (SMILES code for the monomer)
- 2nd column: "conductivity" ("1" is high conductivity, "0" is low conductivity)
### Demo
The demo notebook is ```minGPT_pipeline.ipynb```

### Steps: 
- data preprocessing
- build the model
- train the model
- generate candidates
- evaluation

### Configurations:
- data preprocessing (data_config):
  - length (default=5): length of input labels, for conditional case, it is set to 5 (conductivity label). For unconditional case, it is set to 1 (random number).
  - block_size (default=64): the max length of the whole sequence.
  - train_test_split (default=(0.8, 0.2)): the ratio of train and test set.
  - task (default="conditional"): "unconditional" for unconditional generation.
- build the model (model_config):
  - model_type (default='gpt-nano'): type of model architecture, available pretrained options ('gpt2', 'gpt-mini', 'gpt-nano').
  - n_layer, n_head, n_embd: will auto-fill based on the model type.
  - vocab_size (default=591): size of vocabulary, obtained based on tokenizer. 
  - block_size (default=64): same as data preprocessing.
  - embd_pdrop (default=0.1): dropout prob for embedding.
  - resid_pdrop (default=0.1): dropout prob for residual layer.
  - attn_pdrop (default=0.1): dropout prob for attention layer.    
- train the model (train_config):
  - device (default='auto'): train device.
  - num_workers (default=0): dataloader parameter.
  - max_iters (no default): number of iterations.
  - batch_size (default=64): batch size.
  - learning_rate (default=5e-4): learning rate.
  - betas (default=(0.9, 0.95)): optimizer parameter.
  - weight_decay (default=0.1): scheduler parameter.
  - grad_norm_clip (default=1.0): optimizer parameter.
  - model (default=None): model class.
  - call_back (default=None): callback function.
  - pretrain (default=None):  path to the checkpoint of pretrained model.
- generate candidates (generate_config):
  - ckpts_path (default=None): path to the model checkpoint used for generation.
  - num_samples (default=100): number of samples that will be generated.
  - temperature (default=1.0): temperature for generation (higher leads to higher diversity and lower validity).
  - task (default="conditional"): "unconditional" for unconditional generation.
- evaluation (no config): 6 metrics: novelty, uniqueness, validity, synthesizability, diversity and similarity. 
  
### Citation
If you use PolyGen, please cite the following:

```
@article{lei2023self,
  title={A self-improvable Polymer Discovery Framework Based on Conditional Generative Model},
  author={Lei, Xiangyun and Ye, Weike and Yang, Zhenze and Schweigert, Daniel and Kwon, Ha-Kyung and Khajeh, Arash},
  journal={arXiv preprint arXiv:2312.04013},
  year={2023}
}

@article{yang2023novo,
  title={De novo design of polymer electrolytes with high conductivity using gpt-based and diffusion-based generative models},
  author={Yang, Zhenze and Ye, Weike and Lei, Xiangyun and Schweigert, Daniel and Kwon, Ha-Kyung and Khajeh, Arash},
  journal={arXiv preprint arXiv:2312.06470},
  year={2023}
}
```
