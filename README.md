# De novo designs of polymer electrolytes with high conductivities using Generative AIs
![Generated polymer electrolyte](https://github.com/TRI-AMDD/PolyGen/blob/main/molecule_grid.png)

### Installation
```
cd mingpt/minGPT
pip install -e .
```
### Dataset
Data used for training is stored in .csv file with two columns, the separation marker is ```"\t"```
- 1st column: "mol_smiles" (SMILES code for the monomer)
- 2nd column: "conductivity" ("1" is high conductivity, "0" is low conductivity)
### Demo
The demo notebook is ```./mingpt/minGPT_pipeline.ipynb```

### Steps: 
- data preprocessing
- build the model
- train the model
- generate candidates
- evaluation

### Parameters:
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
  - model_type (default='gpt-nano'): type of model architecture, available pretrained options ('gpt2', 'gpt-mini', 'gpt-nano').
  - n_layer, n_head, n_embd: will auto-fill based on the model type.
  - vocab_size (default=591): size of vocabulary, obtained based on tokenizer. 
  - block_size (default=64): same as data preprocessing.
  - embd_pdrop (default=0.1): dropout prob for embedding.
  - resid_pdrop (default=0.1): dropout prob for residual layer.
  - attn_pdrop (default=0.1): dropout prob for attention layer.


