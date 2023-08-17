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
- data preprocessing:
  - input_col = "mol_smiles"


