# De novo designs of polymer electrolytes with high conductivities using Generative AIs
![Generated polymer electrolyte](https://github.com/TRI-AMDD/PolyGen/blob/main/molecule_grid.png)

## Installation
### the following installation steps have been tested on macOS 14.6.1 with M1 Max chip
#### minGPT
Python version: 3.8

Install the required packages:
```
pip install -r requirements.txt
```
#### diffusion1D
Python version: 3.8

Install the required packages denoising_diffusion_pytorch, rdkit, deepchem and transformers:
```
pip install rdkit deepchem transformers

cd diffusion1D/model
pip install -e .
```

#### diffusionLM
Python version: 3.8

Install the required packages diffusionLM, transformers (customized) and others:
```
pip install mpi4py
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e diffusionLM/improved-diffusion/ 
pip install -e diffusionLM/transformers/
pip install spacy==3.2.6
pip install datasets==2.0.0 
pip install huggingface_hub==0.16.4
pip install wandb deepchem torchsummary
```
## Dataset
#### minGPT & diffusion1D
Prepare the data used for training in .csv file with two columns, the separation marker is ```"\t"```
- 1st column: "mol_smiles" (SMILES code for the monomer)
- 2nd column: "conductivity" ("1" is high conductivity, "0" is low conductivity)

#### diffusionLM
- The datasets are stored in .json format, please check the ```diffusionLM/datasets``` for examples. 

## Training, generation and evaluation pipeline
- data preprocessing (data_config) 
- build the model (model_config)
- train the model (train_config)
- generate candidates (generate_config)
- evaluation (6 metrics including validity, novelty, uniqueness, synthesizability, similarity and diversity)

## Demo
The demos are shown in ```minGPT_pipeline.ipynb```, ```diffusion1D_pipeline.ipynb```, ```diffusionLM_pipeline.ipynb```
#### minGPT & diffusion1D
- For ```minGPT_pipeline.ipynb```, ```diffusion1D_pipeline.ipynb```, all the steps in pipeline can be executed in the notebook.

#### diffusionLM
- For ```diffusionLM_pipeline.ipynb```, the notebook generates the the bash scripts for training and generation. The scripts will be stored under ```diffusionLM/improved-diffusion```.
 
To run the training:
```
cd diffusionLM/improved-diffusion
bash train_conditional.sh or bash train_unconditional.sh
The model checkpoints will be stored in ```diffusionLM/improved-diffusion/diffusion_models```
```
To run the generation:
```
cd diffusionLM/improved-diffusion
bash generate_conditional.sh or bash generate_unconditional.sh
```
The generated output will be stored in ```diffusionLM/improved-diffusion/generation_outputs```

## Pretrained models
#### minGPT
The checkpoints of pretrained model at different epochs can be obtained here:https://drive.google.com/drive/folders/1M1VjgUnFDospbmVSnr17JdUcUa-_4O79?usp=sharing. Please put the checkpoints files under ```minGPT/ckpts/```. 

#### diffusion1D
The checkpoints of pretrained model at different epochs can be obtained here: https://drive.google.com/drive/folders/1kFnKtnmuQLTNDZ7BJG2ZhoJKGWoXlI--?usp=sharing. Please put the checkpoints files under ```diffusion1D/ckpts/```. 

### diffusionLM
The checkpoints of pretrained model at different epochs can be obtained here: https://drive.google.com/drive/folders/1ndLNhRZu8TL2Ni7VL8Q9GRAeX9fFVOq0?usp=sharing. Please put the whole checkpoints folder and files under ```diffusionLM/improved-diffusion/diffusion_models/```. 

<!-- ### Configurations:
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
- evaluation (no config): 6 metrics: novelty, uniqueness, validity, synthesizability, diversity and similarity. -->

## Reference
The github repositories that are referenced for this code include:

```
https://github.com/karpathy/minGPT
https://github.com/lucidrains/denoising-diffusion-pytorch
https://github.com/XiangLi1999/Diffusion-LM
```
In this work, we copied the minGPT model from the original repository by Karpathy at https://github.com/karpathy/minGPT at commit 37baab7 (Jan 8, 2023). This unchanged copy is saved in https://github.com/TRI-AMDD/PolyGen/tree/main/minGPT/model. 

## Citation
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
