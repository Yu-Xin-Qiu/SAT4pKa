# SAT4pKa
The acid-base dissociation constant (pKa) is an essential physicochemical parameter that indicates the extent of proton dissociation. However, accurately predicting pKa values is still challenging due to limited data availability for organic small molecules in aqueous solutions. In this work, we propose an open-source pKa prediction tool based on Graph Transformer, which combines the graph neural network and transformer to take both local and global information into account. 
## The workflow of our work
![image](https://github.com/Violets9527/SAT4pKa/assets/127859234/d81fffcb-89cf-4c0e-9bd9-92a7b54fafcb)
Some commonly used open-source data sets are collected, including a large computational dataset for pre-training, the extended experimental dataset for fine-tuning, and three external test sets for proving the generalization of the model. RDKit and the PyTorch-Geometric (PyG) library are used to represent canonical SMILES as molecular graphs and obtain the adjacency matrix and feature matrix. The graph transformer model is pretrained using computational pKa values, after which transfer learning is implemented to fit high-precision experimental pKa values. Additionally, two applications are conducted to validate the reliability of our model, including the screening of tertiary amines for CO2 absorption and acidic ionic liquids for long-chain esterification in reactive extraction.
## Usage
### Installation
The dependencies are managed by anaconda
```
einops==0.7.0  
gradio==4.2.0  
matplotlib==3.7.1  
networkx==3.1  
numpy==1.24.3  
pandas==2.0.2  
PyYAML==6.0.1  
rdkit==2023.3.1  
scikit_learn==1.2.2  
streamlit==1.28.1  
torch==1.11.0+cu115  
torch_geometric==2.1.0  
torch_scatter==2.0.9  
```
### Using the provided model for pKa prediction
`predict.py` is an example file for using our tool, we provide the trained GCN, GIN and SAT model.

### Training model for Graph Transformer
We provide the model training file `finetune.py`. In the `finetune.yaml`, you can easily change the hyperparameters.

### Model visualization
In `explain.py`, we show how to generate the attention weight graph learned by the SAT model.The following figure shows the attention weights learned by the SAT model, using MDEA as an example.
![image](https://github.com/Violets9527/SAT4pKa/assets/127859234/b55503ae-da33-44c5-b36a-ae26575ca0e7)








