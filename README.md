# SAT4pKa
The acid-base dissociation constant (pKa) is an essential physicochemical parameter that indicates the extent of proton dissociation. However, accurately predicting pKa values is still challenging due to limited data availability for organic small molecules in aqueous solutions. In this work, we propose an open-source pKa prediction tool based on Graph Transformer, which combines the graph neural network and transformer to take both local and global information into account. 
# Requirements
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
