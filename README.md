# H2GnnDTI
## Overview
![image](https://github.com/LiminLi-xjtu/H2GnnDTI/blob/master/H2GnnDTI.png)
 H2GnnDTI is a two-level hierarchical heterogeneous graph learning model, to predict DTIs byintegrating the structures of drugs and proteins via low-level graph neural networks (LGNN) and a high-level graph
neural network(HGNN). The hierarchical graph is a high-level heterogeneous graph with nodes being drugs and proteins and edges being the known DTIs, and each drug or protein node is further represented as a low-level graph with nodes being molecules in each drug or amino acids in each protein with their chemical descriptors. Two low-level graph neural networks are ffrst used to capture the structural and chemical features for drugs and proteins from the low-level graphs, respectively, and a high-level graph encoder is employed to further capture and integrate interactive features for drugs and proteins from the high-level graph. The high-level encoder utilizes a structure and attribute information fusion module which could explicitly merge the representations learned by a feature encoder and a graph encoder for consensus representation learning.

## Installation
```bash
pip install git+https://github.com/LiminLi-xjtu/H2GnnDTI.git
```

## Requirements
* Python 3.8.18
* torch              1.7.0
* torch-geometric    2.0.1
* torch-scatter      2.0.8
* torch-sparse      0.6.12
* tqdm 4.54.0
* rdkit-pypi        2021.9.4
  
## Data preparation
Prepare the data need for train. Get all msa files of the proteins in datasets (download the dataset.rar and unzip it), and using Pconsc4 to predict all the contact map. A script in the repo can be run to do all the steps:
```bash
python scripts.py
```
Then you can generate two foldes called "aln" and "pconsc4", copy two folders from davis to the /data/davis of your repo, so do the KIBA and DrugBank.
## Usage
```python main.py

"""Load preprocessed data."""
DATASET = "Davis"
# DATASET = "KIBA"
# DATASET = "DrugBank"

data_new, nb_drugs, nb_proteins = dataload(DATASET)
nb_all = nb_drugs+nb_proteins
drug_set, protein_set, adj, labels, idx_train, idx_test,edge = process(data_new, nb_drugs, nb_proteins,DATASET,foldcount=5,setting = 2)
'''

*change the dataset:DATASET = "Davis" or DATASET = "KIBA" or DATASET = "KIBA"
*change the settings: New-drug setting = 1,New-target setting = 2,New-dt setting = 3

*change the parameters: opt.py

