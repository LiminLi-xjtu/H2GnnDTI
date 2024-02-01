import torch
import random
import numpy as np
import os
from torch.utils.data import DataLoader


"""select seed"""
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def dataload(DATASET):
    print("Train in " + DATASET)
    weight_CE = None
    dir_input = ('./dataset/{}.txt'.format(DATASET))
    print("load data")
    with open(dir_input, "r") as f:
        train_data_list = f.read().strip().split('\n')
    print("load finished")

    # random shuffle
    print("data shuffle")
    dataset = shuffle_dataset(train_data_list, SEED)

    N = len(dataset)

    drug_ids, protein_ids,data_new = [], [], []
    for i, pair in enumerate(dataset):
        pair = pair.strip().split()
        drug_id, protein_id, compoundstr, proteinstr, label = pair[-5], pair[-4], pair[-3], pair[-2], pair[-1]
        drug_ids.append(drug_id)
        protein_ids.append(protein_id)
        label = np.int(label)
        # labels_new[i] = np.int(label)
        # data_new = np.vstack((drug_new,protein_new,labels_new)).T
        data_index = (drug_id,protein_id,label,compoundstr,proteinstr)
        data_new.append(data_index)  ##hstack

    nb_drugs = len(set(drug_ids))
    nb_proteins = len(set(protein_ids))

    print('All %d pairs across %d drugs and %d proteins.' % (len(dataset), nb_drugs,nb_proteins))
    return data_new,nb_drugs,nb_proteins




