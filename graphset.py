from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric import data as DATA
import torch
import os

class GraphDataset(InMemoryDataset):
    def __init__(self, root='.', dataset='davis', transform=None, pre_transform=None, graphs_dict=None, dttype=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.dttype = dttype
        self.process(graphs_dict)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + f'_data_{self.dttype}.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass
#         if not os.path.exists(self.processed_dir):
#             os.makedirs(self.processed_dir)

    def process(self, graphs_dict):
        data_list = []
        for data_mol in graphs_dict:
            # c_size, features, edge_index = data_mol[0],data_mol[1],data_mol[2]
            features, edge_index = data_mol[1], data_mol[2]
            GCNData = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index).transpose(1, 0))
            # GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            data_list.append(GCNData)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate(data_list):
    batchA = Batch.from_data_list([data for data in data_list])
    return batchA


# initialize the dataset
class DTADataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis',len_proteins = None, transform=None,
                 pre_transform=None,target_key=None, target_graph=None):

        super(DTADataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.process(len_proteins, target_key,target_graph)

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '_data_pro.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, len_proteins, target_key, target_graph):

        data_list_pro = []
        data_len = len_proteins
        for i in range(data_len):
            tar_key = target_key[i]
            # convert SMILES to molecular representation using rdkit
            target_size, target_features, target_edge_index = target_graph[tar_key]
            GCNData_pro = DATA.Data(x=torch.Tensor(target_features),
                                    edge_index=torch.LongTensor(target_edge_index).transpose(1, 0))
            # GCNData_pro.__setitem__('target_size', torch.LongTensor([target_size]))

            data_list_pro.append(GCNData_pro)

        self.data_pro = data_list_pro

    def __len__(self):
        return len(self.data_pro)

    def __getitem__(self, idx):
        return self.data_pro[idx]

#prepare the protein and drug pairs
def collate2(data_list):
    # batchA = Batch.from_data_list([data[0] for data in data_list])  ##这里应该是每个药物的长度
    batchB = Batch.from_data_list([data for data in data_list])
    return batchB
