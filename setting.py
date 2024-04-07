import scipy.sparse as sp
from tqdm import tqdm
import torch.utils.data as Data
from scipy.sparse import coo_matrix
from graphset import *
from pro_graph import *
from rdkit import Chem
import networkx as nx
from data_load import dataload
from sklearn.model_selection import train_test_split


CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

CHARPROTLEN = 25

def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN,dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X

def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN,np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X

def cmask(num, ratio, seed):
    mask = np.ones(num, dtype=bool)
    mask[0:int(ratio * num)] = False
    np.random.seed(seed)
    np.random.shuffle(mask)
    return mask

# mol atom feature for mol graph
def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

# mol smile to mol graph edge index
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()  ##一个药物字符串中原子的数量

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)  ##78个原子特征
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
        # edge_index.append([e1, e2])
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    return c_size, features, edge_index


def process(data_new,nb_drugs,nb_proteins,dataset, foldcount=5,setting = 2):
    # -----construct cell line-drug response pairs
    drugid = list(set([item[0] for item in data_new]));drugid.sort()
    proteinid = list(set([item[1] for item in data_new]));proteinid.sort()
    drugmap = list(zip(drugid, list(range(len(drugid)))))
    proteinmap = list(zip(proteinid, list(range(len(drugid), len(drugid) + len(proteinid)))))
    drug_num = np.squeeze([[j[1] for j in drugmap if i[0] == j[0]] for i in data_new])
    protein_num = np.squeeze([[j[1] for j in proteinmap if i[1] == j[0]] for i in data_new])
    Inter_num = np.squeeze([i[2] for i in data_new])
    allpairs = np.vstack((drug_num, protein_num, Inter_num)).T
    allpairs = allpairs[allpairs[:, 2].argsort()]

    pos_edge = allpairs[allpairs[:, 2] == 1, 0:2]
    neg_edge_ori = allpairs[allpairs[:, 2] == 0, 0:2]
    if dataset == "Davis":
        edge_mask = cmask(len(neg_edge_ori), 0.6032, 666)  ##davis
    elif dataset == "KIBA":
        edge_mask = cmask(len(neg_edge_ori), 0.7648, 666)  ##kiba数据集
    neg_edge = neg_edge_ori[edge_mask][:, 0:2]
    print(len(pos_edge))
    print(len(neg_edge))
    alledges = np.concatenate([pos_edge, neg_edge], axis=0)

    ##构建蛋白质字典
    pro_dict = {}
    for key in proteinid:
        value = list(set([j[4] for j in data_new if key == j[1]]))[0]
        pro_dict[key] = value

    compoundstr = np.squeeze([list(set([j[3] for j in data_new if i[0] == j[0]])) for i in drugmap])
    drug_data = [[] for item in range(len(compoundstr))]
    for i, smile in enumerate(compoundstr):
        g = smile_to_graph(smile)
        drug_data[i] = g
        # print(smile_graph['CN1CCN(C(=O)c2cc3cc(Cl)ccc3[nH]2)CC1']) #for test

    target_key = proteinid
    nb_proteins = len(proteinid)

    # ----cell line_feature_input   379个蛋白质
    msa_path = 'D:/软件/data/data/' + dataset + '/aln'
    contac_path = 'D:/软件/data/data/' + dataset + '/pconsc4'
    # msa_path = '/home/datasets/data_pro/' + dataset + '/aln'
    # contac_path = '/home/datasets/data_pro/' + dataset + '/pconsc4'
    msa_list = []
    contact_list = []
    for key in target_key:
        msa_list.append(os.path.join(msa_path, key + '.aln'))
        contact_list.append(os.path.join(contac_path, key + '.npy'))

    target_graph = {}
    ##无重复的字符串列表   将蛋白质变成数字标签
    for key in target_key:
        g = target_to_graph(key, pro_dict[key], contac_path, msa_path)
        target_graph[key] = g

    # ---compile training set and test set  drug_new
    drug_set = Data.DataLoader(dataset=GraphDataset(graphs_dict=drug_data),collate_fn=collate,batch_size=nb_drugs,shuffle=False)
    protein1 = DTADataset(len_proteins=nb_proteins, target_key=target_key, target_graph=target_graph)
    protein_set = Data.DataLoader(dataset=protein1, collate_fn=collate2, batch_size=nb_proteins, shuffle=False)

    ##划分训练集和测试集
    # setting2 new-drug
    if setting == 2:
        print('------------setting2 new-drug-----------------')
        folds = get_random_folds(nb_drugs, foldcount)
        test_set = folds[4]
        pairs_list = alledges.tolist()
        pairs_test = []
        for x in pairs_list:
            if x[0] in test_set:
                pairs_list.remove(x)
                pairs_test.append(x)
        train = np.array(pairs_list)
        test = np.array(pairs_test)
    if setting == 3:
        print('------------setting3 new-target-----------------')
        folds = get_random_folds(nb_proteins, foldcount)
        test_set = folds[4]
        pairs_list = alledges.tolist()
        pairs_test = []
        for x in pairs_list:
            if x[1] - nb_drugs in test_set:
                pairs_list.remove(x)
                pairs_test.append(x)
        train = np.array(pairs_list)
        test = np.array(pairs_test)
    if setting == 4:
        print('------------setting4 new-dt-----------------')
        folds_drug = get_random_folds(nb_drugs, foldcount)
        folds_protein = get_random_folds(nb_proteins, foldcount)
        test_set_drug = folds_drug[4]
        test_set_pro = folds_protein[4]
        pairs_list = alledges.tolist()
        pairs_test = []
        for x in pairs_list:
            if x[1]-nb_drugs in test_set_pro and x[0] in test_set_drug:
                pairs_list.remove(x)
                pairs_test.append(x)
        train = np.array(pairs_list)
        test = np.array(pairs_test)

    train[:, 1] -= nb_drugs  ##减去药物的数量，才是真正的蛋白质的序号
    test[:, 1] -= nb_drugs
    train_mask = coo_matrix((np.ones(train.shape[0], dtype=bool), (train[:, 0], train[:, 1])),
                            shape=(nb_drugs, nb_proteins)).toarray()
    test_mask = coo_matrix((np.ones(test.shape[0], dtype=bool), (test[:, 0], test[:, 1])),
                           shape=(nb_drugs, nb_proteins)).toarray()
    train_mask = torch.from_numpy(train_mask).view(-1)  ##按顺序展成一维张量
    test_mask = torch.from_numpy(test_mask).view(-1)
    ##构造标签label

    pos_edge = allpairs[allpairs[:, 2] == 1, 0:2]
    neg_edge = allpairs[allpairs[:, 2] == 0, 0:2]

    pos_edge[:, 1] -= nb_drugs
    neg_edge[:, 1] -= nb_drugs
    label_pos = coo_matrix((np.ones(pos_edge.shape[0]), (pos_edge[:, 0], pos_edge[:, 1])),
                           shape=(nb_drugs, nb_proteins)).toarray()
    label_pos = torch.from_numpy(label_pos).type(torch.FloatTensor).view(-1)
    # labels = torch.LongTensor(np.where(label_pos)[1])

    ## 构造邻接矩阵 build graph
    nb_all = nb_drugs+nb_proteins
    train_edge = allpairs[allpairs[:, 2] == 1, 0:2]

    edge = np.vstack((train_edge, train_edge[:, [1, 0]]))  ##交换顺序，还是有相互作用
    adj = coo_matrix((np.ones(edge.shape[0]), (edge[:, 0], edge[:, 1])),
                           shape=(nb_all, nb_all), dtype=np.float32)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))

    positive_adj = torch.zeros((nb_all, nb_all))
    for inter_k in edge:
        drug_node_id = int(inter_k[0])
        protein_node_id = int(inter_k[1])
        positive_adj[drug_node_id][protein_node_id] = 1
    sim = pos_transform_adj(nb_all,positive_adj,sample_type='positive',common_neibor = 5)
    adj1 = positive_adj + sim
    adj1 = normalize_adj(adj1 + np.eye(adj1.shape[0]))
    adj1 = torch.FloatTensor(adj1)

    return drug_set, protein_set, adj1, label_pos, train_mask, test_mask,edge


def pos_transform_adj(node_num, adj, sample_type='positive',common_neibor=3):
    # neighbor_mask = (adj.repeat(1, node_num).view(node_num * node_num, -1) + adj.repeat(node_num, 1))  # n^2, n
    adj_transform = torch.zeros_like(adj)
    ones_vec_0 = torch.ones_like(adj[0])
    zeros_vec_0 = torch.zeros_like(adj[0])
    for row in tqdm(range(node_num)):
        row_adj = adj[row]
        for col in range(node_num):
            col_adj = adj[col]
            neighbor_mask = row_adj + col_adj
            if sample_type == 'positive':
                com_num = torch.where(neighbor_mask == 2, ones_vec_0, zeros_vec_0).sum(0).item()
            elif sample_type == 'negative':
                com_num = torch.where(neighbor_mask == 0, ones_vec_0, zeros_vec_0).sum(0).item()
            else:
                print("wrong_type")
            if com_num > common_neibor: adj_transform[row][col] = 1
    return adj_transform


def normalize_adj(mx):
    rowsum = mx.sum(1)
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    # r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    r_mat_inv_sqrt = np.diag(r_inv_sqrt)
    # r_mat_inv_sqrt = torch.from_numpy(r_inv_sqrt)
    return r_mat_inv_sqrt.dot(mx).dot(r_mat_inv_sqrt)


def get_random_folds(tsize, foldcount):
    folds = []
    indices = set(range(tsize))
    foldsize = tsize / foldcount
    leftover = tsize % foldcount
    for i in range(foldcount):
        sample_size = foldsize
        if leftover > 0:
            sample_size += 1
            leftover -= 1
        fold = random.sample(indices, int(sample_size))  ##从指定序列中随机获取指定长度的片断
        indices = indices.difference(fold)  ##返回一个集合，其中包含仅存在于集合 x 中而不存在于集合 y 中的项目
        folds.append(fold)

    # assert stuff
    foldunion = set([])
    for find in range(len(folds)):
        fold = set(folds[find])
        assert len(fold & foldunion) == 0, str(find)
        foldunion = foldunion | fold
    assert len(foldunion & set(range(tsize))) == tsize

    return folds

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


