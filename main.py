from opt import *
from utils import *
from model import H2GNN
import time
import random
import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import torch.optim as optim
from data_load import dataload
from setting import process
from NodeRepresentation import GNNNet,combined
import torch

args = parser.parse_args()
start_time = time.time()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

"""Load preprocessed data."""
DATASET = "Davis"
# DATASET = "KIBA"
# DATASET = "DrugBank"

data_new, nb_drugs, nb_proteins = dataload(DATASET)
nb_all = nb_drugs+nb_proteins
drug_set, protein_set, adj, labels, idx_train, idx_test,edge = process(data_new, nb_drugs, nb_proteins,DATASET,foldcount=5,setting = 2)
node = GNNNet()

for batch, (drug,pro) in enumerate(zip(drug_set,protein_set)):
    features = node(drug.x, drug.edge_index, drug.batch, pro.x, pro.edge_index, pro.batch)
    # pro_fea = pro_node(pro[0])
features = normalize_features(features.detach().numpy())
features = torch.FloatTensor(features)

model = H2GNN(n_node=features.shape[0])
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr)
myloss = nn.BCEWithLogitsLoss()
gamma_value = 0.3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
features = features.to(device)
adj = adj.to(device)
labels = labels.to(device)
idx_train = idx_train.to(device)
idx_test = idx_test.to(device)

acc_reuslt = []
f1_result = []

def Train(epoch):
    model.train()
    x_hat, z_hat, adj_hat, z_ae, z_igae, z_tilde = model(features, adj)
    output = adj_hat[:nb_drugs, nb_drugs:nb_all + 1]  ##邻接矩阵  四分矩阵的右上角就是预测结果
    pre = output.reshape(-1)
    pre = torch.sigmoid(pre)
    loss_train = myloss(pre[idx_train], labels[idx_train])
    # loss_ae = F.mse_loss(x_hat, features)  ##特征矩阵 重建损失
    # loss_w = F.mse_loss(z_hat, torch.spmm(adj, features))
    # loss_a = F.mse_loss(adj_hat, adj)
    # loss_igae = loss_w + gamma_value * loss_a
    # loss = loss_ae + loss_train +loss_igae
    # loss = loss.requires_grad_(True)
    loss = loss_train.requires_grad_(True)
    print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss.data.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def test():
    model.eval()
    with torch.no_grad():
        x_hat, z_hat, adj_hat, z_ae, z_igae, z_tilde = model(features, adj)
        output = adj_hat[:nb_drugs, nb_drugs:nb_all + 1]
        pre = output.reshape(-1)
        loss_test = myloss(pre[idx_test], labels[idx_test])  ##BCEloss
        # loss_ae = F.mse_loss(x_hat, features)
        loss = loss_test
        yp = pre[idx_test].cpu().detach().numpy()
        ytest = labels[idx_test].cpu().detach().numpy()
        AUC, AUPR, F1, ACC = metrics_graph(ytest,yp)
        print('test loss: ', str(round(loss.item(), 4)))
        print('test auc: ' + str(round(AUC, 4)) + '  test aupr: ' + str(round(AUPR, 4)) +
              '  test f1: ' + str(round(F1, 4)) + '  test acc: ' + str(round(ACC, 4)))
    return AUC, AUPR, F1, ACC

#------main
final_AUC = 0;final_AUPR = 0;final_F1 = 0;final_ACC = 0
for epoch in range(args.epochs):
    print('\nepoch: ' + str(epoch))
    Train(epoch)
    AUC, AUPR, F1, ACC = test()
    if (AUC > final_AUC):
        best_epoch = epoch
        final_AUC = AUC;final_AUPR = AUPR;final_F1 = F1;final_ACC = ACC
elapsed = time.time() - start_time
print('---------------------------------------')
print("Train in " + DATASET)
print('Elapsed time: ', round(elapsed, 4))
print("best_epoch: " + str(best_epoch))
print('Final_AUC: ' + str(round(final_AUC, 4)) + '  Final_AUPR: ' + str(round(final_AUPR, 4)) +
      '  Final_F1: ' + str(round(final_F1, 4)) + '  Final_ACC: ' + str(round(final_ACC, 4)))
print('---------------------------------------')