import argparse

parser = argparse.ArgumentParser(description='H2GnnDTI', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# setting
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=3)
parser.add_argument('--alpha_value', type=float, default=0.2)
parser.add_argument('--lambda_value', type=float, default=10)
parser.add_argument('--gamma_value', type=float, default=1e3)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--n_z', type=int, default=20)
parser.add_argument('--n_input', type=int, default=160)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--show_training_details', type=bool, default=False)
parser.add_argument('--norm', type=str, default='false',
                        help='normalization')

# AE structure parameter from DFCN
parser.add_argument('--ae_n_enc_1', type=int, default=128)
parser.add_argument('--ae_n_enc_2', type=int, default=256)
parser.add_argument('--ae_n_enc_3', type=int, default=512)
parser.add_argument('--ae_n_dec_1', type=int, default=512)
parser.add_argument('--ae_n_dec_2', type=int, default=256)
parser.add_argument('--ae_n_dec_3', type=int, default=128)

# IGAE structure parameter from DFCN
parser.add_argument('--gae_n_enc_1', type=int, default=128)
parser.add_argument('--gae_n_enc_2', type=int, default=256)
parser.add_argument('--gae_n_enc_3', type=int, default=20)
parser.add_argument('--gae_n_dec_1', type=int, default=20)
parser.add_argument('--gae_n_dec_2', type=int, default=256)
parser.add_argument('--gae_n_dec_3', type=int, default=128)


##节点表示的参数
parser.add_argument('--char_dim', type=int, default=64, help='embedding_dim')
parser.add_argument('--conv', type=int, default=40, help='conv')
parser.add_argument('--drug_kernel', type=list, default=[4, 6, 8], help='drug_kernel')
parser.add_argument('--protein_kernel', type=int, default=[4, 8, 12], help='protein_kernel')

args = parser.parse_args()
