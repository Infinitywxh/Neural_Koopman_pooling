import torch
import IPython
import argparse

parser = argparse.ArgumentParser(
        description='classwise')
parser.add_argument('--load-path', type=str)
parser.add_argument('--save-path', type=str)
parser.add_argument('--num-class', type=int)
arg = parser.parse_args()

model = torch.load(arg.load_path)
K_double = model['K'].double()# .cuda()

e, v = torch.linalg.eig(K_double)
indices = e.abs().sort(dim=-1).indices  # 120*256

for i in range(arg.num_class):
    rank = 32
    idx = indices[i, -rank:]  # 120 * 32
    eigen_norm = e[i][idx].abs()
    e[i][idx] = e[i][idx] / eigen_norm * torch.pow(eigen_norm, 0.25)

K_reconstruct = (v @ torch.diag_embed(e) @ torch.linalg.inv(v)).real.float()
model['K'] = K_reconstruct
torch.save(model, arg.save_path)
