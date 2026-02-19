import torch
import torch.nn as nn
import torch.linalg

def generalised_gcca_loss(views, top_k, eps):
    n_views = len(views)
    for i, h in enumerate(views):
        if torch.isnan(h).any():
            raise ValueError(f"NaN detected in view {i}")

    centered_views = [h - torch.mean(h, dim=0) for h in views]

    AT_list = []
    for h_bar in centered_views:
        U, S, V = torch.svd(h_bar, some=True)
        k = min(top_k, S.size(0))
        U_k = U[:, :k]
        S_k = S[:k]
        
        S2 = torch.mul(S_k, S_k)
        T_diag = torch.sqrt(S2 / (S2 + eps))
        T = torch.diag(T_diag)
        
        AT = torch.mm(U_k, T)
        AT_list.append(AT)

    M_tilde = torch.cat(AT_list, dim=1)
    Q, R = torch.linalg.qr(M_tilde)
    U_r, S_r, V_r = torch.linalg.svd(R, full_matrices=False)
    
    k_final = min(top_k, U_r.size(1))
    G = torch.mm(Q, U_r[:, :k_final])
    corr = torch.sum(S_r[:k_final])
    
    return -corr, G

class MlpNet(nn.Module):
    def __init__(self, layer_sizes, input_size):
        super(MlpNet, self).__init__()
        layers = []
        all_sizes = [input_size] + layer_sizes 
        
        for i in range(len(all_sizes) - 1):
            is_last = (i == len(all_sizes) - 2)
            layers.append(nn.Linear(all_sizes[i], all_sizes[i + 1]))
            if is_last:
                layers.append(nn.Sigmoid())
                layers.append(nn.BatchNorm1d(num_features=all_sizes[i + 1], affine=False))
            else:
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DeepGCCA(nn.Module):
    def __init__(self, view_configs, device):
        super(DeepGCCA, self).__init__()
        self.encoders = nn.ModuleList([
            MlpNet(cfg['layer_sizes'], cfg['input_size']) for cfg in view_configs
        ])
        self.to(device)

    def forward(self, views):
        return [encoder(v) for encoder, v in zip(self.encoders, views)]