import torch
import os
from deepchem.models.torch_models import AttentiveFPModel


def load_attfp(name, n_tasks=15, mode='regression'):
    num_layers, num_timesteps, graph_feat_size, dropout = os.path.basename(name).split('_')[-4:]
    if torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'
    model = AttentiveFPModel(num_layers=int(num_layers),
                             num_timesteps=int(num_timesteps),
                             graph_feat_size=int(graph_feat_size),
                             dropout=float(dropout),
                             device=device,
                             mode=mode,
                             n_tasks=n_tasks)
    model.restore(model_dir=name)

    return model