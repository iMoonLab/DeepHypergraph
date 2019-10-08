import numpy as np
import scipy.io as scio
import torch


def load_ft(data_dir, feature_name='GVCNN'):
    data = scio.loadmat(data_dir)
    lbls = data['Y'].astype(np.long)
    if lbls.min() == 1:
        lbls = lbls - 1
    idx = data['indices'].item()

    if feature_name == 'MVCNN':
        fts = data['X'][0].item().astype(np.float32)
    elif feature_name == 'GVCNN':
        fts = data['X'][1].item().astype(np.float32)
    else:
        print(f'wrong feature name{feature_name}!')
        raise IOError

    idx_train = (idx == 1)
    idx_test = (idx == 0)
    return torch.tensor(fts), torch.tensor(lbls).squeeze(), \
           torch.tensor(idx_train).squeeze().bool(), \
           torch.tensor(idx_test).squeeze().bool()
