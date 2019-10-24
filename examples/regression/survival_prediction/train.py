import copy
import os.path as osp
import time

import torch
import torch.nn as nn
import torch.optim as optim
from data_helper import split_train_val, preprocess, get_dataloaders
from model import HGNN_reg
from torch.optim import lr_scheduler

from SuperMoon.hyedge import neighbor_distance
from SuperMoon.utils import check_dir
from SuperMoon.utils.meter import CIndexMeter

# initialize parameters
data_root = '/repository/HyperG_example/example_data/survival_prediction/processed'
result_root = '/repository/HyperG_example/tmp/survival_prediction'
num_sample = 2000
patch_size = 256
mini_frac = 32
k_nearest = 7
n_target = 1
hiddens = [128]
num_epochs = 50

svs_dir = osp.join(data_root, 'svs')
patch_ft_dir = osp.join(data_root, 'patch_ft')
sampled_vis = osp.join(result_root, 'sampled_vis')
patch_coors_dir = osp.join(result_root, 'patch_coors')

split_dir = osp.join(result_root, 'split.pkl')
model_save_dir = osp.join(result_root, 'model_best.pth')

# check directions
assert check_dir(data_root, make=False)
assert check_dir(svs_dir, make=False)
assert check_dir(patch_ft_dir, make=False)
check_dir(result_root)
check_dir(sampled_vis)
check_dir(patch_coors_dir)

data_dict = split_train_val(svs_dir, ratio=0.8, save_split_dir=split_dir, resplit=False)

preprocess(data_dict, patch_ft_dir, patch_coors_dir, num_sample=num_sample,
           patch_size=patch_size, sampled_vis=sampled_vis, mini_frac=mini_frac)

dataloaders, dataset_sizes, len_ft = get_dataloaders(data_dict, patch_ft_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    mini_loss = 1e8
    c_index_train = CIndexMeter()
    c_index_val = CIndexMeter()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set models to training mode
            else:
                model.eval()  # Set models to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            c_index = CIndexMeter()

            # Iterate over data.
            for fts, st in dataloaders[phase]:
                fts = fts.to(device).squeeze(0)
                st = st.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    H = neighbor_distance(fts, k_nearest)
                    pred = model(fts, H)
                    # print(f'pred: {pred.item():.4f}, st: {st.item():.4f}')
                    loss = criterion(pred, st)
                    c_index.add(pred, st)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f}, C Index: {c_index.value():.4f}')

            # deep copy the models
            if phase == 'val' and epoch_loss < mini_loss:
                mini_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Minimize val Loss: {:4f}'.format(mini_loss))

    # load best models weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    model = HGNN_reg(in_ch=len_ft, n_target=n_target, hiddens=hiddens)
    model = model.to(device)

    criterion = nn.MSELoss()

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)

    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)
    torch.save(model.cpu().state_dict(), model_save_dir)
