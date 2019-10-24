from __future__ import print_function, division

import copy
import os.path as osp
import time

import torch
import torch.nn as nn
import torch.optim as optim
from data_helper import get_dataloaders
from torch.optim import lr_scheduler

from SuperMoon.models import ResNet_HGNN
from SuperMoon.utils import check_dir

data_root = '/repository/HyperG_example/example_data/breast_pathology/processed'
result_root = '/repository/HyperG_example/tmp/breast_pathology'
n_class = 4
k_nearest = 7
depth = 34
hiddens = [512]
batch_size = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_save_dir = osp.join(result_root, 'model_best.pth')

# check directions
assert check_dir(data_root, False)
check_dir(result_root)

dataloaders, dataset_sizes, _ = get_dataloaders(data_root, batch_size)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the models
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best models weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    model_ft = ResNet_HGNN(n_class=n_class, depth=depth, k_nearest=k_nearest, hiddens=hiddens)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.01, weight_decay=5e-4)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
    torch.save(model_ft.cpu().state_dict(), model_save_dir)
