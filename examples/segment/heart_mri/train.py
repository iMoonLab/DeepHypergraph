import torch
import torch.nn.functional as F

from HyperG.models import HGNN
from .data_helper import preprocess, split_train_val

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

k_nearest = 7
patch_size = (5, 5)
data_list = split_train_val('/repository/HyperG_example/example_data/heart_mri/processed',
                            save_dir='/repository/HyperG_example/tmp/heart_mri', ratio=0.8)
x, H, lbl, mask_train, mask_val = preprocess(data_list, patch_size, k_nearest)

x_ch = x.size(1)
n_class = lbl.max() + 1
model = HGNN(x_ch, n_class, hidens=[16])

model, x, H, lbl, mask_train, mask_val = model.to(device), x.to(device), H.to(device), \
                                         lbl.to(device), mask_train.to(device), mask_val.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()
    output = model(x, H)
    F.nll_loss(output[mask_train], lbl[mask_train]).backward()
    optimizer.step()


def val():
    model.eval()
    output = model(x, H)

    train_pred = output[mask_train].max(1)[1]
    train_acc = train_pred.eq(lbl[mask_train]).sum().item() / mask_train.sum().item()

    val_pred = output[mask_val].max(1)[1]
    val_acc = val_pred.eq(lbl[mask_val]).sum().item() / mask_val.sum().item()

    return train_acc, val_acc


if __name__ == '__main__':
    best_acc = 0.0
    for epoch in range(1, 51):
        train()
        train_acc, val_acc = val()
        if val_acc > best_acc:
            best_acc = val_acc
        print(f'Epoch: {epoch}, Train:{train_acc:.4f}, Val:{val_acc:.4f}, Best Val:{best_acc}:.4f')
