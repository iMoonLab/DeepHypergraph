import openslide
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from SuperMoon.models import ResNetFeature


def extract_ft(slide_dir: str, patch_coors, depth=34, batch_size=16):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    slide = openslide.open_slide(slide_dir)

    model_ft = ResNetFeature(depth=depth, pooling=True, pretrained=True)
    model_ft = model_ft.to(device)
    model_ft.eval()

    dataset = Patches(slide, patch_coors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    fts = []
    with tqdm(total=len(dataset)) as pbar:
        for _patches in dataloader:
            _patches = _patches.to(device)
            with torch.no_grad():
                _fts = model_ft(_patches)
            fts.append(_fts)
            pbar.update(_patches.size(0))

    fts = torch.cat(fts, dim=0)
    assert fts.size(0) == len(patch_coors)
    return fts


class Patches(Dataset):

    def __init__(self, slide: openslide, patch_coors) -> None:
        super().__init__()
        self.slide = slide
        self.patch_coors = patch_coors
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx: int):
        coor = self.patch_coors[idx]
        img = self.slide.read_region((coor[0], coor[1]), 0, (coor[2], coor[3])).convert('RGB')
        return self.transform(img)

    def __len__(self) -> int:
        return len(self.patch_coors)
