import torch.nn as nn
import torchvision


class ResNetFeature(nn.Module):

    def __init__(self, depth=34, pooling=False, pretrained=True):
        super().__init__()
        assert depth in [18, 34, 50, 101, 152]
        self.pooling = pooling

        if depth == 18:
            base_model = torchvision.models.resnet18(pretrained=pretrained)
            self.len_feature = 512
            self.features = nn.Sequential(*list(base_model.children())[:-2])
        elif depth == 34:
            base_model = torchvision.models.resnet34(pretrained=pretrained)
            self.len_feature = 512
            self.features = nn.Sequential(*list(base_model.children())[:-2])
        elif depth == 50:
            base_model = torchvision.models.resnet50(pretrained=pretrained)
            self.len_feature = 2048
            self.features = nn.Sequential(*list(base_model.children())[:-2])
        elif depth == 101:
            base_model = torchvision.models.resnet101(pretrained=pretrained)
            self.len_feature = 2048
            self.features = nn.Sequential(*list(base_model.children())[:-2])
        elif depth == 152:
            base_model = torchvision.models.resnet152(pretrained=pretrained)
            self.len_feature = 2048
            self.features = nn.Sequential(*list(base_model.children())[:-2])
        else:
            raise NotImplementedError(f'ResNet-{depth} is not implemented!')

    def forward(self, x):
        x = self.features(x)

        if self.pooling:
            # -> batch_size x C x N
            x = x.view(x.size(0), x.size(1), -1)
            # -> batch_size x C
            x = x.mean(dim=-1)
            return x
        else:
            # Attention! No reshape!
            return x


class ResNetClassifier(nn.Module):

    def __init__(self, n_class, len_feature):
        super().__init__()
        self.len_feature = len_feature
        self.classifier = nn.Linear(self.len_feature, n_class)

    def forward(self, x):
        # -> batch_size x C x N
        x = x.view(x.size(0), x.size(1), -1)

        # -> batch_size x C
        x = x.mean(dim=-1)

        x = self.classifier(x)
        return x
