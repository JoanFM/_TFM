import torch
import torch.nn as nn

import numpy as np


class DenseVisualFeatureExtractor:

    def __init__(self,
                 backbone_model: str = 'vgg16',
                 pool_strategy: str = 'mean',
                 ):
        super().__init__()
        import torchvision.models as models

        self.pool_fn = None
        self.model = getattr(models, backbone_model)(pretrained=True)
        self.model = self.model.features.eval()
        if pool_strategy not in ('mean', 'max', None):
            raise NotImplementedError(f'unknown pool_strategy: {pool_strategy}')
        else:
            self.pool_fn = getattr(torch, pool_strategy)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    @property
    def on_gpu(self):
        return False

    @property
    def output_dim(self):
        return 512

    def _get_features(self, content):
        return self.model(content)

    def _get_pooling(self, feature_map):
        if feature_map.ndim == 2 or self.pool_fn is None:
            return feature_map
        return self.pool_fn(feature_map, axis=(2, 3))

    def encode(self, content, *args, **kwargs):
        _feature = self._get_features(content).detach()
        if self.on_gpu:
            _feature = _feature.cpu()
        out = self._get_pooling(_feature)
        return out


class ImageSiamese(nn.Module):

    def __init__(self, layer_size=None, **kwargs):
        super().__init__()
        if layer_size is None:
            layer_size = [512, 2056, 4128, 13926] #8256, 13926]
        self.feature_extractor = DenseVisualFeatureExtractor(**kwargs)
        modules = []

        previous_layer = self.feature_extractor.output_dim
        for layer in layer_size:
            modules.append(nn.Linear(in_features=previous_layer, out_features=layer))
            modules.append(nn.ReLU(inplace=True))
            previous_layer = layer
        self.sparse_encoder = nn.Sequential(*modules)

    def forward_one(self, x):
        x = self.feature_extractor.encode(x)
        x = x.view(x.size()[0], -1)
        x = self.sparse_encoder(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        return out1, out2
