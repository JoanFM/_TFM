import torch
import torch.nn as nn

import numpy as np


class DenseVisualFeatureExtractor:

    def __init__(self,
                 backbone_model: str = 'resnet101',
                 ):
        super().__init__()
        import torchvision.models as models
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        device = torch.device(dev)

        self.pool_fn = None
        self.model = getattr(models, backbone_model)(pretrained=True)
        self.model = self.model.eval()
        self.layer = self.model._modules.get('avgpool')
        self.model.to(device)

    @property
    def output_dim(self):
        return 2048

    def _get_features(self, content):
        feature_map = None

        def get_activation(model, model_input, output):
            nonlocal feature_map
            feature_map = output.detach()

        handle = self.layer.register_forward_hook(get_activation)
        self.model(content)
        handle.remove()

        return feature_map

    def encode(self, content, *args, **kwargs):
        _feature = self._get_features(content)
        return _feature


class ImageEncoder(nn.Module):

    def __init__(self, layer_size=None, **kwargs):
        super().__init__()
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        device = torch.device(dev)
        if layer_size is None:
            layer_size = [4096, 8192, 13439]
        self.feature_extractor = DenseVisualFeatureExtractor(**kwargs)
        modules = []

        previous_layer = self.feature_extractor.output_dim
        for layer in layer_size:
            modules.append(nn.Linear(in_features=previous_layer, out_features=layer))
            modules.append(nn.ReLU(inplace=True))
            previous_layer = layer
        self.sparse_encoder = nn.Sequential(*modules)
        self.sparse_encoder.to(device)

    def forward(self, x):
        x = self.feature_extractor.encode(x)
        x = x.view(x.size()[0], -1)
        return self.sparse_encoder(x)
