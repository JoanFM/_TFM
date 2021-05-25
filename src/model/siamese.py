from typing import List

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
        self.model.features.eval()
        if pool_strategy not in ('mean', 'max', None):
            raise NotImplementedError(f'unknown pool_strategy: {pool_strategy}')
        else:
            self.pool_fn = getattr(np, pool_strategy)

    @property
    def on_gpu(self):
        return False

    @property
    def output_dim(self):
        return 256

    def _get_features(self, content):
        return self.model(content)

    def _get_pooling(self, feature_map: 'np.ndarray') -> 'np.ndarray':
        if feature_map.ndim == 2 or self.pool_fn is None:
            return feature_map
        return self.pool_fn(feature_map, axis=(2, 3))

    def encode(self, content: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':

        _input = torch.from_numpy(content.astype('float32'))
        if self.on_gpu:
            _input = _input.cuda()
        _feature = self._get_features(_input).detach()
        if self.on_gpu:
            _feature = _feature.cpu()
        _feature = _feature.numpy()
        return self._get_pooling(_feature)


class ImageSiamese(nn.Module):

    def __init__(self, layer_size=None, **kwargs):
        super().__init__()
        if layer_size is None:
            layer_size = [512, 1024, 2056, 4128, 8256]
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


class TextEncoder:

    def __init__(self, vocab_path, **kwargs):
        super().__init__()
        from sklearn.feature_extraction.text import CountVectorizer
        import pickle
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

        def mytokenizer(text):
            return text.split()

        self.vectorizer = CountVectorizer(vocabulary=self.vocab, tokenizer=mytokenizer)

    def forward(self, x):
        return self.vectorizer.transform(x)
