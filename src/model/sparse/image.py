import torch
import numpy as np
import torch.nn as nn


class DenseVisualFeatureExtractor:

    """
    The class in charge of extracting the dense features from a pretrained model
    """

    def __init__(self,
                 backbone_model: str = 'resnet50',
                 ):
        super().__init__()
        import torchvision.models as models
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        self.device = torch.device(dev)

        self.pool_fn = None
        self.model = getattr(models, backbone_model)(pretrained=True)
        self.model = self.model.eval()
        self.layer = self.model._modules.get('avgpool')
        self.model.to(self.device)

    @property
    def output_dim(self):
        random_image_array = np.random.random((8, 3, 224, 224))
        content = torch.Tensor(random_image_array)
        content = content.to(self.device)
        return self.encode(content).shape[1]

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
        return self._get_features(content)


class ImageEncoder(nn.Module):
    """
    The Encoder to train that adds a set of Fully Connected Layers expanding from the ouptut of a dense embedding into a sparse one
    """

    def __init__(self, layer_size=None, **kwargs):
        super().__init__()
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        self.device = torch.device(dev)
        if layer_size is None:
            layer_size = [4096, 1062]
        self.feature_extractor = DenseVisualFeatureExtractor(**kwargs)
        modules = []

        previous_layer = self.feature_extractor.output_dim
        for i, layer in enumerate(layer_size):
            modules.append(nn.Linear(in_features=previous_layer, out_features=layer))
            modules.append(nn.ReLU(inplace=True))
            previous_layer = layer
        #
        modules.pop()
        self.sparse_encoder = nn.Sequential(*modules)
        print(f' sparse encoder {self.sparse_encoder}')
        self.sparse_encoder.to(self.device)
        self.extra_relu = nn.ReLU(inplace=True)
        self.extra_relu.to(self.device)

    def forward(self, x):
        x = self.feature_extractor.encode(x)
        x = x.view(x.size()[0], -1)
        result = self.sparse_encoder(x)
        if self.training:
            return result
        else:
            return self.extra_relu(result)
