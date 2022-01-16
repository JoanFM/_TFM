import torch
import numpy as np
import torch.nn as nn


class DenseVisualFeatureExtractor(nn.Module):
    """
    The class in charge of extracting the dense features from a pretrained model
    """

    def __init__(self,
                 backbone_model: str = 'resnet50',
                 ):
        super().__init__()
        import torchvision.models as models
        self.pool_fn = None
        self.model = getattr(models, backbone_model)(pretrained=True)
        self.layer = self.model._modules.get('avgpool')

    @property
    def output_dim(self):
        random_image_array = np.random.random((8, 3, 224, 224))
        content = torch.Tensor(random_image_array)
        return self(content).shape[1]

    def _get_features(self, content):
        feature_map = None

        def get_activation(model, model_input, output):
            nonlocal feature_map
            feature_map = output

        handle = self.layer.register_forward_hook(get_activation)
        self.model(content)
        handle.remove()

        return feature_map

    def forward(self, x):
        return self._get_features(x)


class ImageEncoder(nn.Module):
    """
    The Encoder to train that adds a set of Fully Connected Layers expanding from the ouptut of a dense embedding into a sparse one
    """

    def __init__(self, common_layer_size=512, **kwargs):
        super().__init__()
        self.feature_extractor = DenseVisualFeatureExtractor(**kwargs)
        previous_layer = self.feature_extractor.output_dim
        self.common_space_embedding = nn.Linear(in_features=previous_layer, out_features=common_layer_size)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size()[0], -1)
        result = self.common_space_embedding(x)
        return result
