import torch
import numpy as np
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ImageEncoder(nn.Module):
    """
    The Encoder to train that adds a set of Fully Connected Layers expanding from the ouptut of a dense embedding into a sparse one
    """

    def __init__(self, common_layer_size=512, backbone_model: str = 'resnet50', **kwargs):
        super().__init__()
        import torchvision.models as models
        self.model = getattr(models, backbone_model)(pretrained=True)
        self.model.fc = Identity()
        previous_layer = self.pre_linear_dim()
        self.common_space_embedding = nn.Linear(in_features=previous_layer, out_features=common_layer_size)

    def pre_linear_dim(self):
        random_image_array = np.random.random((8, 3, 224, 224))
        content = torch.Tensor(random_image_array)
        return self(content).shape[1]

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        x = self.common_space_embedding(x)
        result = torch.nn.functional.normalize(x, p=2, dim=1)
        return result
