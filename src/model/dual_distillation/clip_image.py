from transformers import CLIPFeatureExtractor, CLIPModel


class CLIPImageEncoder:

    def __init__(self, **kwargs):
        self.model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.model.eval()
        self.preprocessor = CLIPFeatureExtractor.from_pretrained('openai/clip-vit-base-patch32')
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.model.to(device)

    def __call__(self, X, *args, **kwargs):
        return self.forward(X)

    def forward(self, images):
        # input_tokens = self.preprocessor(
        #     images=images,
        #     return_tensors='pt',
        # )
        # input_tokens = {
        #     k: v.to(self.device) for k, v in input_tokens.items()
        # }
        a =  self.model.get_image_features(images)
        return self.model.get_image_features(images)
