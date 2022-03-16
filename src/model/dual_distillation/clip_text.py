from transformers import CLIPModel, CLIPTokenizer


class CLIPTextEncoder:

    def __init__(self, **kwargs):
        self.model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.model.eval()
        self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.model.to(device)

    def __call__(self, X, *args, **kwargs):
        return self.forward(X)

    def forward(self, texts):
        input_tokens = self.tokenizer(
            texts,
            max_length=77,
            padding='longest',
            truncation=True,
            return_tensors='pt',
        )
        input_tokens = {k: v.to(self.device) for k, v in input_tokens.items()}
        return self.model.get_text_features(**input_tokens)
