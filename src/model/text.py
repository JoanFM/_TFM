from sklearn.feature_extraction.text import CountVectorizer
import pickle
import torch


class TextEncoder:

    def __init__(self, vocab_path, **kwargs):
        super().__init__()
        # if vocab_path:
        #     with open(vocab_path, 'rb') as f:
        #         self.vocab = pickle.load(f)
        #
        # def mytokenizer(text):
        #     return text.split()
        #
        # self.vectorizer = CountVectorizer(vocabulary=self.vocab, tokenizer=mytokenizer)

    def forward(self, x):
        import numpy as np
        return torch.Tensor(np.random.random((len(x), 8256)))
        #return self.vectorizer.transform(x)
