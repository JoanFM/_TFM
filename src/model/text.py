import os
import pickle

import torch
from sklearn.feature_extraction.text import CountVectorizer

from src.dataset.dataset import Flickr30kDataset

cur_dir = os.path.dirname(os.path.abspath(__file__))


class TextEncoder:

    def __init__(self, model_path, **kwargs):
        super().__init__()
        if model_path:
            with open(model_path, 'rb') as f:
                self.vectorizer = pickle.load(f)

    def forward(self, x):
        return torch.Tensor(self.vectorizer.transform(x).toarray())


def get_model():
    # 4154 words appear at least 10 times in the full 30k dataset
    import spacy
    nlp = spacy.load('en_core_web_sm')
    dataset = Flickr30kDataset(root=os.path.join(cur_dir, '../../flickr30k_images'))
    vocab = set()
    c = []
    for (_, captions) in dataset:
        c.extend(captions)
    for i, doc in enumerate(nlp.pipe(c)):
        for token in doc:
            if not token.is_punct and not token.is_space:
                vocab.add(token.lemma_.lower())
        if i % 500 == 0:
            print(f' vocab size {len(vocab)} when processed {int(i / 5)} images')

    print(f' vocab size {len(vocab)}')

    def corpus():
        for lemma in vocab:
            yield lemma

    vectorizer = CountVectorizer()
    vectorizer.fit(corpus())
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)


if __name__ == '__main__':
    get_model()
