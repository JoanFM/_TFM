import os
import sys
import pickle

import torch
from sklearn.feature_extraction.text import CountVectorizer
from src.model.spacy_tokenizer import spacy_tokenizer

from src.dataset.dataset import CaptionFlickr30kDataset

cur_dir = os.path.dirname(os.path.abspath(__file__))


class TextEncoder:
    """
    The encoder that given a text, extracts a sparse vector where every index corresponds to an image or token in the vocabulary
    """

    def __init__(self, model_path, **kwargs):
        super().__init__()
        if model_path:
            with open(model_path, 'rb') as f:
                self.vectorizer = pickle.load(f)

    def forward(self, x):
        return torch.Tensor(self.vectorizer.transform(x).toarray())


def get_model(min_df=10, max_df=1.0):
    """
    Given a DataLoader loading only captions fit into a CountVectorizer to obtain the TextEncoder model to use for training
    """
    # 4154 words appear at least 10 times in the full 30k dataset
    train_dataset = CaptionFlickr30kDataset(root='/hdd/master/tfm/flickr30k_images',
                                            split_root='/hdd/master/tfm/flickr30k_images/flickr30k_entities',
                                            split='train')

    corpus = [train_dataset[i][1] for i in range(len(train_dataset))]

    print(f' Fitting a vectorizer with a corpus of {len(corpus)} sentences with min_df {min_df} and max_df {max_df}')
    vectorizer = CountVectorizer(tokenizer=spacy_tokenizer, stop_words='english', min_df=min_df, max_df=max_df, binary=True)
    vectorizer.fit(corpus)
    with open(f'vectorizer_tokenizer_stop_words_{min_df}_{max_df}.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f' length of vocabulary is {len(vectorizer.vocabulary_)}')


if __name__ == '__main__':
    min_df = int(sys.argv[1])
    max_df = float(sys.argv[2])
    get_model(min_df, max_df)
