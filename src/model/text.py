import os
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


def get_model():
    """
    Given a DataLoader loading only captions fit into a CountVectorizer to obtain the TextEncoder model to use for training
    """
    # 4154 words appear at least 10 times in the full 30k dataset
    # train_dataset = CaptionFlickr30kDataset(root='/hdd/master/tfm/flickr30k_images',
    #                                         split_root='/hdd/master/tfm/flickr30k_images/flickr30k_entities',
    #                                         split='train')
    test_dataset = CaptionFlickr30kDataset(root='/hdd/master/tfm/flickr30k_images',
                                           split_root='/hdd/master/tfm/flickr30k_images/flickr30k_entities',
                                           split='test')
    # val_dataset = CaptionFlickr30kDataset(root='/hdd/master/tfm/flickr30k_images',
    #                                       split_root='/hdd/master/tfm/flickr30k_images/flickr30k_entities',
    #                                       split='val')

    corpus = [test_dataset[i][1] for i in range(len(test_dataset))]
    # corpus = [train_dataset[i][1] for i in range(len(train_dataset))] + \
    #          [test_dataset[i][1] for i in range(len(test_dataset))] + \
    #          [val_dataset[i][1] for i in range(len(val_dataset))]

    vectorizer = CountVectorizer(tokenizer=spacy_tokenizer, stop_words='english')
    vectorizer.fit(corpus)
    with open('vectorizer_tokenizer_stop_words-test.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)


if __name__ == '__main__':
     get_model()
