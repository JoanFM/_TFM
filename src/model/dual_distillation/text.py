import os
import sys
import torch
import torch.nn as nn
import pickle

import numpy as np
from gensim.models import KeyedVectors
import spacy

nlp = spacy.load('en_core_web_sm')

# Load word2vec pre-train model


cur_dir = os.path.dirname(os.path.abspath(__file__))

TEXT_EMBEDDING_VECTORIZER_PATH = os.getenv('TEXT_EMBEDDING_VECTORIZER_PATH', 'vectorizer.pkl')
# The root path where the flickr30k dataset is found
DATASET_ROOT_PATH = os.getenv('DATASET_ROOT_PATH', '/hdd/master/tfm/flickr30k_images')
# The root path where the flickr30k entities per split is kept
DATASET_SPLIT_ROOT_PATH = os.getenv('DATASET_SPLIT_ROOT_PATH', '/hdd/master/tfm/flickr30k_images/flickr30k_entities')


class TextEncoder(nn.Module):
    def __init__(
            self,
            embd_dim=512,
            model_path='filtered_f30k_word2vec.model',
            output_dim=2048,
    ):
        super().__init__()
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        self.device = torch.device(dev)
        self.gensim_model = KeyedVectors.load(model_path)
        weights = torch.FloatTensor(self.gensim_model.vectors)
        self.word_embd = nn.Embedding.from_pretrained(weights)
        self.word_embd.to(self.device)
        self.fc1 = nn.Linear(self.word_embd.embedding_dim, output_dim)
        self.fc1.to(self.device)
        self.relu = nn.ReLU(inplace=True)
        self.relu.to(self.device)
        self.fc2 = nn.Linear(output_dim, embd_dim)
        self.fc2.to(self.device)

    def _words_to_token_ids(self, x):
        def _get_lemma(token):
            if not token.is_punct and not token.is_space:
                lower_lemma = token.lemma_.lower()
                return lower_lemma

        tokens = [self.gensim_model.get_index(_get_lemma(token)) for sent in x for token in nlp(sent)]
        return torch.stack(tokens, dim=0)

    def forward(self, x):
        x = self._words_to_token_ids(x)
        x = self.word_embd(x)
        x = self.relu(self.fc1(x))
        # what is this torch max?
        x = torch.max(x, dim=1)[0]
        x = self.fc2(x)
        return x


def get_word2vec_for_vocabulary(countvectorizer_path: str = 'count_vectorizer.pkl',
                                original_word2vec: str = 'word2vec-google-news-300',
                                output_model_path: str = 'filtered_f30k_word2vec.model'):
    """
    Given a DataLoader loading only captions fit into a CountVectorizer to obtain the TextEncoder model to use for training
    """

    import gensim.downloader as api

    original_model = api.load(original_word2vec)
    with open(countvectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    new_vectors = []
    new_vocab = {}
    new_index2entity = []
    new_model = original_model
    words_to_keep = list(vectorizer.vocabulary_.keys())

    for word, index in original_model.key_to_index.items():
        if word in words_to_keep:
            new_vocab[word] = len(new_index2entity)
            new_vectors.append(original_model.get_vector(word))
            new_index2entity.append(word)

    new_model.key_to_index = new_vocab
    new_model.vectors = np.array(new_vectors)
    new_model.index_to_key = np.array(new_index2entity)
    new_model.save(output_model_path)


if __name__ == '__main__':
    countvectorizer_path = sys.argv[1] if len(sys.argv) > 1 else f'count_vectorizer.pkl'
    original_word2vec = sys.argv[2] if len(sys.argv) > 2 else f'word2vec-google-news-300'
    output_model_path = sys.argv[3] if len(sys.argv) > 3 else f'filtered_f30k_word2vec.model'
    get_word2vec_for_vocabulary(countvectorizer_path, original_word2vec, output_model_path)
