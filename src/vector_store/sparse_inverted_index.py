import os
from typing import List, Optional
from collections import defaultdict
import pickle

import scipy
import numpy as np


class InvertedIndex:

    def __init__(self):
        self.inverted_index = defaultdict(set)
        self.document_frequencies = defaultdict(int)
        self.document_sparse_vectors = {}
        self.idfs = {}

    def cache_idfs(self):
        for term_idx in self.document_frequencies.keys():
            num = len(self.document_sparse_vectors.keys())
            den = 1 + self.document_frequencies[term_idx]
            self.idfs[term_idx] = np.log(num / den)

    def add(self, document_id, document_vector):
        def _add(term_id, document_id, value):
            self.inverted_index[term_id].add(document_id)
            self.document_frequencies[term_id] += value

        for term_id, term_value in zip(document_vector.indices, document_vector.data):
            _add(term_id, document_id, term_value)
        self.document_sparse_vectors[document_id] = document_vector

    def get_candidates(self, term_index):
        return self.inverted_index[term_index]

    def match(self, query, top_k, return_scores=False):
        candidates = set()

        for term_index in query.indices:
            candidates.update(self.get_candidates(term_index))

        scores = []
        candidates = list(candidates)
        for candidate in candidates:
            scores.append(self._relevance(query, candidate))

        results = sorted(zip(scores, candidates), reverse=True)
        if top_k:
            if return_scores:
                return results[: top_k]
            else:
                return [element for _, element in results[: top_k]]
        else:
            if return_scores:
                return results
            else:
                return [element for _, element in results]

    def _relevance(self, query_vec, candidate):
        candidate_vector = self.document_sparse_vectors[candidate]
        candidate_dense = np.array(candidate_vector.todense())[0]
        number_words = len(candidate_vector.indices)
        prod = 1
        for term_index in query_vec.indices:
            tf = self._tf(candidate_dense, term_index, number_words)
            idf = self._idf(term_index)
            prod = prod * tf * idf
        return prod

    def _tf(self, candidate_dense, term_idx, number_words):
        return candidate_dense[term_idx] / number_words

    def _idf(self, term_idx):
        return self.idfs.get(term_idx, 0)


class StorageLink:

    @staticmethod
    def load_from_file(base_path):
        inverted_index_path = os.path.join(base_path, 'inverted_index.pickle')
        with open(inverted_index_path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def store_to_file(base_path, inverted_idx):
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        inverted_index_path = os.path.join(base_path, 'inverted_index.pickle')
        with open(inverted_index_path, 'wb') as f:
            pickle.dump(inverted_idx, f)


class QuerySparseInvertedIndexer:
    def __init__(self, base_path: str, **kwargs):
        super().__init__(**kwargs)
        self.inverted_index = InvertedIndex()
        self.inverted_index = StorageLink.load_from_file(base_path)

    def search(self, query: 'scipy.sparse.csr_matrix', top_k: Optional[int], **kwargs):
        return self.inverted_index.match(query, top_k, **kwargs)

    def analyze(self, **kwargs):
        count = 0
        for key, bucket in self.inverted_index.inverted_index.items():
            if len(bucket) > 0 and len(set(bucket)) > 0:
                count += 1
                print(f' key {key} => len {len(bucket)}, {len(set(bucket))}')
        print(f' COUNT => {count}')


class AddSparseInvertedIndexer:
    def __init__(self, base_path: str, **kwargs):
        super().__init__(**kwargs)
        self.base_path = base_path
        self.inverted_index = InvertedIndex()

    def add(self, indexes: List[str], vectors: 'scipy.sparse.csr_matrix', **kwargs):
        for i, (document_id, vec) in enumerate(zip(indexes, vectors)):
            self.inverted_index.add(document_id, vec)

    def save(self):
        StorageLink.store_to_file(self.base_path, self.inverted_index)

    def analyze(self, **kwargs):
        count = 0
        for key, bucket in self.inverted_index.inverted_index.items():
            count += 1
            print(f' key {key} => len {len(bucket)}, {len(set(bucket))}')
        print(f' COUNT => {count}')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.inverted_index.cache_idfs()
        self.save()
