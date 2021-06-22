import os
from typing import List, Optional
from collections import defaultdict
import pickle

import scipy


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
        self.inverted_index = defaultdict(list)
        self.inverted_index = StorageLink.load_from_file(base_path)

    def search(self, vector: 'scipy.sparse.csr_matrix', top_k: Optional[int], **kwargs):
        result = []
        for index in vector.indices:
            result.extend(self.inverted_index[index])

        if top_k:
            return result[:top_k]
        else:
            return result

    def analyze(self, **kwargs):
        count = 0
        for key, bucket in self.inverted_index.items():
            if len(bucket) > 0 and len(set(bucket)) > 0:
                count += 1
                print(f' key {key} => len {len(bucket)}, {len(set(bucket))}')
        print(f' COUNT => {count}')


class AddSparseInvertedIndexer:
    def __init__(self, base_path: str, **kwargs):
        super().__init__(**kwargs)
        self.base_path = base_path
        self.inverted_index = defaultdict(list)

    def add(self, indexes: List[str], vectors: 'scipy.sparse.csr_matrix', **kwargs):
        for i, (id, vec) in enumerate(zip(indexes, vectors)):
            for index in vec.indices:
                self.inverted_index[index].append(id)

    def save(self):
        StorageLink.store_to_file(self.base_path, self.inverted_index)

    def analyze(self, **kwargs):
        count = 0
        for key, bucket in self.inverted_index.items():
            count += 1
            print(f' key {key} => len {len(bucket)}, {len(set(bucket))}')
        print(f' COUNT => {count}')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()
