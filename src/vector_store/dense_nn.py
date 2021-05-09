import os
from typing import Tuple, List
import pickle

import numpy as np


def _get_ones(x, y):
    return np.ones((x, y))


def _ext_A(A):
    nA, dim = A.shape
    A_ext = _get_ones(nA, dim * 3)
    A_ext[:, dim: 2 * dim] = A
    A_ext[:, 2 * dim:] = A ** 2
    return A_ext


def _ext_B(B):
    nB, dim = B.shape
    B_ext = _get_ones(dim * 3, nB)
    B_ext[:dim] = (B ** 2).T
    B_ext[dim: 2 * dim] = -2.0 * B.T
    del B
    return B_ext


def _euclidean(A_ext, B_ext):
    sqdist = A_ext.dot(B_ext).clip(min=0)
    return np.sqrt(sqdist)


def _norm(A):
    return A / np.linalg.norm(A, ord=2, axis=1, keepdims=True)


def _cosine(A_norm_ext, B_norm_ext):
    return A_norm_ext.dot(B_norm_ext).clip(min=0) / 2


class StorageLink:

    @staticmethod
    def load_from_file(base_path):
        id_path = os.path.join(base_path, 'ids.pickle')
        with open(id_path, 'rb') as id_f:
            ids = pickle.load(id_f)
        vecs_path = os.path.join(base_path, 'vecs.pickle')
        with open(vecs_path, 'rb') as vecs_f:
            vecs = pickle.load(vecs_f)
        return ids, vecs

    @staticmethod
    def store_to_file(base_path, ids, vecs):
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        id_path = os.path.join(base_path, 'ids.pickle')

        with open(id_path, 'wb') as f:
            pickle.dump(ids, f)

        vecs_path = os.path.join(base_path, 'vecs.pickle')
        with open(vecs_path, 'wb') as f:
            pickle.dump(np.vstack(vecs), f)


class QueryDenseNumpyIndexer:
    def __init__(self, base_path: str, **kwargs):
        super().__init__(**kwargs)
        self.base_path = base_path
        ids, vecs = StorageLink.load_from_file(base_path)
        self._ids = np.array(ids)
        self._vecs = np.array(vecs)

    def search(self, vector: 'np.array', top_k: int, **kwargs):
        q_emb = _ext_A(_norm(vector))
        d_emb = _ext_B(_norm(self._vecs))
        dists = _cosine(q_emb, d_emb)
        positions, dists = self._get_sorted_top_k(dists, top_k)
        for position, dist in zip(positions, dists):
            return self._ids[position]

    @staticmethod
    def _get_sorted_top_k(
            dist: 'np.array', top_k: int
    ) -> Tuple['np.ndarray', 'np.ndarray']:
        if top_k >= dist.shape[1]:
            idx = dist.argsort(axis=1)[:, :top_k]
            dist = np.take_along_axis(dist, idx, axis=1)
        else:
            idx_ps = dist.argpartition(kth=top_k, axis=1)[:, :top_k]
            dist = np.take_along_axis(dist, idx_ps, axis=1)
            idx_fs = dist.argsort(axis=1)
            idx = np.take_along_axis(idx_ps, idx_fs, axis=1)
            dist = np.take_along_axis(dist, idx_fs, axis=1)

        return idx, dist


class AddDenseNumpyIndexer:
    def __init__(self, base_path: str, **kwargs):
        super().__init__(**kwargs)
        self.base_path = base_path
        self._ids = []
        self._vecs = []

    def add(self, indexes: List[str], vectors: 'np.array', **kwargs):
        for id, vec in zip(indexes, vectors):
            self._ids.append(id)
            self._vecs.append(vec)

    def save(self):
        StorageLink.store_to_file(self.base_path, self._ids, self._vecs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()
