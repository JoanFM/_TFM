import os
import pytest

import random
import numpy as np
from scipy.sparse import csr_matrix

from src.vector_store.sparse_inverted_index import AddSparseInvertedIndexer, QuerySparseInvertedIndexer


@pytest.fixture()
def base_path(tmpdir):
    return os.path.join(str(tmpdir), 'sparse_inverted_index')


@pytest.mark.repeat(10)
def test_sparse_inverted_indexed(base_path):
    VOCAB_SIZE = 8000
    NUM_ENTRIES = VOCAB_SIZE
    # in this case we are having like a diagonal matrix
    row = np.array(list(range(VOCAB_SIZE)))
    col = np.array(list(range(NUM_ENTRIES)))
    data = np.array([1] * VOCAB_SIZE)
    matrix = csr_matrix((data, (row, col)), shape=(VOCAB_SIZE, VOCAB_SIZE))
    id_list = list(range(VOCAB_SIZE))

    with AddSparseInvertedIndexer(base_path=base_path) as add_indexer:
        add_indexer.add(id_list, matrix)

    assert len(add_indexer.inverted_index.keys()) == VOCAB_SIZE

    for key, bucket in add_indexer.inverted_index.items():
        assert len(bucket) == 1
        assert key == bucket[0]

    QUERY_ID = random.randint(0, VOCAB_SIZE)
    query_indexer = QuerySparseInvertedIndexer(base_path=base_path)
    assert len(query_indexer.inverted_index.keys()) == VOCAB_SIZE

    for key, bucket in query_indexer.inverted_index.items():
        assert len(bucket) == 1
        assert key == bucket[0]

    results = query_indexer.search(matrix.getrow(QUERY_ID), top_k=1)
    assert len(results) == 1
    assert results[0] == QUERY_ID

