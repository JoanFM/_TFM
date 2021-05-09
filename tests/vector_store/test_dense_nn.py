import os
import pytest

import random
import numpy as np

from src.vector_store.dense_nn import AddDenseNumpyIndexer, QueryDenseNumpyIndexer


@pytest.fixture()
def base_path(tmpdir):
    return os.path.join(str(tmpdir), 'dense_nn')


def test_dense_nn_search(base_path):

    id_list = list(range(30000))
    ids = np.array(id_list)
    vecs = np.random.rand(30000, 512)

    with AddDenseNumpyIndexer(base_path=base_path) as add_indexer:
        add_indexer.add(ids, vecs)

    QUERY_ID = random.randint(0, 30000)

    query_indexer = QueryDenseNumpyIndexer(base_path=base_path)
    results = query_indexer.search(vecs[QUERY_ID], top_k=1)
    assert len(results) == 1
    result_id = results[0]
    print(f' QUERY_ID {QUERY_ID}, result_id {result_id}')

