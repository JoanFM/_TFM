import os
import pytest

import random
import numpy as np

from src.vector_store.dense_nn import AddDenseNumpyIndexer, QueryDenseNumpyIndexer


@pytest.fixture()
def base_path(tmpdir):
    return os.path.join(str(tmpdir), 'dense_nn')


@pytest.mark.repeat(10)
def test_dense_nn_search(base_path):

    id_list = list(range(30000))
    ids = np.array(id_list)
    vecs = np.random.rand(30000, 512)

    with AddDenseNumpyIndexer(base_path=base_path) as add_indexer:
        add_indexer.add(ids, vecs)

    assert len(add_indexer._vecs) == 30000
    assert add_indexer._vecs[0].shape == (512, )
    QUERY_ID = random.randint(0, 30000)

    query_indexer = QueryDenseNumpyIndexer(base_path=base_path)
    results = query_indexer.search(np.expand_dims(vecs[QUERY_ID], axis=0), top_k=1)
    assert len(results) == 1
    result_id = results[0]
    assert result_id == QUERY_ID

