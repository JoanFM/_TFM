import os
from typing import List, Optional, Union
from collections import defaultdict
import pickle

import scipy
import numpy as np

_ATTRIBUTES = {
    'bold': 1,
    'dark': 2,
    'underline': 4,
    'blink': 5,
    'reverse': 7,
    'concealed': 8,
}

_HIGHLIGHTS = {
    'on_grey': 40,
    'on_red': 41,
    'on_green': 42,
    'on_yellow': 43,
    'on_blue': 44,
    'on_magenta': 45,
    'on_cyan': 46,
    'on_white': 47,
}

_COLORS = {
    'black': 30,
    'red': 31,
    'green': 32,
    'yellow': 33,
    'blue': 34,
    'magenta': 35,
    'cyan': 36,
    'white': 37,
}

_RESET = '\033[0m'


def colored(
        text: str,
        color: Optional[str] = None,
        on_color: Optional[str] = None,
        attrs: Optional[Union[str, list]] = None,
):
    fmt_str = '\033[%dm%s'
    if color:
        text = fmt_str % (_COLORS[color], text)
    if on_color:
        text = fmt_str % (_HIGHLIGHTS[on_color], text)

    if attrs:
        if isinstance(attrs, str):
            attrs = [attrs]
        if isinstance(attrs, list):
            for attr in attrs:
                text = fmt_str % (_ATTRIBUTES[attr], text)
    text += _RESET
    return text


class InvertedIndex:
    """
    Class implementing an inverted index.

    Stores a dictionary with key as the indices of sparse vectors and values the ids of the images

    It implements a `proxy` of a TFIDF algorith.

    Term frequency (TF (term (index), document (image))) is proxied by the activation of the specific word (sparse index) in the vector. It tries to mean
    how important or strong that feature is for that image.

    Inverted Document Frequency (IDF (term)) of a term (index, feature, word) inside the collection is inversed to the sum of all the frequencies of the documents
    with that term. It means, if many images in the collection activate this `feature`, the importance of this term, feature or word is not so important
    """

    def __init__(self):
        self.inverted_index = defaultdict(set)
        self.document_frequencies = defaultdict(int)
        self.document_sparse_vectors = {}
        self.idfs = {}

    def cache_idfs(self):
        """
        Cache the IDFS for each term into an inmemory dictionary for fast access
        """
        for term_idx in self.document_frequencies.keys():
            num = len(self.document_sparse_vectors.keys())
            den = 1 + self.document_frequencies[term_idx]
            self.idfs[term_idx] = np.log(num / den)

    def add(self, document_id, document_vector):
        """
        Add into the inverted index a document_id with a specific sparse vector

        :param document_id: The id of the document (image) to insert
        :param document_vector: The sparse vector representing the image
        :return: None
        """
        def _add(term_id, document_id, value):
            self.inverted_index[term_id].add(document_id)
            self.document_frequencies[term_id] += value

        for term_id, term_value in zip(document_vector.indices, document_vector.data):
            _add(term_id, document_id, term_value)
        self.document_sparse_vectors[document_id] = document_vector

    def get_candidates(self, term_index):
        """
        Get all the images inside a bucket

        :param term_index: the term, feature or bucket from where to extract the images
        :return: list of candidate ids
        """
        return self.inverted_index[term_index]

    def match(self, query, top_k, return_scores=False):
        """
        Find the most relevance images given a query, the query being a sparse vector representing a tokenized caption or embedded image

        :param query: The vector query
        :param top_k: The max number of documents to return
        :param return_scores: Option to return the relevance scores for better analysis
        :return: list of results
        """
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
        """
        Compute the tfidf (the relevance score) given a query and a candidate result

        :param query_vec: The query sparse vector
        :param candidate: THe candidate sparse vector
        :return: The relevance (TFIDF) score
        """
        candidate_vector = self.document_sparse_vectors[candidate]
        candidate_dense = np.array(candidate_vector.todense())[0]
        number_words = len(candidate_vector.indices)
        sum = 0
        for term_index in query_vec.indices:
            tf = self._tf(candidate_dense, term_index, number_words)
            idf = self._idf(term_index)
            sum = sum + tf * idf
        return sum

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
    """
    Class used to query an inverted index loaded from disk
    """
    def __init__(self, base_path: str, **kwargs):
        super().__init__(**kwargs)
        self.base_path = base_path
        self.inverted_index = InvertedIndex()
        self.inverted_index = StorageLink.load_from_file(base_path)
        self.log_queries = open(os.path.join(base_path, 'logs.txt'), 'a')

    def search(self, query: 'scipy.sparse.csr_matrix', top_k: Optional[int], **kwargs):
        """
        Add into the inverted index a set of documents with ids that have sparse vectors from which to extract the inverted index keys

        :param query: A query represented by a sparse vector (like a tokenized sentence)
        :param top_k: The top_k documents to return, if None all the candidates
        :param kwargs: Extra kwargs
        :return: the candidates ids
        """
        self.log_queries.write(f'{query}\n')
        self.log_queries.flush()
        return self.inverted_index.match(query, top_k, **kwargs)

    def analyze(self, **kwargs):
        """
        Print some analysis data on the inverted index.

        How big is the vocabulary,
        How many words (tokens) (keys) have at least one document attached
        How many words do not have a document (image) attached
        How many images are at each backet (with and without counting empty bins)
        """
        count = 0
        counts = []
        counts_ignoring_0_sized = []
        for key, bucket in self.inverted_index.inverted_index.items():
            counts.append(len(set(bucket)))
            if len(bucket) > 0 and len(set(bucket)) > 0:
                counts_ignoring_0_sized.append(len(set(bucket)))
                count += 1
        print(colored(f' Analysis for {self.base_path}, query', 'cyan'))
        print(colored(f' Size of vocabulary => {len(self.inverted_index.inverted_index.keys())}', 'cyan'))
        print(colored(f' Number of keys with at least one candidate => {count}', 'cyan'))
        shape_1 = list(self.inverted_index.document_sparse_vectors.values())[0].shape[1]
        print(colored(f' Shape of sparse vector => {shape_1}', 'cyan'))
        print(colored(f' Number of empty bins => {shape_1 - count}', 'cyan'))
        print(colored(f' Average amount of documents per bucket, ignoring 0-sized buckets => mean: {np.mean(counts_ignoring_0_sized)}, std: {np.std(counts_ignoring_0_sized)}', 'cyan'))
        print(colored(f' Average amount of documents per bucket, counting 0-sized buckets => mean: {np.mean(counts)}, std: {np.std(counts)}', 'cyan'))


class AddSparseInvertedIndexer:
    """
    Class used to store embeddings into an inverted index, this is only used for indexing and dumping the data
    """
    def __init__(self, base_path: str, **kwargs):
        super().__init__(**kwargs)
        self.base_path = base_path
        self.inverted_index = InvertedIndex()
        self._s = set()

    def add(self, indexes: List[str], vectors: 'scipy.sparse.csr_matrix', **kwargs):
        """
        Add into the inverted index a set of documents with ids that have sparse vectors from which to extract the inverted index keys

        :param indexes: The batch of document ids to insert
        :param vectors: The corresponding vector of each id in indexes
        :param kwargs: Extra kwargs
        :return: None
        """
        for i, (document_id, vec) in enumerate(zip(indexes, vectors)):
            for j in vec.indices:
                self._s.add(j)
            self.inverted_index.add(document_id, vec)

    def save(self):
        """
        Save the inverted index into a file
        """
        StorageLink.store_to_file(self.base_path, self.inverted_index)

    def analyze(self, **kwargs):
        """
        Print some analysis data on the inverted index.

        How big is the vocabulary,
        How many words (tokens) (keys) have at least one document attached
        How many words do not have a document (image) attached
        How many images are at each backet (with and without counting empty bins)
        """
        count = 0
        counts = []
        counts_ignoring_0_sized = []
        for key, bucket in self.inverted_index.inverted_index.items():
            counts.append(len(set(bucket)))
            if len(bucket) > 0 and len(set(bucket)) > 0:
                counts_ignoring_0_sized.append(len(set(bucket)))
                count += 1
        print(colored(f' Analysis for {self.base_path}, query', 'cyan'))
        print(colored(f' Size of vocabulary => {len(self.inverted_index.inverted_index.keys())}', 'cyan'))
        print(colored(f' Number of keys with at least one candidate => {count}', 'cyan'))
        shape_1 = list(self.inverted_index.document_sparse_vectors.values())[0].shape[1]
        print(colored(f' Shape of sparse vector => {shape_1}', 'cyan'))
        print(colored(f' Number of empty bins => {shape_1 - count}', 'cyan'))
        print(colored(f' Average amount of documents per bucket, ignoring 0-sized buckets => mean: {np.mean(counts_ignoring_0_sized)}, std: {np.std(counts_ignoring_0_sized)}', 'cyan'))
        print(colored(f' Average amount of documents per bucket, counting 0-sized buckets => mean: {np.mean(counts)}, std: {np.std(counts)}', 'cyan'))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(colored(f' length of set {len(self._s)}', 'cyan'))
        self.inverted_index.cache_idfs()
        self.save()
