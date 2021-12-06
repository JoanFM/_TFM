import os
import torch
import numpy as np

from src.model import ImageEncoder, TextEncoder
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from jina import Executor, Document, requests, Flow, DocumentArrayMemmap, DocumentArray
from scipy.sparse import csr_matrix

os.environ['JINA_LOG_LEVEL'] = 'DEBUG'

# File where Jina will store the index
INDEX_FILE_PATH = os.getenv('INDEX_FILE_PATH', 'tmp/sparse_index')
# Model to expose for indexing, the path where the state_dict of the model is stored
IMAGE_EMBEDDING_MODEL_PATH = os.getenv('IMAGE_EMBEDDING_MODEL_PATH', '/hdd/master/tfm/output-image-encoders/model-inter-9-final.pt')
# Text embedding model to expose for querying, the path where the CountVectorizer is stored of the model is stored
TEXT_EMBEDDING_VECTORIZER_PATH = os.getenv('TEXT_EMBEDDING_VECTORIZER_PATH', 'vectorizer.pkl')
# The root path where the flickr30k dataset is found
DATASET_ROOT_PATH = os.getenv('DATASET_ROOT_PATH', '/hdd/master/tfm/flickr30k_images')
# The root path where the flickr30k entities per split is kept
DATASET_SPLIT_ROOT_PATH = os.getenv('DATASET_SPLIT_ROOT_PATH', '/hdd/master/tfm/flickr30k_images/flickr30k_entities')
# The split to use for indexing (test, val, train)
DATASET_SPLIT = os.getenv('DATASET_SPLIT', 'test')


class JinaImageEncoder(Executor):

    def __init__(self, model_path, vectorizer_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._image_encoder = ImageEncoder(layer_size=[4096, 3537])
        self._image_encoder.load_state_dict(torch.load(model_path))
        self._image_encoder.training = False
        text_encoder = TextEncoder(vectorizer_path)
        self._vocab = {v: k for k, v in text_encoder.vectorizer.vocabulary_.items()}

    @requests
    def encode(self, docs, **kwargs):
        with torch.no_grad():
            images = docs.get_attributes('blob')
            embeddings = self._image_encoder(torch.from_numpy(np.array(images)))
            for doc, embedding in zip(docs, embeddings):
                embedding = csr_matrix(embedding.detach().numpy())
                sort_indices = np.argsort(-embedding.data)
                doc.tags['words'] = [self._vocab[embedding.indices[i]] for i in sort_indices]
                doc.tags['num_words_in_image'] = len(embedding.indices)
                doc.embedding = embedding


class JinaTextEncoder(Executor):

    def __init__(self, vectorizer_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._text_encoder = TextEncoder(vectorizer_path)
        self._vocab = {v: k for k, v in self._text_encoder.vectorizer.vocabulary_.items()}

    @requests
    def encode(self, docs, **kwargs):
        texts = docs.get_attributes('text')
        embeddings = self._text_encoder.forward(texts)
        for doc, embedding in zip(docs, embeddings):
            text_embedding = embedding.detach().numpy()
            text_embedding = text_embedding / text_embedding
            text_embedding[text_embedding != text_embedding] = 0
            doc.embedding = csr_matrix(text_embedding)
            doc.tags['words'] = [self._vocab[i] for i in doc.embedding.indices]


def csr_vappend(a, b):
    """ Takes in 2 csr_matrices and appends the second one to the bottom of the first one.
    Much faster than scipy.sparse.vstack but assumes the type to be csr and overwrites
    the first matrix instead of copying it. The data, indices, and indptr still get copied."""
    if a is None:
        return b
    a.data = np.hstack((a.data, b.data))
    a.indices = np.hstack((a.indices, b.indices))
    a.indptr = np.hstack((a.indptr, (b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0] + b.shape[0], b.shape[1])
    return a


class JinaIndexer(Executor):

    def __init__(self, index_path, top_k=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._index_path = index_path
        self._docs = DocumentArrayMemmap(self._index_path)
        self._tfidf_transformer = None
        self._tfidf_index = None
        self.top_k = top_k

    @requests(on='/index')
    def index(self, docs, **kwargs):
        for doc in docs:
            doc.set_image_blob_shape(channel_axis=0, shape=(doc.blob.shape[1], doc.blob.shape[2]))
            doc.convert_image_blob_to_uri()
            doc.pop('blob')
            self._docs.append(doc)

    @requests(on='/search')
    def query(self, docs, **kwargs):
        if self._tfidf_transformer is None:
            self._tfidf_transformer = TfidfTransformer()
            self._tfidf_index = DocumentArray([doc for doc in self._docs])
            all_image_embeddings = None
            for doc in self._docs:
                image_embedding = doc.embedding
                if all_image_embeddings is None:
                    all_image_embeddings = csr_matrix(image_embedding)
                else:
                    all_image_embeddings = csr_vappend(all_image_embeddings, csr_matrix(image_embedding))
            self._embds = self._tfidf_transformer.fit_transform(all_image_embeddings)

        all_query_embeddings = None
        for doc in docs:
            doc.embedding = self._tfidf_transformer.transform(doc.embedding)
            text_embedding = doc.embedding
            if all_query_embeddings is None:
                all_query_embeddings = csr_matrix(text_embedding)
            else:
                all_query_embeddings = csr_vappend(all_query_embeddings, csr_matrix(text_embedding))

        X = all_query_embeddings
        Y = self._embds
        cosine_scores = cosine_similarity(X, Y)
        for i, doc in enumerate(docs):
            results = sorted(zip(self._docs, cosine_scores[i]), key=lambda pair: pair[1], reverse=True)
            for m, score in results[0: self.top_k]:
                match = Document(m, copy=True)
                match.scores['tfidf'] = score
                doc.matches.append(match)

    def close(self) -> None:
        self._docs.flush()


def _get_index_documents(root=DATASET_ROOT_PATH,
                         split_root=DATASET_SPLIT_ROOT_PATH,
                         split=DATASET_SPLIT,
                         batch_size=8):
    from src.dataset import get_image_data_loader
    print(f'\n Indexing docs from split {split}\n')
    data_loader = get_image_data_loader(root, split_root=split_root, split=split, batch_size=batch_size, shuffle=False)
    count = 0
    for filenames, images in data_loader:
        for filename, image in zip(filenames, images):
            yield Document(blob=image.numpy(), tags={'filename': filename})
            count += 1
            if count % 1000 == 0:
                print(f'\n Indexed {count} docs\n')


def index():
    f = Flow().add(uses=JinaImageEncoder,
                   uses_with={'model_path': IMAGE_EMBEDDING_MODEL_PATH,
                              'vectorizer_path': TEXT_EMBEDDING_VECTORIZER_PATH}).add(
        uses=JinaIndexer, uses_with={
            'index_path': INDEX_FILE_PATH})
    with f:
        f.index(_get_index_documents(), request_size=8, show_progress=True)


def search():
    f = Flow(protocol='http', port_expose=45678).add(uses=JinaTextEncoder, uses_with={
        'vectorizer_path': TEXT_EMBEDDING_VECTORIZER_PATH}).add(
        uses=JinaIndexer,
        uses_with={'index_path': INDEX_FILE_PATH})
    with f:
        f.block()


if __name__ == '__main__':
    index()
    search()
