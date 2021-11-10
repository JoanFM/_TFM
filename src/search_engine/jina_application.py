import os
import torch
import numpy as np

from src.model import ImageEncoder, TextEncoder
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from jina import Executor, Document, requests, Flow, DocumentArrayMemmap, DocumentArray
from scipy.sparse import csr_matrix

os.environ['JINA_LOG_LEVEL'] = 'DEBUG'


class MatchConverter(Executor):

    @requests
    def convert(self, docs, **kwargs):
        for doc in docs:
            for match in doc.matches:
                match.set_image_blob_shape(channel_axis=0, shape=(match.blob.shape[1], match.blob.shape[2]))
                match.convert_image_blob_to_uri()


class JinaImageEncoder(Executor):

    def __init__(self, model_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._image_encoder = ImageEncoder(layer_size=[4096, 3537])
        self._image_encoder.load_state_dict(torch.load(model_path))
        self._image_encoder.training = False

    @requests
    def encode(self, docs, **kwargs):
        with torch.no_grad():
            images = docs.get_attributes('blob')
            embeddings = self._image_encoder(torch.from_numpy(np.array(images)))
            for doc, embedding in zip(docs, embeddings):
                doc.embedding = csr_matrix(embedding.detach().numpy())


class JinaTextEncoder(Executor):

    def __init__(self, vectorizer_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._text_encoder = TextEncoder(vectorizer_path)

    @requests
    def encode(self, docs, **kwargs):
        texts = docs.get_attributes('text')
        embeddings = self._text_encoder.forward(texts)
        for doc, embedding in zip(docs, embeddings):
            doc.embedding = csr_matrix(embedding.detach().numpy())


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
        self._docs.extend(docs)

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


def _get_index_documents(root='/hdd/master/tfm/flickr30k_images',
                         split_root='/hdd/master/tfm/flickr30k_images/flickr30k_entities',
                         split='test',
                         batch_size=8):
    from src.dataset import get_image_data_loader
    data_loader = get_image_data_loader(root, split_root=split_root, split=split, batch_size=batch_size, shuffle=False)

    for filenames, images in data_loader:
        for filename, image in zip(filenames, images):
            yield Document(blob=image.numpy(), tags={'filename': filename})


def index():
    f = Flow().add(uses=JinaImageEncoder,
                   uses_with={'model_path': '/hdd/master/tfm/output_models-test/model-inter-9-final.pt'}).add(
        uses=JinaIndexer, uses_with={
            'index_path': 'tmp/sparse_index'})
    with f:
        f.index(_get_index_documents(), request_size=8, show_progress=True)


def search():
    f = Flow(protocol='http', port_expose=45678).add(uses=JinaTextEncoder, uses_with={
        'vectorizer_path': f'/hdd/master/tfm/vectorizer_tokenizer_stop_words_all_words_filtered_10.pkl'}).add(
        uses=JinaIndexer,
        uses_with={'index_path': 'tmp/sparse_index'}).add(
        uses=MatchConverter)
    with f:
        f.block()
        #resp = f.search(Document(text='hey here boy'), return_results=True)
        #print(f' matches {resp[0].docs[0].matches}')


if __name__ == '__main__':
    #index()
    search()
