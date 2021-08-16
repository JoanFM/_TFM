import os
import torch
import numpy as np

from src.model import ImageEncoder, TextEncoder
from src.vector_store.sparse_inverted_index import AddSparseInvertedIndexer, QuerySparseInvertedIndexer
from jina import Executor, Document, DocumentArray, requests, Flow
from scipy.sparse import csr_matrix

os.environ['JINA_LOG_LEVEL'] = 'DEBUG'


class MatchConverter(Executor):

    @requests
    def convert(self, docs, **kwargs):
        for doc in docs:
            for match in doc.matches:
                match.convert_blob_to_buffer()
                match.convert_buffer_to_uri()


class JinaImageEncoder(Executor):

    def __init__(self, model_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._image_encoder = ImageEncoder(layer_size=[4096, 3537])
        self._image_encoder.load_state_dict(torch.load(model_path))
        self._image_encoder.training = False

    @requests
    def encode(self, docs, **kwargs):
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
            print(f' embedding {embedding.shape}')
            doc.embedding = csr_matrix(embedding.detach().numpy())
            print(f' doc.embedding {csr_matrix(doc.embedding).indices}')


class JinaIndexer(Executor):

    def __init__(self, index_path, top_k=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._index_path = index_path
        self._query_indexer = None
        self._add_indexer = None
        if os.path.exists(self._index_path):
            self._query_indexer = QuerySparseInvertedIndexer(index_path)
            self._docs = DocumentArray.load(f'{self._index_path}/docs.bin', file_format='binary')
            self._query_indexer.analyze()
            print(f' len {len(self._docs)}')
        else:
            self._add_indexer = AddSparseInvertedIndexer(index_path)
            self._docs = DocumentArray()
        self.top_k = top_k

    @requests(on='/index')
    def index(self, docs, **kwargs):
        embeddings = docs.get_attributes('embedding')
        ids = docs.get_attributes('id')
        for id, embedding in zip(ids, embeddings):
            self._add_indexer.add([id], [csr_matrix(embedding)])
        self._docs.extend(docs)

    @requests(on='/search')
    def query(self, docs, **kwargs):
        for query in docs:
            results = self._query_indexer.search(csr_matrix(query.embedding), top_k=self.top_k)
            print(f' results {results} => {self._docs[0].id}')
            for res in results:
                doc = self._docs[res]
                query.matches.append(doc)

    def close(self) -> None:
        if self._add_indexer is not None:
            self._add_indexer.save()
            self._docs.save(f'{self._index_path}/docs.bin', file_format='binary')


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
                   uses_with={'model_path': '/hdd/master/tfm/output_models-test/model-inter-11-final.pt'}).add(
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
        resp = f.search(Document(text='hey here boy'), return_results=True)
        print(f' matches {resp[0].docs[0].matches}')


if __name__ == '__main__':
    #index()
    search()
