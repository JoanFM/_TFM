from src.vector_store.sparse_inverted_index import QuerySparseInvertedIndexer
from src.model import TextEncoder
from scipy.sparse import csr_matrix

IMAGE_BASE_PATH = f'/hdd/master/tfm/flickr30k_images/flickr30k_images'


def display(results):
    from PIL import Image
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 8))
    columns = 2
    rows = 2
    for i, (score, file) in enumerate(results):
        img = Image.open(open(f'{IMAGE_BASE_PATH}/{file}', 'rb'))
        sublot = fig.add_subplot(rows, columns, i + 1)
        sublot.title.set_text(f'{file} - {score}')
        plt.imshow(img)
    plt.show()


def load_index_and_model(inverted_index_path: str = '/hdd/master/tfm/sparse_indexers_tmp-test/epoch-90',
                         vectorizer_path: str = '/hdd/master/tfm'
                                                '/vectorizer_tokenizer_stop_words_all_words_filtered_3000.pkl'):
    """
    Loads both a text encoder

    :param inverted_index_path: The inverted index path from where to load the inverted index to query
    :param vectorizer_path: The vectorizer path from where to load the TextEncoder
    :return:
    """
    query_indexer = QuerySparseInvertedIndexer(
        base_path=inverted_index_path)
    model = TextEncoder(vectorizer_path)
    return query_indexer, model


def search(query, indexer, text_encoder, top_k):
    """
    Compute a search query against an indexer

    :param query: The sentence query
    :param indexer: The Inverted Index with the images indexed
    :param text_encoder: The Text Encoder
    :param top_k: THe amount of images to return
    :return:
    """
    query_embedding = text_encoder.forward([query])
    query_embedding = query_embedding / query_embedding
    query_embedding[query_embedding != query_embedding] = 0
    query_embedding = csr_matrix(query_embedding)
    print(f' query_embedding {query_embedding.shape}')

    results = indexer.search(query_embedding, top_k, return_scores=True)

    display(results[:4])


if __name__ == '__main__':
    """
    Application trying to show the results to some queries
    """
    indexer, text_encoder = load_index_and_model()
    while True:
        query = input("Please enter your query: ")
        search(query, indexer, text_encoder, None)
