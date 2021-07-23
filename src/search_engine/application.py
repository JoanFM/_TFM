import os
from src.vector_store.sparse_inverted_index import QuerySparseInvertedIndexer
from src.model import TextEncoder
from scipy.sparse import csr_matrix

IMAGE_BASE_PATH = f'/hdd/master/tfm/flickr30k_images/flickr30k_images'


# def display(results):
#     from tkinter import Tk, Label
#     from PIL import Image, ImageTk
#
#     root = Tk()
#     #root.geometry("800x600")
#     photos = []
#
#     def displayImg(img):
#
#         image = Image.open(img)
#         photo = ImageTk.PhotoImage(image)
#         photos.append(photo)  # keep references!
#         newPhoto_label = Label(image=photo)
#         newPhoto_label.pack()
#
#     for file in results[: 4]:
#         print(f' file {file}')
#         displayImg(open(f'{IMAGE_BASE_PATH}/{file}', mode='rb'))
#     root.mainloop()


def display(results):
    from PIL import Image
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 8))
    columns = 2
    rows = 2
    for i, (score, file) in enumerate(results):
        img = Image.open(open(f'{IMAGE_BASE_PATH}/{file}', 'rb'))
        sublot = fig.add_subplot(rows, columns, i + 1)
        sublot.title.set_text(f'Relevance {score}')
        plt.imshow(img)
    plt.show()


def load_index_and_model(inverted_index_path: str = '/hdd/master/tfm/sparse_indexers_tmp-test/epoch-90',
                         vectorizer_path: str = '/hdd/master/tfm/vectorizer_tokenizer_stop_words_all_words_filtered_3000.pkl', ):
    query_indexer = QuerySparseInvertedIndexer(
        base_path=inverted_index_path)
    model = TextEncoder(vectorizer_path)
    return query_indexer, model


def search(query, indexer, model, top_k):
    query_embedding = model.forward([query])
    query_embedding = query_embedding / query_embedding
    query_embedding[query_embedding != query_embedding] = 0
    query_embedding = csr_matrix(query_embedding)
    print(f' query_embedding {query_embedding.shape}')

    results = indexer.search(query_embedding, top_k, return_scores=True)

    display(results[:4])


if __name__ == '__main__':
    indexer, model = load_index_and_model()
    while True:
        query = input("Please enter your query: ")
        search(query, indexer, model, None)
