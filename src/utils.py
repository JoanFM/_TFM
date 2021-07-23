import os
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from src.model.spacy_tokenizer import spacy_tokenizer

from src.dataset.dataset import CaptionFlickr30kDataset
from src.model import TextEncoder
from src.dataset import get_captions_data_loader
from src.vector_store.sparse_inverted_index import AddSparseInvertedIndexer, QuerySparseInvertedIndexer
from scipy.sparse import csr_matrix
from progress.bar import Bar


cur_dir = os.path.dirname(os.path.abspath(__file__))


def build_inverted_index_text(vectorizer_path: str = '/hdd/master/tfm/vectorizer_tokenizer_stop_words_analyze.pkl',
                              validation_indexers_path: str = '/hdd/master/tfm/sparse_indexers_tmp_text_analyze',
                              batch_size=16):
    train_data_loader = get_captions_data_loader(root='/hdd/master/tfm/flickr30k_images',
                                                 split_root='/hdd/master/tfm/flickr30k_images/flickr30k_entities',
                                                 split='train',
                                                 shuffle=True,
                                                 batch_size=batch_size)
    test_data_loader = get_captions_data_loader(root='/hdd/master/tfm/flickr30k_images',
                                                split_root='/hdd/master/tfm/flickr30k_images/flickr30k_entities',
                                                split='test',
                                                shuffle=True,
                                                batch_size=batch_size)
    val_data_loader = get_captions_data_loader(root='/hdd/master/tfm/flickr30k_images',
                                               split_root='/hdd/master/tfm/flickr30k_images/flickr30k_entities',
                                               split='val',
                                               shuffle=True,
                                               batch_size=batch_size)

    text_encoder = TextEncoder(vectorizer_path)

    with AddSparseInvertedIndexer(base_path=f'{validation_indexers_path}') as text_indexer:
        for batch_id, (filenames, captions) in enumerate(train_data_loader):
            text_embedding_query = csr_matrix(text_encoder.forward(captions).detach().numpy())
            text_indexer.add(filenames, text_embedding_query)
        for batch_id, (filenames, captions) in enumerate(test_data_loader):
            text_embedding_query = csr_matrix(text_encoder.forward(captions).detach().numpy())
            text_indexer.add(filenames, text_embedding_query)
        for batch_id, (filenames, captions) in enumerate(val_data_loader):
            text_embedding_query = csr_matrix(text_encoder.forward(captions).detach().numpy())
            text_indexer.add(filenames, text_embedding_query)


def analyze_word_frequencies(indexer_path='/hdd/master/tfm/sparse_indexers_tmp_text_analyze'):
    """
     3638 words appear more than 10 times in the vocabulary
     1583 words appear more than 50 times in the vocabulary
     1051 words appear more than 100 times in the vocabulary <-- Lets try with this. TODO(Select a dataset with the captions and images having these words, and build a dataset with these words)
     634 words appear more than 200 times in the vocabulary
     294 words appear more than 500 times in the vocabulary
    """
    v = QuerySparseInvertedIndexer(indexer_path)
    for threshold in [10, 50, 100, 200, 500, 1000, 2000, 2500, 3000, 3500, 4000]:
        num = 0
        for k in v.inverted_index.keys():
            if len(v.inverted_index[k]) > threshold:
                num += 1
        print(f' {num} words appear more than {threshold} times in the vocabulary')


def analyze_vocab_learnt(vectorizer_path: str, inverted_index_base_path: str):
    import pickle
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    query_indexer = QuerySparseInvertedIndexer(base_path=inverted_index_base_path)
    inverse_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
    for b in query_indexer.inverted_index.keys():
        if len(query_indexer.inverted_index[b]) > 0:
            print(f' {inverse_vocab[b]}')


def get_model(vectorizer_path='/hdd/master/tfm/vectorizer_tokenizer_stop_words_analyze.pkl'):
    train_dataset = CaptionFlickr30kDataset(root='/hdd/master/tfm/flickr30k_images',
                                            split_root='/hdd/master/tfm/flickr30k_images/flickr30k_entities',
                                            split='train')
    test_dataset = CaptionFlickr30kDataset(root='/hdd/master/tfm/flickr30k_images',
                                           split_root='/hdd/master/tfm/flickr30k_images/flickr30k_entities',
                                           split='test')
    val_dataset = CaptionFlickr30kDataset(root='/hdd/master/tfm/flickr30k_images',
                                          split_root='/hdd/master/tfm/flickr30k_images/flickr30k_entities',
                                          split='val')

    corpus = [train_dataset[i][1] for i in range(len(train_dataset))] + \
             [test_dataset[i][1] for i in range(len(test_dataset))] + \
             [val_dataset[i][1] for i in range(len(val_dataset))]

    vectorizer = CountVectorizer(tokenizer=spacy_tokenizer, stop_words='english')
    vectorizer.fit(corpus)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)


def filter_vectorizers(vectorizer_path: str, indexer_path='/hdd/master/tfm/sparse_indexers_tmp_text_analyze'):
    """
     3638 words appear more than 10 times in the vocabulary
     1583 words appear more than 50 times in the vocabulary
     1051 words appear more than 100 times in the vocabulary <-- Lets try with this. TODO(Select a dataset with the captions and images having these words, and build a dataset with these words)
     634 words appear more than 200 times in the vocabulary
     294 words appear more than 500 times in the vocabulary
    """
    import pickle
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    v = QuerySparseInvertedIndexer(indexer_path)
    for threshold in [10, 50, 100, 200, 500, 1000, 2000, 2500, 3000, 3500, 4000]:
        num = 0
        keys_to_keep_in_vocab = []
        for k in v.inverted_index.keys():
            if len(v.inverted_index[k]) > threshold:
                num += 1
                keys_to_keep_in_vocab.append(k)
        print(f' {num} words appear more than {threshold} times in the vocabulary')
        inverted_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
        words_to_keep = [inverted_vocab[key] for key in keys_to_keep_in_vocab]
        filtered_vectorizer = CountVectorizer(tokenizer=spacy_tokenizer, stop_words='english')
        filtered_vectorizer.fit(words_to_keep)
        path = vectorizer_path.split('.pkl')[0]
        with open(f'{path}_filtered_{threshold}.pkl', 'wb') as f:
            pickle.dump(filtered_vectorizer, f)


def create_filtered_datasets(vectorizer_path: str = '/hdd/master/tfm/vectorizer_tokenizer_stop_words_all_words.pkl'):
    train_dataset = CaptionFlickr30kDataset(root='/hdd/master/tfm/flickr30k_images',
                                            split_root='/hdd/master/tfm/flickr30k_images/flickr30k_entities',
                                            split='train')
    test_dataset = CaptionFlickr30kDataset(root='/hdd/master/tfm/flickr30k_images',
                                           split_root='/hdd/master/tfm/flickr30k_images/flickr30k_entities',
                                           split='test')
    val_dataset = CaptionFlickr30kDataset(root='/hdd/master/tfm/flickr30k_images',
                                          split_root='/hdd/master/tfm/flickr30k_images/flickr30k_entities',
                                          split='val')

    corpus = [train_dataset[i] for i in range(len(train_dataset))] + \
             [test_dataset[i] for i in range(len(test_dataset))] + \
             [val_dataset[i] for i in range(len(val_dataset))]

    for threshold in [10, 50, 100, 200, 500, 1000, 2000, 2500, 3000]:
        vectorizer_path_threshold = f'{vectorizer_path.split(".pkl")[0]}_filtered_{threshold}.pkl'
        text_encoder = TextEncoder(vectorizer_path_threshold)

        with Bar(f'Element in corpus', max=len(corpus)) as bar:
            filenames = set()
            for filename, caption in corpus:
                tag = filename.split('.jpg')[0]
                text_embedding = text_encoder.forward([caption])
                if csr_matrix(text_embedding.detach().numpy()).getnnz() > 0:
                    if tag not in filenames:
                        filenames.add(tag)
                bar.next()
        print(f' Number of elements in the filtered dataset for threshold {threshold} ==> {len(filenames)}')

        with Bar(f'Writing in filter_{threshold} split', max=len(filenames)) as bar:
            with open(f'/hdd/master/tfm/flickr30k_images/flickr30k_entities/filter_{threshold}.txt', 'w') as f:
                for tag in filenames:
                    f.write(tag)
                    f.write('\n')


if __name__ == '__main__':
    get_model(vectorizer_path='/hdd/master/tfm/vectorizer_tokenizer_stop_words_all_words.pkl')
    build_inverted_index_text(vectorizer_path='/hdd/master/tfm/vectorizer_tokenizer_stop_words_all_words.pkl')
    analyze_word_frequencies()
    filter_vectorizers(vectorizer_path='/hdd/master/tfm/vectorizer_tokenizer_stop_words_all_words.pkl')
    create_filtered_datasets(vectorizer_path='/hdd/master/tfm/vectorizer_tokenizer_stop_words_all_words.pkl')
