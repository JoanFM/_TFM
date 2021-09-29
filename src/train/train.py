import torch
import time
import os
import pickle
from typing import List
from progress.bar import Bar

from src.model import ImageEncoder
from src.model import TextEncoder
from src.dataset import get_data_loader, get_image_data_loader, get_captions_data_loader
from src.vector_store.sparse_inverted_index import AddSparseInvertedIndexer, QuerySparseInvertedIndexer
from src.evaluate import evaluate
from scipy.sparse import csr_matrix
import numpy as np

cur_dir = os.path.dirname(os.path.abspath(__file__))

l1_regularization_weight = 1e-7


def run_evaluations(image_encoder, text_encoder, validation_indexers_path, batch_size, root, split_root, split,
                    top_ks=[5, 10, 20, None]):
    image_data_loader = get_image_data_loader(root=root, split_root=split_root, split=split, batch_size=batch_size)
    text_data_loader = get_captions_data_loader(root=root, split_root=split_root, split=split, batch_size=batch_size)
    image_encoder.feature_extractor.model.eval()
    image_encoder.sparse_encoder.eval()
    image_encoder.eval()
    eval_start = time.time()
    with Bar(f'Indexing images into Sparse Index',
             max=len(image_data_loader)) as image_indexing_bar:
        with AddSparseInvertedIndexer(
                base_path=f'{validation_indexers_path}-{split}-image') as image_indexer:
            for batch_id, (filenames, images) in enumerate(image_data_loader):
                image_embedding = image_encoder.forward(images).detach().numpy()
                image_indexer.add(filenames, csr_matrix(image_embedding))
                image_indexing_bar.next()
            print(f' Analyze Image Indexer Add')
            image_indexer.analyze()

    retrieved_image_filenames = []  # it should be a list of lists
    groundtruth_expected_image_filenames = []  # it should be a list of lists
    num_buckets_query = []
    with Bar(f'Querying Image Sparse Index with text',
             max=len(text_data_loader)) as querying_bar:
        with AddSparseInvertedIndexer(base_path=f'{validation_indexers_path}-{split}-text') as text_indexer:
            query_indexer = QuerySparseInvertedIndexer(
                base_path=f'{validation_indexers_path}-{split}-image')
            for batch_id, (filenames, captions) in enumerate(text_data_loader):
                # print(f' captions {captions}, filenames {filenames}')
                text_embedding_query = csr_matrix(text_encoder.forward(captions).detach().numpy())
                text_indexer.add(list(zip(captions, filenames)), text_embedding_query)

                for i in range(text_embedding_query.shape[0]):
                    num_buckets_query.append(len(text_embedding_query.getrow(i).indices))
                    results = query_indexer.search(text_embedding_query.getrow(i), None)
                    retrieved_image_filenames.append(results)
                    groundtruth_expected_image_filenames.append([filenames[i]])
                querying_bar.next()
        print(f' Analyze Image Indexer Query')
        query_indexer.analyze()

    t2i_evaluations = evaluate(['recall', 'reciprocal_rank', 'num_candidates'], retrieved_image_filenames,
                               groundtruth_expected_image_filenames,
                               top_ks)
    print('#' * 70)
    print(f'{split} EVALUATION text2Image retrieval: ', t2i_evaluations)
    print(f' Average number of buckets for query {np.average(num_buckets_query)}')
    print(f'time elapsed:\t {time.time() - eval_start}')

    # eval_start = time.time()
    #
    # with Bar(f'Querying Text Sparse Index with images',
    #          max=len(image_data_loader)) as querying_bar:
    #     retrieved_image_filenames = []  # it should be a list of lists
    #     groundtruth_expected_image_filenames = []  # it should be a list of lists
    #     query_indexer = QuerySparseInvertedIndexer(
    #         base_path=f'{validation_indexers_path}-{split}-text')
    #     print(f' Analyze Text Index')
    #     query_indexer.analyze()
    #     for batch_id, (filenames, images) in enumerate(image_data_loader):
    #         image_embedding = image_encoder.forward(images).detach().numpy()
    #         image_embedding_query = csr_matrix(image_embedding)
    #         print(f' image_embedding_query shape {image_embedding_query.shape}')
    #         for i in range(image_embedding_query.shape[0]):
    #             results = query_indexer.search(image_embedding_query.getrow(i), None)
    #             print(f' results length {len(results)}')
    #             retrieved_image_filenames.append([result[1] for result in results])
    #             groundtruth_expected_image_filenames.append([filenames[i]])
    #         querying_bar.next()
    #     print(f' Analyze Text Index')
    #     query_indexer.analyze()
    #
    # i2t_evaluations = evaluate(['recall', 'reciprocal_rank', 'num_candidates'], retrieved_image_filenames,
    #                            groundtruth_expected_image_filenames,
    #                            top_ks)
    # print('#' * 70)
    # print(f'{split} EVALUATION image2Text retrieval: ', i2t_evaluations)
    # print(f'time elapsed:\t {time.time() - eval_start}')


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


def validation_loop(image_encoder, text_encoder, dataloader, device, loss_fn, training_batch_id):
    # positive_tensor = torch.Tensor([1]).to(device)
    val_loss = []
    with Bar(f'Validation for training batch {training_batch_id} Batch',
             max=len(dataloader)) as validation_bar:
        for batch_id, (_, image, caption) in enumerate(dataloader):
            image_encoder.feature_extractor.model.eval()
            image_encoder.sparse_encoder.eval()
            image = image.to(device)
            image_embedding = image_encoder.forward(image)
            text_embedding = text_encoder.forward(caption)
            text_embedding = text_embedding.to(device)
            text_embedding = text_embedding / text_embedding
            text_embedding[text_embedding != text_embedding] = 0
            loss = loss_fn(image_embedding, text_embedding)
            val_loss.append(loss.item())
            validation_bar.next()

    return val_loss


def compute_average_positives_in_vocab(vectorizer_path,
                                       device='cpu'):
    batch_size = 1
    text_encoder = TextEncoder(vectorizer_path)
    train_data_loader = get_data_loader(root='/hdd/master/tfm/flickr30k_images',
                                        split_root='/hdd/master/tfm/flickr30k_images/flickr30k_entities',
                                        split='train',
                                        shuffle=True,
                                        batch_size=batch_size)

    num_positives = []
    num_negatives = []
    num_totals = []
    with Bar(f'Caption ', max=len(train_data_loader.dataset) / batch_size) as bar:
        for batch_id, (_, image, caption) in enumerate(train_data_loader):
            text_embedding = text_encoder.forward(caption).to(device)
            text_embedding = text_embedding / text_embedding
            text_embedding[text_embedding != text_embedding] = 0
            sparse_text_embedding = csr_matrix(text_embedding)
            num_positives.append(len(sparse_text_embedding.indices))
            num_negatives.append(sparse_text_embedding.shape[1] - len(sparse_text_embedding.indices))
            num_totals.append(sparse_text_embedding.shape[1])
            bar.next()
    return np.mean(num_positives), np.mean(num_negatives), np.mean(num_totals)


def train(output_model_path: str = '/hdd/master/tfm/output_models-test',
          vectorizer_path: str = '/hdd/master/tfm/vectorizer_tokenizer_stop_words_analyze_filtered.pkl',
          validation_indexers_base_path: str = '/hdd/master/tfm/sparse_indexers',
          positive_weights: float = 1.0,
          num_epochs: int = 100,
          batch_size: int = 8,
          layers: List[int] = [1062]
          ):
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    pos_weight = torch.ones([layers[-1]]) * positive_weights
    os.makedirs(output_model_path, exist_ok=True)
    image_encoder = ImageEncoder(layer_size=layers)
    text_encoder = TextEncoder(vectorizer_path)
    optimizer = torch.optim.SGD(image_encoder.parameters(), lr=0.5)
    optimizer.zero_grad()
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    with Bar('Epochs', max=num_epochs) as epoch_bar:

        for epoch in range(num_epochs):
            train_data_loader = get_data_loader(root='/hdd/master/tfm/flickr30k_images',
                                                split_root='/hdd/master/tfm/flickr30k_images/flickr30k_entities',
                                                split='train',
                                                shuffle=True,
                                                batch_size=batch_size)

            val_data_loader = get_data_loader(root='/hdd/master/tfm/flickr30k_images',
                                              split_root='/hdd/master/tfm/flickr30k_images/flickr30k_entities',
                                              split='val',
                                              shuffle=True,
                                              batch_size=batch_size)

            train_loss = []
            time_start = time.time()
            # positive_tensor = torch.Tensor([1]).to(device)

            with Bar(f'Batch in epoch {epoch}', max=len(train_data_loader.dataset) / batch_size) as training_bar:

                for batch_id, (_, image, caption) in enumerate(train_data_loader):
                    image_encoder.feature_extractor.model.eval()
                    image_encoder.train()
                    image_encoder.sparse_encoder.train()
                    optimizer.zero_grad()
                    image = image.to(device)
                    image_embedding = image_encoder.forward(image)

                    text_embedding = text_encoder.forward(caption).to(device)
                    text_embedding = text_embedding / text_embedding
                    text_embedding[text_embedding != text_embedding] = 0
                    loss = loss_fn(image_embedding, text_embedding)
                    train_loss.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    if batch_id % 200 == 0:
                        print(
                            f'[{batch_id}] \t training loss:\t {np.mean(np.array(train_loss))} \t time elapsed:\t {time.time() - time_start} s')
                        train_loss.clear()
                        time_start = time.time()
                    if batch_id % 1000 == 0 and batch_id != 0:
                        torch.save(image_encoder.state_dict(),
                                   output_model_path + '/model-inter-' + str(epoch + 1) + '-' + str(batch_id) + '.pt')
                    if batch_id % 200 == 0 and batch_id != 0:
                        val_loss = validation_loop(image_encoder, text_encoder, val_data_loader, device, loss_fn,
                                                   batch_id)
                        print(
                            f'[{batch_id}]\tvalidation loss:\t{np.mean(np.array(val_loss))}\ttime lapsed:\t{time.time() - time_start} s')

                        time_start = time.time()
                    training_bar.next()
                torch.save(image_encoder.state_dict(),
                           output_model_path + '/model-inter-' + str(epoch + 1) + '-final.pt')

            with open(f'train_loss-{epoch}', 'wb') as f:
                pickle.dump(train_loss, f)

            if epoch % 1 == 0:
                run_evaluations(image_encoder, text_encoder,
                                os.path.join(validation_indexers_base_path, f'epoch-{epoch}'),
                                batch_size, root='/hdd/master/tfm/flickr30k_images',
                                split_root='/hdd/master/tfm/flickr30k_images/flickr30k_entities',
                                split='test')
                run_evaluations(image_encoder, text_encoder,
                                os.path.join(validation_indexers_base_path, f'epoch-{epoch}'),
                                batch_size, root='/hdd/master/tfm/flickr30k_images',
                                split_root='/hdd/master/tfm/flickr30k_images/flickr30k_entities',
                                split='val')
                # run_evaluations(image_encoder, text_encoder,
                #                 os.path.join(validation_indexers_base_path, f'epoch-{epoch}'),
                #                 batch_size, root='/hdd/master/tfm/flickr30k_images',
                #                 split_root='/hdd/master/tfm/flickr30k_images/flickr30k_entities',
                #                 split='train')
            epoch_bar.next()


if __name__ == '__main__':
    min_num_appareances = 10
    filter_layer_size = {'3000': 47, '2000': 80, '1000': 165, '500': 311, '100': 1062, '10': 3537, None: 13439}
    vectorizer_path = f'/hdd/master/tfm/vectorizer_tokenizer_stop_words_all_words_filtered_{min_num_appareances}.pkl' if min_num_appareances is not None else f'/hdd/master/tfm/vectorizer_tokenizer_stop_words_all_words.pkl'
    mean_positives, mean_negatives, mean_totals = compute_average_positives_in_vocab(vectorizer_path, 'cpu')
    positive_weights = mean_negatives / mean_positives
    # positive_weights = 190.2681631496955
    print(
        f' mean_positives {mean_positives}, mean_negatives {mean_negatives}, num_totals {mean_negatives} => positive_weights {positive_weights}')
    train(
        vectorizer_path=vectorizer_path,
        layers=[4096, filter_layer_size[str(min_num_appareances) if min_num_appareances is not None else None]],
        positive_weights=positive_weights,
        batch_size=16)
