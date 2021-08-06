import torch
import time
import os
import pickle
from typing import List
from progress.bar import Bar

from src.model import ImageEncoder
from src.model import TextEncoder
from src.dataset import get_data_loader, get_image_data_loader
from src.vector_store.sparse_inverted_index import AddSparseInvertedIndexer, QuerySparseInvertedIndexer
from src.evaluate import evaluate
from scipy.sparse import csr_matrix
import numpy as np

cur_dir = os.path.dirname(os.path.abspath(__file__))

l1_regularization_weight = 1e-7


def evaluate_t2i(image_encoder, text_encoder, validation_indexers_path, batch_size, root, split_root, split):
    image_data_loader = get_image_data_loader(root=root, split_root=split_root, split=split, batch_size=batch_size)
    text_data_loader = get_data_loader(root=root, split_root=split_root, split=split, batch_size=batch_size)
    image_encoder.feature_extractor.model.eval()
    image_encoder.sparse_encoder.eval()
    image_encoder.eval()

    with Bar(f'Indexing images into Sparse Indexer',
             max=len(image_data_loader)) as indexing_bar:
        with AddSparseInvertedIndexer(
                base_path=validation_indexers_path) as indexer:
            for batch_id, (filenames, images) in enumerate(image_data_loader):
                image_embedding = image_encoder.forward(images).detach().numpy()
                indexer.add(filenames, csr_matrix(image_embedding))
                indexing_bar.next()
            indexer.analyze()

    with Bar(f'Querying Sparse Indexer with text',
             max=len(text_data_loader)) as querying_bar:
        retrieved_image_filenames = []  # it should be a list of lists
        groundtruth_expected_image_filenames = []  # it should be a list of lists
        with AddSparseInvertedIndexer(base_path=f'{validation_indexers_path}-text') as text_indexer:
            query_indexer = QuerySparseInvertedIndexer(
                base_path=validation_indexers_path)
            for batch_id, (filenames, _, captions) in enumerate(text_data_loader):
                text_embedding_query = csr_matrix(text_encoder.forward(captions).detach().numpy())
                text_indexer.add(filenames, text_embedding_query)
                for i in range(text_embedding_query.shape[0]):
                    results = query_indexer.search(text_embedding_query.getrow(i), None)
                    retrieved_image_filenames.append(results)
                    groundtruth_expected_image_filenames.append([filenames[i]])
                querying_bar.next()
            print(f' Analyze Query Indexer')
            query_indexer.analyze()
        print(f' Analyze Add Text Indexer')
        text_indexer.analyze()

        return evaluate(['recall', 'reciprocal_rank'], retrieved_image_filenames, groundtruth_expected_image_filenames,
                        None)


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


def train(output_model_path: str = '/hdd/master/tfm/output_models-test',
          vectorizer_path: str = '/hdd/master/tfm/vectorizer_tokenizer_stop_words_analyze_filtered.pkl',
          validation_indexers_base_path: str = '/hdd/master/tfm/sparse_indexers_tmp-test',
          num_epochs: int = 100,
          batch_size: int = 8,
          layers: List[int] = [1062]
          ):
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    # pos_weight = torch.ones([layers[-1]])
    os.makedirs(output_model_path, exist_ok=True)
    image_encoder = ImageEncoder(layer_size=layers)
    text_encoder = TextEncoder(vectorizer_path)
    optimizer = torch.optim.SGD(image_encoder.parameters(), lr=0.5)
    optimizer.zero_grad()
    loss_fn = torch.nn.BCEWithLogitsLoss()

    with Bar('Epochs', max=num_epochs) as epoch_bar:

        for epoch in range(num_epochs):
            train_data_loader = get_data_loader(root='/hdd/master/tfm/flickr30k_images',
                                                split_root='/hdd/master/tfm/flickr30k_images/flickr30k_entities',
                                                split='filter_small',
                                                shuffle=True,
                                                batch_size=batch_size)

            val_data_loader = get_data_loader(root='/hdd/master/tfm/flickr30k_images',
                                              split_root='/hdd/master/tfm/flickr30k_images/flickr30k_entities',
                                              split='filter_small',
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
                    if batch_id % 20 == 0:
                        for j in range(len(image_embedding[0])):
                            if text_embedding[0][j] == 1:
                                print(f' image {image_embedding[0][j]} => text {text_embedding[0][j]}')
                    loss = loss_fn(image_embedding, text_embedding)
                    train_loss.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    if batch_id % 20 == 0:
                        print(
                            f'[{batch_id}] \t training loss:\t {np.mean(np.array(train_loss))} \t time elapsed:\t {time.time() - time_start} s')
                        train_loss.clear()
                        time_start = time.time()
                    if batch_id % 100 == 0 and batch_id != 0:
                        torch.save(image_encoder.state_dict(),
                                   output_model_path + '/model-inter-' + str(epoch + 1) + '-' + str(batch_id) + '.pt')
                    if batch_id % 1000 == 0 and batch_id != 0:
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

            if epoch % 10 == 0 and epoch != 0:
                evaluation_results = evaluate_t2i(image_encoder, text_encoder,
                                                  os.path.join(validation_indexers_base_path, f'epoch-{epoch}'),
                                                  batch_size, root='/hdd/master/tfm/flickr30k_images',
                                                  split_root='/hdd/master/tfm/flickr30k_images/flickr30k_entities',
                                                  split='filter_small')
            print('#' * 70)
            print(f'Buckets eval at the end of epoch {epoch}/{num_epochs}: ', evaluation_results)
            epoch_bar.next()


if __name__ == '__main__':
    min_num_appareances = 1000
    filter_layer_size = {'3000': 47, '2000': 80, '1000': 165, '500': 311, '100': 1062}
    train(
        vectorizer_path=f'/hdd/master/tfm/vectorizer_tokenizer_stop_words_all_words_filtered_{min_num_appareances}.pkl',
        layers=[256, filter_layer_size[str(min_num_appareances)]], batch_size=8)
