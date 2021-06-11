import torch
import time
import os
import pickle
from progress.bar import Bar

from src.model import ImageEncoder
from src.model import TextEncoder
from src.dataset import get_data_loader, get_image_data_loader, get_captions_data_loader
from src.vector_store.sparse_inverted_index import AddSparseInvertedIndexer, QuerySparseInvertedIndexer
from scipy.sparse import csr_matrix
import numpy as np

cur_dir = os.path.dirname(os.path.abspath(__file__))

l1_regularization_weight = 1e-6


def evaluate_in_buckets_t2i(image_encoder, text_encoder, validation_indexers_path, batch_size, root, split_root, split):
    image_data_loader = get_image_data_loader(root=root, split_root=split_root, split=split, batch_size=batch_size)
    text_data_loader = get_captions_data_loader(root=root, split_root=split_root, split=split, batch_size=batch_size)
    image_encoder.feature_extractor.model.eval()
    image_encoder.sparse_encoder.eval()

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
        total_found = 0
        total_query_cardinality = 0
        num_of_queries_where_expected_result_would_be_candidate = 0
        indexer = QuerySparseInvertedIndexer(
            base_path=validation_indexers_path)
        for batch_id, (filenames, captions) in enumerate(text_data_loader):
            text_embedding_query = csr_matrix(text_encoder.forward(captions).detach().numpy())
            for i in range(text_embedding_query.shape[0]):
                query_cardinality = text_embedding_query.getrow(i).getnnz()
                results = indexer.search(text_embedding_query.getrow(i), None)
                number_of_buckets_found = 0
                for result in results:
                    if result == filenames[i]:
                        number_of_buckets_found += 1

                if number_of_buckets_found > 1:
                    num_of_queries_where_expected_result_would_be_candidate += 1

                total_found += number_of_buckets_found
                total_query_cardinality += query_cardinality
            querying_bar.next()
        indexer.analyze()
        print(f'On average, the image has fallen in {(total_found / total_query_cardinality) * 100}% of the '
              f'expected text buckets')
        print(
            f'On average, the image would be an overall candidate (TOP_K NONE) {(num_of_queries_where_expected_result_would_be_candidate / len(text_data_loader.dataset)) * 100}% of the '
            f'queries')

    return total_found / total_query_cardinality


def evaluate_t2i(image_encoder, text_encoder, dataloader, validation_indexers_path):
    image_encoder.feature_extractor.model.eval()
    image_encoder.sparse_encoder.eval()
    number_correct_top_1 = 0

    ids = []
    matrix = None
    queries = None
    query_expected_ids = []
    num_entry = 0
    for batch_id, (filename, image_test, caption_test) in enumerate(dataloader):
        ids.extend(filename)
        image_embedding = image_encoder.forward(image_test).detach().numpy()
        text_embedding = text_encoder.forward(caption_test).detach().numpy()
        text_sparse_embedding = csr_matrix(text_embedding)
        # change to put image_embedding
        matrix = csr_vappend(matrix, csr_matrix(image_embedding))
        queries = csr_vappend(queries, text_sparse_embedding)
        for caption in caption_test:
            query_expected_ids.append(filename)
        num_entry += len(caption_test)
    print("#" * 70)
    print(f'Indexing image sparse embeddings')
    with AddSparseInvertedIndexer(
            base_path=validation_indexers_path) as indexer:
        indexer.add(ids, matrix)

    query_indexer = QuerySparseInvertedIndexer(
        base_path=validation_indexers_path)
    print("#" * 70)
    print(f'Let\'s query with all text sparse embeddings')
    for i in range(queries.shape[0]):
        results = query_indexer.search(queries.getrow(i), top_k=1)
        if results[0] == query_expected_ids[i]:
            number_correct_top_1 += 1
    r_at_1 = number_correct_top_1 / len(query_expected_ids)
    return r_at_1


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
    positive_tensor = torch.Tensor([1]).to(device)
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
            l1_regularization = torch.sum(torch.mean(image_embedding, dim=1))
            loss = loss_fn(image_embedding, text_embedding,
                           positive_tensor) + l1_regularization_weight * l1_regularization
            val_loss.append(loss.item())
            validation_bar.next()

    return val_loss


def train(output_model_path: str = '/hdd/master/tfm/output_models',
          vectorizer_path: str = '/hdd/master/tfm/vectorizer.pkl',
          validation_indexers_base_path: str = '/hdd/master/tfm/sparse_indexers_tmp',
          num_epochs: int = 3,
          batch_size: int = 16):
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    os.makedirs(output_model_path, exist_ok=True)
    image_encoder = ImageEncoder()
    text_encoder = TextEncoder(vectorizer_path)
    optimizer = torch.optim.Adam(image_encoder.parameters(), lr=0.02)
    optimizer.zero_grad()
    loss_fn = torch.nn.CosineEmbeddingLoss(margin=0)

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
            positive_tensor = torch.Tensor([1]).to(device)

            with Bar(f'Batch in epoch {epoch}', max=len(train_data_loader.dataset) / batch_size) as training_bar:

                for batch_id, (_, image, caption) in enumerate(train_data_loader):
                    image_encoder.feature_extractor.model.eval()
                    image_encoder.sparse_encoder.train()
                    optimizer.zero_grad()
                    image = image.to(device)
                    image_embedding = image_encoder.forward(image)
                    text_embedding = text_encoder.forward(caption).to(device)
                    l1_regularization = torch.sum(torch.mean(image_embedding, dim=1))
                    loss = loss_fn(image_embedding, text_embedding,
                                   positive_tensor) + l1_regularization_weight * l1_regularization
                    train_loss.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    if batch_id % 100 == 0:
                        print(
                            f'[{batch_id}] \t training loss:\t {np.mean(np.array(train_loss))} \t time elapsed:\t {time.time() - time_start} s')
                        train_loss.clear()
                        time_start = time.time()
                    if batch_id % 3000 == 0 and batch_id != 0:
                        torch.save(image_encoder.state_dict(),
                                   output_model_path + '/model-inter-' + str(epoch + 1) + '-' + str(batch_id) + '.pt')
                    if batch_id % 4500 == 0 and batch_id != 0:
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

            buckets_eval = evaluate_in_buckets_t2i(image_encoder, text_encoder,
                                                   os.path.join(validation_indexers_base_path, f'epoch-{epoch}'),
                                                   batch_size, root='/hdd/master/tfm/flickr30k_images',
                                                   split_root='/hdd/master/tfm/flickr30k_images/flickr30k_entities',
                                                   split='test')
            print('#' * 70)
            print(f'Buckets eval at the end of epoch {epoch}/{num_epochs}: ', buckets_eval)
            epoch_bar.next()


def evaluate_buckets_t2i(output_model_path: str = '/hdd/master/tfm/output_models',
                         vectorizer_path: str = '/hdd/master/tfm/vectorizer.pkl',
                         validation_indexers_base_path: str = '/hdd/master/tfm/sparse_indexers_tmp',
                         batch_size: int = 16):
    image_encoder = ImageEncoder()
    text_encoder = TextEncoder(vectorizer_path)
    image_encoder.load_state_dict(torch.load(os.path.join(output_model_path, 'model-inter-1-final.pt')))

    buckets_eval = evaluate_in_buckets_t2i(image_encoder, text_encoder,
                                           os.path.join(validation_indexers_base_path, f'eval-test'),
                                           batch_size, root='/hdd/master/tfm/flickr30k_images',
                                           split_root='/hdd/master/tfm/flickr30k_images/flickr30k_entities',
                                           split='test')

    print(f' buckets_eval {buckets_eval}')


def analyze_vocab_learnt(vectorizer_path: str, inverted_index_base_path: str):
    import pickle
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    query_indexer = QuerySparseInvertedIndexer(base_path=inverted_index_base_path)
    inverse_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
    for b in query_indexer.inverted_index.keys():
        if len(query_indexer.inverted_index[b]) > 0:
            print(f' {inverse_vocab[b]}')


if __name__ == '__main__':
    train()
