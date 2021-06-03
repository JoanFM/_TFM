import torch
import time
import os
import pickle
from progress.bar import Bar

from src.model import ImageEncoder
from src.model import TextEncoder
from src.dataset import get_data_loader
from src.vector_store.sparse_inverted_index import AddSparseInvertedIndexer, QuerySparseInvertedIndexer
from scipy.sparse import csr_matrix
import numpy as np

cur_dir = os.path.dirname(os.path.abspath(__file__))

l1_regularization_weight = 1e-7


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


def validation_loop(image_encoder, text_encoder, dataloader, device, loss_fn, batch_size, training_batch_id):
    positive_tensor = torch.Tensor([1]).to(device)
    val_loss = []
    with Bar(f'Validation for training batch {training_batch_id} Batch',
             max=len(dataloader.dataset) / batch_size) as validation_bar:
        for batch_id, (_, image, caption) in enumerate(dataloader):
            image_encoder.feature_extractor.model.eval()
            image_encoder.sparse_encoder.eval()
            image = image.to(device)
            image_embedding = image_encoder.forward(image)
            text_embedding = text_encoder.forward(caption)
            text_embedding = text_embedding.to(device)
            l1_regularization = torch.mean(torch.sum(image_embedding, dim=1))
            loss = loss_fn(image_embedding, text_embedding,
                           positive_tensor) + l1_regularization_weight * l1_regularization
            val_loss.append(loss.item())
            validation_bar.next()

    return val_loss


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
    optimizer = torch.optim.Adam(image_encoder.parameters(), lr=0.01)
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

            test_data_loader = get_data_loader(root='/hdd/master/tfm/flickr30k_images',
                                               split_root='/hdd/master/tfm/flickr30k_images/flickr30k_entities',
                                               split='test',
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
                    text_embedding = text_encoder.forward(caption)
                    text_embedding = text_embedding.to(device)
                    l1_regularization = torch.mean(torch.sum(image_embedding, dim=1))
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
                    if batch_id % 3000 == 0:
                        torch.save(image_encoder.state_dict(),
                                   output_model_path + '/model-inter-' + str(epoch + 1) + '-' + str(batch_id) + '.pt')
                    if batch_id % 4500 == 0:
                        val_loss = validation_loop(image_encoder, text_encoder, val_data_loader, device, loss_fn,
                                                   batch_size, batch_id)
                        print(
                            f'[{batch_id}]\tvalidation loss:\t{np.mean(np.array(val_loss))}\ttime lapsed:\t{time.time() - time_start} s')

                        time_start = time.time()
                    training_bar.next()
                torch.save(image_encoder.state_dict(),
                           output_model_path + '/model-inter-' + str(epoch + 1) + '-final.pt')

            with open(f'train_loss-{epoch}', 'wb') as f:
                pickle.dump(train_loss, f)

            r_at_1 = evaluate_t2i(image_encoder, text_encoder, test_data_loader,
                                  os.path.join(validation_indexers_base_path, f'epoch-{epoch}'))
            print('#' * 70)
            print(f'accuracy at the end of epoch {epoch}/{num_epochs}: ', r_at_1)
            epoch_bar.next()


if __name__ == '__main__':
    train()
