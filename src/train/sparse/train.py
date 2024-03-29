import torch
import time
import os
import pickle
from typing import List, Optional, Union

import numpy as np
from progress.bar import Bar

from src.model.sparse.image import ImageEncoder
from src.model.sparse.text import TextEncoder

from src.dataset import get_concatenated_captions_image_data_loader, get_image_data_loader, get_captions_data_loader
from src.evaluate import evaluate
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cur_dir = os.path.dirname(os.path.abspath(__file__))

# The base directory where models are stored after every epoch
IMAGE_EMBEDDING_BASE_PATH = os.getenv('IMAGE_EMBEDDING_BASE_PATH', '/hdd/master/tfm/output-image-encoders')

TEXT_EMBEDDING_VECTORIZER_PATH = os.getenv('TEXT_EMBEDDING_VECTORIZER_PATH', 'vectorizer.pkl')
# The root path where the flickr30k dataset is found
DATASET_ROOT_PATH = os.getenv('DATASET_ROOT_PATH', '/hdd/master/tfm/flickr30k_images')
# The root path where the flickr30k entities per split is kept
DATASET_SPLIT_ROOT_PATH = os.getenv('DATASET_SPLIT_ROOT_PATH', '/hdd/master/tfm/flickr30k_images/flickr30k_entities')

l1_regularization_weight = 1e-3

_ATTRIBUTES = {
    'bold': 1,
    'dark': 2,
    'underline': 4,
    'blink': 5,
    'reverse': 7,
    'concealed': 8,
}

_HIGHLIGHTS = {
    'on_grey': 40,
    'on_red': 41,
    'on_green': 42,
    'on_yellow': 43,
    'on_blue': 44,
    'on_magenta': 45,
    'on_cyan': 46,
    'on_white': 47,
}

_COLORS = {
    'black': 30,
    'red': 31,
    'green': 32,
    'yellow': 33,
    'blue': 34,
    'magenta': 35,
    'cyan': 36,
    'white': 37,
}

_RESET = '\033[0m'


def colored(
        text: str,
        color: Optional[str] = None,
        on_color: Optional[str] = None,
        attrs: Optional[Union[str, list]] = None,
):
    fmt_str = '\033[%dm%s'
    if color:
        text = fmt_str % (_COLORS[color], text)
    if on_color:
        text = fmt_str % (_HIGHLIGHTS[on_color], text)

    if attrs:
        if isinstance(attrs, str):
            attrs = [attrs]
        if isinstance(attrs, list):
            for attr in attrs:
                text = fmt_str % (_ATTRIBUTES[attr], text)
    text += _RESET
    return text


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


def analyse_index(index_images: csr_matrix):
    """
    Print some analysis data on the inverted index.

    How big is the vocabulary,
    How many words (tokens) (keys) have at least one document attached
    How many words do not have a document (image) attached
    How many images are at each backet (with and without counting empty bins)
    """
    count_by_token = {}
    size_vocabulary = index_images.shape[1]
    num_images = index_images.shape[0]
    num_active_features = []
    print(f' Analyzing index with {num_images} images with vocabulary size {size_vocabulary}')

    for i in range(num_images):
        num_active_features.append(len(index_images.getrow(i).indices))
        for _i in index_images.getrow(i).indices:
            if _i not in count_by_token:
                count_by_token[_i] = 0
            count_by_token[_i] += 1

    counts = []
    counts_ignoring_0_sized = []
    for i in range(size_vocabulary):
        if i in count_by_token:
            counts_ignoring_0_sized.append(count_by_token[i])
            counts.append(count_by_token[i])
        else:
            counts.append(0.0)
    print(
        colored(f' Number of keys with at least one candidate => {len(count_by_token.keys())} out of {size_vocabulary}',
                'cyan'))
    print(colored(f' Size of vocabulary {size_vocabulary}', 'cyan'))
    print(colored(f' Number of empty bins => {size_vocabulary - len(count_by_token.keys())}', 'cyan'))
    print(colored(
        f' Average amount of documents per bucket, ignoring 0-sized buckets => mean: {np.mean(counts_ignoring_0_sized)}, std: {np.std(counts_ignoring_0_sized)}',
        'cyan'))
    print(colored(
        f' Average amount of documents per bucket, counting 0-sized buckets => mean: {np.mean(counts)}, std: {np.std(counts)}',
        'cyan'))
    print(colored(
        f' Average active features in images => mean: {np.mean(num_active_features)}, std: {np.std(num_active_features)}',
        'cyan'))


def extract_all_image_embeddings_and_filenames(image_encoder, batch_size, root, split_root, split):
    """
    Extracts all the image embeddings as a csr_matrix and all the image filenames from a split of data with a given image encoder

    :param image_encoder: The ImageEncoder to extract the embeddings from where to compute evaluation
    :param batch_size: The batch_size to load
    :param root: The root file path of the data loader, where the images are found
    :param split_root: The root file path of the data loader specific to the split, where the indices of the split are kept
    :param split: The split of data to evaluate on (eval, val, train)
    :return:
    """
    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    device = torch.device(dev)
    image_encoder.feature_extractor.model.eval()
    image_encoder.sparse_encoder.eval()
    image_encoder.eval()
    image_data_loader = get_image_data_loader(root=root, split_root=split_root, split=split, batch_size=batch_size)
    all_image_embeddings = None
    image_filenames = []
    print(colored(f' Extract the sparse embeddings of all the images', 'black', 'on_yellow'))
    with Bar(f'Indexing images into Sparse Index',
             max=len(image_data_loader)) as image_indexing_bar:
        for batch_id, (filenames, images) in enumerate(image_data_loader):
            images = images.to(device)
            image_embedding = image_encoder.forward(images).detach().cpu().numpy()
            sparse_embedding = csr_matrix(image_embedding)
            if all_image_embeddings is None:
                all_image_embeddings = sparse_embedding
            else:
                all_image_embeddings = csr_vappend(all_image_embeddings, sparse_embedding)
            image_filenames.extend(filenames)
            image_indexing_bar.next()
            if batch_id % 100 == 0:
                print(
                    f' Indexed {batch_size * (batch_id + 1)} images out of {len(image_data_loader) * batch_size}')
    return all_image_embeddings, image_filenames


def analyse(image_encoder, batch_size, root, split_root, split):
    """
    Analyses the index resulting from extracting all the embeddings from the images in a split of data with a given image encoder

    :param image_encoder: The ImageEncoder to extract the embeddings from where to compute evaluation
    :param batch_size: The batch_size to load
    :param root: The root file path of the data loader, where the images are found
    :param split_root: The root file path of the data loader specific to the split, where the indices of the split are kept
    :param split: The split of data to evaluate on (eval, val, train)
    :return:
    """
    all_image_embeddings, image_filenames = extract_all_image_embeddings_and_filenames(image_encoder, batch_size, root,
                                                                                       split_root, split)
    tfidf_transformer = TfidfTransformer()
    image_tfidf_index = tfidf_transformer.fit_transform(all_image_embeddings)
    analyse_index(image_tfidf_index)


def run_evaluations(image_encoder, text_encoder, batch_size, root, split_root, split,
                    top_ks=[5, 10, 20, 30, 40, 50, 100, None]):
    """
    Runs evaluations of an ImageEncoder resulting from some training

    :param image_encoder: The ImageEncoder to extract the embeddings from where to compute evaluation
    :param text_encoder: The TextEncoder to extract the embeddings from where to compute evaluation
    :param batch_size: The batch_size to load
    :param root: The root file path of the data loader, where the images are found
    :param split_root: The root file path of the data loader specific to the split, where the indices of the split are kept
    :param split: The split of data to evaluate on (eval, val, train)
    :param top_ks: The top_ks parameters to run evaluation on
    :return:
    """
    print(colored('#' * 70, 'black', 'on_red'))
    print(colored(f'{split} EVALUATION text2Image retrieval start ', 'black', 'on_red'))
    image_encoder.feature_extractor.model.eval()
    image_encoder.sparse_encoder.eval()
    with torch.no_grad():
        query_batch_size = 1000
        eval_start = time.time()
        all_image_embeddings, image_filenames = extract_all_image_embeddings_and_filenames(image_encoder, batch_size, root,
                                                                                           split_root, split)

        print(colored(f'\n Image embeddings collected in {time.time() - eval_start}s', 'black', 'on_yellow'))

        fit_start = time.time()
        print(colored(f'\n Fit transform a TFIDFTransformer with the images in the collection', 'black', 'on_yellow'))
        tfidf_transformer = TfidfTransformer()
        image_tfidf_index = tfidf_transformer.fit_transform(all_image_embeddings)
        print(colored(f'\n TFIDF Transform fitting finished in {time.time() - fit_start}s', 'black', 'on_yellow'))
        analyse_start = time.time()
        print(colored(f'\n Analyse computed index', 'black', 'on_yellow'))
        analyse_index(image_tfidf_index)
        print(colored(f'\n Analysis on computed index done in {time.time() - analyse_start}', 'black', 'on_yellow'))
        all_image_embeddings = None
        image_encoder = None

        accum_evaluation_results = {}  # {'metric': {'sum': 0, 'num': 0}

        querying_start = time.time()
        num_buckets_query = []
        text_data_loader = get_captions_data_loader(root=root, split_root=split_root, split=split,
                                                    batch_size=query_batch_size)

        with Bar(f'Querying Image Sparse Index with text',
                 max=len(text_data_loader)) as querying_bar:
            print(colored(f' Query the TFIDF', 'black', 'on_yellow'))
            for batch_id, (filenames, captions) in enumerate(text_data_loader):
                text_embedding_query = csr_matrix(text_encoder.forward(captions).detach().cpu().numpy())
                text_embedding_query = tfidf_transformer.transform(text_embedding_query)
                cosine_scores = cosine_similarity(text_embedding_query, image_tfidf_index)
                retrieved_image_filenames = []  # it should be a list of lists
                groundtruth_expected_image_filenames = []  # it should be a list of lists
                for i in range(text_embedding_query.shape[0]):
                    num_buckets_query.append(len(text_embedding_query.getrow(i).indices))
                    results = [filename for filename, score in
                               sorted(zip(image_filenames, cosine_scores[i]), key=lambda pair: pair[1], reverse=True)]
                    retrieved_image_filenames.append(results)
                    groundtruth_expected_image_filenames.append([filenames[i]])
                evaluate(['recall', 'reciprocal_rank'], retrieved_image_filenames,
                         groundtruth_expected_image_filenames,
                         top_ks,
                         accum_evaluation_results, print_results=False)

                if batch_id % 100 == 0:
                    print(
                        f' Queried {query_batch_size * (batch_id + 1)} captions out of {len(text_data_loader) * query_batch_size}')
                querying_bar.next()

        print(colored(f' Results collected in {time.time() - querying_start}s', 'black', 'on_yellow'))
        compute_start = time.time()

        t2i_evaluations = {k: v['sum'] / v['num'] for k, v in accum_evaluation_results.items()}
        print(colored(f' Evaluation computed in {time.time() - compute_start}s', 'black', 'on_yellow'))

        print(colored('#' * 70, 'black', 'on_red'))
        print(colored(f'RESULTS of {split} EVALUATION text2Image retrieval: {t2i_evaluations}', 'black', 'on_red'))
        print(colored(f' Average number of buckets for query {np.average(num_buckets_query)}', 'black', 'on_red'))
        print(colored(f'Total run_evaluation time elapsed:\t {time.time() - eval_start}s', 'black', 'on_red'))
        return t2i_evaluations


def validation_loop(image_encoder, text_encoder, dataloader, device, loss_fn, training_batch_id):
    """
    Runs a loop to compute the validation loss

    :param image_encoder: The ImageEncoder to extract the embeddings from to compute the loss
    :param text_encoder: The TextEncoder to extract the embeddings from to compute the loss
    :param dataloader: The dataloader from which to extract the validation pairs of images and captions
    :param device: the device where to run
    :param loss_fn: The loss function to compute based on the output from image_encoder and text_encoder
    :param training_batch_id: The training batch_id at wich the validaiton is performed

    :return: The validation loss
    """
    image_encoder.feature_extractor.model.eval()
    image_encoder.sparse_encoder.eval()
    with torch.no_grad():
        is_tfidf_vectorizer = isinstance(text_encoder.vectorizer, TfidfVectorizer)
        val_loss = []
        with Bar(f'Validation for training batch {training_batch_id} Batch',
                 max=len(dataloader)) as validation_bar:
            for batch_id, (_, image, caption) in enumerate(dataloader):
                image = image.to(device)
                image_embedding = image_encoder.forward(image)
                text_embedding = text_encoder.forward(caption)
                text_embedding = text_embedding.to(device)
                if not is_tfidf_vectorizer:
                    text_embedding = text_embedding / text_embedding
                    text_embedding[text_embedding != text_embedding] = 0
                loss = loss_fn(image_embedding, text_embedding)
                val_loss.append(loss.item())
                validation_bar.next()

        return val_loss


def compute_average_positives_in_vocab(vectorizer_path,
                                       device='cpu'):
    """
    Compute the average amount of words (indices) of the sparse vector inside the captions in train

    :param vectorizer_path: The path to the vectorizer path from where to load the TextEncoder
    :param device: The device where to load the model
    :return: The average of positives
    """
    batch_size = 1
    text_encoder = TextEncoder(vectorizer_path)
    train_data_loader = get_concatenated_captions_image_data_loader(root=DATASET_ROOT_PATH,
                                                                    split_root=DATASET_SPLIT_ROOT_PATH,
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


def train(output_model_path: str,
          vectorizer_path: str,
          positive_weights: float = 1.0,
          num_epochs: int = 100,
          batch_size: int = 8,
          layers: List[int] = [1062],
          learning_rate: float = 0.5,
          ):
    """
    Train the model to have an image encoder that encodes into sparse embeddings matching the text encoder's outputs

    :param output_model_path: The base directory path where output models of each epoch are saved
    :param vectorizer_path: The vectorizer path from where the TextEncoder's vectorizer is loaded
    :param positive_weights: The positive weights to pass to `torch.nn.BCEWithLogitsLoss` to compensate for an unbalanced dataset, so that the model can
        focus on bringing 1 to 1 and not only setting 0 to 0
    :param num_epochs: The number of epochs to run the training for
    :param batch_size: The batch size to load the images from
    :param layers: The size of layers to be added after the dense feature extractor
    :param learning_rate: The learning rate for the optimizer

    :return: Nothing, the outputs are models stored and prints of validation and training loss
    """
    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    device = torch.device(dev)

    pos_weight = torch.ones([layers[-1]]) * positive_weights
    pos_weight = pos_weight.to(device)
    os.makedirs(output_model_path, exist_ok=True)
    image_encoder = ImageEncoder(layer_size=layers)
    text_encoder = TextEncoder(vectorizer_path)
    optimizer = torch.optim.SGD(image_encoder.parameters(), lr=learning_rate)
    optimizer.zero_grad()

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_losses_epochs = []
    val_losses_epochs = []
    test_evals_epochs = []
    val_evals_epochs = []
    train_evals_epochs = []
    extra_relu = torch.nn.ReLU(inplace=True)
    is_tfidf_vectorizer = isinstance(text_encoder.vectorizer, TfidfVectorizer)
    if is_tfidf_vectorizer:
        print(f' Training with TfidfVectorizer as text vectorizer')
    else:
        print(f' Training with CountVectorizer as text vectorizer')

    print(f' Run evaluation on random model with no training:')
    run_evaluations(image_encoder, text_encoder,
                    batch_size, root=DATASET_ROOT_PATH,
                    split_root=DATASET_SPLIT_ROOT_PATH,
                    split='test')

    with Bar('Epochs', max=num_epochs) as epoch_bar:

        for epoch in range(num_epochs):
            train_data_loader = get_concatenated_captions_image_data_loader(root=DATASET_ROOT_PATH,
                                                                            split_root=DATASET_SPLIT_ROOT_PATH,
                                                                            split='train',
                                                                            shuffle=True,
                                                                            batch_size=batch_size)

            val_data_loader = get_concatenated_captions_image_data_loader(root=DATASET_ROOT_PATH,
                                                                          split_root=DATASET_SPLIT_ROOT_PATH,
                                                                          split='val',
                                                                          shuffle=True,
                                                                          batch_size=batch_size)

            train_loss = []
            epoch_start = time.time()
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
                    activations = extra_relu(image_embedding)
                    l1_regularization = torch.mean(torch.sum(activations, dim=1))
                    text_embedding = text_encoder.forward(caption).to(device)
                    if not is_tfidf_vectorizer:
                        text_embedding = text_embedding / text_embedding
                        text_embedding[text_embedding != text_embedding] = 0
                    loss = loss_fn(image_embedding, text_embedding) + l1_regularization * l1_regularization_weight
                    # print(f' loss {loss} vs l1_reg {l1_regularization}:{l1_regularization_weight*l1_regularization}
                    # with  {len(csr_matrix(activations.detach().cpu().numpy()).data)}')
                    train_loss.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    if batch_id % 200 == 0:
                        print(colored(
                            f'\n[{batch_id}] \t training loss:\t {np.mean(np.array(train_loss))} \t time elapsed:\t {time.time() - time_start} s',
                            'green'))
                        train_loss.clear()
                        time_start = time.time()
                    if batch_id % 1000 == 0 and batch_id != 0:
                        torch.save(image_encoder.state_dict(),
                                   output_model_path + '/model-inter-' + str(epoch + 1) + '-' + str(batch_id) + '.pt')
                    if batch_id % 200 == 0 and batch_id != 0:
                        val_loss = validation_loop(image_encoder, text_encoder, val_data_loader, device, loss_fn,
                                                   batch_id)
                        print(colored(
                            f'\n[{batch_id}]\tvalidation loss:\t{np.mean(np.array(val_loss))}\ttime elapsed:\t{time.time() - time_start} s',
                            'yellow'))

                        time_start = time.time()
                    training_bar.next()
                file_path_dump = output_model_path + '/model-inter-' + str(epoch + 1) + '-final.pt'
                print(colored(f'\nEpoch finished in {time.time() - epoch_start} s, saving model to {file_path_dump}',
                              'green'))
                val_loss = validation_loop(image_encoder, text_encoder, val_data_loader, device, loss_fn,
                                           batch_id)
                val_losses_epochs.append(np.mean(np.array(val_loss)))
                train_losses_epochs.append(np.mean(np.array(train_loss)))
                print(colored(
                    f'\n[{epoch}]\tEnd of epoch validation loss:\t{np.mean(np.array(val_loss))}', 'yellow'))
                print(colored(
                    f'\n[{epoch}]\tEnd of epoch training loss:\t{np.mean(np.array(train_loss))}', 'yellow'))

                print(colored(
                    f'\n[{batch_id}]\tBest epoch w.r.t validation loss:\t{val_losses_epochs.index(min(val_losses_epochs))}',
                    'yellow'))
                print(colored(
                    f'\n[{batch_id}]\tBest epoch w.r.t training loss:\t{train_losses_epochs.index(min(train_losses_epochs))}',
                    'yellow'))

                torch.save(image_encoder.state_dict(), file_path_dump)

            with open(f'train_loss-{epoch}', 'wb') as f:
                pickle.dump(train_loss, f)

            if epoch % 1 == 0:
                test_evaluations = run_evaluations(image_encoder, text_encoder,
                                                   batch_size, root=DATASET_ROOT_PATH,
                                                   split_root=DATASET_SPLIT_ROOT_PATH,
                                                   split='test')
                test_evals_epochs.append(test_evaluations)

                val_evaluations = run_evaluations(image_encoder, text_encoder,
                                                  batch_size, root=DATASET_ROOT_PATH,
                                                  split_root=DATASET_SPLIT_ROOT_PATH,
                                                  split='val')
                val_evals_epochs.append(val_evaluations)

                # train_evaluations = run_evaluations(image_encoder, text_encoder,
                #                                     batch_size, root=DATASET_ROOT_PATH,
                #                                     split_root=DATASET_SPLIT_ROOT_PATH,
                #                                     split='train',
                #                                     top_ks=[5, 10, 20])
                # train_evals_epochs.append(train_evaluations)

            for key in test_evaluations.keys():
                test_keys_evals_list = [d[key] for d in test_evals_epochs]
                val_keys_evals_list = [d[key] for d in val_evals_epochs]
                train_keys_evals_list = [d[key] for d in train_evals_epochs]
                if len(test_keys_evals_list) > 0:
                    print(colored(
                        f'\n[{epoch}]\tBest epoch w.r.t evaluation {key} in test:\t{test_keys_evals_list.index(max(test_keys_evals_list))}',
                        'yellow'))
                if len(val_keys_evals_list):
                    print(colored(
                        f'\n[{epoch}]\tBest epoch w.r.t evaluation {key} in validation:\t{val_keys_evals_list.index(max(val_keys_evals_list))}',
                        'yellow'))
                if len(train_keys_evals_list):
                    print(colored(
                        f'\n[{epoch}]\tBest epoch w.r.t evaluation {key} in train:\t{train_keys_evals_list.index(max(train_keys_evals_list))}',
                        'yellow'))

        epoch_bar.next()


def main(*args, **kwargs):
    if len(args) > 1:
        task = args[1]
        if task == 'evaluate':
            split = args[2]
            path = args[3]
            vectorizer_path = args[4] if len(args) > 4 else TEXT_EMBEDDING_VECTORIZER_PATH
            text_encoder = TextEncoder(vectorizer_path)
            last_layer_size = text_encoder.forward(['test']).shape[1]
            layers = [4096, last_layer_size]
            image_encoder = ImageEncoder(layer_size=layers)
            image_encoder.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            image_encoder.eval()
            text_encoder = TextEncoder(vectorizer_path)
            run_evaluations(image_encoder, text_encoder,
                            batch_size=16, root=DATASET_ROOT_PATH,
                            split_root=DATASET_SPLIT_ROOT_PATH,
                            split=split,
                            top_ks=[1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, None])
            import sys

            sys.exit(0)

        if task == 'analyse':
            split = args[2]
            path = args[3]
            vectorizer_path = args[4] if len(args) > 4 else TEXT_EMBEDDING_VECTORIZER_PATH
            text_encoder = TextEncoder(vectorizer_path)
            last_layer_size = text_encoder.forward(['test']).shape[1]
            layers = [4096, last_layer_size]

            image_encoder = ImageEncoder(layer_size=layers)
            image_encoder.load_state_dict(torch.load(path))
            image_encoder.eval()
            analyse(image_encoder, batch_size=16, root=DATASET_ROOT_PATH,
                    split_root=DATASET_SPLIT_ROOT_PATH,
                    split=split)
            import sys
            sys.exit(0)

    vectorizer_path = args[1] if len(args) > 1 else TEXT_EMBEDDING_VECTORIZER_PATH
    text_encoder = TextEncoder(vectorizer_path)
    last_layer_size = text_encoder.forward(['test']).shape[1]
    layers = [4096, last_layer_size]
    # mean_positives, mean_negatives, mean_totals = compute_average_positives_in_vocab(vectorizer_path, 'cpu')
    # positive_weights = mean_negatives / mean_positives
    positive_weights = 200
    # print(
    #     f' mean_positives {mean_positives}, mean_negatives {mean_negatives}, num_totals {mean_negatives} => positive_weights {positive_weights}')
    train(
        output_model_path=IMAGE_EMBEDDING_BASE_PATH,
        vectorizer_path=vectorizer_path,
        layers=layers,
        positive_weights=positive_weights,
        batch_size=16)


if __name__ == '__main__':
    import sys
    main(*sys.argv)
