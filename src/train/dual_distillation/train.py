import torch
import time
import os
import random
import copy
import numpy as np
import math
from typing import Optional, Union

from progress.bar import Bar

from src.model.dual_distillation.image import ImageEncoder
from src.model.dual_distillation.text import TextEncoder, TextTokenizer

from src.model.vilt_model import get_vilt_model
from vilt.transforms.pixelbert import pixelbert_transform

from src.dataset import get_captions_image_data_loader, get_image_data_loader, get_captions_data_loader
from src.dataset.dataset import DEFAULT_TRANSFORM
from src.evaluate import evaluate
import transformers

cur_dir = os.path.dirname(os.path.abspath(__file__))

vilt_transform = pixelbert_transform(384)

softmax_dim_0 = torch.nn.Softmax(dim=0)
softmax_dim_1 = torch.nn.Softmax(dim=1)
log_softmax_dim_0 = torch.nn.LogSoftmax(dim=0)
log_softmax_dim_1 = torch.nn.LogSoftmax(dim=1)

# The base directory where models are stored after every epoch
IMAGE_EMBEDDING_BASE_PATH = os.getenv('IMAGE_EMBEDDING_BASE_PATH', '/hdd/master/tfm/output-image-encoders')
# The word2vec model base
TEXT_WORD2_VEC_MODEL_PATH = os.getenv('TEXT_WORD2_VEC_MODEL_PATH', 'filtered_f30k_word2vec.model')

VILT_BASE_MODEL_LOAD_PATH = os.getenv('VILT_BASE_MODEL_LOAD_PATH', 'vilt_irtr_f30k.ckpt')

# The root path where the flickr30k dataset is found
DATASET_ROOT_PATH = os.getenv('DATASET_ROOT_PATH', '/hdd/master/tfm/flickr30k_images')
# The root path where the flickr30k entities per split is kept
DATASET_SPLIT_ROOT_PATH = os.getenv('DATASET_SPLIT_ROOT_PATH', '/hdd/master/tfm/flickr30k_images/flickr30k_entities')

# The root path where the flickr30k dataset is found
FLICKR_DATASET_ROOT_PATH = os.getenv('FLICKR_DATASET_ROOT_PATH', '/hdd/master/tfm/flickr30k_images')
# The root path where the flickr30k entities per split is kept
FLICKR_DATASET_SPLIT_ROOT_PATH = os.getenv('FLICKR_DATASET_SPLIT_ROOT_PATH', '/hdd/master/tfm/flickr30k_images/flickr30k_entities')

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


def run_evaluations(image_encoder, text_encoder, text_tokenizer, vilt_model, batch_size, root, split_root, split,
                    top_k_first_phase=None, top_ks=None, cache_query_image_slow_scores=None,
                    dual_encoder_transform=DEFAULT_TRANSFORM, dset='flickr'):
    """
    Runs evaluations of an ImageEncoder resulting from some training

    :param dual_encoder_transform:
    :param cache_query_image_slow_scores:
    :param top_k_first_phase:
    :param image_encoder: The ImageEncoder to extract the embeddings from where to compute evaluation
    :param text_encoder: The TextEncoder to extract the embeddings from where to compute evaluation
    :param vilt_model: The ViLT model to compute the last step of the reranking with the top-k candidates
    :param batch_size: The batch_size to load
    :param root: The root file path of the data loader, where the images are found
    :param split_root: The root file path of the data loader specific to the split, where the indices of the split are kept
    :param split: The split of data to evaluate on (eval, val, train)
    :param top_ks: The top_ks parameters to run evaluation on
    :return:
    """
    if top_ks is None:
        top_ks = [1, 5, 10, 50, 100, 500]
    if top_k_first_phase is None:
        top_k_first_phase = [1, 5, 10, 50]
    if torch.cuda.is_available():
        dev = 'cuda'
    else:
        dev = 'cpu'
    device = torch.device(dev)
    try:
        if isinstance(image_encoder, torch.nn.DataParallel):
            image_encoder.train(False)
            image_encoder.module.common_space_embedding.train(False)
        else:
            image_encoder.train(False)
            image_encoder.common_space_embedding.train(False)
        if isinstance(text_encoder, torch.nn.DataParallel):
            text_encoder.train(False)
            text_encoder.module.word_embd.train(False)
            text_encoder.module.fc1.train(False)
            text_encoder.module.fc2.train(False)
        else:
            text_encoder.train(False)
            text_encoder.word_embd.train(False)
            text_encoder.fc1.train(False)
            text_encoder.fc2.train(False)
    except:
        pass
    with torch.no_grad():
        image_data_loader = get_image_data_loader(root=root, split_root=split_root, split=split, batch_size=batch_size,
                                                  collate_fn=collate_images, shuffle=False,
                                                  force_transform_to_none=True, dset=dset if split != 'test' else 'flickr')

        all_image_embeddings = []
        image_filenames = []
        pixel_bert_transformed_images = []
        all_images_ids = []
        with Bar(f'Computing the embedding of all the images', check_tty=False,
                 max=len(image_data_loader)) as image_embedding_bar:
            for batch_id, (image_ids, filenames, images) in enumerate(image_data_loader):
                image_tensors = []
                for i in images:
                    pixel_bert_transformed_images.append(vilt_transform(i).to(device))
                    image_tensors.append(dual_encoder_transform(i))

                images_embeddings = image_encoder(torch.stack(image_tensors).to(device))
                all_image_embeddings.append(images_embeddings)
                image_filenames.extend(filenames)
                all_images_ids.extend(image_ids)
                image_embedding_bar.next()

        all_image_embeddings = torch.cat(all_image_embeddings)

        text_data_loader = get_captions_data_loader(root=root, split_root=split_root, split=split,
                                                    batch_size=batch_size, shuffle=False, collate_fn=collate_captions, dset=dset if split != 'test' else 'flickr')

        dot_products = []
        groundtruth_expected_image_filenames = []
        queries_ids = []
        queries = []
        with Bar(f'Computing dot products for every query', check_tty=False,
                 max=len(text_data_loader)) as dot_prod_bar:
            for batch_id, (caption_ids, filenames, captions) in enumerate(text_data_loader):
                tokenized_captions = text_tokenizer(captions).to(device)
                texts_embeddings = text_encoder(tokenized_captions).to(device)
                d_product = texts_embeddings.matmul(all_image_embeddings.T)
                dot_products.append(d_product)
                for filename in filenames:
                    groundtruth_expected_image_filenames.append([filename])
                queries.extend(captions)
                queries_ids.extend(caption_ids)
                # for q, f, d in zip(captions, filenames, d_product):
                #     list_scores = d.cpu().detach().numpy().tolist()
                #     print(f' Expected: "{q}": {f} with {sorted(list_scores, reverse=True)}')

                dot_prod_bar.next()
        dot_products = torch.cat(dot_products)
        assert dot_products.shape[0] == len(groundtruth_expected_image_filenames)  # 1 matching filename for caption
        # query
        assert dot_products.shape[1] == len(image_filenames)
        assert len(groundtruth_expected_image_filenames) == len(queries)

        for first_phase_top_k in top_k_first_phase:
            retrieved_image_filenames = []
            with Bar(f'Second phase query for first_phase {first_phase_top_k}', check_tty=False,
                     max=len(queries)) as query_bar:
                for i, (query, query_id, dot_prod) in enumerate(zip(queries, queries_ids, dot_products)):
                    list_scores = dot_prod.cpu().detach().numpy().tolist()
                    assert len(list_scores) == len(pixel_bert_transformed_images)
                    sorted_images_indices = [i for _, i in
                                             sorted(zip(list_scores, range(len(pixel_bert_transformed_images))),
                                                    reverse=True)]
                    candidate_images_indices = sorted_images_indices[:first_phase_top_k]
                    non_candidate_images_indices = sorted_images_indices[first_phase_top_k:]
                    candidate_images = [pixel_bert_transformed_images[i] for i in candidate_images_indices]
                    if cache_query_image_slow_scores is None:
                        scores = vilt_model.rank_query_vs_images(query, candidate_images)
                    else:
                        scores = cache_query_image_slow_scores.get_scores(query_id, candidate_images_indices)
                    best_indices = [i for _, i in sorted(zip(scores, range(len(scores))), reverse=True)]
                    resulting_filenames = [image_filenames[candidate_images_indices[i]] for i in best_indices]
                    resulting_filenames.extend([image_filenames[i] for i in non_candidate_images_indices])
                    retrieved_image_filenames.append(resulting_filenames)
                    # print(f' Result: "{query}": {resulting_filenames}')
                    query_bar.next()

            t2i_evaluations = evaluate(['recall', 'reciprocal_rank'], retrieved_image_filenames,
                                       groundtruth_expected_image_filenames,
                                       top_ks,
                                       None, print_results=True)

        return t2i_evaluations


def validation_loop(image_encoder, text_encoder, text_tokenizer, vilt_model, dataloader, negative_batch_size, temperature, alpha, beta,
                    reduction_in_loss, cache_query_image_slow_scores, dual_encoder_transform=DEFAULT_TRANSFORM):
    """
    Runs a loop to compute the validation loss

    :param reduction_in_loss:
    :param alpha:
    :param temperature:
    :param negative_batch_size:
    :param image_encoder: The ImageEncoder to extract the embeddings from to compute the loss
    :param text_encoder: The TextEncoder to extract the embeddings from to compute the loss
    :param vilt_model: The ViLT model from which the distillation is done
    :param dataloader: The dataloader from which to extract the validation pairs of images and captions

    :return: The validation loss
    """
    if torch.cuda.is_available():
        dev = 'cuda'
    else:
        dev = 'cpu'

    device = torch.device(dev)
    with torch.no_grad():
        if isinstance(image_encoder, torch.nn.DataParallel):
            image_encoder.train(False)
            image_encoder.module.common_space_embedding.train(False)
        else:
            image_encoder.train(False)
            image_encoder.common_space_embedding.train(False)
        if isinstance(text_encoder, torch.nn.DataParallel):
            text_encoder.train(False)
            text_encoder.module.word_embd.train(False)
            text_encoder.module.fc1.train(False)
            text_encoder.module.fc2.train(False)
        else:
            text_encoder.train(False)
            text_encoder.word_embd.train(False)
            text_encoder.fc1.train(False)
            text_encoder.fc2.train(False)
        val_loss = []
        for batch_id, (caption_indices, images_indices, matching_filenames, images, captions) in enumerate(dataloader):

            image_tensors = []
            for i in images:
                image_tensors.append(dual_encoder_transform(i))

            original_images = images
            print(f' len of image_tensors {image_tensors}')
            deviced_tensors = torch.stack(image_tensors).to(device)
            try:
                print(f' deviced tensors device {deviced_tensors.get_device()}')
            except:
                pass

            images_embeddings = image_encoder(deviced_tensors).to(device)
            tokenized_captions = text_tokenizer(captions).to(device)
            texts_embeddings = text_encoder(tokenized_captions).to(device)
            loss = compute_loss(images, captions, images_indices, caption_indices, matching_filenames, original_images,
                                vilt_model, images_embeddings,
                                texts_embeddings,
                                negative_batch_size, temperature, alpha, beta, reduction_in_loss,
                                cache_query_image_slow_scores)
            val_loss.append(loss.item())

    return val_loss


def collate(batch, *args, **kwargs):
    caption_indices = []
    images_indices = []
    filenames = []
    pil_images = []
    captions = []
    for ci, ii, f, i, c in batch:
        caption_indices.append(ci)
        images_indices.append(ii)
        filenames.append(f)
        pil_images.append(i)
        captions.append(c)
    return caption_indices, images_indices, filenames, pil_images, captions


def collate_images(batch, *args, **kwargs):
    filenames = []
    pil_images = []
    indices = []
    for ii, f, i in batch:
        indices.append(ii)
        filenames.append(f)
        pil_images.append(i)
    return indices, filenames, pil_images


def collate_captions(batch, *args, **kwargs):
    filenames = []
    captions = []
    indices = []
    for ii, f, c in batch:
        indices.append(ii)
        filenames.append(f)
        captions.append(c)
    return indices, filenames, captions


def compute_loss(images, captions, images_indices, captions_indices, matching_filenames, original_images, vilt_model,
                 images_embeddings, texts_embeddings,
                 negative_batch_size, temperature, alpha, beta, reduction_in_loss, cache_query_image_slow_scores):
    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction=reduction_in_loss)
    if torch.cuda.is_available():
        dev = 'cuda'
    else:
        dev = 'cpu'
    device = torch.device(dev)
    sample_set = list(range(len(images)))
    all_dot_products = texts_embeddings.matmul(images_embeddings.T)

    target = torch.LongTensor(list(range(len(images)))).to(device)  # 0, 1, 2, 3, 4.. 127
    dual_encoder_loss = cross_entropy_loss(all_dot_products, target)

    if beta > 0:
        transformed_images = []
        if cache_query_image_slow_scores is not None:
            transformed_images = [vilt_transform(original_image).to(device) for original_image in original_images]

        list_of_student_scores_with_temperature = []
        list_of_teacher_distributions = []

        # setting device on GPU if available, else CPU
        for i, (image, caption) in enumerate(zip(images, captions)):
            csample_set = copy.copy(sample_set)
            csample_set.remove(i)
            negative_images_indices = random.sample(csample_set, min(negative_batch_size, len(images)) - 1)

            student_scores = all_dot_products[i, [i] + negative_images_indices]

            list_of_student_scores_with_temperature.append(student_scores / temperature)

            with torch.no_grad():
                if cache_query_image_slow_scores is not None:
                    transformed_negative_images = [transformed_images[j] for j in negative_images_indices]
                    transformed_positive_image = transformed_images[i]
                    teacher_scores = vilt_model.score_query_vs_images(caption,
                                                                      [
                                                                          transformed_positive_image] + transformed_negative_images)

                else:
                    absolute_images_indices_in_dataset = [images_indices[i]] + [images_indices[j] for j in
                                                                                negative_images_indices]
                    teacher_scores = cache_query_image_slow_scores.get_scores(captions_indices[i],
                                                                              absolute_images_indices_in_dataset)

                teacher_distrib_p_bi = softmax_dim_0(
                    teacher_scores / temperature)

                list_of_teacher_distributions.append(teacher_distrib_p_bi)

        student_scores_with_temperature = torch.stack(list_of_student_scores_with_temperature)
        teacher_distributions_from_temperature = torch.stack(list_of_teacher_distributions)

        distillation_loss = cross_entropy_loss(student_scores_with_temperature, teacher_distributions_from_temperature)
    else:
        distillation_loss = 0

    if os.getenv('PRINT_DOT_PRODUCTS') == 'True':
        with torch.no_grad():
            print(f' num captions: {len(captions)}, num images: {len(images)}')
            print(f' \ndot_products diagonal {torch.diagonal(all_dot_products, 0)}')
            ordinals = []
            sorts = torch.sort(all_dot_products, descending=True, dim=1)
            for i, s in enumerate(sorts.indices):
                ordinals.append(s.cpu().detach().numpy().tolist().index(i))
            print(f' ordinals {ordinals}')
    loss = beta * distillation_loss + alpha * dual_encoder_loss
    return loss


def train(output_model_path: str,
          vilt_model_path: str,
          word2vec_model_path: str,
          image_encoder_backbone_model: str,
          num_epochs: int = 10000,
          batch_size: int = 8,
          negative_batch_size: int = 4,
          learning_rate: float = 0.001,
          num_warmup_steps: int = 5000,
          num_training_steps: int = 150000,
          temperature=10,
          dataloader_num_worker=1,
          alpha=0.1,
          beta=1,
          reduction_in_loss='mean',
          cache_scores={'train': None, 'val': None, 'test': None},
          dual_encoder_transform=DEFAULT_TRANSFORM,
          dset='flickr',
          clip_gradient=30.0,
          weight_decay=0.0001
          ):
    """
    Train the model to have an image encoder that encodes into sparse embeddings matching the text encoder's outputs

    :param dset:
    :param dual_encoder_transform:
    :param num_training_steps:
    :param num_warmup_steps:
    :param cache_scores:
    :param beta:
    :param alpha:
    :param reduction_in_loss:
    :param dataloader_num_worker: num worker to use in dataloader
    :param temperature:
    :param negative_batch_size:
    :param image_encoder_backbone_model: The dense backbone model on which the image embedding is based
    :param output_model_path: The base directory path where output models of each epoch are saved
    :param vilt_model_path: The vilt model path from where the ViltModel is loaded
    :param word2vec_model_path: The word2vec model from which to load the text encoder word2vec part
    :param num_epochs: The number of epochs to run the training for
    :param batch_size: The batch size to load the images from
    :param learning_rate: The learning rate for the optimizer

    :return: Nothing, the outputs are models stored and prints of validation and training loss
    """
    if torch.cuda.is_available():
        dev = 'cuda'
    else:
        dev = 'cpu'
    device = torch.device(dev)

    os.makedirs(output_model_path, exist_ok=True)
    image_encoder = ImageEncoder(backbone_model=image_encoder_backbone_model)
    text_tokenizer = TextTokenizer(word2vec_model_path=word2vec_model_path)
    text_encoder = TextEncoder(gensim_model_vector_weights=torch.FloatTensor(text_tokenizer.gensim_model.vectors))

    vilt_model = None
    if beta > 0:
        vilt_model = get_vilt_model(load_path=vilt_model_path)

    if torch.cuda.device_count() > 1:
        print("Let's use ", torch.cuda.device_count(), " GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        image_encoder = torch.nn.DataParallel(image_encoder)
        text_encoder = torch.nn.DataParallel(text_encoder)
        if beta > 0:
            vilt_model = torch.nn.DataParallel(vilt_model)

    image_encoder.to(device)
    text_encoder.to(device)

    if beta > 0:
        vilt_model.to(device)

    optimizer = torch.optim.Adam([{'params': image_encoder.parameters()}, {'params': text_encoder.parameters()}],
                                 lr=learning_rate, weight_decay=weight_decay)

    lr_scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=optimizer,
                                                                                   num_warmup_steps=num_warmup_steps,
                                                                                   num_training_steps=num_training_steps)
    optimizer.zero_grad()

    train_losses_epochs = []
    val_losses_epochs = []
    test_evals_epochs = []
    val_evals_epochs = []
    train_evals_epochs = []

    run_evaluations(image_encoder, text_encoder, text_tokenizer, vilt_model,
                    batch_size, root=FLICKR_DATASET_ROOT_PATH,
                    split_root=FLICKR_DATASET_SPLIT_ROOT_PATH,
                    split='test',
                    cache_query_image_slow_scores=cache_scores['test'],
                    dset=dset)

    with Bar('Epochs', max=num_epochs, check_tty=False) as epoch_bar:

        for epoch in range(num_epochs):
            train_data_loader = get_captions_image_data_loader(root=DATASET_ROOT_PATH,
                                                               split_root=DATASET_SPLIT_ROOT_PATH,
                                                               split='train',
                                                               shuffle=True,
                                                               num_workers=dataloader_num_worker,
                                                               batch_size=batch_size,
                                                               collate_fn=collate,
                                                               batch_sampling=True,
                                                               dset=dset)

            val_data_loader = get_captions_image_data_loader(root=DATASET_ROOT_PATH,
                                                             split_root=DATASET_SPLIT_ROOT_PATH,
                                                             split='val',
                                                             shuffle=True,
                                                             num_workers=dataloader_num_worker,
                                                             batch_size=batch_size,
                                                             collate_fn=collate,
                                                             batch_sampling=True,
                                                             dset=dset)

            train_loss = []
            epoch_start = time.time()
            time_start = time.time()

            with Bar(f'Batch in epoch {epoch}', max=math.ceil(len(train_data_loader.dataset) / batch_size),
                     check_tty=False) as training_bar:
                sum_images = 0
                sum_captions = 0
                for batch_id, (caption_indices, images_indices, matching_filenames, images, captions) in enumerate(
                        train_data_loader):
                    image_tensors = []
                    for i in images:
                        image_tensors.append(dual_encoder_transform(i))

                    if isinstance(image_encoder, torch.nn.DataParallel):
                        image_encoder.train()
                        image_encoder.module.common_space_embedding.train()
                    else:
                        image_encoder.train()
                        image_encoder.common_space_embedding.train()
                    if isinstance(text_encoder, torch.nn.DataParallel):
                        text_encoder.train()
                        text_encoder.module.word_embd.train()
                        text_encoder.module.fc1.train()
                        text_encoder.module.fc2.train()
                    else:
                        text_encoder.train()
                        text_encoder.word_embd.train()
                        text_encoder.fc1.train()
                        text_encoder.fc2.train()

                    optimizer.zero_grad()
                    original_images = images
                    image_tensors = torch.stack(image_tensors).to(device)
                    images_embeddings = image_encoder(image_tensors).to(device)
                    tokenized_captions = text_tokenizer(captions).to(device)
                    texts_embeddings = text_encoder(tokenized_captions).to(device)

                    if batch_id % 200 == 0:
                        os.environ['PRINT_DOT_PRODUCTS'] = 'False'
                    else:
                        os.environ['PRINT_DOT_PRODUCTS'] = 'False'
                    sum_images += len(images)
                    sum_captions += len(captions)
                    loss = compute_loss(images, captions, images_indices, caption_indices, matching_filenames,
                                        original_images, vilt_model,
                                        images_embeddings,
                                        texts_embeddings,
                                        negative_batch_size, temperature, alpha, beta, reduction_in_loss,
                                        cache_scores['val'])

                    loss.backward()
                    train_loss.append(loss.item())
                    torch.nn.utils.clip_grad_norm_(image_encoder.parameters(), clip_gradient)
                    torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), clip_gradient)
                    optimizer.step()
                    lr_scheduler.step()

                    if batch_id % 50 == 0:
                        print(colored(
                            f'\n[{batch_id}] \t training loss:\t {np.mean(np.array(train_loss))} \t time elapsed:\t {time.time() - time_start} s',
                            'green'))
                        train_loss.clear()
                        time_start = time.time()
                    if batch_id % (math.ceil(len(train_data_loader.dataset) / batch_size) / 4) == 0 and batch_id != 0:
                        os.environ['PRINT_DOT_PRODUCTS'] = 'False'
                        val_loss = validation_loop(image_encoder, text_encoder, text_tokenizer, vilt_model, val_data_loader,
                                                   negative_batch_size, temperature, alpha, beta, reduction_in_loss,
                                                   cache_scores['val'])
                        print(colored(
                            f'\n[{batch_id}]\tvalidation loss:\t{np.mean(np.array(val_loss))}\ttime elapsed:\t{time.time() - time_start} s',
                            'yellow'))

                        time_start = time.time()
                    training_bar.next()

                print(colored(
                    f'\nNum processed captions: {sum_captions}. Num processed images: {sum_images}',
                    'green'))
                image_file_path_dump = output_model_path + '/model-inter-' + str(epoch + 1) + '-final-image.pt'
                text_file_path_dump = output_model_path + '/model-inter-' + str(epoch + 1) + '-final-text.pt'
                print(colored(
                    f'\nEpoch finished in {time.time() - epoch_start} s, saving model to {image_file_path_dump}',
                    'green'))
                os.environ['PRINT_DOT_PRODUCTS'] = 'False'
                val_loss = validation_loop(image_encoder, text_encoder, text_tokenizer, vilt_model, val_data_loader,
                                           negative_batch_size, temperature, alpha, beta, reduction_in_loss,
                                           cache_scores['val'])
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

                if epoch % 1 == 0 and epoch != 0:
                    torch.save(image_encoder.state_dict(), image_file_path_dump)
                    torch.save(text_encoder.state_dict(), text_file_path_dump)

            test_evaluations = {}
            val_evaluations = {}
            test_evaluations = run_evaluations(image_encoder,
                                               text_encoder,
                                               text_tokenizer,
                                               vilt_model,
                                               batch_size,
                                               root=FLICKR_DATASET_ROOT_PATH,
                                               split_root=FLICKR_DATASET_SPLIT_ROOT_PATH,
                                               split='test',
                                               cache_query_image_slow_scores=cache_scores['test'],
                                               dset=dset)
            test_evals_epochs.append(test_evaluations)

            # val_evaluations = run_evaluations(image_encoder, text_encoder, text_tokenizer, vilt_model,
            #                                   batch_size, root=DATASET_ROOT_PATH,
            #                                   split_root=DATASET_SPLIT_ROOT_PATH,
            #                                   split='val',
            #                                   cache_query_image_slow_scores=cache_scores['val'])
            # val_evals_epochs.append(val_evaluations)

            # train_evaluations = run_evaluations(image_encoder, text_encoder, vilt_model,
            #                                     batch_size, root=DATASET_ROOT_PATH,
            #                                     split_root=DATASET_SPLIT_ROOT_PATH,
            #                                     split='train',
            #                                     top_ks=[5, 10, 20])
            # train_evals_epochs.append(train_evaluations)

            for key in test_evaluations.keys():
                test_keys_evals_list = [d[key] for d in test_evals_epochs]
                #val_keys_evals_list = [d[key] for d in val_evals_epochs]
                #train_keys_evals_list = [d[key] for d in train_evals_epochs]
                if len(test_keys_evals_list) > 0:
                    print(colored(
                        f'\n[{epoch}]\tBest epoch w.r.t evaluation {key} in test:\t{test_keys_evals_list.index(max(test_keys_evals_list))}',
                        'yellow'))
                # if len(val_keys_evals_list):
                #     print(colored(
                #         f'\n[{epoch}]\tBest epoch w.r.t evaluation {key} in validation:\t{val_keys_evals_list.index(max(val_keys_evals_list))}',
                #         'yellow'))
                # if len(train_keys_evals_list):
                #     print(colored(
                #         f'\n[{epoch}]\tBest epoch w.r.t evaluation {key} in train:\t{train_keys_evals_list.index(max(train_keys_evals_list))}',
                #         'yellow'))
        epoch_bar.next()


def main_train(*args, **kwargs):
    from src.model.cached_scores import CachedScores
    import os
    os.path.dirname(os.path.abspath(__file__))

    val_cache_scores = CachedScores(os.path.join(cur_dir, '../../model/slow_scores/val.th'))
    test_cache_scores = CachedScores(os.path.join(cur_dir, '../../model/slow_scores/test.th'))
    train(
        output_model_path=IMAGE_EMBEDDING_BASE_PATH,
        word2vec_model_path=TEXT_WORD2_VEC_MODEL_PATH,
        image_encoder_backbone_model='resnet50',
        vilt_model_path=VILT_BASE_MODEL_LOAD_PATH,
        batch_size=8,
        negative_batch_size=4,
        dataloader_num_worker=1,
        learning_rate=0.001,
        alpha=1,
        temperature=10,
        beta=1,
        reduction_in_loss='mean',
        cache_scores={'train': None, 'val': val_cache_scores, 'test': test_cache_scores},
        dset='coco',
        clip_gradient=30.0,
    )


def main_evaluate(*args, **kwargs):
    from src.model.cached_scores import CachedScores
    import os
    os.path.dirname(os.path.abspath(__file__))

    val_cache_scores = CachedScores(os.path.join(cur_dir, '../../model/slow_scores/val.th'))
    test_cache_scores = CachedScores(os.path.join(cur_dir, '../../model/slow_scores/test.th'))
    image_encoder = ImageEncoder(backbone_model='resnet50')
    image_encoder.load_state_dict(
        torch.load(os.path.join(cur_dir, '../../model/checkpoints/model-inter-401-final-image.pt'),
                   map_location=torch.device('cpu')))
    image_encoder.eval()
    text_tokenizer = TextTokenizer(word2vec_model_path=TEXT_WORD2_VEC_MODEL_PATH)
    text_encoder = TextEncoder(gensim_model_vector_weights=torch.FloatTensor(text_tokenizer.gensim_model.vectors))
    text_encoder.load_state_dict(
        torch.load(os.path.join(cur_dir, '../../model/checkpoints/model-inter-401-final-text.pt'),
                   map_location=torch.device('cpu')))
    text_encoder.eval()

    vilt_model = get_vilt_model(load_path=VILT_BASE_MODEL_LOAD_PATH)
    run_evaluations(image_encoder,
                    text_encoder,
                    text_tokenizer,
                    vilt_model,
                    8,
                    root=DATASET_ROOT_PATH,
                    split_root=DATASET_SPLIT_ROOT_PATH,
                    split='test',
                    top_ks=[1],
                    top_k_first_phase=[1],
                    cache_query_image_slow_scores=test_cache_scores)


def main_evaluate_clip(*args, **kwargs):
    from src.model.cached_scores import CachedScores
    import os
    from src.model.dual_distillation.clip_image import CLIPImageEncoder
    from src.model.dual_distillation.clip_text import CLIPTextEncoder
    os.path.dirname(os.path.abspath(__file__))

    test_cache_scores = CachedScores(os.path.join(cur_dir, '../../model/slow_scores/test.th'))
    image_encoder = CLIPImageEncoder()
    text_encoder = CLIPTextEncoder()
    run_evaluations(image_encoder, text_encoder, None,
                    8, root=DATASET_ROOT_PATH,
                    split_root=DATASET_SPLIT_ROOT_PATH,
                    split='test',
                    top_ks=[1, 5, 10, 50],
                    top_k_first_phase=[1, 5, 10, 50],
                    cache_query_image_slow_scores=test_cache_scores)


if __name__ == '__main__':
    import sys

    main_train(*sys.argv)
