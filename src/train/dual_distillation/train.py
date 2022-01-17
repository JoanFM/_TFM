import torch
import time
import os
import random
import copy
import numpy as np
import pickle
from typing import Optional, Union

from progress.bar import Bar

from src.model.dual_distillation.image import ImageEncoder
from src.model.dual_distillation.text import TextEncoder
from src.model.vilt_model import get_vilt_model
from vilt.transforms.pixelbert import pixelbert_transform

from src.dataset import get_captions_image_data_loader, get_image_data_loader, get_captions_data_loader
from src.dataset.dataset import DEFAULT_TRANSFORM
from src.evaluate import evaluate

cur_dir = os.path.dirname(os.path.abspath(__file__))

dual_encoder_transform = DEFAULT_TRANSFORM
vilt_transform = pixelbert_transform(384)

softmax_dim_0 = torch.nn.Softmax(dim=0)
softmax_dim_1 = torch.nn.Softmax(dim=1)
log_softmax_dim_0 = torch.nn.LogSoftmax(dim=0)
log_softmax_dim_1 = torch.nn.LogSoftmax(dim=1)
cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='mean')

# The base directory where models are stored after every epoch
IMAGE_EMBEDDING_BASE_PATH = os.getenv('IMAGE_EMBEDDING_BASE_PATH', '/hdd/master/tfm/output-image-encoders')
# The word2vec model base
TEXT_WORD2_VEC_MODEL_PATH = os.getenv('TEXT_WORD2_VEC_MODEL_PATH', 'filtered_f30k_word2vec.model')

VILT_BASE_MODEL_LOAD_PATH = os.getenv('VILT_BASE_MODEL_LOAD_PATH', 'vilt_irtr_f30k.ckpt')

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


def run_evaluations(image_encoder, text_encoder, vilt_model, batch_size, root, split_root, split,
                    top_k_first_phase=None, top_ks=None):
    """
    Runs evaluations of an ImageEncoder resulting from some training

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
        top_ks = [5, 10, 100, 500]
    if top_k_first_phase is None:
        top_k_first_phase = [10]
    if torch.cuda.is_available():
        dev = 'cuda:0'
    else:
        dev = 'cpu'
    device = torch.device(dev)
    image_encoder.train(False)
    image_encoder.feature_extractor.train(False)
    image_encoder.common_space_embedding.train(False)
    text_encoder.train(False)
    text_encoder.word_embd.train(False)
    text_encoder.fc1.train(False)
    text_encoder.fc2.train(False)
    with torch.no_grad():
        image_data_loader = get_image_data_loader(root=root, split_root=split_root, split=split, batch_size=batch_size,
                                                  collate_fn=collate_images, force_transform_to_none=True)

        all_image_embeddings = []
        image_filenames = []
        pixel_bert_transformed_images = []
        with Bar(f'Computing the embedding of all the images', check_tty=False,
                 max=len(image_data_loader)) as image_embedding_bar:
            for batch_id, (filenames, images) in enumerate(image_data_loader):
                image_tensors = []
                for i in images:
                    pixel_bert_transformed_images.append(vilt_transform(i).to(device))
                    image_tensors.append(dual_encoder_transform(i))

                images_embeddings = image_encoder(torch.stack(image_tensors).to(device)).to(device)
                all_image_embeddings.append(images_embeddings)
                image_filenames.extend(filenames)
                image_embedding_bar.next()

        all_image_embeddings = torch.cat(all_image_embeddings)

        text_data_loader = get_captions_data_loader(root=root, split_root=split_root, split=split,
                                                    batch_size=batch_size)

        dot_products = []
        groundtruth_expected_image_filenames = []
        queries = []
        with Bar(f'Computing dot products for every query', check_tty=False,
                 max=len(text_data_loader)) as dot_prod_bar:
            for batch_id, (filenames, captions) in enumerate(text_data_loader):
                texts_embeddings = text_encoder(captions).to(device)
                d_product = texts_embeddings.matmul(all_image_embeddings.T)
                dot_products.append(d_product)
                groundtruth_expected_image_filenames.extend(filenames)
                queries.extend(captions)
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
                for i, (query, dot_prod) in enumerate(zip(queries, dot_products)):
                    now = time.time()
                    list_scores = dot_prod.cpu().detach().numpy().tolist()
                    assert len(list_scores) == len(pixel_bert_transformed_images)
                    sorted_images_indices = [i for _, i in
                                             sorted(zip(list_scores, range(len(pixel_bert_transformed_images))),
                                                    reverse=True)]
                    candidate_images_indices = sorted_images_indices[:first_phase_top_k]
                    non_candidate_images_indices = sorted_images_indices[first_phase_top_k:]
                    candidate_images = [pixel_bert_transformed_images[i] for i in candidate_images_indices]
                    pre_score = time.time()
                    scores = vilt_model.rank_query_vs_images(query, candidate_images)
                    # if i % 50 == 0:
                    #     print(f' Computing the score for {len(candidate_images)} took {time.time() - pre_score}s')
                    best_indices = [i for _, i in sorted(zip(scores, range(len(scores))), reverse=True)]
                    resulting_filenames = [image_filenames[candidate_images_indices[i]] for i in best_indices]
                    # if i % 50 == 0:
                    #     print(
                    #         f' groundtruth {groundtruth_expected_image_filenames[i]} in resulting {resulting_filenames}')
                    resulting_filenames.extend([image_filenames[i] for i in non_candidate_images_indices])
                    retrieved_image_filenames.append(resulting_filenames)
                    query_bar.next()
                    # if i % 50 == 0:
                    #     print(f' Computing the re-ranking for a single query took {time.time() - now}s')

            t2i_evaluations = evaluate(['recall', 'reciprocal_rank'], retrieved_image_filenames,
                                       groundtruth_expected_image_filenames,
                                       top_ks,
                                       None, print_results=True)

        return t2i_evaluations


def validation_loop(image_encoder, text_encoder, vilt_model, dataloader, negative_batch_size, temperature, alpha):
    """
    Runs a loop to compute the validation loss

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
        dev = 'cuda:0'
    else:
        dev = 'cpu'

    device = torch.device(dev)
    image_encoder.train(False)
    image_encoder.feature_extractor.train(False)
    image_encoder.common_space_embedding.train(False)
    text_encoder.train(False)
    text_encoder.word_embd.train(False)
    text_encoder.fc1.train(False)
    text_encoder.fc2.train(False)
    image_encoder.to(device)
    text_encoder.to(device)
    with torch.no_grad():
        val_loss = []
        for batch_id, (matching_filenames, images, captions) in enumerate(dataloader):

            image_tensors = []
            for i in images:
                image_tensors.append(dual_encoder_transform(i))

            original_images = images
            images_embeddings = image_encoder(torch.stack(image_tensors).to(device)).to(device)
            texts_embeddings = text_encoder(captions).to(device)
            loss = compute_loss(images, captions, original_images, vilt_model, images_embeddings,
                                texts_embeddings,
                                negative_batch_size, temperature, alpha)
            val_loss.append(loss.item())

    return val_loss


def collate(batch, *args, **kwargs):
    filenames = []
    pil_images = []
    captions = []
    for f, i, c in batch:
        filenames.append(f)
        pil_images.append(i)
        captions.append(c)
    return filenames, pil_images, captions


def collate_images(batch, *args, **kwargs):
    filenames = []
    pil_images = []
    for f, i in batch:
        filenames.append(f)
        pil_images.append(i)
    return filenames, pil_images


def compute_loss(images, captions, original_images, vilt_model, images_embeddings, texts_embeddings,
                 negative_batch_size, temperature, alpha):
    if torch.cuda.is_available():
        dev = 'cuda:0'
    else:
        dev = 'cpu'
    device = torch.device(dev)
    sample_set = list(range(len(images)))
    all_dot_products = texts_embeddings.matmul(images_embeddings.T)
    dual_encoder_loss = torch.mean(torch.diagonal(-log_softmax_dim_1(all_dot_products), 0))
    transformed_images = [vilt_transform(original_image).to(device) for original_image in original_images]
    list_of_student_scores_with_temperature = []
    list_of_teacher_distributions = []
    for i, (image, caption) in enumerate(zip(images, captions)):
        csample_set = copy.copy(sample_set)
        csample_set.remove(i)
        negative_images_indices = random.sample(csample_set, negative_batch_size - 1)
        transformed_negative_images = [transformed_images[j] for j in negative_images_indices]
        transformed_positive_image = vilt_transform(image)
        student_scores = all_dot_products[i, [i] + negative_images_indices]
        list_of_student_scores_with_temperature.append(student_scores / temperature)

        with torch.no_grad():
            teacher_scores = vilt_model.score_query_vs_images(caption,
                                                              [
                                                                  transformed_positive_image] + transformed_negative_images)
            teacher_distrib_p_bi = softmax_dim_0(
                teacher_scores / temperature)
            list_of_teacher_distributions.append(teacher_distrib_p_bi)

    student_scores_with_temperature = torch.stack(list_of_student_scores_with_temperature)
    teacher_distributions_from_temperature = torch.stack(list_of_teacher_distributions)

    distillation_loss = cross_entropy_loss(student_scores_with_temperature, teacher_distributions_from_temperature)
    loss = distillation_loss + alpha * dual_encoder_loss
    return loss


def train(output_model_path: str,
          vilt_model_path: str,
          word2vec_model_path: str,
          image_encoder_backbone_model: str,
          num_epochs: int = 100,
          batch_size: int = 8,
          negative_batch_size: int = 4,
          learning_rate: float = 0.001,
          temperature=10,
          dataloader_num_worker=1
          ):
    """
    Train the model to have an image encoder that encodes into sparse embeddings matching the text encoder's outputs

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
        dev = 'cuda:0'
    else:
        dev = 'cpu'
    device = torch.device(dev)

    os.makedirs(output_model_path, exist_ok=True)
    image_encoder = ImageEncoder(backbone_model=image_encoder_backbone_model)
    text_encoder = TextEncoder(model_path=word2vec_model_path)
    vilt_model = get_vilt_model(load_path=vilt_model_path)

    if dev == 'cuda:0' and os.getenv('CHECK_CUDA_MEM_USAGE', 'False') == 'True':
        t = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
        r = torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024
        a = torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024
        print(f'Before moving models to GPU total_memory {t} GB, reserved {r} GB, allocated {a} GB')

    image_encoder.to(device)
    if dev == 'cuda:0' and os.getenv('CHECK_CUDA_MEM_USAGE', 'False') == 'True':
        t = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
        r = torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024
        a = torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024
        print(f'After moving image encoder model to GPU total_memory {t} GB, reserved {r} GB, allocated {a} GB')
    text_encoder.to(device)
    if dev == 'cuda:0' and os.getenv('CHECK_CUDA_MEM_USAGE', 'False') == 'True':
        t = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
        r = torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024
        a = torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024
        print(f'After moving text encoder model to GPU total_memory {t} GB, reserved {r} GB, allocated {a} GB')
    vilt_model.to(device)
    if dev == 'cuda:0' and os.getenv('CHECK_CUDA_MEM_USAGE', 'False') == 'True':
        t = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
        r = torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024
        a = torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024
        print(f'After moving models to GPU total_memory {t} GB, reserved {r} GB, allocated {a} GB')

    optimizer = torch.optim.Adam([{'params': image_encoder.parameters()}, {'params': text_encoder.parameters()}],
                                lr=learning_rate)
    optimizer.zero_grad()

    train_losses_epochs = []
    val_losses_epochs = []
    test_evals_epochs = []
    val_evals_epochs = []
    train_evals_epochs = []
    alpha = temperature * temperature * 0.001

    # run_evaluations(image_encoder, text_encoder, vilt_model,
    #                 batch_size, root=DATASET_ROOT_PATH,
    #                 split_root=DATASET_SPLIT_ROOT_PATH,
    #                 split='test')

    if dev == 'cuda:0' and os.getenv('CHECK_CUDA_MEM_USAGE', 'False') == 'True':
        t = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
        r = torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024
        a = torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024
        print(f'After run evaluations total_memory {t} GB, reserved {r} GB, allocated {a} GB')
    with Bar('Epochs', max=num_epochs, check_tty=False) as epoch_bar:

        for epoch in range(num_epochs):
            train_data_loader = get_captions_image_data_loader(root=DATASET_ROOT_PATH,
                                                               split_root=DATASET_SPLIT_ROOT_PATH,
                                                               split='train',
                                                               shuffle=True,
                                                               num_workers=dataloader_num_worker,
                                                               batch_size=batch_size,
                                                               collate_fn=collate)

            val_data_loader = get_captions_image_data_loader(root=DATASET_ROOT_PATH,
                                                             split_root=DATASET_SPLIT_ROOT_PATH,
                                                             split='val',
                                                             shuffle=True,
                                                             num_workers=dataloader_num_worker,
                                                             batch_size=batch_size,
                                                             collate_fn=collate)

            train_loss = []
            epoch_start = time.time()
            time_start = time.time()

            with Bar(f'Batch in epoch {epoch}', max=len(train_data_loader.dataset) / batch_size,
                     check_tty=False) as training_bar:

                for batch_id, (matching_filenames, images, captions) in enumerate(train_data_loader):

                    image_tensors = []
                    for i in images:
                        image_tensors.append(dual_encoder_transform(i))

                    image_encoder.train()
                    image_encoder.feature_extractor.train()
                    image_encoder.common_space_embedding.train()
                    text_encoder.train()
                    text_encoder.word_embd.train()
                    text_encoder.fc1.train()
                    text_encoder.fc2.train()
                    optimizer.zero_grad()
                    original_images = images
                    if dev == 'cuda:0' and os.getenv('CHECK_CUDA_MEM_USAGE', 'False') == 'True':
                        t = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
                        r = torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024
                        a = torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024
                        print(
                            f'Before stacking {len(image_tensors)} image tensors to GPU total_memory {t} GB, reserved {r} GB, allocated {a} GB')
                    image_tensors = torch.stack(image_tensors).to(device)
                    if dev == 'cuda:0' and os.getenv('CHECK_CUDA_MEM_USAGE', 'False') == 'True':
                        t = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
                        r = torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024
                        a = torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024
                        print(
                            f'Before encoding {len(image_tensors)} images to GPU total_memory {t} GB, reserved {r} GB, allocated {a} GB')
                    images_embeddings = image_encoder(image_tensors).to(device)
                    if dev == 'cuda:0' and os.getenv('CHECK_CUDA_MEM_USAGE', 'False') == 'True':
                        t = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
                        r = torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024
                        a = torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024
                        print(
                            f'Before encoding {len(captions)} texts to GPU total_memory {t} GB, reserved {r} GB, allocated {a} GB')
                    texts_embeddings = text_encoder(captions).to(device)
                    if dev == 'cuda:0' and os.getenv('CHECK_CUDA_MEM_USAGE', 'False') == 'True':
                        t = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
                        r = torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024
                        a = torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024
                        print(
                            f'After computing images and texts embeddings for all batch total_memory {t} GB, reserved {r} GB, allocated {a} GB')
                    loss = compute_loss(images, captions, original_images, vilt_model, images_embeddings,
                                        texts_embeddings,
                                        negative_batch_size, temperature, alpha)

                    # before_update_image = {}
                    # for param_name, param in image_encoder.named_parameters():
                    #     before_update_image[param_name] = 100*param
                    # before_update_text = {}
                    # for param_name, param in text_encoder.named_parameters():
                    #     before_update_text[param_name] = 100*param
                    loss.backward()
                    train_loss.append(loss.item())
                    optimizer.step()
                    # after_update_image = {}
                    # for param_name, param in image_encoder.named_parameters():
                    #     after_update_image[param_name] = 100*param
                    # after_update_text = {}
                    # for param_name, param in text_encoder.named_parameters():
                    #     after_update_text[param_name] = 100*param
                    #
                    # for param_after_name, param_after in after_update_image.items():
                    #     param_before = before_update_image[param_after_name]
                    #     diff = param_after - param_before
                    #     no_zeros = torch.count_nonzero(diff)
                    #     if no_zeros == 0:
                    #         print(f'\nParam image_encoder.{param_after_name} is not updated')
                    # for param_after_name, param_after in after_update_text.items():
                    #     param_before = before_update_text[param_after_name]
                    #     diff = param_after - param_before
                    #     no_zeros = torch.count_nonzero(diff)
                    #     if no_zeros == 0:
                    #         print(f'\nParam text_encoder.{param_after_name} is not updated')

                    if batch_id % 200 == 0:
                        print(colored(
                            f'\n[{batch_id}] \t training loss:\t {np.mean(np.array(train_loss))} \t time elapsed:\t {time.time() - time_start} s',
                            'green'))
                        train_loss.clear()
                        time_start = time.time()
                    if batch_id % 1000 == 0 and batch_id != 0:
                        torch.save(image_encoder.state_dict(),
                                   output_model_path + '/model-inter-' + str(epoch + 1) + '-' + str(
                                       batch_id) + '-image.pt')
                        torch.save(text_encoder.state_dict(),
                                   output_model_path + '/model-inter-' + str(epoch + 1) + '-' + str(
                                       batch_id) + '-text.pt')
                    if batch_id % 200 == 0 and batch_id != 0:
                        val_loss = validation_loop(image_encoder, text_encoder, vilt_model, val_data_loader,
                                                   negative_batch_size, temperature, alpha)
                        print(colored(
                            f'\n[{batch_id}]\tvalidation loss:\t{np.mean(np.array(val_loss))}\ttime elapsed:\t{time.time() - time_start} s',
                            'yellow'))

                        time_start = time.time()
                    training_bar.next()
                image_file_path_dump = output_model_path + '/model-inter-' + str(epoch + 1) + '-final-image.pt'
                text_file_path_dump = output_model_path + '/model-inter-' + str(epoch + 1) + '-final-text.pt'

                print(colored(
                    f'\nEpoch finished in {time.time() - epoch_start} s, saving model to {image_file_path_dump}',
                    'green'))
                val_loss = validation_loop(image_encoder, text_encoder, vilt_model, val_data_loader,
                                           negative_batch_size, temperature, alpha)
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

                torch.save(image_encoder.state_dict(), image_file_path_dump)
                torch.save(text_encoder.state_dict(), text_file_path_dump)

            with open(f'train_loss-{epoch}', 'wb') as f:
                pickle.dump(train_loss, f)

            if epoch % 1 == 0:
                test_evaluations = run_evaluations(image_encoder, text_encoder, vilt_model,
                                                   batch_size, root=DATASET_ROOT_PATH,
                                                   split_root=DATASET_SPLIT_ROOT_PATH,
                                                   split='test')
                test_evals_epochs.append(test_evaluations)

                val_evaluations = run_evaluations(image_encoder, text_encoder, vilt_model,
                                                  batch_size, root=DATASET_ROOT_PATH,
                                                  split_root=DATASET_SPLIT_ROOT_PATH,
                                                  split='val')
                val_evals_epochs.append(val_evaluations)

                # train_evaluations = run_evaluations(image_encoder, text_encoder, vilt_model,
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
    train(
        output_model_path=IMAGE_EMBEDDING_BASE_PATH,
        word2vec_model_path=TEXT_WORD2_VEC_MODEL_PATH,
        image_encoder_backbone_model='resnet50',
        vilt_model_path=VILT_BASE_MODEL_LOAD_PATH,
        batch_size=8,
        negative_batch_size=4,
        dataloader_num_worker=1)


if __name__ == '__main__':
    import sys

    main(*sys.argv)
