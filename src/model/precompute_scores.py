import os
import torch
import time

from src.model.vilt_model import get_vilt_model
from vilt.transforms.pixelbert import pixelbert_transform
from src.dataset import get_image_data_loader, get_captions_data_loader
from progress.bar import Bar

vilt_transform = pixelbert_transform(384)

# The base directory where models are stored after every epoch
IMAGE_EMBEDDING_BASE_PATH = os.getenv('IMAGE_EMBEDDING_BASE_PATH', '/hdd/master/tfm/output-image-encoders')
# The word2vec model base
TEXT_WORD2_VEC_MODEL_PATH = os.getenv('TEXT_WORD2_VEC_MODEL_PATH', 'filtered_f30k_word2vec.model')

VILT_BASE_MODEL_LOAD_PATH = os.getenv('VILT_BASE_MODEL_LOAD_PATH', 'vilt_irtr_f30k.ckpt')

# The root path where the flickr30k dataset is found
DATASET_ROOT_PATH = os.getenv('DATASET_ROOT_PATH', '/hdd/master/tfm/flickr30k_images')
# The root path where the flickr30k entities per split is kept
DATASET_SPLIT_ROOT_PATH = os.getenv('DATASET_SPLIT_ROOT_PATH', '/hdd/master/tfm/flickr30k_images/flickr30k_entities')


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


def _get_num_captions_in_partition(partition_to_compute, number_partitions, total_number_of_captions):
    processed_captions = 0
    for i in range(partition_to_compute):
        processed_captions += total_number_of_captions // number_partitions

    left_captions = total_number_of_captions - processed_captions
    if partition_to_compute + 1 == number_partitions:
        return left_captions
    else:
        return total_number_of_captions // number_partitions


def precompute_scores(output_file_path: str, vilt_model_path: str = VILT_BASE_MODEL_LOAD_PATH, split: str = 'val',
                      batch_size=4, number_partitions=1, partition_to_compute=0):
    with torch.no_grad():
        if torch.cuda.is_available():
            dev = 'cuda:0'
        else:
            dev = 'cpu'
        device = torch.device(dev)
        vilt_model = get_vilt_model(load_path=vilt_model_path)
        vilt_model.to(device)
        with torch.no_grad():
            text_data_loader = get_captions_data_loader(root=DATASET_ROOT_PATH, split_root=DATASET_SPLIT_ROOT_PATH,
                                                        split=split,
                                                        batch_size=1, collate_fn=collate_captions,
                                                        shuffle=False)

            image_data_loader = get_image_data_loader(root=DATASET_ROOT_PATH, split_root=DATASET_SPLIT_ROOT_PATH,
                                                      split=split, batch_size=batch_size, shuffle=False,
                                                      collate_fn=collate_images, force_transform_to_none=True)

            total_num_captions = len(text_data_loader.dataset)
            total_num_images = len(image_data_loader.dataset)

            partition = 0

            scores = torch.zeros(
                _get_num_captions_in_partition(partition_to_compute, number_partitions, total_num_captions),
                total_num_images)
            with Bar(f'Caption progress', check_tty=False,
                     max=len(text_data_loader)) as caption_bar:
                processed_captions = 0
                offset_caption_in_partition = 0
                for caption_batch_id, (caption_indices, filenames, captions) in enumerate(text_data_loader):
                    if partition > partition_to_compute:
                        break
                    for c_id, caption in zip(caption_indices, captions):
                        start = time.time()
                        if partition == partition_to_compute:
                            with Bar(f'Image progress', check_tty=False,
                                     max=len(image_data_loader)) as image_bar:
                                for image_batch_id, (images_indices, filenames, images) in enumerate(image_data_loader):
                                    slow_batch_time = time.time()
                                    pixel_bert_transformed_images = []

                                    for i in images:
                                        pixel_bert_transformed_images.append(vilt_transform(i).to(device))

                                    slow_scores = vilt_model.score_query_vs_images(caption,
                                                                                   pixel_bert_transformed_images)
                                    scores[offset_caption_in_partition, images_indices] = slow_scores
                                    print(f' Scoring {batch_size} images for caption {c_id} took {time.time() - slow_batch_time}s')

                                image_bar.next()
                        offset_caption_in_partition += 1
                        processed_captions += 1
                        print(f' Scoring every image for caption {c_id} took {time.time() - start}s')
                        if number_partitions > 1 and (partition + 1 < number_partitions) and (
                                processed_captions // (total_num_captions // number_partitions) > partition):
                            offset_caption_in_partition = 0
                            partition += 1
                        if partition > partition_to_compute:
                            break
                    caption_bar.next()
        torch.save(scores, f'{output_file_path.split(".th")[0]}-{partition_to_compute}.th')


def precompute_scores_inverted(output_file_path: str, vilt_model_path: str = VILT_BASE_MODEL_LOAD_PATH,
                               split: str = 'val',
                               batch_size=4, number_partitions=1, partition_to_compute=0):
    with torch.no_grad():
        if torch.cuda.is_available():
            dev = 'cuda:0'
        else:
            dev = 'cpu'
        device = torch.device(dev)
        vilt_model = get_vilt_model(load_path=vilt_model_path)
        vilt_model.to(device)
        with torch.no_grad():
            text_data_loader = get_captions_data_loader(root=DATASET_ROOT_PATH, split_root=DATASET_SPLIT_ROOT_PATH,
                                                        split=split,
                                                        batch_size=batch_size, collate_fn=collate_captions,
                                                        shuffle=False)

            image_data_loader = get_image_data_loader(root=DATASET_ROOT_PATH, split_root=DATASET_SPLIT_ROOT_PATH,
                                                      split=split, batch_size=1, shuffle=False,
                                                      collate_fn=collate_images, force_transform_to_none=True)

            total_num_captions = len(text_data_loader.dataset)
            total_num_images = len(image_data_loader.dataset)

            partition = 0

            scores = torch.zeros(
                _get_num_captions_in_partition(partition_to_compute, number_partitions, total_num_images),
                total_num_captions).to(device)
            print(f' scores.shape {scores.shape}')
            with Bar(f'Image progress', check_tty=False,
                     max=len(image_data_loader)) as image_bar:
                processed_images = 0
                offset_image_in_partition = 0
                for image_batch_id, (images_indices, filenames, images) in enumerate(image_data_loader):
                    if partition > partition_to_compute:
                        break
                    for image_id, image in zip(images_indices, images):
                        start = time.time()
                        pixel_bert_transformed_image = vilt_transform(image).to(device)
                        if partition == partition_to_compute:
                            with Bar(f'Caption progress', check_tty=False,
                                     max=len(text_data_loader)) as caption_bar:
                                for caption_batch_id, (caption_ids, filenames, captions) in enumerate(text_data_loader):
                                    slow_batch_time = time.time()
                                    slow_scores = vilt_model.score_image_vs_texts(pixel_bert_transformed_image, captions)
                                    scores[offset_image_in_partition, caption_ids] = slow_scores
                                    print(f' Scoring {batch_size} captions for image {image_id} took {time.time() - slow_batch_time}s')
                                caption_bar.next()
                        offset_image_in_partition += 1
                        processed_images += 1
                        print(f' Scoring every caption for image {image_id} took {time.time() - start}s')
                        if number_partitions > 1 and (partition + 1 < number_partitions) and (
                                processed_images // (total_num_images // number_partitions) > partition):
                            offset_image_in_partition = 0
                            partition += 1
                        if partition > partition_to_compute:
                            break
                    image_bar.next()
        torch.save(scores, f'{output_file_path.split(".th")[0]}-{partition_to_compute}.th')


def main(*args, **kwargs):
    for i in range(10):
        precompute_scores_inverted(
            output_file_path='val-inverted.th',
            vilt_model_path=VILT_BASE_MODEL_LOAD_PATH,
            split='val',
            batch_size=8,
            number_partitions=10,
            partition_to_compute=i)


if __name__ == '__main__':
    import sys

    main(*sys.argv)
