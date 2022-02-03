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


def precompute_scores(output_file_path: str, vilt_model_path: str = VILT_BASE_MODEL_LOAD_PATH, split: str = 'val',
                      batch_size=4):
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
                                                      split=split, batch_size=batch_size, shuffle=False,
                                                      collate_fn=collate_images, force_transform_to_none=True)

            scores = torch.zeros(len(text_data_loader.dataset), len(image_data_loader.dataset))

            with Bar(f'Caption progress', check_tty=False,
                     max=len(text_data_loader)) as caption_bar:

                for caption_batch_id, (caption_indices, filenames, captions) in enumerate(text_data_loader):

                    for c_id, caption in zip(caption_indices, captions):
                        start = time.time()
                        with Bar(f'Image progress', check_tty=False,
                                 max=len(image_data_loader)) as image_bar:
                            for image_batch_id, (images_indices, filenames, images) in enumerate(image_data_loader):
                                pixel_bert_transformed_images = []

                                for i in images:
                                    pixel_bert_transformed_images.append(vilt_transform(i).to(device))

                                slow_scores = vilt_model.score_query_vs_images(caption, pixel_bert_transformed_images)
                                for image_ind, sc in zip(images_indices, slow_scores):
                                    scores[c_id, image_ind] = sc.cpu().item()
                            image_bar.next()
                        print(f' Scoring every image for caption {c_id} took {time.time() - start}s')
                    caption_bar.next()
        torch.save(scores, output_file_path)


def main(*args, **kwargs):
    precompute_scores(
        output_file_path='val.th',
        vilt_model_path=VILT_BASE_MODEL_LOAD_PATH,
        split='val',
        batch_size=4)


if __name__ == '__main__':
    import sys

    main(*sys.argv)
