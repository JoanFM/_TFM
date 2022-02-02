import os
import torch

from src.model.vilt_model import get_vilt_model
from vilt.transforms.pixelbert import pixelbert_transform
from src.dataset import get_captions_image_data_loader



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


def precompute_scores(output_file_path: str, vilt_model_path: str = VILT_BASE_MODEL_LOAD_PATH, split: str = 'val'):
    print(f' hey joan')
    with torch.no_grad():
        if torch.cuda.is_available():
            dev = 'cuda:0'
        else:
            dev = 'cpu'
        device = torch.device(dev)
        vilt_model = get_vilt_model(load_path=vilt_model_path)
        print(f' hey joan HEY')
        vilt_model.to(device)

        data_loader = get_captions_image_data_loader(root=DATASET_ROOT_PATH,
                                                     split_root=DATASET_SPLIT_ROOT_PATH,
                                                     split=split,
                                                     shuffle=False,
                                                     num_workers=1,
                                                     batch_size=4,
                                                     collate_fn=collate,
                                                     batch_sampling=True)

        scores = torch.zeros(len(data_loader.dataset), data_loader.dataset.num_images)
        for batch_id, (caption_indices, images_indices, matching_filenames, images, captions) in enumerate(data_loader):
            transformed_images = [vilt_transform(image).to(device) for image in images]
            for c_id, caption in enumerate(captions):
                slow_scores = vilt_model.score_query_vs_images(caption, transformed_images)
                for i, sc in enumerate(slow_scores):
                    scores[caption_indices[c_id], images_indices[i]] = sc.cpu().item()
            if batch_id > 5:
                break

        torch.save(scores, output_file_path)


def main(*args, **kwargs):
    precompute_scores(
        output_file_path='val.th',
        vilt_model_path=VILT_BASE_MODEL_LOAD_PATH,
        split='val')


if __name__ == '__main__':
    print(f' hahaha')
    import sys

    main(*sys.argv)
