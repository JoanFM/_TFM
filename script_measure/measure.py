import os
import torch

torch.cuda.empty_cache()
IMAGE_EMBEDDING_BASE_PATH = os.getenv('IMAGE_EMBEDDING_BASE_PATH', 'output-image-encoders')
# The word2vec model base
TEXT_WORD2_VEC_MODEL_PATH = os.getenv('TEXT_WORD2_VEC_MODEL_PATH', 'filtered_coco_word2vec_with_stop_words.model')

VILT_BASE_MODEL_LOAD_PATH = os.getenv('VILT_BASE_MODEL_LOAD_PATH', '/data/users/jfontanals/vilt_irtr_f30k.ckpt')

# The root path where the flickr30k dataset is found
DATASET_ROOT_PATH = os.getenv('DATASET_ROOT_PATH', '/datatmp/users/jfontanals/flickr30k_images')
# The root path where the flickr30k entities per split is kept
DATASET_SPLIT_ROOT_PATH = os.getenv('DATASET_SPLIT_ROOT_PATH', '/datatmp/users/jfontanals/flickr30k_images/flickr30k_entities')


os.environ['DATASET_ROOT_PATH'] = '/data/users/jfontanals/datasets/coco-2014' # path to the files downloaded from kaggle
os.environ['DATASET_SPLIT_ROOT_PATH'] = '/data/users/jfontanals/flickr30k_images/flickr30k_entities'
os.environ['FLICKR_DATASET_ROOT_PATH'] = '/data/users/jfontanals/flickr30k_images' # path to the files downloaded from kaggle
os.environ['FLICKR_DATASET_SPLIT_ROOT_PATH'] = '/data/users/jfontanals/flickr30k_images/flickr30k_entities'
os.environ['VILT_BASE_MODEL_LOAD_PATH'] = '/data/users/jfontanals/vilt_irtr_f30k.ckpt'

from src.train.dual_distillation.train import main_display

main_display()
