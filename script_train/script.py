import os
import torch

torch.cuda.empty_cache()
IMAGE_EMBEDDING_BASE_PATH = os.getenv('IMAGE_EMBEDDING_BASE_PATH', 'output-image-encoders')
# The word2vec model base
TEXT_WORD2_VEC_MODEL_PATH = os.getenv('TEXT_WORD2_VEC_MODEL_PATH', 'filtered_f30k_word2vec.model')

VILT_BASE_MODEL_LOAD_PATH = os.getenv('VILT_BASE_MODEL_LOAD_PATH', '/datatmp/users/jfontanals/vilt_irtr_f30k.ckpt')

# The root path where the flickr30k dataset is found
DATASET_ROOT_PATH = os.getenv('DATASET_ROOT_PATH', '/datatmp/users/jfontanals/flickr30k_images')
# The root path where the flickr30k entities per split is kept
DATASET_SPLIT_ROOT_PATH = os.getenv('DATASET_SPLIT_ROOT_PATH', '/datatmp/users/jfontanals/flickr30k_images/flickr30k_entities')


os.environ['DATASET_ROOT_PATH'] = '/datatmp/users/jfontanals/flickr30k_images' # path to the files downloaded from kaggle
os.environ['DATASET_SPLIT_ROOT_PATH'] = '/datatmp/users/jfontanals/flickr30k_images/flickr30k_entities'

from src.train.dual_distillation.train import train
from src.model.cached_scores import CachedScores

val_cache_scores = CachedScores('src/model/slow_scores/val.th')
test_cache_scores = CachedScores('src/model/slow_scores/test.th')

train(
    output_model_path='output-image-encoders',
    word2vec_model_path='filtered_f30k_word2vec.model',
    image_encoder_backbone_model='resnet50',
    vilt_model_path='/datatmp/users/jfontanals/vilt_irtr_f30k.ckpt', # path to the vilt model downloaded in a previous step
    batch_size=128, # try to make it as large as possible
    negative_batch_size=4, # if beta = 0, no distillation and therefore does not impact
    alpha=1,
    beta=0,
    learning_rate=0.001,
    num_warmup_steps=500, # this should be adjusted with the batch size, I was expecting to have around 2 epochs of
    # `warmup`
    num_training_steps=500000, # this should set to num training steps
    dataloader_num_worker=1,
    temperature=10, # since beta 0 does not affect
    reduction_in_loss='mean',
    cache_scores={'train': None, 'val': val_cache_scores, 'test': test_cache_scores}
)