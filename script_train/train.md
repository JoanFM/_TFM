# How to run code to train the model

## Clone project

Clone the project of the Master's Thesis

```bash
git clone https://{your_user_name}:{your_token_password}@github.com/JoanFM/TFM_Sparse_Embeddings
```


## Install ViLT model

Clone and install the ViLT Slow model package

```bash
git clone https://github.com/dandelin/ViLT
pip install -r ViLT/requirements.txt
cd ViLT;
pip install e ViLT/.
```

## Install extra packages

```bash
pip install torchvision==0.9.1
pip install scipy==1.6.1
pip install numpy==1.19.5
pip install scikit-learn==0.22.2.post1
pip install progress==1.5
pip install pandas==1.0.3
pip install spacy==3.0.6
pip install gensim==4.1.2
pip install torch==1.10.0
```

### Download Kaggle dataset

With the kaggle client installed and the proper username, download flickr dataset and clean a badly formated user:

```bash
kaggle datasets download hsankesara/flickr-image-dataset -d flickr-image-dataset
unzip flickr-image-dataset
grep -rl "2199200615.jpg| 4   A dog runs across the grass ." flickr30k_images/results.csv | xargs sed -i 's/4   A dog runs across the grass ./4| A dog runs across the grass ./g'
```

Organize the files properly and download the entities
```
cd flickr30k_images/
mv flickr30k_images/ flickr30k-images
git clone https://github.com/BryanPlummer/flickr30k_entities
cd ..
```

### Download pretrained model checkpoint

```bash
wget https://github.com/dandelin/ViLT/releases/download/200k/vilt_irtr_f30k.ckpt
```

### Do some needed tweaks

```bash
pip uninstall -y torchtext
pip uninstall -y keras
pip install tensorflow==1.15.0
pip install keras==2.2.5
pip install keras --upgrade
```

### Download spacy model

```bash
python -m spacy download en_core_web_sm
```

### Make sure torch 1.10.0 is still installed and gensim 4.1.2

```bash
pip show torch
pip show gensim
```

### Run the code to train models. It will dump one model per epoch for image and for text.

Assume run from the root of the repository

```python
import os
import torch

torch.cuda.empty_cache()
IMAGE_EMBEDDING_BASE_PATH = os.getenv('IMAGE_EMBEDDING_BASE_PATH', 'output-image-encoders')
# The word2vec model base
TEXT_WORD2_VEC_MODEL_PATH = os.getenv('TEXT_WORD2_VEC_MODEL_PATH', 'filtered_f30k_word2vec.model')

VILT_BASE_MODEL_LOAD_PATH = os.getenv('VILT_BASE_MODEL_LOAD_PATH', 'vilt_irtr_f30k.ckpt')

# The root path where the flickr30k dataset is found
DATASET_ROOT_PATH = os.getenv('DATASET_ROOT_PATH', '/hdd/master/tfm/flickr30k_images')
# The root path where the flickr30k entities per split is kept
DATASET_SPLIT_ROOT_PATH = os.getenv('DATASET_SPLIT_ROOT_PATH', '/hdd/master/tfm/flickr30k_images/flickr30k_entities')


IMAGE_EMBEDDING_BASE_PATH = os.getenv('IMAGE_EMBEDDING_BASE_PATH',  '/content/gdrive/MyDrive/MasterCVC/Thesis/models')
TEXT_WORD2_VEC_MODEL_PATH = os.getenv('TEXT_WORD2_VEC_MODEL_PATH', 'filtered_f30k_word2vec.model')

VILT_BASE_MODEL_LOAD_PATH = os.getenv('VILT_BASE_MODEL_LOAD_PATH', '../vilt_irtr_f30k.ckpt')

os.environ['DATASET_ROOT_PATH'] = '../flickr30k_images' # path to the files downloaded from kaggle
os.environ['DATASET_SPLIT_ROOT_PATH'] = '../flickr30k_images/flickr30k_entities'

from src.train.dual_distillation.train import train
from src.model.cached_scores import CachedScores

val_cache_scores = CachedScores('src/model/slow_scores/val.th')
test_cache_scores = CachedScores('src/model/slow_scores/test.th')

train(
    output_model_path='output-image-encoders',
    word2vec_model_path='filtered_f30k_word2vec.model',
    image_encoder_backbone_model='resnet50',
    vilt_model_path='vilt_irtr_f30k.ckpt', # path to the vilt model downloaded in a previous step
    batch_size=128, # try to make it as large as possible
    negative_batch_size=4, # if beta = 0, no distillation and therefore does not impact
    alpha=1,
    beta=0,
    learning_rate=0.001,
    num_warmup_steps=10000, # this should be adjusted with the batch size, I was expecting to have around 2 epochs of `warmup`
    num_training_steps=500000, # this should set to num training steps
    dataloader_num_worker=1,
    temperature=10, # since beta 0 does not affect
    reduction_in_loss='mean',
    cache_scores={'train': None, 'val': val_cache_scores, 'test': test_cache_scores}
)
```
