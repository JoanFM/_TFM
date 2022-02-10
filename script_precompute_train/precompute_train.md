# How to run code to precompute scores 

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

### Download spacy model

```bash
python -m spacy download en_core_web_sm
```

### Run the code to precompute scores

In python run the following script, from the root of the `TFM` project.

This code will run the computation of scores separated in 10 steps (to see some progress).

It will create files `train-0.th`, `train-1.th`, etc ... 

```python
import os
import torch

torch.cuda.empty_cache()

IMAGE_EMBEDDING_BASE_PATH = os.getenv('IMAGE_EMBEDDING_BASE_PATH', 'output-encoders')
# The word2vec model base
TEXT_WORD2_VEC_MODEL_PATH = os.getenv('TEXT_WORD2_VEC_MODEL_PATH', 'filtered_f30k_word2vec.model')

VILT_BASE_MODEL_LOAD_PATH = os.getenv('VILT_BASE_MODEL_LOAD_PATH', '../vilt_irtr_f30k.ckpt')

os.environ['DATASET_ROOT_PATH'] = '../flickr30k_images' 
os.environ['DATASET_SPLIT_ROOT_PATH'] = '../flickr30k_images/flickr30k_entities'

from src.model.precompute_scores import precompute_scores

output_file_path = 'train.th'

for i in range(0, 10):
  precompute_scores(
              output_file_path=output_file_path,
              vilt_model_path=VILT_BASE_MODEL_LOAD_PATH,
              split='train',
              batch_size=128,
              number_partitions=10,
              partition_to_compute=i)
```