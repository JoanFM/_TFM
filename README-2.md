# TFM_Sparse_Embeddings

This repository contains the code for the implementation of the Master Thesis related to learning Sparse Representations for Images in CrossModal Search.

The main idea of the project is to merge the goods and the bads of dense and sparse representations for image search, taking inspiration from traditinal text search.

- Dense embeddings extracted from Neural Networks tend to excel at extracting semantic meaning and concepts from images or text. This is currently the only technique widely used
for Image Retrieval, even in Image2Image retrieval or in Text2Image.

- Sparse embeddings have been successfully used in text search and have the advantage of being more robust, easier to interpret and generally faster and more scalable.

In traditional text search, `sentences` are tokenized and represented as a sparse array of the size of the vocabulary. What this project tries to do is to use a dataset of images
and its related captions to learn sparse representations from the image features matching the words or tokens from the captions.

Then this sparse representations for images can be used to do Text2Image retrieval using `TFIDF` techniques where the frequency of a `term` is proxied by the activation of that sparse feature in 
a given image sparse embedding.

## Dataset

The dataset used for this project is the Flickr30K that can be obtained from this link.

https://www.kaggle.com/hsankesara/flickr-image-dataset

## Train the model

In order to train the model, some configuration must be set.

You can set the following `environment` variables if the default values do not fit.

- IMAGE_EMBEDDING_BASE_PATH (The base directory where models are stored after every epoch)
- TEXT_EMBEDDING_VECTORIZER_BASE_PATH (The base directory where models where CountVectorizer are stored. Different CountVectorizers correspond to different preprocessings of the corpus)
- DATASET_ROOT_PATH (The root path where the flickr30k dataset is found)
- DATASET_SPLIT_ROOT_PATH (The root path where the flickr30k entities per split is kept)


```python
# The base directory where models are stored after every epoch
IMAGE_EMBEDDING_BASE_PATH = os.getenv('IMAGE_EMBEDDING_BASE_PATH', '/hdd/master/tfm/output-image-encoders')

# The base directory where models where CountVectorizer are stored. Different CountVectorizers correspond to
# different preprocessings of the corpus
TEXT_EMBEDDING_VECTORIZER_BASE_PATH = os.getenv('TEXT_EMBEDDING_VECTORIZER_PATH',
                                                '/hdd/master/tfm/vectorizers')

# The root path where the flickr30k dataset is found
DATASET_ROOT_PATH = os.getenv('DATASET_ROOT_PATH', '/hdd/master/tfm/flickr30k_images')

# The root path where the flickr30k entities per split is kept
DATASET_SPLIT_ROOT_PATH = os.getenv('DATASET_SPLIT_ROOT_PATH', '/hdd/master/tfm/flickr30k_images/flickr30k_entities')
```

Then you can run the following command: 

```bash
  python3.7 src/train/train.py
```

You should see logs informing about the evolution of `training` and `validation` loss. Also you should see evaluation results after each epoch.

## Use the model with a frontend:

