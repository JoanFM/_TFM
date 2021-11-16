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
  python src/train/train.py
```

You should see logs informing about the evolution of `training` and `validation` loss. Also you should see evaluation results after each epoch.

![image](https://user-images.githubusercontent.com/19825685/142002361-ea3b5753-66e5-4170-a60b-ce8c7c7e3360.png)

![image](https://user-images.githubusercontent.com/19825685/142002516-3e13b3cb-e95f-4f76-a0ab-9232f2dde43e.png)

You can also run evaluation on a single model by running the following command:


```bash
  python src/train/train.py evaluate {SPLIT} {MODEL_PATH}
```

```bash
  python src/train/train.py evaluate test /hdd/master/tfm/output_models-test/model-inter-9-final.pt
```

## Use the model with a frontend:

In order to expose the model and play with it, some configuration must be set. For this purpose we use `Jina` and `Streamlit`:

- https://github.com/jina-ai/jina
- https://github.com/streamlit

You can set the following `environment` variables if the default values do not fit.

- INDEX_FILE_PATH (The path where the embedding index will be stored)
- IMAGE_EMBEDDING_MODEL_PATH (Model to expose for indexing, the path where the state_dict of the model is stored)
- TEXT_EMBEDDING_VECTORIZER_PATH (Text embedding model to expose for querying, the path where the CountVectorizer is stored of the model is stored)
- DATASET_ROOT_PATH (The root path where the flickr30k dataset is found)
- DATASET_SPLIT_ROOT_PATH (The root path where the flickr30k entities per split is kept)
- DATASET_SPLIT (The split to use for indexing (test, val, train))

```python
# File where Jina will store the index
INDEX_FILE_PATH = os.getenv('INDEX_FILE_PATH', 'tmp/sparse_index')
# Model to expose for indexing, the path where the state_dict of the model is stored
IMAGE_EMBEDDING_MODEL_PATH = os.getenv('IMAGE_EMBEDDING_MODEL_PATH', '/hdd/master/tfm/output-image-encoders/model-inter-9-final.pt')
# Text embedding model to expose for querying, the path where the CountVectorizer is stored of the model is stored
TEXT_EMBEDDING_VECTORIZER_PATH = os.getenv('TEXT_EMBEDDING_VECTORIZER_PATH', '/hdd/master/tfm/vectorizers/vectorizer_tokenizer_stop_words_all_words_filtered_10.pkl')
# The root path where the flickr30k dataset is found
DATASET_ROOT_PATH = os.getenv('DATASET_ROOT_PATH', '/hdd/master/tfm/flickr30k_images')
# The root path where the flickr30k entities per split is kept
DATASET_SPLIT_ROOT_PATH = os.getenv('DATASET_SPLIT_ROOT_PATH', '/hdd/master/tfm/flickr30k_images/flickr30k_entities')
# The split to use for indexing (test, val, train)
DATASET_SPLIT = os.getenv('DATASET_SPLIT', 'test')
```

```bash
  python src/search_engine/jina_application.py
```

Once the indexing is done and the Query is ready to receive, you should see something like this:

```
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:45678 (Press CTRL+C to quit)
        gateway@22946[D]:ready and listening
           Flow@22946[I]:🎉 Flow is ready to use!                                                   
	🔗 Protocol: 		HTTP
	🏠 Local access:	0.0.0.0:45678
	🔒 Private network:	192.168.1.187:45678
	🌐 Public address:	212.231.186.65:45678
	💬 Swagger UI:		http://localhost:45678/docs
	📚 Redoc:		http://localhost:45678/redoc
           Flow@22946[D]:3 Pods (i.e. 3 Peas) are running in this Flow

```

At this point, the Jina Flow is ready and we can start the Frontend that will connect to Jina, so let's open a different tab and run:

- Note: You may want to check the configuration at `src/search_engine/frontend/config.py`

```bash
  python -m streamlit run src/search_engine/frontend/app.py
```


You should see:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.187:8501
```

Then you can open the browser and play with the model. 

You can do 2 things, enter a text, and see the results obtained, or enter one of the texts from the groundtruth provided, which will also print the expected match.

![image](https://user-images.githubusercontent.com/19825685/142001792-8d770665-4985-4f85-8120-db0e84ee494a.png)

![image](https://user-images.githubusercontent.com/19825685/142001136-00c5343e-0b60-4ca8-98f0-b08397708920.png)

![image](https://user-images.githubusercontent.com/19825685/142001866-0c395880-b7eb-41e6-8c0d-112b67950b92.png)

![image](https://user-images.githubusercontent.com/19825685/142001446-37d83ef3-a33c-4164-b27e-bfc75f286712.png)

