from src.search_engine.sparse.jina_application import DATASET_SPLIT_ROOT_PATH, DATASET_ROOT_PATH, DATASET_SPLIT

# Text search
text_endpoint = "http://0.0.0.0:45678/search"

# Image search
image_endpoint = "http://0.0.0.0:45678/search"
images_path = DATASET_ROOT_PATH + '/flickr30k_images'
split = DATASET_SPLIT
split_root = DATASET_SPLIT_ROOT_PATH
image_size = 128  # How big to render images

# General
top_k = 10

